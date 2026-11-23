"""Id-stable IoU tracker for RF-DETR detections.

RF-DETR produces boxes but no track ids, so this module assigns ids before the
zone-transition counter sees the observations. It is a small SORT-style tracker:
Hungarian matching on predicted boxes (``lap``) with a constant-velocity model
and a centre-distance gate. Tracks stay pending for ``max_age`` missed frames,
so a CCTV dropout of ~30 frames is healed with the *same* id when detection
resumes. Only real detections are returned to the counter; pending predictions
are never fed back as sightings, which would otherwise move a bag across the
count line without any visual evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import lap
import numpy as np

Box = tuple[float, float, float, float]


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _diagonal(box: Box) -> float:
    return max(float(np.hypot(box[2] - box[0], box[3] - box[1])), 1e-6)


@dataclass
class _Track:
    track_id: int
    box: Box  # last *confirmed* box, never a prediction
    vx: float = 0.0  # confirmed-box centre velocity, box units per frame
    vy: float = 0.0
    missed: int = 0  # consecutive frames with no detection before this frame


class IoUTracker:
    def __init__(
        self,
        match_iou: float = 0.5,
        max_age: int = 30,
        max_center_dist: float = 1.0,
        velocity_smoothing: float = 0.7,
    ):
        if max_age < 0:
            raise ValueError("max_age must be non-negative")
        if not 0.0 <= velocity_smoothing <= 1.0:
            raise ValueError("velocity_smoothing must be in 0..1")
        self.match_iou = float(match_iou)
        self.max_age = int(max_age)
        self.max_center_dist = float(max_center_dist)
        self.velocity_smoothing = float(velocity_smoothing)
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def pending_ids(self) -> set[int]:
        """Ids still owned by a track, including temporarily missed ones."""
        return set(self._tracks)

    def predicted_box(self, t: _Track, steps: int | None = None) -> Box:
        """Where a confirmed box is expected now (used only for matching)."""
        n = t.missed if steps is None else steps
        return (
            t.box[0] + t.vx * n,
            t.box[1] + t.vy * n,
            t.box[2] + t.vx * n,
            t.box[3] + t.vy * n,
        )

    def _match_cost(self, t: _Track, box: Box) -> tuple[float, bool]:
        predicted = self.predicted_box(t, t.missed + 1)
        iou = box_iou(predicted, box)
        tcx, tcy = _center(predicted)
        bcx, bcy = _center(box)
        dist = float(np.hypot(tcx - bcx, tcy - bcy)) / _diagonal(box)
        # A strong overlap is enough on a live track. After a dropout, allow a
        # fast-moving bag to be re-matched near its predicted position.
        acceptable = iou >= self.match_iou or (t.missed > 0 and dist <= self.max_center_dist)
        return (1.0 - iou) + 0.4 * min(dist, 1.0), acceptable

    def _update_velocity(self, t: _Track, box: Box, steps: int) -> None:
        ocx, ocy = _center(t.box)
        ncx, ncy = _center(box)
        vx, vy = (ncx - ocx) / steps, (ncy - ocy) / steps
        s = self.velocity_smoothing
        t.vx = s * vx + (1.0 - s) * t.vx
        t.vy = s * vy + (1.0 - s) * t.vy

    def _prune(self) -> None:
        self._tracks = {tid: t for tid, t in self._tracks.items() if t.missed <= self.max_age}

    def update(self, boxes: list[Box]) -> dict[int, Box]:
        """Assign ids to this frame's real detections; returns track_id -> box.

        Missed tracks keep their id internally for up to ``max_age`` frames but
        are not returned, so the counter sees an honest detection gap.
        """
        tracks = list(self._tracks.values())
        if not tracks:
            return {self._new(b).track_id: b for b in boxes}
        if not boxes:
            for t in tracks:
                t.missed += 1
            self._prune()
            return {}

        costs = np.empty((len(tracks), len(boxes)), dtype=np.float64)
        acceptable = np.zeros_like(costs, dtype=bool)
        for i, t in enumerate(tracks):
            for j, box in enumerate(boxes):
                costs[i, j], acceptable[i, j] = self._match_cost(t, box)

        _, assign, _ = lap.lapjv(costs, extend_cost=True, cost_limit=2.0)
        returned: dict[int, Box] = {}
        matched_detections: set[int] = set()
        for i, t in enumerate(tracks):
            j = int(assign[i])
            if j < 0 or j in matched_detections or not acceptable[i, j]:
                t.missed += 1
                continue
            steps = t.missed + 1
            self._update_velocity(t, boxes[j], steps)
            t.box = boxes[j]
            t.missed = 0
            matched_detections.add(j)
            returned[t.track_id] = t.box

        for j, box in enumerate(boxes):
            if j not in matched_detections:
                t = self._new(box)
                returned[t.track_id] = t.box

        self._prune()
        return returned

    def _new(self, box: Box) -> _Track:
        t = _Track(self._next_id, box)
        self._next_id += 1
        self._tracks[t.track_id] = t
        return t
