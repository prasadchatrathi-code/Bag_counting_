"""Detect, track, and count bags crossing between dock and truck.

Detection is RF-DETR (Roboflow); a small IoU tracker assigns track ids before
the zone-transition counter sees the observations. Frames are read sequentially
from any OpenCV source (file now, RTSP later), so the same code path serves
offline analysis and the eventual live deployment.

Weights may be a native RF-DETR checkpoint (``.pt`` / ``.pth``) or an
exported ONNX artifact (an ``.onnx`` file or an export directory).
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from rfdetr import RFDETR
from rfdetr.inference import RFDETRInference

from .config import Config
from .counter import Direction, Event, TransitionCounter
from .tracker import IoUTracker


@dataclass
class RunSummary:
    source: str
    frames: int
    fps: float
    loaded: int
    unloaded: int
    on_truck: int
    events_csv: str
    wall_start: str | None = None  # camera clock at frame 0 (ISO), if known
    start_s: float = 0.0           # requested window start, source-relative
    duration_s: float = 0.0        # requested window length (0 = to source end)


_TIME_UNITS = {
    "": 1.0,
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
}
_COMPOUND_TIME = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)?", re.IGNORECASE
)


def parse_time(value: str | float) -> float:
    """Convert a video position to seconds.

    Accepts plain seconds (``90``, ``1.5``), compound durations
    (``2m``, ``1hr26m30``, ``1h 26m 30s``) or a clock string
    (``MM:SS`` / ``HH:MM:SS``). This lets operators ask for "one hour and
    twenty-six minutes" without doing frame-rate arithmetic themselves.
    """
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("empty time value")
        if ":" in text:
            parts = text.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"invalid time {value!r}: use SECONDS, MM:SS or HH:MM:SS")
            try:
                numbers = [float(part) for part in parts]
            except ValueError:
                raise ValueError(f"invalid time {value!r}: use SECONDS, MM:SS or HH:MM:SS") from None
            if any(n < 0 for n in numbers) or any(n >= 60 for n in numbers[1:]):
                raise ValueError(f"invalid time {value!r}: minutes/seconds must be below 60")
            seconds = (numbers[0] * 3600 + numbers[1] * 60 + numbers[2] if len(numbers) == 3
                       else numbers[0] * 60 + numbers[1])
        elif any(ch.isalpha() for ch in text):
            parts = list(_COMPOUND_TIME.finditer(text))
            covered = "".join(part.group(0) for part in parts)
            if not parts or "".join(covered.split()) != "".join(text.split()):
                raise ValueError(f"invalid time {value!r}: use SECONDS, 2m, 1hr26m30 or MM:SS")
            seconds = sum(float(part.group(1)) * _TIME_UNITS[(part.group(2) or "").lower()]
                          for part in parts)
        else:
            try:
                seconds = float(text)
            except ValueError:
                raise ValueError(f"invalid time {value!r}: use SECONDS, 2m or MM:SS") from None
    if seconds < 0:
        raise ValueError(f"time must be non-negative: {value!r}")
    return seconds


def parse_line(value: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Parse ``x1,y1,x2,y2`` normalised (0..1) count-line coordinates."""
    parts = [part for part in re.split(r"[,\s]+", str(value).strip()) if part]
    try:
        nums = [float(part) for part in parts]
    except ValueError:
        raise ValueError(f"invalid line {value!r}: use x1,y1,x2,y2") from None
    if len(nums) != 4 or not all(0.0 <= v <= 1.0 for v in nums):
        raise ValueError(f"invalid line {value!r}: use four normalised coordinates in 0..1")
    if nums[0] == nums[2] and nums[1] == nums[3]:
        raise ValueError(f"invalid line {value!r}: endpoints must differ")
    return (nums[0], nums[1]), (nums[2], nums[3])


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Prefer browser-friendly H.264; fall back to mp4v.

    On Linux, OpenCV's bundled FFmpeg often has no H.264 encoder and prints
    scary v4l2m2m errors while trying, so try mp4v first there.
    """
    fourccs = ("mp4v", "avc1") if sys.platform.startswith("linux") else ("avc1", "mp4v")
    for fourcc in fourccs:
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w
        w.release()
    raise RuntimeError(f"cannot open video writer for {path}")


def _is_checkpoint(path: str) -> bool:
    """True for native RF-DETR weights (``.pt`` or ``.pth``)."""
    return Path(path).suffix.lower() in {".pt", ".pth"}


def _is_onnx(path: str) -> bool:
    """True for an ``.onnx`` file or an export directory containing one."""
    p = Path(path)
    return p.suffix.lower() == ".onnx" or (p.is_dir() and any(p.glob("*.onnx")))


def resolve_onnx_path(path: str | Path) -> str:
    """Resolve an export directory to its single ``*.onnx`` file.

    ``RFDETRInference`` needs the actual file, not the directory that contains
    it. Explicit ``.onnx`` paths are returned unchanged.
    """
    p = Path(path)
    if not p.is_dir():
        return str(p)
    files = sorted(p.glob("*.onnx"))
    if not files:
        raise RuntimeError(f"no .onnx file in export directory {p}")
    if len(files) > 1:
        names = ", ".join(f.name for f in files)
        raise RuntimeError(f"multiple .onnx files in {p}: {names} — pass the file explicitly")
    return str(files[0])


def _use_onnx(weights: str, backend: str = "auto") -> bool:
    """Resolve the detector backend (``auto`` / ``checkpoint`` / ``onnx``)."""
    if backend not in {"auto", "checkpoint", "onnx"}:
        raise RuntimeError(f"unknown backend {backend!r}: use auto, checkpoint or onnx")
    onnx, checkpoint = _is_onnx(weights), _is_checkpoint(weights)
    if backend == "onnx":
        if not onnx:
            raise RuntimeError(f"--backend onnx needs an .onnx file or export dir, got {weights!r}")
        return True
    if backend == "checkpoint":
        if not checkpoint:
            raise RuntimeError(f"--backend checkpoint needs a .pt/.pth file, got {weights!r}")
        return False
    if onnx:
        return True
    if checkpoint:
        return False
    raise RuntimeError(
        f"cannot infer backend from {weights!r}: expected a .pt/.pth checkpoint "
        "or an .onnx file/export directory (pass --backend to force one)"
    )


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        out_dir: Path,
        save_video: bool = False,
        wall_start: datetime | None = None,
        source_fps: float | None = None,
        on_frame: Callable[[int, float, np.ndarray], None] | None = None,
        on_event: Callable[[dict], None] | None = None,
    ):
        """on_frame(frame_idx, t_seconds, annotated_bgr) and on_event(row) are
        called from the processing thread; they let a server stream the run
        live. Set `self.stop = True` from another thread to end the run early."""
        self.cfg = cfg
        self.on_frame = on_frame
        self.on_event = on_event
        self.stop = False
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "events").mkdir(exist_ok=True)
        self.save_video = save_video
        self.wall_start = wall_start
        self.source_fps = source_fps

        d = cfg.detector
        self.onnx = _use_onnx(d.weights, d.backend)
        if self.onnx:
            self.model = RFDETRInference(resolve_onnx_path(d.weights), device=d.device)
        else:
            self.model = RFDETR.from_checkpoint(d.weights, device=d.device)
        names = self.model.class_names
        if isinstance(names, dict):
            name_map = {str(v).lower(): int(k) for k, v in names.items()}
        else:
            name_map = {str(v).lower(): i for i, v in enumerate(names)}
        self.bag_cls = name_map.get(d.bag_class.lower())
        if self.bag_cls is None:
            raise RuntimeError(f"class {d.bag_class!r} not found in {d.weights}: {names}")
        self.tracker = IoUTracker(match_iou=d.iou, max_age=d.tracker_max_age,
                                  max_center_dist=d.tracker_max_center_dist)

    # -- drawing helpers ---------------------------------------------------
    def _draw_zones(self, frame: np.ndarray) -> None:
        H, W = frame.shape[:2]
        for poly, col in ((self.cfg.zones.truck, (255, 160, 0)), (self.cfg.zones.dock, (0, 200, 255))):
            pts = np.array([(int(x * W), int(y * H)) for x, y in poly], np.int32)
            cv2.polylines(frame, [pts], True, col, 2)

    def _draw_count_line(self, frame: np.ndarray) -> None:
        """Draw the exact line that gates counts, so operators can audit it."""
        if not self.cfg.counter.require_line_crossing or not hasattr(self, "counter"):
            return
        H, W = frame.shape[:2]
        (ax, ay), (bx, by) = self.counter.zc.cross_line
        p1, p2 = (int(ax * W), int(ay * H)), (int(bx * W), int(by * H))
        cv2.line(frame, p1, p2, (0, 255, 255), 3)
        tx = int(max(4, min(W - 160, (p1[0] + p2[0]) // 2 - 70)))
        ty = int(max(18, min(H - 6, (p1[1] + p2[1]) // 2 - 8)))
        cv2.putText(frame, "COUNT LINE", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

    def _draw_hud(self, frame: np.ndarray, t: float) -> None:
        c = self.counter
        txt = f"t={t:7.1f}s  loaded={c.loaded}  unloaded={c.unloaded}  on_truck={c.on_truck}"
        cv2.rectangle(frame, (0, 0), (560, 28), (0, 0, 0), -1)
        cv2.putText(frame, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _clock(self, t: float) -> str:
        if self.wall_start is None:
            return ""
        return (self.wall_start + timedelta(seconds=t)).strftime("%H:%M:%S")

    # -- main loop ---------------------------------------------------------
    def run(self, source: str, max_frames: int = 0, start_s: float = 0.0,
            duration_s: float = 0.0) -> RunSummary:
        """Process a time window of ``source`` and return a run summary.

        ``start_s`` seeks that many seconds into the source; ``duration_s``
        limits how many seconds are processed from there (0 = to the end).
        ``max_frames`` additionally caps the number of processed frames.
        Event times and clock stamps stay source-relative, so a windowed run
        can be compared directly with full runs and hand counts.

        All resources are released in a ``finally`` block, so Ctrl+C, a stop
        request from the web UI, or an exception still closes the capture,
        the video writer and the event CSV, and still writes ``summary.json``.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video source: {source}")
        fps = self.source_fps or cap.get(cv2.CAP_PROP_FPS) or 8.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_frame = int(round(start_s * fps))
        if start_frame:
            if total_frames and start_frame >= total_frames:
                cap.release()
                raise RuntimeError(
                    f"--start {start_s:g}s (frame {start_frame}) is beyond the source "
                    f"({total_frames} frames, ~{total_frames / fps:.1f}s)"
                )
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        window_frames = int(round(duration_s * fps)) if duration_s > 0 else 0
        # max_frames and duration may both be given; honour the tighter cap.
        limit_frames = min(max_frames, window_frames) if max_frames and window_frames else (max_frames or window_frames)
        remaining = max(0, total_frames - start_frame) if total_frames else 0
        progress_total = min(limit_frames, remaining) if limit_frames and remaining else (limit_frames or remaining)
        # The counter's stitching/history windows are time-based; give it the
        # source fps now that the capture has reported it.
        self.counter = TransitionCounter(self.cfg.zones, self.cfg.counter, fps=fps)
        events_path = self.out / "events.csv"
        # Recent annotated frames so an inferred event (raised when a track is
        # pruned) can still be illustrated with the frame it was last seen in.
        recent: deque[tuple[int, np.ndarray]] = deque(maxlen=self.counter.history_gap + 4)
        writer = None
        ef = None
        tf = None
        fi = 0
        last_progress = -1
        run_started = time.perf_counter()
        frame_elapsed: list[float] = []
        inference_elapsed: list[float] = []
        bag_detections = 0
        track_observations = 0
        d = self.cfg.detector
        if start_s or window_frames:
            end_desc = f"{start_s + duration_s:g}s" if window_frames else "end"
            limit = f"source {start_s:g}s .. {end_desc} ({progress_total or '?'} frames)"
        else:
            limit = f"{limit_frames} frames" if limit_frames else "full video"
        line_desc = ("line " + "->".join(f"({x:.2f},{y:.2f})" for x, y in self.counter.zc.cross_line)
                     if self.cfg.counter.require_line_crossing else "line counting disabled")
        print(f"processing {source} ({limit}, fps={fps:.2f}, {line_desc})", flush=True)
        try:
            if self.save_video:
                writer = _open_writer(self.out / "annotated.mp4", fps, (W, H))
            ef = events_path.open("w", newline="")
            ew = csv.writer(ef)
            ew.writerow(["idx", "track_id", "frame", "time_s", "clock", "direction", "inferred", "x", "y", "loaded", "unloaded", "on_truck"])
            tf = (self.out / "frame_times.csv").open("w", newline="")
            tw = csv.writer(tf)
            tw.writerow(["frame", "inference_ms", "processing_ms"])

            while not self.stop:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_started = time.perf_counter()
                source_fi = start_frame + fi
                t = source_fi / fps  # source-relative, so windows align with full runs
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                predict_kwargs = {"threshold": d.conf}
                if not self.onnx:  # ONNX has a fixed input shape recorded at export
                    predict_kwargs["shape"] = (d.resolution, d.resolution)
                inference_started = time.perf_counter()
                detections = self.model.predict(frame_rgb, **predict_kwargs)
                inference_ms = (time.perf_counter() - inference_started) * 1000
                inference_elapsed.append(inference_ms)
                boxes_px: list[tuple[float, float, float, float]] = []
                if detections is not None and len(detections) > 0:
                    xyxy = np.asarray(detections.xyxy)
                    classes = np.asarray(detections.class_id)
                    for box, cls in zip(xyxy, classes):
                        if int(cls) == self.bag_cls:
                            boxes_px.append(tuple(float(v) for v in box))
                bag_detections += len(boxes_px)

                tracks = self.tracker.update(boxes_px)
                track_observations += len(tracks)
                obs: dict[int, tuple[float, float]] = {}
                obs_boxes: dict[int, tuple[float, float, float, float]] = {}
                for tid, (x1, y1, x2, y2) in tracks.items():
                    obs[tid] = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H)
                    obs_boxes[tid] = (x1 / W, y1 / H, x2 / W, y2 / H)
                new_events = self.counter.update(fi, obs, obs_boxes)

                annotated = frame.copy()
                for tid, (x1, y1, x2, y2) in tracks.items():
                    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
                    cv2.putText(annotated, f"bag {tid}", (p1[0], max(14, p1[1] - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                self._draw_zones(annotated)
                self._draw_count_line(annotated)
                self._draw_hud(annotated, t)
                recent.append((fi, annotated))
                for ev in new_events:
                    idx = len(self.counter.events) - len(new_events) + new_events.index(ev) + 1
                    et = (start_frame + ev.frame_idx) / fps
                    row = dict(idx=idx, track_id=ev.track_id, frame=start_frame + ev.frame_idx,
                               time_s=round(et, 2),
                               clock=self._clock(et), direction=ev.direction.value, inferred=int(ev.inferred),
                               x=round(ev.xy[0], 3), y=round(ev.xy[1], 3), loaded=self.counter.loaded,
                               unloaded=self.counter.unloaded, on_truck=self.counter.on_truck)
                    ew.writerow(list(row.values()))
                    ef.flush()
                    src = next((fr for i, fr in recent if i == ev.frame_idx), annotated)
                    thumb = src.copy()
                    cx, cy = int(ev.xy[0] * W), int(ev.xy[1] * H)
                    col = (0, 255, 0) if ev.direction is Direction.LOADED else (0, 0, 255)
                    cv2.circle(thumb, (cx, cy), 28, col, 3)
                    if ev.inferred:
                        cv2.putText(thumb, "INFERRED (lost in transit)", (cx - 120, cy + 56),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                    tag = "inferred_" if ev.inferred else ""
                    thumb_path = self.out / "events" / f"{idx:04d}_{tag}{ev.direction.value}_{et:07.1f}s.jpg"
                    cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if self.on_event is not None:
                        self.on_event({**row, "thumb": thumb_path.name})
                if writer is not None:
                    writer.write(annotated)
                if self.on_frame is not None:
                    self.on_frame(fi, t, annotated)
                processing_ms = (time.perf_counter() - frame_started) * 1000
                frame_elapsed.append(processing_ms)
                tw.writerow([source_fi, round(inference_ms, 3), round(processing_ms, 3)])
                fi += 1
                if progress_total:
                    pct = min(100, int(fi * 100 / progress_total))
                    if pct != last_progress:
                        width = 30
                        filled = int(width * pct / 100)
                        sys.stderr.write(f"\r[{('=' * filled) + (' ' * (width - filled))}] {pct:3d}% "
                                         f"{fi}/{progress_total} frames")
                        sys.stderr.flush()
                        last_progress = pct
                elif fi % 25 == 0:
                    sys.stderr.write(f"\rProcessing: {fi} frames")
                    sys.stderr.flush()
                if limit_frames and fi >= limit_frames:
                    break
        finally:
            if fi:
                sys.stderr.write("\n")
                sys.stderr.flush()
            if ef is not None:
                ef.close()
            if tf is not None:
                tf.close()
            if writer is not None:
                writer.release()
            cap.release()
            summary = RunSummary(source, fi, fps, self.counter.loaded, self.counter.unloaded,
                                 self.counter.on_truck, str(events_path),
                                 self.wall_start.isoformat() if self.wall_start else None,
                                 start_s, duration_s)
            elapsed_s = time.perf_counter() - run_started
            def timing_stats(values: list[float]) -> dict[str, float | None]:
                if not values:
                    return {"mean_ms": None, "median_ms": None, "p95_ms": None,
                            "min_ms": None, "max_ms": None}
                arr = np.asarray(values, dtype=float)
                return {"mean_ms": round(float(np.mean(arr)), 3),
                        "median_ms": round(float(np.median(arr)), 3),
                        "p95_ms": round(float(np.percentile(arr, 95)), 3),
                        "min_ms": round(float(np.min(arr)), 3),
                        "max_ms": round(float(np.max(arr)), 3)}
            report = asdict(summary)
            report.update({
                "model": {"weights": d.weights, "backend": "onnx" if self.onnx else "checkpoint",
                          "device": d.device, "confidence": d.conf, "resolution": d.resolution,
                          "bag_class": d.bag_class},
                "video": {"width": W, "height": H, "source_fps": fps,
                          "estimated_total_frames": total_frames or None},
                "counting_line": {
                    "enabled": self.cfg.counter.require_line_crossing,
                    "a": list(self.counter.zc.cross_line[0]),
                    "b": list(self.counter.zc.cross_line[1]),
                    "source": self.counter.zc.line_source,
                    "truck_side_sign": self.counter.zc.truck_side,
                    "tolerance": self.cfg.counter.cross_line_tolerance,
                    "max_frames": self.cfg.counter.cross_max_frames,
                },
                "window": {"start_frame": start_frame, "end_frame": start_frame + fi,
                           "source_relative_times": True},
                "timing": {"elapsed_seconds": round(elapsed_s, 3),
                           "processing_fps": round(fi / elapsed_s, 3) if elapsed_s else None,
                           "frame_processing": timing_stats(frame_elapsed),
                           "model_inference": timing_stats(inference_elapsed)},
                "detections": {"bag_boxes_total": bag_detections,
                               "average_bag_boxes_per_frame": round(bag_detections / fi, 3) if fi else 0,
                               "track_observations_total": track_observations},
                "counts": {"loaded": self.counter.loaded, "unloaded": self.counter.unloaded,
                           "on_truck": self.counter.on_truck, "events": len(self.counter.events)},
                "artifacts": {"events_csv": str(events_path),
                              "frame_times_csv": str(self.out / "frame_times.csv"),
                              "annotated_video": str(self.out / "annotated.mp4") if self.save_video else None},
            })
            (self.out / "summary.json").write_text(json.dumps(report, indent=2))
            print(f"finished {fi} frames; files in {self.out}", flush=True)
        return summary
