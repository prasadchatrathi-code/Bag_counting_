"""Line-gated zone-transition counter.

Pure logic, no video or model dependencies, so it is unit-testable with
synthetic tracks. Feed it the tracked bag positions for each frame; it emits
an Event only after a bag crosses the configured count line in the matching
direction: dock -> truck is `loaded`, truck -> dock is `unloaded`. Zone dwell
is still required before an event is trusted, so a bag shaking on the line or
briefly entering a polygon does not count.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
import cv2

from .config import CounterConfig, Point, Zones

Box = tuple[float, float, float, float]


def is_carrier_box(loose: Box, tight: Box, x_margin: float, top_tol: float, min_extra: float) -> bool:
    """True if `loose` looks like a bag + carrier box around the tight bag box
    `tight`: same x-range (within margin), same top edge, much taller."""
    return (loose[0] >= tight[0] - x_margin and loose[2] <= tight[2] + x_margin
            and abs(loose[1] - tight[1]) <= top_tol
            and loose[3] >= tight[3] + min_extra)


def _dist(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to segment ab."""
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def _nearest_edge(poly: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polygon edge closest to `point`, used to derive the count line."""
    edges = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    return min(edges, key=lambda e: _seg_dist(point, e[0], e[1]))


log = logging.getLogger("bagcount.counter")


class Zone(str, Enum):
    TRUCK = "truck"
    DOCK = "dock"
    TRANSIT = "transit"


class Direction(str, Enum):
    LOADED = "loaded"
    UNLOADED = "unloaded"


@dataclass
class Event:
    track_id: int
    frame_idx: int
    direction: Direction
    origin_frame: int  # last frame seen in the origin zone
    xy: Point  # anchor position (normalised) at the moment of the event
    inferred: bool = False  # True if the track vanished in transit and the
    # destination was inferred from its heading rather than observed


@dataclass
class TrackState:
    track_id: int
    root: int = -1  # id of the first tracklet in this bag's chain (for de-dup)
    zone: Zone = Zone.TRANSIT  # last *settled* zone
    settled_frames: int = 0  # consecutive frames observed in `zone`
    pending: Zone | None = None  # zone we are tentatively in
    pending_frames: int = 0
    last_seen: int = 0
    last_zone_frame: int = 0
    xy: Point = (0.0, 0.0)
    heading: Zone | None = None  # while in transit: the zone we are nearer to
    adopted_frame: int | None = None  # frame a successor first inherited this track
    box: Box | None = None  # normalised (x1, y1, x2, y2) when the caller supplies boxes
    born_xy: Point = (0.0, 0.0)
    born_frame: int = 0
    born_zone: Zone = Zone.TRANSIT
    provisional: bool = False  # dock history assumed from a handover, not observed
    dock_frames: int = 0  # provisional truck-born track: frames seen on the dock so far
    last_z: Zone = Zone.TRANSIT  # zone of the most recent observation (settled or not)
    inferred_emitted: bool = False  # an inferred event has already been emitted for this loss
    # Line-crossing state. `line_side` uses the classifier's raw sign (+/-1);
    # `crossed_direction` latches the most recent dock->truck / truck->dock
    # crossing so a later zone settle can be gated on it.
    line_side: int | None = None
    crossed_direction: Direction | None = None
    crossed_frame: int = -1
    # Direction of the last settled zone change. Used when a custom line sits
    # inside a zone: the crossing can be the final gate for that same change.
    zone_change_direction: Direction | None = None


class ZoneClassifier:
    def __init__(self, zones: Zones, cross_line_tolerance: float = 0.01):
        self._truck = np.array(zones.truck, dtype=np.float32)
        self._dock = np.array(zones.dock, dtype=np.float32)
        self.tolerance = float(cross_line_tolerance)
        # The tailgate is the truck edge nearest the dock's centre.
        dc = self._dock.mean(axis=0)
        tc = self._truck.mean(axis=0)
        self._tailgate = _nearest_edge(self._truck, dc)

        if zones.cross_line is not None:
            self.cross_line: tuple[Point, Point] = (
                (float(zones.cross_line[0][0]), float(zones.cross_line[0][1])),
                (float(zones.cross_line[1][0]), float(zones.cross_line[1][1])),
            )
            self.line_source = "config"
            truck_ref = tc
        else:
            self.cross_line = self._derive_cross_line(tc)
            self.line_source = "derived"
            truck_ref = _mid(*self._tailgate)

        self._la = np.array(self.cross_line[0], dtype=np.float32)
        self._lb = np.array(self.cross_line[1], dtype=np.float32)
        self._lv = self._lb - self._la
        self._line_len = float(np.linalg.norm(self._lv)) or 1.0
        # Raw side sign of the truck reference point. Everything on this side
        # is truck-side; the opposite sign is dock-side.
        self.truck_side = self._raw_side(truck_ref) or self._raw_side(tc) or 1

    def _derive_cross_line(self, truck_centre: np.ndarray) -> tuple[Point, Point]:
        """Build a line halfway between the truck tailgate and the near dock
        edge, parallel to them and wide enough to cover the whole transition."""
        near_dock = _nearest_edge(self._dock, truck_centre)
        tail_mid, dock_mid = _mid(*self._tailgate), _mid(*near_dock)
        direction = self._tailgate[1] - self._tailgate[0]
        if float(np.linalg.norm(direction)) < 1e-9:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        unit = direction / float(np.linalg.norm(direction))
        dock_direction = near_dock[1] - near_dock[0]
        if float(np.dot(unit, dock_direction)) < 0:
            near_dock = (near_dock[1], near_dock[0])
        centre = _mid(tail_mid, dock_mid)
        offsets = [float(np.dot(p - tail_mid, unit)) for p in (*self._tailgate, *near_dock)]
        a = centre + unit * min(offsets)
        b = centre + unit * max(offsets)
        return ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))

    def _cross(self, xy: Point) -> float:
        p = np.array(xy, dtype=np.float32)
        return float(self._lv[0] * (p[1] - self._la[1]) - self._lv[1] * (p[0] - self._la[0]))

    def _raw_side(self, xy: Point | np.ndarray) -> int:
        p = np.asarray(xy, dtype=np.float32)
        cross = float(self._lv[0] * (p[1] - self._la[1]) - self._lv[1] * (p[0] - self._la[0]))
        return 1 if cross > 0 else -1 if cross < 0 else 0

    def line_distance(self, xy: Point) -> float:
        return abs(self._cross(xy)) / self._line_len

    def line_side(self, xy: Point) -> int:
        """+1 / -1 in raw cross-product terms, 0 when within tolerance.
        Use `truck_side` to convert this to a zone name."""
        if self.line_distance(xy) <= self.tolerance:
            return 0
        return self._raw_side(xy)

    def classify(self, xy: Point) -> Zone:
        if cv2.pointPolygonTest(self._truck, xy, False) >= 0:
            return Zone.TRUCK
        if cv2.pointPolygonTest(self._dock, xy, False) >= 0:
            return Zone.DOCK
        return Zone.TRANSIT

    def tailgate_distance(self, xy: Point) -> float:
        """Distance from a point to the tailgate edge of the truck polygon."""
        return _seg_dist(np.array(xy, dtype=np.float32), *self._tailgate)

    def nearest(self, xy: Point) -> Zone:
        """For a point in transit: which zone it is closer to."""
        dt = -cv2.pointPolygonTest(self._truck, xy, True)
        dd = -cv2.pointPolygonTest(self._dock, xy, True)
        return Zone.TRUCK if dt <= dd else Zone.DOCK


class TransitionCounter:
    def __init__(self, zones: Zones, cfg: CounterConfig | None = None, fps: float = 30.0):
        self.cfg = cfg or CounterConfig()
        self.zc = ZoneClassifier(zones, self.cfg.cross_line_tolerance)
        # Time-based windows keep the same config usable at 8 fps and 30 fps.
        self.stitch_gap = max(int(self.cfg.stitch_max_gap), round(self.cfg.stitch_seconds * fps))
        self.history_gap = max(int(self.cfg.max_lost_frames), round(self.cfg.history_seconds * fps))
        self.tracks: dict[int, TrackState] = {}
        self.events: list[Event] = []
        self._emitted: dict[tuple[int, Direction], int] = {}  # (root, dir) -> frame
        self._fresh_root = 0  # negative ids for roots re-assigned to revived tracks

    def _counted_neighbour(self, st: TrackState, frame_idx: int, xy: Point) -> bool:
        """True if this root was counted very recently and its live fragment is
        now settled in the destination zone, close to the newcomer. That is an
        ID switch just after a crossing. A stale fragment still sitting in the
        origin zone is the next bag being picked up from the same spot and must
        NOT inherit the counted root."""
        if _dist(st.xy, xy) > self.cfg.stitch_switch_radius:
            return False
        for direction in (Direction.LOADED, Direction.UNLOADED):
            seen = self._emitted.get((st.root, direction))
            if seen is None or frame_idx - seen > self.cfg.stitch_switch_frames:
                continue
            settled = Zone.TRUCK if direction is Direction.LOADED else Zone.DOCK
            if st.zone is settled:
                return True
        return False

    @property
    def loaded(self) -> int:
        return sum(e.direction is Direction.LOADED for e in self.events)

    @property
    def unloaded(self) -> int:
        return sum(e.direction is Direction.UNLOADED for e in self.events)

    @property
    def on_truck(self) -> int:
        return self.loaded - self.unloaded

    # -- count-line helpers -------------------------------------------------
    def _side_for(self, direction: Direction) -> int:
        """Raw line sign expected after a given crossing."""
        return self.zc.truck_side if direction is Direction.LOADED else -self.zc.truck_side

    def _direction_for_side(self, side: int) -> Direction:
        return Direction.LOADED if side == self.zc.truck_side else Direction.UNLOADED

    def _init_line_state(self, st: TrackState, xy: Point, frame_idx: int) -> None:
        """Seed line state for a newly born track. A provisional handover is
        assumed to have started on the dock, so a first sighting already on the
        truck side counts as the crossing."""
        if not self.cfg.require_line_crossing:
            return
        side = self.zc.line_side(xy)
        if side == 0:
            return
        if st.provisional:
            st.line_side = -self.zc.truck_side
            if side == self.zc.truck_side:
                st.crossed_direction = Direction.LOADED
                st.crossed_frame = frame_idx
                log.debug("f%d track %d provisional handover crossed line (loaded)", frame_idx, st.track_id)
        else:
            st.line_side = side

    def _observe_line(self, st: TrackState, xy: Point, frame_idx: int,
                      out: list[Event] | None = None) -> None:
        """Update the side/crossing latch from a real observation. Points on
        the line keep the previous side, so jitter cannot flip the direction.
        If the destination zone was settled *before* the line was crossed (a
        custom line deeper in the truck), emit at the crossing itself."""
        if not self.cfg.require_line_crossing:
            return
        side = self.zc.line_side(xy)
        if side == 0:
            return
        if st.line_side is None:
            self._init_line_state(st, xy, frame_idx)
            return
        if side == st.line_side:
            return
        st.line_side = side
        st.crossed_direction = self._direction_for_side(side)
        st.crossed_frame = frame_idx
        log.debug("f%d track %d crossed count line -> %s", frame_idx, st.track_id, st.crossed_direction.value)
        if out is None or st.provisional:
            return
        destination = Zone.TRUCK if st.crossed_direction is Direction.LOADED else Zone.DOCK
        if (st.zone is destination and st.zone_change_direction is st.crossed_direction
                and st.last_zone_frame < frame_idx):
            self._emit(Event(st.track_id, frame_idx, st.crossed_direction,
                             st.last_zone_frame, xy), st, out)

    def _line_gate_ok(self, st: TrackState, direction: Direction, frame_idx: int) -> bool:
        """The final gate for every event: no crossing, no count."""
        if not self.cfg.require_line_crossing:
            return True
        if st.crossed_direction is not direction or st.crossed_frame < 0:
            log.debug("f%d BLOCK %s track %d root %d (no line crossing)",
                      frame_idx, direction.value, st.track_id, st.root)
            return False
        if frame_idx - st.crossed_frame > self.cfg.cross_max_frames:
            log.debug("f%d BLOCK %s track %d root %d (stale crossing at f%d)",
                      frame_idx, direction.value, st.track_id, st.root, st.crossed_frame)
            return False
        if st.line_side is not None and st.line_side != self._side_for(direction):
            log.debug("f%d BLOCK %s track %d root %d (on opposite side)",
                      frame_idx, direction.value, st.track_id, st.root)
            return False
        return True

    def _adopt_lost_track(self, tid: int, xy: Point, frame_idx: int, box: Box | None = None,
                          out: list[Event] | None = None) -> TrackState:
        """A brand-new track id appearing next to a recently lost one is almost
        always the same bag after a detection gap (ID switch). Inherit the lost
        track's zone history so the transition is still counted. The history is
        copied, not moved: a bag can fragment into several new ids in one frame,
        and the chain root de-duplicates whatever they go on to do."""
        # Duplicate box on a bag that is tracked right now: copy the live track.
        for st in self.tracks.values():
            if st.last_seen != frame_idx:
                continue
            d = _dist(st.xy, xy)
            carrier = False
            if box is not None and st.box is not None:
                c = self.cfg
                carrier = (is_carrier_box(box, st.box, c.carrier_x_margin, c.carrier_top_tol, c.carrier_min_extra_height)
                           or is_carrier_box(st.box, box, c.carrier_x_margin, c.carrier_top_tol, c.carrier_min_extra_height))
            if d <= self.cfg.duplicate_radius or carrier:
                log.debug("f%d new %d duplicates live %d (root %d, d=%.3f, carrier=%s)",
                          frame_idx, tid, st.track_id, st.root, d, carrier)
                return replace(st, track_id=tid, adopted_frame=None, born_frame=frame_idx)
        live_roots = {st.root for st in self.tracks.values()
                      if frame_idx - st.last_seen <= 1 and st.born_frame < frame_idx}
        best, best_score = None, 1.0
        counted_switch = False
        for st in self.tracks.values():
            gap = frame_idx - st.last_seen
            if gap < 0 or gap > self.stitch_gap:
                continue
            d = _dist(st.xy, xy)
            radius = min(self.cfg.stitch_max_radius, self.cfg.stitch_radius + self.cfg.stitch_speed * max(gap, 0))
            # Mid-carry loss: the bag may reappear far inside its destination
            # zone after a second or two. Zone/heading agreement is a much
            # stronger signal than distance, so widen the gate for that case.
            if (st.heading is not None and not st.inferred_emitted
                    and self.zc.classify(xy) is st.heading):
                radius = max(radius, self.cfg.stitch_heading_radius)
            # A live fragment normally blocks adoption (it is still tracking the
            # bag). Exception: it was just counted and the newcomer is close —
            # that is an ID switch after the crossing, so inherit its root and
            # zone instead of creating a second, countable bag.
            switch = st.root in live_roots and self._counted_neighbour(st, frame_idx, xy)
            if (st.root in live_roots and not switch) or (gap == 0 and not switch):
                continue
            if switch:
                radius = max(radius, self.cfg.stitch_switch_radius)
            if self._already_counted(st, frame_idx) and not switch:
                # Counted leaving this zone already: a newcomer beside it is the
                # next bag picked up from the same spot, not the same bag.
                continue
            if st.adopted_frame is not None and frame_idx - st.adopted_frame > self.cfg.stitch_sibling_frames:
                continue
            score = d / radius
            if score < best_score:
                best, best_score, counted_switch = st, score, switch
        if best is None:
            return self._handover_or_new(tid, xy, frame_idx, out)
        if best.adopted_frame is None:
            best.adopted_frame = frame_idx
        log.debug("f%d new %d adopts lost %d (root %d, zone %s, gap %d, score %.2f%s)",
                  frame_idx, tid, best.track_id, best.root, best.zone.value,
                  frame_idx - best.last_seen, best_score, ", counted-switch" if counted_switch else "")
        inherited = replace(best, track_id=tid, adopted_frame=None, born_frame=frame_idx)
        # A track that was lost mid-carry before it ever settled has no origin
        # zone. Use its heading to reconstruct one, so the eventual arrival in
        # the destination zone can still produce a count.
        if inherited.zone is Zone.TRANSIT and inherited.heading is not None:
            inherited.zone = Zone.DOCK if inherited.heading is Zone.TRUCK else Zone.TRUCK
            inherited.settled_frames = self.cfg.min_origin_frames
        # The re-detection may itself be the crossing (bag lost just before the
        # line, found on the far side). Inherit the old latch, else latch here.
        self._observe_line(inherited, xy, frame_idx, out)
        return inherited

    def _handover_or_new(self, tid: int, xy: Point, frame_idx: int,
                         out: list[Event] | None = None) -> TrackState:
        """A track born in the transit band or just inside the truck zone is a
        bag that was handed over the tailgate before its own track started;
        give it provisional dock history. If it was born inside the truck and
        then travels into the dock instead, that *may* be a hand-down unload
        (see `update`), but the heuristic is opt-in: a dock bag misdetected at
        the tailgate during a pickup looks the same. Anything else starts with
        no zone history."""
        z = self.zc.classify(xy)
        if z is Zone.TRANSIT or (z is Zone.TRUCK and self.zc.tailgate_distance(xy) <= self.cfg.handover_edge_dist):
            log.debug("f%d new %d born at the tailgate (%s): provisional dock history", frame_idx, tid, z.value)
            st = TrackState(track_id=tid, root=tid, zone=Zone.DOCK, settled_frames=self.cfg.min_origin_frames,
                            last_zone_frame=frame_idx, born_xy=xy, born_frame=frame_idx, born_zone=z,
                            provisional=True)
        else:
            st = TrackState(track_id=tid, root=tid, born_xy=xy, born_frame=frame_idx, born_zone=z)
        self._init_line_state(st, xy, frame_idx)
        return st

    def _travel(self, st: TrackState, xy: Point) -> float:
        """How much further from the tailgate a track is now than where it was born."""
        return self.zc.tailgate_distance(xy) - self.zc.tailgate_distance(st.born_xy)

    def _merge_carrier_boxes(self, frame_idx: int, boxes: dict[int, Box]) -> None:
        """Two live tracks whose boxes are a tight bag box and its carrier box
        are one bag: give the younger the older's root so it is counted once.
        Checked every frame because the geometry only holds once the bag is
        lifted, which may be after both tracks were born."""
        c = self.cfg
        live = [self.tracks[t] for t in boxes if t in self.tracks]
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if a.root == b.root:
                    continue
                ba, bb = boxes[a.track_id], boxes[b.track_id]
                if (is_carrier_box(bb, ba, c.carrier_x_margin, c.carrier_top_tol, c.carrier_min_extra_height)
                        or is_carrier_box(ba, bb, c.carrier_x_margin, c.carrier_top_tol, c.carrier_min_extra_height)):
                    young, old = (a, b) if a.track_id > b.track_id else (b, a)
                    log.debug("f%d track %d is a carrier box of %d: root %d -> %d",
                              frame_idx, young.track_id, old.track_id, young.root, old.root)
                    young.root = old.root

    def _already_counted(self, st: TrackState, frame_idx: int) -> bool:
        """True if an event has recently been emitted for this track's root in
        the direction it would take when leaving its settled zone."""
        if st.zone is Zone.DOCK:
            direction = Direction.LOADED
        elif st.zone is Zone.TRUCK:
            direction = Direction.UNLOADED
        else:
            return False
        prev = self._emitted.get((st.root, direction))
        return prev is not None and frame_idx - prev <= self.cfg.dedup_frames

    def _emit(self, ev: Event, st: TrackState, out: list[Event]) -> None:
        if not self._line_gate_ok(st, ev.direction, ev.frame_idx):
            return
        key = (st.root, ev.direction)
        prev = self._emitted.get(key)
        if prev is not None and ev.frame_idx - prev <= self.cfg.dedup_frames:
            log.debug("f%d SUPPRESS %s track %d root %d (already emitted at f%d)",
                      ev.frame_idx, ev.direction.value, ev.track_id, st.root, prev)
            return  # same bag, already counted via another fragment of its track
        self._emitted[key] = ev.frame_idx
        log.debug("f%d EVENT %s track %d root %d inferred=%s", ev.frame_idx, ev.direction.value, ev.track_id, st.root, ev.inferred)
        out.append(ev)

    def update(self, frame_idx: int, observations: dict[int, Point],
               boxes: dict[int, Box] | None = None) -> list[Event]:
        """observations: track_id -> normalised anchor (x, y) seen this frame.
        boxes (optional): track_id -> normalised (x1, y1, x2, y2); enables the
        overlap test for duplicate boxes on one bag."""
        new_events: list[Event] = []
        for tid, xy in observations.items():
            box = boxes.get(tid) if boxes else None
            st = self.tracks.get(tid)
            z = self.zc.classify(xy)
            if st is None:
                st = self.tracks[tid] = self._adopt_lost_track(tid, xy, frame_idx, box, new_events)
            elif (frame_idx - st.last_seen >= self.cfg.revive_gap_frames and z is st.zone
                  and self._already_counted(st, frame_idx)):
                # The tracker revived this id on a new bag at the pickup spot.
                self._fresh_root -= 1
                log.debug("f%d track %d revived in %s after its bag was counted: new root %d",
                          frame_idx, tid, z.value, self._fresh_root)
                st.root = self._fresh_root
                st.pending, st.pending_frames, st.heading = None, 0, None
                st.provisional, st.inferred_emitted = False, False
                st.line_side, st.crossed_direction, st.crossed_frame = None, None, -1
                st.zone_change_direction = None
            st.last_seen = frame_idx
            st.xy = xy
            st.box = box
            st.last_z = z
            self._observe_line(st, xy, frame_idx, new_events)
            if z is Zone.TRANSIT:
                st.pending = None
                st.pending_frames = 0
                st.heading = self.zc.nearest(xy)
                continue
            st.heading = None
            if z is st.zone:
                st.pending = None
                st.pending_frames = 0
                if st.provisional and z is Zone.DOCK and st.born_zone is Zone.TRUCK:
                    # Born at the tailgate on the truck side, now on the dock.
                    # This is either a genuine hand-down or a dock bag briefly
                    # misdetected at the tailgate during a pickup; the two are
                    # observationally identical, so the hand-down reading is
                    # opt-in. Keep the bag provisional either way: a dip below
                    # the tailgate that comes straight back must not become a
                    # fresh `loaded` count.
                    st.dock_frames += 1
                    if (self.cfg.allow_handover_unload
                            and self._travel(st, xy) >= self.cfg.handover_min_travel
                            and st.dock_frames >= self.cfg.handover_unload_frames):
                        st.provisional = False
                        st.settled_frames = 1
                        self._emit(Event(tid, frame_idx, Direction.UNLOADED, st.born_frame, xy), st, new_events)
                        st.last_zone_frame = frame_idx
                    continue
                st.settled_frames += 1
                st.last_zone_frame = frame_idx
                st.provisional = False  # observed in the zone we assumed
                continue
            # Observed in a zone different from the settled one.
            if st.pending is z:
                st.pending_frames += 1
            else:
                st.pending, st.pending_frames = z, 1
            if st.pending_frames < self.cfg.min_origin_frames:
                continue
            if st.provisional and self._travel(st, xy) < self.cfg.handover_min_travel:
                continue  # assumed handover: wait until the bag has moved away from the tailgate
            st.provisional = False
            prev = st.zone
            st.zone, st.settled_frames = z, st.pending_frames
            st.pending, st.pending_frames = None, 0
            if prev is Zone.DOCK and z is Zone.TRUCK:
                ev = Event(tid, frame_idx, Direction.LOADED, st.last_zone_frame, xy)
            elif prev is Zone.TRUCK and z is Zone.DOCK:
                ev = Event(tid, frame_idx, Direction.UNLOADED, st.last_zone_frame, xy)
            else:
                ev = None  # first settle from TRANSIT: no event
            st.last_zone_frame = frame_idx
            if ev is not None:
                st.zone_change_direction = ev.direction
                self._emit(ev, st, new_events)
        if boxes:
            self._merge_carrier_boxes(frame_idx, boxes)
        # A bag lost mid-carry is assumed to have arrived once it has been
        # unseen for max_lost_frames. Emit that inferred event once, but keep
        # the track's history until history_gap so a late re-detection (e.g.
        # 2 s at 30 fps) can still inherit its zone and be counted.
        for st in self.tracks.values():
            if st.inferred_emitted or frame_idx - st.last_seen <= self.cfg.max_lost_frames:
                continue
            if (st.provisional and st.born_zone is Zone.TRUCK and st.last_z is Zone.DOCK
                    and self.cfg.allow_handover_unload
                    and self._travel(st, st.xy) >= self.cfg.handover_min_travel):
                # Handed down over the tailgate and last seen on the dock: it
                # did not come back, so it is an unload even if the detector
                # lost it before the dwell was reached.
                self._emit(Event(st.track_id, st.last_seen, Direction.UNLOADED, st.born_frame, st.xy), st, new_events)
                st.inferred_emitted = True
                continue
            # A confirmed line crossing is enough to infer arrival even when
            # the detector lost the bag before the zone dwell completed.
            crossed = st.crossed_direction if frame_idx - st.crossed_frame <= self.cfg.cross_max_frames else None
            if not self.cfg.infer_lost_in_transit or st.provisional:
                continue
            if st.heading is None and crossed is None:
                continue
            inferred = None
            if st.zone is Zone.DOCK and (st.heading is Zone.TRUCK or crossed is Direction.LOADED):
                inferred = Event(st.track_id, st.last_seen, Direction.LOADED, st.last_zone_frame, st.xy, inferred=True)
            elif st.zone is Zone.TRUCK and (st.heading is Zone.DOCK or crossed is Direction.UNLOADED):
                inferred = Event(st.track_id, st.last_seen, Direction.UNLOADED, st.last_zone_frame, st.xy, inferred=True)
            if inferred is not None:
                self._emit(inferred, st, new_events)
                st.inferred_emitted = True
        # Bound memory: drop tracks only after the (time-based) history window.
        stale = [t for t, s in self.tracks.items() if frame_idx - s.last_seen > self.history_gap]
        for t in stale:
            self.tracks.pop(t)
        if frame_idx % 500 == 0:
            self._emitted = {k: f for k, f in self._emitted.items() if frame_idx - f <= self.cfg.dedup_frames}
        self.events.extend(new_events)
        return new_events
