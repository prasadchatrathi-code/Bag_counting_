"""Scene configuration: zones, model settings, and run options.

Coordinates are normalised (0..1) so the same config works for any frame size.
Zones are polygons; a bag whose anchor point is inside TRUCK or DOCK is "in"
that zone, anything else is in transit. Counting happens on zone transitions,
which gives natural hysteresis (a bag hovering near the tailgate edge does not
flip-flop).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

Point = tuple[float, float]

# Eyeballed from the "MDP Loading Point top view" camera. Truck bed is the
# trapezoid running up the middle of the frame; dock is the yellow tarp area
# below the tailgate.
DEFAULT_TRUCK: list[Point] = [(0.42, 0.0), (0.60, 0.0), (0.64, 0.64), (0.33, 0.64)]
DEFAULT_DOCK: list[Point] = [(0.28, 0.72), (0.78, 0.72), (0.78, 1.0), (0.28, 1.0)]


@dataclass
class Zones:
    truck: list[Point] = field(default_factory=lambda: list(DEFAULT_TRUCK))
    dock: list[Point] = field(default_factory=lambda: list(DEFAULT_DOCK))


@dataclass
class DetectorConfig:
    # Native RF-DETR checkpoint (.pt/.pth) or exported ONNX artifact (.onnx).
    weights: str = "runs/detect/bags_full/checkpoint_best_ema.pth"
    # "checkpoint", "onnx", or "auto" (infer from the --weights path).
    backend: str = "auto"
    # Model size used for training; inference reads the class from the checkpoint.
    variant: str = "nano"
    bag_class: str = "bag"
    person_class: str = "person"
    # RF-DETR predict() confidence threshold.
    conf: float = 0.25
    # IoU gate for the lightweight tracker (RF-DETR does not use NMS).
    iou: float = 0.5
    # Square input resolution. RF-DETR Nano defaults to 384 (must be divisible by 32).
    resolution: int = 384
    device: str = "cpu"
    # Frames a missing detection keeps its track alive.
    tracker_max_age: int = 8
    # Legacy keys, kept so older JSON configs still load; ignored now.
    prompts: list[str] = field(default_factory=lambda: ["person", "white sack"])
    imgsz: int = 0
    tracker: str = ""


@dataclass
class CounterConfig:
    # Minimum number of frames a track must be observed in its origin zone
    # before a transition is trusted. Filters one-frame flickers.
    min_origin_frames: int = 2
    # Frames a track may be unseen before we finalise it.
    max_lost_frames: int = 24
    # ID-switch stitching: a new track appearing within
    # min(stitch_max_radius, stitch_radius + stitch_speed * gap_frames)
    # (normalised units) of a track lost no more than stitch_max_gap frames ago
    # inherits its zone history. A bag being lifted onto a shoulder jumps up to
    # ~0.05 of frame height per frame at 8 fps.
    stitch_radius: float = 0.08
    stitch_speed: float = 0.03
    stitch_max_radius: float = 0.30
    stitch_max_gap: int = 16
    # Once a lost track has been adopted, only fragments born within this many
    # frames of that adoption may adopt it too (siblings of one bag); a later
    # newcomer is a different bag picked up from the same spot.
    stitch_sibling_frames: int = 0
    # A newcomer this close to a track seen in the same frame is a duplicate
    # box on the same bag: it copies that live track rather than a lost one.
    duplicate_radius: float = 0.06
    # ...or whose box is a "carrier box" of a live track's box (or vice versa):
    # the detector sometimes boxes bag + carrier as one bag. Such a box shares
    # the top edge and x-range of the tight bag box and extends further down.
    carrier_x_margin: float = 0.03
    carrier_top_tol: float = 0.08
    carrier_min_extra_height: float = 0.08
    # Hand-up handover: a new track born in the transit band, or inside the
    # truck zone within handover_edge_dist of its boundary (the tailgate), is
    # a bag lifted off the pile whose own track only started after the lift
    # (the pile's track stays on the next bag beneath). It is given dock
    # history provisionally and counted once it has moved at least
    # handover_min_travel further from the tailgate than where it was born.
    # A static bag on the bed re-detected near the tailgate never moves, and a
    # false "bag" on a worker's head drifting along the tailgate never gets
    # further from it, so neither counts.
    handover_edge_dist: float = 0.25
    handover_min_travel: float = 0.06
    # The reverse handover (stack-top phase): a track born inside the truck
    # zone at the tailgate that then travels handover_min_travel into the dock
    # was handed *down* and counts as unloaded, once it has been seen on the
    # dock for this many frames. A bag that dips below the tailgate for a
    # moment and comes back is not an unload (and not a reload).
    handover_unload_frames: int = 8
    # A track id the tracker revives after at least this many unseen frames,
    # inside the zone it was settled in, after its bag was already counted
    # leaving that zone, is the next bag picked up from the same spot.
    revive_gap_frames: int = 4
    # If a settled track is last seen in the transit band heading for the
    # other zone and is then lost, count the transition (flagged inferred).
    infer_lost_in_transit: bool = True
    # Fragments of one bag's track share a root id; a second event from the
    # same root and direction within this many frames is dropped.
    dedup_frames: int = 80


@dataclass
class Config:
    zones: Zones = field(default_factory=Zones)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    counter: CounterConfig = field(default_factory=CounterConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        raw = json.loads(Path(path).read_text())
        z = raw.get("zones", {})
        return cls(
            zones=Zones(
                truck=[tuple(p) for p in z.get("truck", DEFAULT_TRUCK)],
                dock=[tuple(p) for p in z.get("dock", DEFAULT_DOCK)],
            ),
            detector=DetectorConfig(**raw.get("detector", {})),
            counter=CounterConfig(**raw.get("counter", {})),
        )

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
