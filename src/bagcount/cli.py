"""Command line entry point.

  mise run count -- clips/mid_1300.mp4 --out out/mid --video
"""
from __future__ import annotations

import argparse
import copy
import logging
import signal
from datetime import datetime
from pathlib import Path

from .config import Config
from .pipeline import Pipeline, parse_line, parse_time


def main() -> None:
    ap = argparse.ArgumentParser(description="Count bags loaded/unloaded from CCTV video")
    ap.add_argument("source", help="video file or stream URL")
    ap.add_argument("--out", type=Path, default=None, help="output directory (legacy alternative to --project/--name)")
    ap.add_argument("--project", type=Path, default=None, help="root output folder; combine with --name")
    ap.add_argument("--name", default=None, help="output subfolder name inside --project")
    ap.add_argument("--config", default=None, help="JSON config (zones, detector, counter)")
    ap.add_argument("--weights", default=None, help="override detector weights (.pt/.pth or .onnx)")
    ap.add_argument("--backend", choices=["auto", "checkpoint", "onnx"], default=None,
                    help="detector backend (default: auto-detect from --weights)")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--conf-series", type=float, nargs="+", metavar="THRESHOLD",
                    help="run each threshold in a new numbered folder under OUT")
    ap.add_argument("--iou", type=float, default=None, help="tracker IoU gate (default 0.5)")
    ap.add_argument("--tracker-max-age", type=int, default=None, metavar="FRAMES",
                    help="keep a lost track id alive for this many frames (default 30)")
    ap.add_argument("--line", type=parse_line, default=None, metavar="X1,Y1,X2,Y2",
                    help="count line in normalised 0..1 coordinates, e.g. 0.28,0.68,0.78,0.68")
    ap.add_argument("--no-line", action="store_true",
                    help="disable line gating (legacy zone-only counting)")
    ap.add_argument("--device", default=None, help="inference device: cpu, cuda, cuda:0, mps")
    ap.add_argument("--resolution", type=int, default=None, help="RF-DETR input resolution (default 384)")
    ap.add_argument("--video", action="store_true", help="write annotated.mp4")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (default 0 = process the full video)")
    ap.add_argument("--start", type=parse_time, default=0.0, metavar="TIME",
                    help="seek to TIME in the source: seconds, 1hr26m30 or MM:SS (default 0)")
    ap.add_argument("--duration", type=parse_time, default=0.0, metavar="TIME",
                    help="process only TIME seconds from --start (0 = to end of source)")
    ap.add_argument("--end", type=parse_time, default=None, metavar="TIME",
                    help="stop at this absolute source time (alternative to --duration)")
    ap.add_argument("--fps", type=float, default=None, help="override source fps (for timestamps)")
    ap.add_argument("--wall-start", default=None, help="camera clock at frame 0, e.g. 2026-06-10T11:35:00")
    ap.add_argument("--debug", action="store_true", help="log track adoptions and suppressed events to each run's debug.log")
    a = ap.parse_args()
    if a.project is not None or a.name is not None:
        if a.project is None or not a.name:
            ap.error("--project and --name must be supplied together")
        if a.out is not None:
            ap.error("use either --out or --project with --name")
        out_root = a.project / a.name
    else:
        out_root = a.out or Path("out/run")
    if a.conf_series and a.conf is not None:
        ap.error("use either --conf or --conf-series, not both")
    thresholds = a.conf_series or [a.conf]
    if a.conf_series and any(not 0 <= value <= 1 for value in thresholds):
        ap.error("confidence thresholds must be between 0 and 1")
    if a.max_frames < 0:
        ap.error("--max-frames must be non-negative")
    if a.tracker_max_age is not None and a.tracker_max_age < 0:
        ap.error("--tracker-max-age must be non-negative")
    if a.no_line and a.line is not None:
        ap.error("use either --line or --no-line, not both")
    if a.duration and a.end is not None:
        ap.error("use either --duration or --end, not both")
    if a.max_frames and a.duration:
        ap.error("use either --max-frames or --duration, not both")
    if a.end is not None:
        if a.end <= a.start:
            ap.error("--end must be greater than --start")
        duration_s = a.end - a.start
    else:
        duration_s = a.duration
    base_cfg = Config.load(a.config)
    if a.weights:
        base_cfg.detector.weights = a.weights
    if a.backend is not None:
        base_cfg.detector.backend = a.backend
    if a.iou is not None:
        base_cfg.detector.iou = a.iou
    if a.tracker_max_age is not None:
        base_cfg.detector.tracker_max_age = a.tracker_max_age
    if a.line is not None:
        base_cfg.zones.cross_line = a.line
    if a.no_line:
        base_cfg.counter.require_line_crossing = False
    if a.device is not None:
        base_cfg.detector.device = a.device
    if a.resolution is not None:
        base_cfg.detector.resolution = a.resolution
    ws = datetime.fromisoformat(a.wall_start) if a.wall_start else None
    existing_run_ids = []
    if a.conf_series and out_root.exists():
        for candidate in out_root.glob("run_*"):
            suffix = candidate.name.removeprefix("run_")
            if suffix.isdigit():
                existing_run_ids.append(int(suffix))
    first_run_id = max(existing_run_ids, default=0) + 1
    for series_idx, threshold in enumerate(thresholds, start=1):
        cfg = copy.deepcopy(base_cfg)
        if threshold is not None:
            cfg.detector.conf = threshold
        run_id = first_run_id + series_idx - 1
        run_out = out_root / f"run_{run_id}" if a.conf_series else out_root
        print(f"Run {series_idx}/{len(thresholds)} (run_{run_id}): confidence={cfg.detector.conf:g}, output={run_out}", flush=True)
        if a.debug:
            run_out.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(filename=run_out / "debug.log", filemode="w", level=logging.DEBUG,
                                format="%(message)s", force=True)
        p = Pipeline(cfg, run_out, save_video=a.video, wall_start=ws, source_fps=a.fps)

        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

        def _request_stop(signum, _frame) -> None:
            print(f"\n{signal.Signals(signum).name} received; finishing current frame and closing files...",
                  flush=True)
            p.stop = True

        for sig in previous:
            signal.signal(sig, _request_stop)
        try:
            s = p.run(a.source, max_frames=a.max_frames, start_s=a.start, duration_s=duration_s)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if s.start_s or s.duration_s:
            window = (f" window={s.start_s:g}s..{s.start_s + s.duration_s:g}s" if s.duration_s
                      else f" window={s.start_s:g}s..end")
        else:
            window = ""
        print(f"frames={s.frames}{window} loaded={s.loaded} unloaded={s.unloaded} on_truck={s.on_truck}")
        print(f"events: {s.events_csv}")


if __name__ == "__main__":
    main()
