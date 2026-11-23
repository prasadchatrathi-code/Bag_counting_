"""Extract a downsampled clip from the source CCTV video.

Usage: mise run clip -- <start_sec> <duration_sec> <name> [--fps 8] [--width 1280]
Writes clips/<name>.mp4 (clips/ is git-ignored). Uses OpenCV, so no external
ffmpeg binary is required.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

SOURCE = Path("assets/bags-loading-clip-5min.mp4")


def _writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """mp4v first on Linux (no H.264 encoder in bundled FFmpeg)."""
    fourccs = ("mp4v", "avc1") if sys.platform.startswith("linux") else ("avc1", "mp4v")
    for fourcc in fourccs:
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w
        w.release()
    raise RuntimeError(f"cannot open video writer for {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("start", type=float)
    p.add_argument("duration", type=float)
    p.add_argument("name")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--source", type=Path, default=SOURCE)
    a = p.parse_args()

    out = Path("clips") / f"{a.name}.mp4"
    out.parent.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(a.source))
    if not cap.isOpened():
        raise SystemExit(f"cannot open source video: {a.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, a.width / sw) if sw else 1.0
    size = (int(sw * scale) // 2 * 2, int(sh * scale) // 2 * 2)  # even dims for mp4v
    step = max(1, round(src_fps / a.fps))
    max_frames = max(1, int(a.duration * a.fps))

    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    writer = _writer(out, a.fps, size)
    written = read = 0
    try:
        while written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if read % step == 0:
                if size != (sw, sh):
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                writer.write(frame)
                written += 1
            read += 1
    finally:
        cap.release()
        writer.release()
    print(f"{out} ({written} frames, {size[0]}x{size[1]}, {a.fps} fps)")


if __name__ == "__main__":
    main()
