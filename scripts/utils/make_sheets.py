"""Contact sheets for hand-counting ground truth.

Usage: python scripts/utils/make_sheets.py clips/mid_1300.mp4 out/sheets_mid --start 0 --duration 180 --fps 2
Each tile is stamped with the clip time in seconds; count the moment a bag's
centre crosses the tailgate edge and record it in data/truth/<name>.csv.
"""
import argparse
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--fps", type=float, default=2.0, help="tiles per second of video")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--crop", type=float, nargs=2, default=(0.22, 0.80), metavar=("X0", "X1"),
                    help="horizontal crop as fractions of width")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(src_fps / a.fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * src_fps))
    per_sheet = a.cols * a.rows
    tiles, sheet = [], 0
    n = int(a.duration * src_fps)
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        if i % step:
            continue
        t = a.start + i / src_fps
        H, W = f.shape[:2]
        c = f[:, int(a.crop[0] * W):int(a.crop[1] * W)]
        c = cv2.resize(c, (372, 360))
        cv2.rectangle(c, (0, 0), (90, 22), (0, 0, 0), -1)
        cv2.putText(c, f"{t:5.1f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        tiles.append(c)
        if len(tiles) == per_sheet:
            rows = [cv2.hconcat(tiles[j:j + a.cols]) for j in range(0, per_sheet, a.cols)]
            cv2.imwrite(str(a.out / f"sheet_{sheet:02d}.jpg"), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
            sheet += 1
            tiles = []
    if tiles:
        while len(tiles) % a.cols:
            tiles.append(tiles[-1] * 0)
        rows = [cv2.hconcat(tiles[j:j + a.cols]) for j in range(0, len(tiles), a.cols)]
        cv2.imwrite(str(a.out / f"sheet_{sheet:02d}.jpg"), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
        sheet += 1
    print(f"wrote {sheet} sheets to {a.out}")


if __name__ == "__main__":
    main()
