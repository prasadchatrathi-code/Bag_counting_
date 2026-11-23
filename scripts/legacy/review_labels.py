"""Render contact sheets of labelled frames to eyeball label quality.

Usage: python scripts/legacy/review_labels.py datasets/bags train out/review --per-sheet 12 --max-sheets 6
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

COL = {0: (255, 120, 0), 1: (0, 0, 255)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("split")
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--max-sheets", type=int, default=6)
    ap.add_argument("--only-with-bags", action="store_true")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--cols", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((a.root / "images" / a.split).glob("*.jpg"))
    tiles, sheet = [], 0

    def write_sheet(batch: list) -> None:
        """Tile a batch into equal-width rows, padding the last row."""
        nonlocal sheet
        if not batch:
            return
        blank = np.zeros_like(batch[0])
        rows_ = []
        for i in range(0, len(batch), a.cols):
            row = batch[i:i + a.cols]
            row = row + [blank] * (a.cols - len(row))
            rows_.append(cv2.hconcat(row))
        cv2.imwrite(str(a.out / f"sheet_{sheet:02d}.jpg"), cv2.vconcat(rows_))
        sheet += 1

    for p in imgs:
        lab = a.root / "labels" / a.split / (p.stem + ".txt")
        rows = [l.split() for l in lab.read_text().splitlines()] if lab.exists() else []
        if a.only_with_bags and not any(r[0] == "1" for r in rows):
            continue
        f = cv2.imread(str(p))
        H, W = f.shape[:2]
        for c, cx, cy, w, h in rows:
            c = int(c)
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            cv2.rectangle(f, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), COL[c], 3)
        cv2.putText(f, p.stem, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(cv2.resize(f, (a.tile, a.tile * H // W)))
        if len(tiles) == a.per_sheet:
            write_sheet(tiles)
            tiles = []
            if sheet >= a.max_sheets:
                break
    if sheet < a.max_sheets:
        write_sheet(tiles)  # flush a final partial sheet
    print(f"wrote {sheet} sheets to {a.out}")


if __name__ == "__main__":
    main()
