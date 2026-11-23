"""Count labelled boxes in a dataset split.

Usage:
  mise run check-labels -- datasets/bags_full train
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Count labelled boxes in a dataset split")
    ap.add_argument("root", type=Path, help="dataset directory (contains labels/)")
    ap.add_argument("split", nargs="?", default="train", help="train or val (default: train)")
    a = ap.parse_args()

    files = sorted((a.root / "labels" / a.split).glob("*.txt"))
    person = bag = other = 0
    for f in files:
        for line in f.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "0":
                person += 1
            elif parts[0] == "1":
                bag += 1
            else:
                other += 1
    print(f"{a.root} [{a.split}]: images={len(files)} person_boxes={person} bag_boxes={bag} other={other}")
    if len(files) == 0:
        raise SystemExit("no label files found — check the dataset path and split")
    if bag == 0:
        print("WARNING: no bag boxes — the detector cannot learn to count bags from this split")


if __name__ == "__main__":
    main()
