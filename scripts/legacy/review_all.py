"""Render label contact sheets for both train and val splits.

Usage:
  mise run review-all -- datasets/bags_full out/review_full --per-sheet 12
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Review train and val label sheets")
    ap.add_argument("root", type=Path, help="dataset directory (contains images/ and labels/)")
    ap.add_argument("out", type=Path, help="parent output directory")
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--cols", type=int, default=3)
    a = ap.parse_args()

    script = Path(__file__).with_name("review_labels.py")
    for split in ("train", "val"):
        cmd = [sys.executable, str(script), str(a.root), split, str(a.out / split),
               "--per-sheet", str(a.per_sheet), "--cols", str(a.cols)]
        print("$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
