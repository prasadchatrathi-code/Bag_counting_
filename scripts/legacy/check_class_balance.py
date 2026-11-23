"""Verify label counts and class balance before training.

Usage:
  mise run check-balance -- datasets/bags_full --class bag

Exits non-zero when a check fails, so it can gate training in scripts/CI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bagcount.label_check import (
    DEFAULT_CLASS,
    DEFAULT_MAX_CLASS_RATIO,
    DEFAULT_MAX_SPLIT_RATIO,
    DEFAULT_MIN_CLASS_FRACTION,
    DEFAULT_MIN_TRAIN,
    DEFAULT_MIN_VAL,
    verify_labels,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify label counts and class balance")
    ap.add_argument("root", type=Path, help="dataset directory (contains data.yaml and labels/)")
    ap.add_argument("--data", type=Path, default=None,
                    help="path to data.yaml (default: <root>/data.yaml)")
    ap.add_argument("--class", dest="class_spec", default=DEFAULT_CLASS,
                    help="class id or name to verify (default: bag)")
    ap.add_argument("--min-train", type=int, default=DEFAULT_MIN_TRAIN)
    ap.add_argument("--min-val", type=int, default=DEFAULT_MIN_VAL)
    ap.add_argument("--min-class-fraction", type=float, default=DEFAULT_MIN_CLASS_FRACTION)
    ap.add_argument("--max-split-ratio", type=float, default=DEFAULT_MAX_SPLIT_RATIO)
    ap.add_argument("--max-class-ratio", type=float, default=DEFAULT_MAX_CLASS_RATIO,
                    help="max most/least class count ratio within a split")
    a = ap.parse_args()

    data_yaml = a.data or a.root / "data.yaml"
    report = verify_labels(
        data_yaml,
        class_spec=a.class_spec,
        min_train=a.min_train,
        min_val=a.min_val,
        min_class_fraction=a.min_class_fraction,
        max_split_ratio=a.max_split_ratio,
        max_class_ratio=a.max_class_ratio,
    )
    print(report.render())
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
