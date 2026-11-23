"""Merge labelled sources into one training set, with per-source frame filters.

Each source is a YOLO dataset dir (images/{train,val}, labels/{train,val}).
Rules let us drop frames that a particular labeller is known to get wrong,
e.g. a self-labelled round from a model that was blind to the first hour.

Usage:
  python scripts/legacy/build_dataset.py datasets/bags_v5 \
      --src datasets/bags_v3 --drop-prefix early_ --drop-full-before 4200 \
      --src datasets/bags_v4 --drop-prefix early_ --drop-full-before 4200 --keep-every 2 \
      --src datasets/owl_early \
      --src datasets/probe_owl

--drop-full-before N drops frames named full_<idx> with idx < N (idx is the
frame number in the source clip the labeller read). Later sources win on a
duplicate file name. Files are hard-linked, not copied.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


class SrcAction(argparse.Action):
    def __call__(self, parser, ns, values, option_string=None):
        ns.sources.append({"path": Path(values), "drop_prefix": [], "drop_full_before": 0, "keep_every": 1})


class RuleAction(argparse.Action):
    def __call__(self, parser, ns, values, option_string=None):
        if not ns.sources:
            parser.error(f"{option_string} must follow --src")
        key = option_string.lstrip("-").replace("-", "_")
        if key == "drop_prefix":
            ns.sources[-1][key].append(values)
        else:
            ns.sources[-1][key] = int(values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", action=SrcAction, dest="sources", default=[])
    ap.add_argument("--drop-prefix", action=RuleAction, dest="sources")
    ap.add_argument("--drop-full-before", action=RuleAction, dest="sources")
    ap.add_argument("--keep-every", action=RuleAction, dest="sources")
    a = ap.parse_args()

    for split in ("train", "val"):
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (a.out / "data.yaml").write_text(
        f"path: {a.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: bag\n"
    )
    full_re = re.compile(r"^full_(\d+)$")
    totals = {}
    for src in a.sources:
        kept = dropped = bags = 0
        for split in ("train", "val"):
            imgs = sorted((src["path"] / "images" / split).glob("*.jpg"))
            for i, img in enumerate(imgs):
                stem = img.stem
                drop = any(stem.startswith(p) for p in src["drop_prefix"])
                m = full_re.match(stem)
                if m and int(m.group(1)) < src["drop_full_before"]:
                    drop = True
                if i % src["keep_every"]:
                    drop = True
                if drop:
                    dropped += 1
                    continue
                lab = src["path"] / "labels" / split / f"{stem}.txt"
                if not lab.exists():
                    continue
                for dst, s in ((a.out / "images" / split / img.name, img), (a.out / "labels" / split / lab.name, lab)):
                    if dst.exists():
                        dst.unlink()
                    os.link(s, dst)
                kept += 1
                bags += sum(1 for l in lab.read_text().splitlines() if l.startswith("1 "))
        totals[src["path"].name] = (kept, dropped, bags)
        print(f"{src['path'].name}: kept {kept} frames ({bags} bag boxes), dropped {dropped}")
    n_train = len(list((a.out / "images" / "train").glob("*.jpg")))
    n_val = len(list((a.out / "images" / "val").glob("*.jpg")))
    print(f"total train={n_train} val={n_val} -> {a.out}")


if __name__ == "__main__":
    main()
