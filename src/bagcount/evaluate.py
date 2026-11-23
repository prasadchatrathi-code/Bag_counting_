"""Compare predicted events against hand-counted ground truth.

Ground truth CSV columns: time_s, direction (loaded|unloaded). Times are in the
same clock as the predicted events (seconds from clip start). Predicted and true
events are matched by direction within ±tolerance seconds (each prediction,
in time order, takes the earliest unmatched truth event in range).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalResult:
    tp: int
    fp: int
    fn: int
    pred_loaded: int
    pred_unloaded: int
    true_loaded: int
    true_unloaded: int
    unmatched_pred: list[tuple[float, str]]
    unmatched_true: list[tuple[float, str]]

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0


def _read(path: Path, time_col: str) -> list[tuple[float, str]]:
    with Path(path).open() as f:
        return [(float(r[time_col]), r["direction"]) for r in csv.DictReader(f)]


def evaluate(pred_csv: Path, truth_csv: Path, tolerance_s: float = 4.0,
             window: tuple[float, float] | None = None) -> EvalResult:
    pred = _read(pred_csv, "time_s")
    true = _read(truth_csv, "time_s")
    if window:
        lo, hi = window
        pred = [p for p in pred if lo <= p[0] <= hi]
        true = [t for t in true if lo <= t[0] <= hi]
    used = set()
    tp = 0
    unmatched_pred = []
    # Predictions in time order each take the *earliest* unmatched truth event
    # within tolerance (not the nearest): with events a few seconds apart the
    # nearest-first choice can steal a later truth and leave an earlier one
    # unmatched, understating the score.
    order = sorted(range(len(true)), key=lambda i: true[i][0])
    for pt, pd in sorted(pred):
        best = None
        for i in order:
            tt, td = true[i]
            if i in used or td != pd or abs(tt - pt) > tolerance_s:
                continue
            best = i
            break
        if best is None:
            unmatched_pred.append((pt, pd))
        else:
            used.add(best)
            tp += 1
    unmatched_true = [t for i, t in enumerate(true) if i not in used]
    return EvalResult(
        tp=tp, fp=len(unmatched_pred), fn=len(unmatched_true),
        pred_loaded=sum(d == "loaded" for _, d in pred),
        pred_unloaded=sum(d == "unloaded" for _, d in pred),
        true_loaded=sum(d == "loaded" for _, d in true),
        true_unloaded=sum(d == "unloaded" for _, d in true),
        unmatched_pred=unmatched_pred, unmatched_true=unmatched_true,
    )


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_csv", type=Path)
    ap.add_argument("truth_csv", type=Path)
    ap.add_argument("--tolerance", type=float, default=4.0)
    ap.add_argument("--window", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    a = ap.parse_args()
    r = evaluate(a.pred_csv, a.truth_csv, a.tolerance, tuple(a.window) if a.window else None)
    print(f"pred loaded={r.pred_loaded} unloaded={r.pred_unloaded} | true loaded={r.true_loaded} unloaded={r.true_unloaded}")
    print(f"TP={r.tp} FP={r.fp} FN={r.fn}  precision={r.precision:.3f} recall={r.recall:.3f}")
    if r.unmatched_pred:
        print("false positives:", ", ".join(f"{t:.1f}s {d}" for t, d in r.unmatched_pred))
    if r.unmatched_true:
        print("missed:", ", ".join(f"{t:.1f}s {d}" for t, d in r.unmatched_true))


if __name__ == "__main__":
    main()
