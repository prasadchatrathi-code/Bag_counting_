"""Verify the environment, dependencies and model artifacts.

Usage:
  mise run verify
  mise run verify -- --smoke --device cpu

With --smoke, each discovered checkpoint / ONNX artifact is actually loaded
and run on a blank frame, so a broken export is caught before a full video run.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _check(name: str, ok: bool, detail: str = "", required: bool = True) -> bool:
    """Print a check result. Artifacts are informational (WARN) on a fresh setup."""
    status = "PASS" if ok else ("WARN" if not required else "FAIL")
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify the bag-counting setup")
    ap.add_argument("--smoke", action="store_true", help="load each model and predict on a blank frame")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    results: list[bool] = []

    # 1. Python / tools -----------------------------------------------------
    results.append(_check("python >= 3.12", sys.version_info >= (3, 12),
                          ".".join(map(str, sys.version_info[:3]))))
    for tool in ("mise", "uv"):
        results.append(_check(f"{tool} on PATH", shutil.which(tool) is not None,
                              shutil.which(tool) or "not found"))

    # 2. Dependencies -------------------------------------------------------
    for mod in ("numpy", "cv2", "torch", "rfdetr", "fastapi", "lap", "onnx", "onnxruntime"):
        try:
            m = importlib.import_module(mod)
            detail = getattr(m, "__version__", "")
        except Exception as e:
            m, detail = None, str(e)
        results.append(_check(f"import {mod}", m is not None, detail))

    # 3. Artifacts ----------------------------------------------------------
    datasets = sorted((ROOT / "datasets").rglob("data.yaml"))
    _check("labelled datasets", bool(datasets),
           ", ".join(p.parent.name for p in datasets) or "none (run mise run label first)",
           required=False)

    checkpoints = sorted([* (ROOT / "runs" / "detect").glob("*/*.pth"),
                          * (ROOT / "runs" / "detect").glob("*/*.pt")])
    _check("native checkpoints", bool(checkpoints),
           ", ".join(str(p.relative_to(ROOT)) for p in checkpoints) or "none (run mise run train first)",
           required=False)

    onnx = sorted((ROOT / "exports").glob("**/*.onnx"))
    _check("ONNX artifacts", bool(onnx),
           ", ".join(str(p.relative_to(ROOT)) for p in onnx) or "none (run mise run export first)",
           required=False)

    # 4. Optional smoke inference ------------------------------------------
    if a.smoke:
        import cv2
        import numpy as np

        frame = np.zeros((384, 384, 3), dtype=np.uint8)
        for p in checkpoints:
            try:
                from rfdetr import RFDETR
                model = RFDETR.from_checkpoint(str(p), device=a.device)
                d = model.predict(frame, threshold=0.25, shape=(384, 384))
                results.append(_check(f"smoke {p.name}", True,
                                      f"classes={model.class_names}, boxes={len(d)}"))
            except Exception as e:
                results.append(_check(f"smoke {p.name}", False, str(e)))
        for p in onnx:
            try:
                from rfdetr.inference import RFDETRInference
                model = RFDETRInference(str(p), device=a.device)
                d = model.predict(frame, threshold=0.25)
                results.append(_check(f"smoke {p.name}", True,
                                      f"classes={model.class_names}, boxes={len(d)}"))
            except Exception as e:
                results.append(_check(f"smoke {p.name}", False, str(e)))

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed" + (f" ({failed} FAILED)" if failed else ""))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
