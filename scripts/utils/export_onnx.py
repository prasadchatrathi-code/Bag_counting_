"""Export a fine-tuned RF-DETR checkpoint (.pt/.pth) to ONNX.

Usage:
  mise run export -- --weights runs/detect/bags_full/checkpoint_best_ema.pth \
                      --out exports/bags_full_onnx --device cpu

Run inference on the exported artifact with:
  mise run count -- clips/demo.mp4 \
    --weights exports/bags_full_onnx/rfdetr-nano.onnx --backend onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr import RFDETR

from bagcount.pipeline import _is_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description="Export an RF-DETR checkpoint to ONNX")
    ap.add_argument("--weights", type=Path, required=True, help="RF-DETR .pt/.pth checkpoint")
    ap.add_argument("--out", type=Path, default=Path("exports/bags_onnx"))
    ap.add_argument("--resolution", type=int, default=384,
                    help="square export size (Nano default 384, divisible by 32)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if not _is_checkpoint(str(a.weights)):
        raise SystemExit(f"expected a .pt/.pth checkpoint, got {a.weights}")
    if not a.weights.exists():
        raise SystemExit(f"checkpoint not found: {a.weights}")
    model = RFDETR.from_checkpoint(str(a.weights), device=a.device)
    out = model.export(
        format="onnx",
        output_dir=str(a.out),
        shape=(a.resolution, a.resolution),
        opset_version=a.opset,
    )
    print(f"exported ONNX artifact: {out}")
    print(f"count with: mise run count -- <video> --weights {out} --backend onnx --device {a.device}")


if __name__ == "__main__":
    main()
