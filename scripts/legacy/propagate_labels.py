"""Densify bag labels by propagating confident detections through each carry.

The fine-tuned detector fires on a carried bag only in some frames. For every
confident detection (seed) we run a visual tracker (CSRT) forward and backward
through neighbouring frames and keep the tracked box while it stays inside the
scene ROI, keeps a sane size, and keeps moving. Those propagated boxes become
labels for frames where the detector was blind, which is exactly the data the
next training round needs.

Usage:
  mise run propagate -- clips/early_1140.mp4 datasets/prop_early \
      --weights runs/detect/bags_full/checkpoint_best_ema.pth \
      --seed-conf 0.6 --extend 24 --every 3 --prefix early_ --device cpu

Classes: 0 = person, 1 = bag.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from bagcount.config import Config

PERSON, BAG = 0, 1


def _load_model(weights: str, device: str):
    """Load a native RF-DETR checkpoint (.pt/.pth) or an ONNX artifact."""
    from bagcount.pipeline import _is_onnx, resolve_onnx_path

    onnx = _is_onnx(weights)
    if onnx:
        from rfdetr.inference import RFDETRInference
        model = RFDETRInference(resolve_onnx_path(weights), device=device)
    else:
        from rfdetr import RFDETR
        model = RFDETR.from_checkpoint(weights, device=device)
    return model, onnx


def in_roi(cfg: Config, cx: float, cy: float) -> bool:
    pts = [np.array(cfg.zones.truck, np.float32), np.array(cfg.zones.dock, np.float32)]
    if any(cv2.pointPolygonTest(p, (cx, cy), False) >= 0 for p in pts):
        return True
    truck_bottom = max(p[1] for p in cfg.zones.truck)
    dock_top = min(p[1] for p in cfg.zones.dock)
    xs = [p[0] for p in cfg.zones.truck + cfg.zones.dock]
    return truck_bottom <= cy <= dock_top and min(xs) <= cx <= max(xs)


def moving(a: np.ndarray, b: np.ndarray, box, thresh: float) -> bool:
    H, W = a.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return False
    ga = cv2.cvtColor(a[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.abs(ga - gb).mean() >= thresh


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def propagate(frames: list[np.ndarray], start: int, box, direction: int, a, cfg: Config):
    """Yield (frame_idx, box) walking from `start` in `direction` (+1/-1)."""
    H, W = frames[start].shape[:2]
    x1, y1, x2, y2 = box
    w0, h0 = x2 - x1, y2 - y1
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frames[start], (int(x1), int(y1), int(w0), int(h0)))
    i = start
    for _ in range(a.extend):
        i += direction
        if i < 0 or i >= len(frames):
            return
        ok, (x, y, w, h) = tracker.update(frames[i])
        if not ok:
            return
        b = np.array([x, y, x + w, y + h], dtype=np.float32)
        cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
        if not in_roi(cfg, cx, cy):
            return
        if not (0.6 <= w / w0 <= 1.7 and 0.6 <= h / h0 <= 1.7):
            return
        j = max(0, min(len(frames) - 1, i - direction * a.motion_gap))
        if not moving(frames[i], frames[j], b, a.motion_thresh):
            return
        yield i, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out", type=Path)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds (0 = to end)")
    ap.add_argument("--seed-conf", type=float, default=0.6)
    ap.add_argument("--person-conf", type=float, default=0.3)
    ap.add_argument("--extend", type=int, default=24, help="max frames to propagate each way")
    ap.add_argument("--motion-gap", type=int, default=6)
    ap.add_argument("--motion-thresh", type=float, default=12.0)
    ap.add_argument("--every", type=int, default=3, help="write every Nth frame that has labels")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--prefix", default="p")
    ap.add_argument("--chunk", type=int, default=400, help="frames held in memory at once")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    cfg = Config()
    model, is_onnx = _load_model(a.weights or cfg.detector.weights, a.device)
    names = model.class_names
    if isinstance(names, dict):
        name_map = {str(v).lower(): int(k) for k, v in names.items()}
    else:
        name_map = {str(v).lower(): i for i, v in enumerate(names)}
    person_id, bag_id = name_map.get("person", PERSON), name_map.get("bag", BAG)
    for split in ("train", "val"):
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (a.out / "data.yaml").write_text(
        f"path: {a.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: bag\n"
    )

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = int(a.start * fps)
    last = int((a.start + a.duration) * fps) if a.duration else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    stats = defaultdict(int)
    sample_i = 0
    overlap = a.extend
    base = first
    buf: list[np.ndarray] = []
    done = False
    while not done:
        while len(buf) < a.chunk + overlap and base + len(buf) < last:
            ok, f = cap.read()
            if not ok:
                done = True
                break
            buf.append(f)
        else:
            if base + len(buf) >= last:
                done = True
        if not buf:
            break
        H, W = buf[0].shape[:2]
        persons: dict[int, list] = {}
        seeds: dict[int, list] = {}
        for i, f in enumerate(buf):
            kwargs = {"threshold": min(a.seed_conf, a.person_conf)}
            if not is_onnx:  # ONNX has a fixed input shape from export
                kwargs["shape"] = (cfg.detector.resolution, cfg.detector.resolution)
            det = model.predict(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), **kwargs)
            ps, bs = [], []
            if det is not None and len(det) > 0:
                for box, sc, c in zip(np.asarray(det.xyxy), np.asarray(det.confidence),
                                      np.asarray(det.class_id)):
                    if int(c) == person_id and sc >= a.person_conf:
                        ps.append(box)
                    elif int(c) == bag_id and sc >= a.seed_conf:
                        bs.append(box)
            persons[i], seeds[i] = ps, bs
        bags: dict[int, list] = defaultdict(list)
        for i, bs in seeds.items():
            for b in bs:
                bags[i].append(b)
                stats["seeds"] += 1
        for i in sorted(seeds):
            for b in seeds[i]:
                for direction in (1, -1):
                    for j, pb in propagate(buf, i, b, direction, a, cfg):
                        if any(iou(pb, q) > 0.5 for q in bags[j]):
                            break  # reached a frame that already has this bag
                        bags[j].append(pb)
                        stats["propagated"] += 1
        # Write frames from the non-overlap region (the tail is re-read next chunk).
        limit = len(buf) if done else a.chunk
        for i in range(limit):
            if not bags.get(i):
                continue
            if sample_i % a.every:
                sample_i += 1
                continue
            split = "val" if (sample_i // a.every) % a.val_every == 0 else "train"
            sample_i += 1
            name = f"{a.prefix}{base + i:06d}"
            cv2.imwrite(str(a.out / "images" / split / f"{name}.jpg"), buf[i], [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = []
            for cls, boxes in ((PERSON, persons[i]), (BAG, bags[i])):
                for x1, y1, x2, y2 in boxes:
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                    lines.append(f"{cls} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            (a.out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
            stats["frames"] += 1
            stats["bags"] += len(bags[i])
        print(f"chunk @ {base} ({base / fps:.0f}s): {dict(stats)}", flush=True)
        if done:
            break
        buf = buf[a.chunk:]
        base += a.chunk
    print("done", dict(stats))


if __name__ == "__main__":
    main()
