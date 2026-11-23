"""Bootstrap a YOLO training set with Grounding DINO pseudo-labels.

Samples frames from a video, runs Grounding DINO with the prompts
"white sack. person.", filters the sack boxes (size, scene ROI, must be near a
person so stacked sacks are excluded), and writes YOLO-format labels.

Usage:
  python scripts/legacy/pseudo_label.py clips/full_1fps.mp4 datasets/bags --every 20 --val-every 5

Classes: 0 = person, 1 = bag (a sack being carried / handled).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from bagcount.config import Config

MODEL_ID = "IDEA-Research/grounding-dino-base"
PERSON, BAG = 0, 1


def iou_expand(person: np.ndarray, bag: np.ndarray, grow: float) -> bool:
    """True if `bag` box intersects `person` box grown by `grow` (fraction)."""
    px1, py1, px2, py2 = person
    w, h = px2 - px1, py2 - py1
    px1, px2 = px1 - grow * w, px2 + grow * w
    py1, py2 = py1 - grow * h, py2 + grow * h * 0.5
    bx1, by1, bx2, by2 = bag
    return bx1 < px2 and bx2 > px1 and by1 < py2 and by2 > py1


def moving(frame, prev_f, next_f, box, thresh: float) -> bool:
    """True if the box region differs from both the earlier and the later frame."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return False
    g = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    d_prev = np.abs(g - cv2.cvtColor(prev_f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)).mean()
    d_next = np.abs(g - cv2.cvtColor(next_f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)).mean()
    return min(d_prev, d_next) >= thresh


def in_roi(cfg: Config, cx: float, cy: float, W: int, H: int) -> bool:
    pts = [np.array(cfg.zones.truck, np.float32), np.array(cfg.zones.dock, np.float32)]
    x, y = cx / W, cy / H
    # Accept truck, dock, and the transit band between them (y between the two).
    if any(cv2.pointPolygonTest(p, (x, y), False) >= 0 for p in pts):
        return True
    truck_bottom = max(p[1] for p in cfg.zones.truck)
    dock_top = min(p[1] for p in cfg.zones.dock)
    xs = [p[0] for p in cfg.zones.truck + cfg.zones.dock]
    return truck_bottom <= y <= dock_top and min(xs) <= x <= max(xs)


def detect(proc, model, device, img: Image.Image, text: str, thresh: float):
    inp = proc(images=img, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    res = proc.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=thresh, text_threshold=0.25,
        target_sizes=[(img.height, img.width)],
    )[0]
    labels = res.get("text_labels", res.get("labels"))
    return list(zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), [str(l) for l in labels]))


def nms_union(boxes: list[np.ndarray], iou_thresh: float = 0.5) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        ok = True
        for k in kept:
            ix1, iy1 = max(b[0], k[0]), max(b[1], k[1])
            ix2, iy2 = min(b[2], k[2]), min(b[3], k[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (b[2]-b[0])*(b[3]-b[1]) + (k[2]-k[0])*(k[3]-k[1]) - inter
            if union > 0 and inter / union > iou_thresh:
                ok = False
                break
        if ok:
            kept.append(b)
    return kept


def crop_pass(proc, model, device, frame_rgb: Image.Image, persons, thresh: float, W: int, H: int):
    """Run the detector on an enlarged crop around each person; return bag boxes in frame coords."""
    found = []
    for px1, py1, px2, py2 in persons:
        w, h = px2 - px1, py2 - py1
        cx1, cy1 = max(0, int(px1 - 0.6 * w)), max(0, int(py1 - 0.8 * h))
        cx2, cy2 = min(W, int(px2 + 0.6 * w)), min(H, int(py2 + 0.2 * h))
        side = max(cx2 - cx1, cy2 - cy1, 256)
        # square-ish crop so the bag is big relative to the input
        cx1 = max(0, min(cx1, W - side)); cy1 = max(0, min(cy1, H - side))
        cx2, cy2 = min(W, cx1 + side), min(H, cy1 + side)
        crop = frame_rgb.crop((cx1, cy1, cx2, cy2))
        for box, sc, lab in detect(proc, model, device, crop, "white sack. bag.", thresh):
            if "sack" in lab or "bag" in lab:
                b = np.array([box[0] + cx1, box[1] + cy1, box[2] + cx1, box[3] + cy1], dtype=np.float32)
                found.append(b)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out", type=Path)
    ap.add_argument("--every", type=int, default=20, help="sample every Nth frame")
    ap.add_argument("--val-every", type=int, default=5, help="every Nth sample goes to val")
    ap.add_argument("--max-bag-area", type=float, default=0.035)
    ap.add_argument("--bag-thresh", type=float, default=0.3)
    ap.add_argument("--person-thresh", type=float, default=0.35)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=0, help="exclusive; 0 = to end")
    ap.add_argument("--prefix", default="f")
    ap.add_argument("--motion", action="store_true",
                    help="keep a sack only if its region changes vs. frames ±motion-gap (rejects stacked bags)")
    ap.add_argument("--motion-gap", type=int, default=12, help="frames before/after to compare (1.5 s at 8 fps)")
    ap.add_argument("--motion-thresh", type=float, default=18.0, help="mean abs grey diff (0-255) to count as moving")
    ap.add_argument("--require-person", action="store_true", help="also require overlap with a person box")
    ap.add_argument("--owl", action="store_true",
                    help="use OWLv2 (google/owlv2-base-patch16-ensemble) instead of Grounding DINO; better recall on bags carried on the head")
    ap.add_argument("--yolo", default=None,
                    help="use this fine-tuned YOLO (classes person/bag) as the candidate detector instead of Grounding DINO")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    yolo = owl = None
    if a.yolo:
        from ultralytics import YOLO
        yolo = YOLO(a.yolo)
    elif a.owl:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        owl_id = "google/owlv2-base-patch16-ensemble"
        proc = Owlv2Processor.from_pretrained(owl_id)
        owl = Owlv2ForObjectDetection.from_pretrained(owl_id).to(a.device).eval()
    else:
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(a.device).eval()

    for split in ("train", "val"):
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (a.out / "data.yaml").write_text(
        f"path: {a.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: bag\n"
    )

    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stats = {"frames": 0, "person": 0, "bag": 0, "bag_rejected": 0}
    sample_i = 0
    for fi in range(a.start_frame, a.end_frame or n_frames, a.every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        persons, bags = [], []
        if yolo is not None:
            r = yolo.predict(frame, conf=min(a.bag_thresh, a.person_thresh), imgsz=1280, device=a.device, verbose=False)[0]
            for box, sc, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                if int(c) == PERSON and sc >= a.person_thresh:
                    persons.append(box)
                elif int(c) == BAG and sc >= a.bag_thresh:
                    bags.append(box)
        elif owl is not None:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            queries = [["a white sack", "a person"]]
            inp = proc(text=queries, images=img, return_tensors="pt").to(a.device)
            with torch.no_grad():
                o = owl(**inp)
            side = max(H, W)  # OWLv2 pads to square; boxes come back in that frame
            res = proc.post_process_grounded_object_detection(
                o, threshold=min(a.bag_thresh, a.person_thresh), target_sizes=torch.tensor([[side, side]]), text_labels=queries)[0]
            for box, sc, lab in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), res["labels"].cpu().numpy()):
                if int(lab) == 1 and sc >= a.person_thresh:
                    persons.append(box)
                elif int(lab) == 0 and sc >= a.bag_thresh:
                    bags.append(box)
        else:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for box, sc, lab in detect(proc, model, a.device, img, "white sack. person.", min(a.bag_thresh, a.person_thresh)):
                if "person" in lab and sc >= a.person_thresh:
                    persons.append(box)
                elif "sack" in lab and sc >= a.bag_thresh:
                    bags.append(box)
        prev_f = next_f = None
        if a.motion:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fi - a.motion_gap)); ok1, prev_f = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(n_frames - 1, fi + a.motion_gap)); ok2, next_f = cap.read()
            if not (ok1 and ok2):
                prev_f = next_f = None
        clamp = lambda b: np.array([max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])], dtype=np.float32)
        persons = [clamp(b) for b in persons]
        bags = [clamp(b) for b in bags if (min(W, b[2]) - max(0, b[0])) > 4 and (min(H, b[3]) - max(0, b[1])) > 4]
        kept_bags = []
        for b in bags:
            x1, y1, x2, y2 = b
            area = (x2 - x1) * (y2 - y1) / (W * H)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ok_bag = area <= a.max_bag_area and in_roi(cfg, cx, cy, W, H)
            if ok_bag and a.require_person:
                ok_bag = any(iou_expand(p, b, 0.3) for p in persons)
            if ok_bag and a.motion and prev_f is not None:
                ok_bag = moving(frame, prev_f, next_f, b, a.motion_thresh)
            if ok_bag:
                kept_bags.append(b)
            else:
                stats["bag_rejected"] += 1
        split = "val" if sample_i % a.val_every == 0 else "train"
        sample_i += 1
        name = f"{a.prefix}{fi:06d}"
        cv2.imwrite(str(a.out / "images" / split / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = []
        for cls, boxes in ((PERSON, persons), (BAG, kept_bags)):
            for x1, y1, x2, y2 in boxes:
                lines.append(f"{cls} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        (a.out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["frames"] += 1
        stats["person"] += len(persons)
        stats["bag"] += len(kept_bags)
        if stats["frames"] % 50 == 0:
            print(stats, flush=True)
        if a.limit and stats["frames"] >= a.limit:
            break
    print("done", stats)


if __name__ == "__main__":
    main()
