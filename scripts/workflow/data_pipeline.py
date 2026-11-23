"""Command-line data preparation and training commands for Bag Counting."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2

from bagcount.workflow import Workflow


def review(w: Workflow, out: Path, limit: int, cols: int, per_sheet: int = 0) -> None:
    stats = w.analyze()
    images = []
    image_dir, label_dir = w.data / "images", w.data / "labels"
    files = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in w.IMAGE_EXTS)
    for split in ("train", "val"):
        files.extend(sorted(p for p in (image_dir / split).glob("*") if p.suffix.lower() in w.IMAGE_EXTS))
    for path in files[:limit]:
        image = cv2.imread(str(path))
        label = (label_dir / path.parent.name / f"{path.stem}.txt"
                 if path.parent.name in ("train", "val")
                 else label_dir / f"{path.stem}.txt")
        if image is None:
            continue
        h, width = image.shape[:2]
        if label.exists():
            for row in label.read_text().splitlines():
                try:
                    _, cx, cy, bw, bh = map(float, row.split())
                    x1, y1 = int((cx-bw/2)*width), int((cy-bh/2)*h)
                    x2, y2 = int((cx+bw/2)*width), int((cy+bh/2)*h)
                    cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(image, "bag", (x1,max(18,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2)
                except ValueError:
                    continue
        tile_w, tile_h = 480, 300
        image = cv2.resize(image, (tile_w, tile_h))
        cv2.putText(image, path.name[:55], (8, tile_h-10), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
        images.append(image)
    out.parent.mkdir(parents=True, exist_ok=True)
    if per_sheet > 0 and images:
        paths = []
        stem, suffix = out.stem, out.suffix or ".jpg"
        for idx, start in enumerate(range(0, len(images), per_sheet), 1):
            sheet = images[start:start + per_sheet]
            rows = []
            for row_start in range(0, len(sheet), cols):
                row = sheet[row_start:row_start + cols]
                row.extend([sheet[0].copy() * 0 for _ in range(cols - len(row))])
                rows.append(cv2.hconcat(row))
            path = out.with_name(f"{stem}_{idx:03d}{suffix}")
            cv2.imwrite(str(path), cv2.vconcat(rows))
            paths.append(str(path))
        result = {"stats": stats, "preview": paths[0] if paths else None, "previews": paths}
    elif images:
        rows = []
        for start in range(0, len(images), cols):
            row = images[start:start+cols]
            row.extend([images[0].copy() * 0 for _ in range(cols-len(row))])
            rows.append(cv2.hconcat(row))
        cv2.imwrite(str(out), cv2.vconcat(rows))
        result = {"stats": stats, "preview": str(out)}
    else:
        result = {"stats": stats, "preview": None}
    print(json.dumps(result, indent=2))


class ProgressBar:
    """Dependency-free CLI progress bar with elapsed time and ETA."""

    def __init__(self, width: int = 28, stream=None):
        self.width = width
        self.stream = stream or sys.stdout
        self.start = time.monotonic()
        self.last_draw = 0.0
        self.last_len = 0
        self.total = 1

    @staticmethod
    def _clock(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def update(self, scanned: int, total: int, images: int, skipped: int, boxes: int) -> None:
        self.total = max(int(total), 1)
        now = time.monotonic()
        # Avoid flooding the terminal on long videos, but always redraw on completion.
        if scanned < self.total and now - self.last_draw < 0.1:
            return
        self.last_draw = now
        pct = min(100.0, 100.0 * scanned / self.total)
        filled = int(self.width * pct / 100)
        bar = "=" * filled + " " * (self.width - filled)
        rate = scanned / max(0.001, now - self.start)
        eta = (self.total - scanned) / rate if rate > 0 else 0.0
        line = (f"\r[{bar}] {pct:5.1f}% | frames {scanned}/{self.total} | "
                f"saved {images} | skipped {skipped} | boxes {boxes} | "
                f"{self._clock(now - self.start)} elapsed, {self._clock(eta)} left")
        self.stream.write(line + " " * max(0, self.last_len - len(line)))
        self.stream.flush()
        self.last_len = len(line)

    def finish(self, stats: dict) -> None:
        total = max(int(stats.get("estimated_frames", self.total)), self.total, 1)
        self.update(total, total, int(stats.get("saved_images", 0)),
                    int(stats.get("empty_frames_skipped", 0)), int(stats.get("bag_boxes", 0)))
        self.stream.write("\n")
        self.stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bag-only dataset preparation and RF-DETR training")
    parser.add_argument("command", choices=("prepare", "review", "split", "augment", "train"))
    parser.add_argument("inputs", nargs="*", help="image/video/ZIP files; required for prepare")
    parser.add_argument("--root", type=Path, default=Path("datasets/workflow"))
    parser.add_argument("--fps", type=float, default=1.0, help="video sample FPS")
    parser.add_argument("--interval", type=int, default=0, help="instead sample every Nth video frame")
    parser.add_argument("--confidence", type=float, default=.30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start", type=float, default=0, help="video segment start seconds")
    parser.add_argument("--duration", type=float, default=0, help="segment seconds; 0 means full video")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--extract-clip", action="store_true")
    parser.add_argument("--clip-fps", type=float, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=.80)
    parser.add_argument("--val-ratio", type=float, default=.20)
    parser.add_argument("--config", type=Path, help="JSON augmentation configuration")
    parser.add_argument("--data", type=Path, default=None,
                        help="YOLO data.yaml for train (default: <root>/dataset/data.yaml); "
                             "use this to train from an explicit dataset config")
    parser.add_argument("--preview", type=Path, default=None,
                        help="preview sheet path (review: out/workflow-review.jpg, augment: out/review/augmentation.jpg)")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--cols", type=int, default=None,
                        help="tiles per row (default: 4, or near-square when --per-sheet is set)")
    parser.add_argument("--per-sheet", type=int, default=0,
                        help="images per preview sheet; 0 writes one sheet")
    parser.add_argument("--variant", choices=("nano","small","medium","large"), default="nano")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="bag_workflow")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    if args.data and args.command != "train":
        parser.error("--data only applies to the train command")

    workflow = Workflow(args.root)
    if args.command == "prepare":
        if not args.inputs:
            parser.error("prepare requires one or more image, video, or ZIP paths")
        for name in args.inputs:
            item = workflow.register_local_file(Path(name))
            print(f"registered {item['type']}: {item['name']} ({item['bytes']} bytes)")
        bar = ProgressBar()
        print("Preparing labels; progress updates once per sampled frame.", flush=True)
        try:
            workflow.prepare({"sample_fps":args.fps,"frame_interval":args.interval,
                              "confidence":args.confidence,"device":args.device,
                              "start_s":args.start,"duration_s":args.duration,
                              "max_images":args.max_samples,"extract_clip":args.extract_clip,
                              "clip_fps":args.clip_fps}, on_progress=bar.update)
        except BaseException:
            bar.stream.write("\n")
            bar.stream.flush()
            raise
        stats = workflow.snapshot()["stats"]
        bar.finish(stats)
        print(json.dumps(stats, indent=2))
    elif args.command == "review":
        cols = args.cols
        if cols is None:
            cols = max(1, round(math.sqrt(args.per_sheet))) if args.per_sheet > 0 else 4
        review(workflow, args.preview or Path("out/workflow-review.jpg"), args.limit, cols, args.per_sheet)
    elif args.command == "split":
        workflow.split({"train":args.train_ratio,"val":args.val_ratio,"seed":args.seed})
        print(json.dumps(workflow.snapshot()["stats"].get("splits", {}), indent=2))
    elif args.command == "augment":
        if not args.config or not args.config.is_file():
            parser.error(
                "augment requires an existing augmentation JSON file "
                f"(got {args.config}). Run from the repo root and copy "
                "configs/augmentations.example.json to configs/augmentations.json first."
            )
        config = json.loads(args.config.read_text())
        workflow.augment(config, preview=args.preview or Path("out/review/augmentation.jpg"))
        print(json.dumps(workflow.snapshot()["stats"], indent=2))
    elif args.command == "train":
        workflow.train({"variant":args.variant,"epochs":args.epochs,"resolution":args.resolution,
                        "batch_size":args.batch,"grad_accum_steps":args.grad_accum,"lr":args.lr,
                        "workers":args.workers,"device":args.device,"name":args.name,
                        "checkpoint":args.checkpoint,
                        "data_yaml":str(args.data) if args.data else None})
        print(json.dumps(workflow.snapshot().get("training", {}), indent=2))


if __name__ == "__main__":
    main()
