"""Persistent, single-workspace data preparation and training workflow."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import threading
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

import cv2
import numpy as np


class Workflow:
    """File-backed workflow state; long operations run in daemon workers."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

    def __init__(self, root: Path):
        self.root = Path(root)
        self.inputs = self.root / "uploads"
        self.data = self.root / "dataset"
        self.state_path = self.root / "state.json"
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.imported_labels: dict[str, list[str]] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self.inputs.mkdir(exist_ok=True)
        self.data.mkdir(exist_ok=True)
        self.state = self._load()

    def _load(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (ValueError, OSError):
                pass
        return {"step": 1, "uploads": [], "status": "idle", "progress": 0,
                "message": "Upload images, a dataset archive, or videos to begin.", "logs": [],
                "stats": {}, "config": {}}

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def store_config(self, key: str, value: dict) -> None:
        with self.lock:
            self.state.setdefault("config", {})[key] = value
            self.save()

    def log(self, message: str) -> None:
        with self.lock:
            self.state.setdefault("logs", []).append(f"{datetime.now().isoformat(timespec='seconds')} {message}")
            self.state["logs"] = self.state["logs"][-300:]
            self.state["message"] = message
            self.save()

    def snapshot(self) -> dict:
        with self.lock:
            result = json.loads(json.dumps(self.state))
            result["uploads"] = [{k:v for k,v in item.items() if k != "path"} for item in result.get("uploads", [])]
            result["running"] = bool(self.thread and self.thread.is_alive())
            return result

    def launch(self, fn, *args) -> None:
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("A workflow task is already running")
            self.state["status"], self.state["progress"] = "running", 0
            self.state.pop("error", None)
            self.save()
            def run():
                try:
                    fn(*args)
                    with self.lock:
                        self.state["status"], self.state["progress"] = "complete", 100
                        self.save()
                except Exception as exc:
                    with self.lock:
                        self.state["status"] = "error"
                        self.state["error"] = str(exc)
                        self.log(f"Error: {exc}")
            self.thread = threading.Thread(target=run, name="bagcount-workflow", daemon=True)
            self.thread.start()

    def add_upload(self, filename: str, content: bytes) -> dict:
        name = Path(filename).name
        ext = Path(name).suffix.lower()
        if ext not in self.IMAGE_EXTS | self.VIDEO_EXTS | {".zip"}:
            raise ValueError(f"Unsupported file type: {ext or '(no extension)'}")
        if not content:
            raise ValueError(f"{name} is empty")
        digest = hashlib.sha256(content).hexdigest()
        dest = self.inputs / f"{digest[:16]}_{name}"
        if not dest.exists():
            dest.write_bytes(content)
        item = {"name": name, "path": str(dest), "type": "archive" if ext == ".zip" else
                "video" if ext in self.VIDEO_EXTS else "image", "bytes": len(content), "sha256": digest}
        return self.register_upload(name, dest, len(content), digest, ext)

    def register_upload(self, name: str, dest: Path, size: int, digest: str, ext: str) -> dict:
        item = {"name": name, "path": str(dest), "type": "archive" if ext == ".zip" else
                "video" if ext in self.VIDEO_EXTS else "image", "bytes": size, "sha256": digest}
        with self.lock:
            if any(x["sha256"] == digest for x in self.state["uploads"]):
                return item
            # A new source invalidates derived annotations and checkpoints in
            # the current workflow session, while retaining prior source files.
            if self.data.exists(): shutil.rmtree(self.data)
            self.data.mkdir(parents=True)
            self.state["uploads"].append(item)
            self.state.update({"step":1,"status":"idle","progress":0,"stats":{},"data_ready":False})
            self.state.pop("training", None); self.state.pop("error", None)
            self.save()
        return item

    def register_local_file(self, source: Path) -> dict:
        """Register an on-disk input without copying a potentially large video."""
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input file not found: {source}")
        ext = source.suffix.lower()
        if ext not in self.IMAGE_EXTS | self.VIDEO_EXTS | {".zip"}:
            raise ValueError(f"Unsupported input type: {source.name}")
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        return self.register_upload(source.name, source, size, digest.hexdigest(), ext)

    def _candidates(self):
        for item in self.state.get("uploads", []):
            p = Path(item["path"])
            if item["type"] == "image":
                yield ("image", p)
            elif item["type"] == "video":
                yield ("video", p)
            elif item["type"] == "archive":
                # Only extract image members; reject traversal paths and nested archives.
                dest = self.inputs / (p.stem + "_contents")
                dest.mkdir(exist_ok=True)
                with zipfile.ZipFile(p) as zf:
                    members = {Path(x.filename).as_posix(): x for x in zf.infolist() if not x.is_dir()}
                    bag_ids = set()
                    yaml_member = next((x for x in zf.namelist() if Path(x).name in ("data.yaml", "data.yml")), None)
                    if yaml_member:
                        try:
                            import yaml
                            names = (yaml.safe_load(zf.read(yaml_member)) or {}).get("names", {})
                            names = names.items() if isinstance(names, dict) else enumerate(names)
                            bag_ids = {int(i) for i, n in names if str(n).lower() in ("bag", "sack")}
                        except Exception:
                            bag_ids = set()
                    for member in zf.infolist():
                        rel = PurePosixPath(member.filename)
                        if member.is_dir() or rel.is_absolute() or ".." in rel.parts:
                            continue
                        if Path(rel.name).suffix.lower() not in self.IMAGE_EXTS:
                            continue
                        target = dest / f"{hashlib.sha1(member.filename.encode()).hexdigest()[:8]}_{rel.name}"
                        if not target.exists():
                            target.write_bytes(zf.read(member))
                        if bag_ids:
                            rel = PurePosixPath(member.filename)
                            guesses = []
                            parts = list(rel.parts)
                            if "images" in parts:
                                pos = parts.index("images"); guesses.append(str(PurePosixPath(*parts[:pos], "labels", *parts[pos+1:]).with_suffix(".txt")))
                            guesses.append(str(rel.with_suffix(".txt")))
                            label_member = next((members[g] for g in guesses if g in members), None)
                            if label_member:
                                lines = []
                                for row in zf.read(label_member).decode("utf-8", errors="ignore").splitlines():
                                    cols = row.split()
                                    if len(cols) != 5: continue
                                    try: class_id = int(cols[0]); coords = [float(x) for x in cols[1:]]
                                    except ValueError: continue
                                    if class_id in bag_ids and all(0 <= x <= 1 for x in coords):
                                        lines.append("0 " + " ".join(f"{x:.6f}" for x in coords))
                                self.imported_labels[str(target)] = lines
                        yield ("image", target)

    @staticmethod
    def _yolo(box, width: int, height: int) -> str:
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            return ""
        return f"0 {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}"

    def prepare(self, config: dict, on_progress=None) -> None:
        """Sample inputs and generate bag-only YOLO labels with Grounding DINO.

        ``on_progress(scanned, estimated, images, skipped, boxes)`` is called once
        per sampled frame so the CLI can render a live progress bar.
        """
        from PIL import Image
        c = {"sample_fps": 1.0, "frame_interval": 0, "confidence": 0.30, "start_s": 0,
             "duration_s": 0, "max_images": 0, "device": "cpu", **config}
        if c["sample_fps"] <= 0 or not 0 <= c["confidence"] <= 1:
            raise ValueError("Sampling FPS must be positive and confidence must be in [0, 1]")
        if self.data.exists():
            shutil.rmtree(self.data)
        image_dir, label_dir = self.data / "images", self.data / "labels"
        image_dir.mkdir(parents=True); label_dir.mkdir(parents=True)
        self.imported_labels = {}
        candidates = list(self._candidates())
        need_detector = any(kind == "video" or str(path) not in self.imported_labels for kind, path in candidates)
        if need_detector:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
            detector = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(c["device"]).eval()
        else:
            torch = processor = detector = None
        estimated = 0
        for item in self.state.get("uploads", []):
            if item["type"] != "video": continue
            cap = cv2.VideoCapture(item["path"])
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            start = max(0, int(float(c["start_s"]) * fps))
            end = total if not float(c["duration_s"]) else min(total, start + int(float(c["duration_s"]) * fps))
            stride = int(c["frame_interval"] or max(1, round(fps / float(c["sample_fps"]))))
            estimated += max(0, (end-start+stride-1)//stride)
        image_count = sum(1 for item in self.state.get("uploads", []) if item["type"] == "image")
        estimated = max(1, estimated + image_count)
        images = skipped = scanned = boxes_n = 0
        seen: set[str] = set()
        extracted_clips = []
        if on_progress is not None:
            on_progress(scanned, estimated, images, skipped, boxes_n)

        def annotate(frame, source, frame_number, provided_labels=None):
            nonlocal images, skipped, scanned, boxes_n
            scanned += 1
            if on_progress is not None:
                on_progress(scanned, estimated, images, skipped, boxes_n)
            candidates = []
            if provided_labels is not None:
                candidates = [(line, 1.0) for line in provided_labels]
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                # Bag-only query: no human/person model or query is invoked.
                inputs = processor(images=pil, text="bag. white sack.", return_tensors="pt").to(c["device"])
                with torch.no_grad():
                    output = detector(**inputs)
                result = processor.post_process_grounded_object_detection(
                    output, inputs.input_ids, threshold=float(c["confidence"]), text_threshold=0.20,
                    target_sizes=[(pil.height, pil.width)],
                )[0]
                for box, score in zip(result["boxes"].cpu().numpy(), result["scores"].cpu().numpy()):
                    if float(score) < float(c["confidence"]): continue
                    line = self._yolo(box, frame.shape[1], frame.shape[0])
                    if line: candidates.append((line, float(score)))
            # Confidence-ordered duplicate removal keeps only one label for a box.
            keep = []
            for line, score in sorted(candidates, key=lambda x: -x[1]):
                vals = list(map(float, line.split()[1:]))
                x1, y1, x2, y2 = (vals[0]-vals[2]/2, vals[1]-vals[3]/2, vals[0]+vals[2]/2, vals[1]+vals[3]/2)
                dup = False
                for prev in keep:
                    q = list(map(float, prev.split()[1:]))
                    a1, b1, a2, b2 = q[0]-q[2]/2, q[1]-q[3]/2, q[0]+q[2]/2, q[1]+q[3]/2
                    inter = max(0, min(x2,a2)-max(x1,a1))*max(0,min(y2,b2)-max(y1,b1))
                    union = vals[2]*vals[3]+q[2]*q[3]-inter
                    if union and inter/union > .65:
                        dup = True; break
                if not dup:
                    keep.append(line)
            if not keep:
                skipped += 1
                return
            digest = hashlib.sha1(frame.tobytes()).hexdigest()
            if digest in seen:
                skipped += 1
                return
            seen.add(digest)
            name = f"bag_{images:06d}"
            cv2.imwrite(str(image_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
            (label_dir / f"{name}.txt").write_text("\n".join(keep) + "\n")
            images += 1; boxes_n += len(keep)
            if on_progress is not None:
                on_progress(scanned, estimated, images, skipped, boxes_n)
            if scanned % 5 == 0:
                with self.lock:
                    self.state["progress"] = min(95, int(95 * scanned / estimated))
                    self.state["stats"] = {"estimated_frames": estimated, "scanned_frames": scanned,
                                           "saved_images": images, "empty_frames_skipped": skipped,
                                           "bag_boxes": boxes_n}
                    self.save()
            if on_progress is None and scanned % 25 == 0:
                print(f"sampled={scanned} bag_images={images} empty_skipped={skipped} bag_boxes={boxes_n}", flush=True)

        interval = int(c["frame_interval"] or 0)
        for kind, path in candidates:
            if kind == "image":
                frame = cv2.imread(str(path))
                if frame is not None:
                    annotate(frame, path.name, 0, self.imported_labels.get(str(path)))
            else:
                cap = cv2.VideoCapture(str(path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                start = max(0, int(float(c["start_s"]) * fps))
                end = total if not float(c["duration_s"]) else min(total, start + int(float(c["duration_s"]) * fps))
                stride = interval if interval > 0 else max(1, round(fps / float(c["sample_fps"])))
                writer = None
                clip_stride = max(1, round(fps / float(c.get("clip_fps") or c["sample_fps"])))
                if c.get("extract_clip"):
                    clip_dir = self.root / "clips"; clip_dir.mkdir(exist_ok=True)
                    clip_path = clip_dir / f"{path.stem}_{len(extracted_clips):03d}.mp4"
                    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             float(c.get("clip_fps") or c["sample_fps"]), (width, height))
                    if not writer.isOpened():
                        cap.release(); raise RuntimeError(f"Cannot write extracted clip: {clip_path}")
                    extracted_clips.append(str(clip_path.relative_to(self.root)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                fi = start
                try:
                    while fi < end:
                        ok, frame = cap.read()
                        if not ok: break
                        if writer is not None and (fi - start) % clip_stride == 0: writer.write(frame)
                        if (fi - start) % stride == 0:
                            annotate(frame, path.name, fi)
                            if c["max_images"] and scanned >= int(c["max_images"]): break
                        fi += 1
                finally:
                    if writer is not None: writer.release()
                cap.release()
            if c["max_images"] and scanned >= int(c["max_images"]): break
        (self.data / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: bag\n")
        stats = {"estimated_frames": estimated, "scanned_frames": scanned, "saved_images": images,
                 "empty_frames_skipped": skipped, "bag_boxes": boxes_n,
                 "source_count": len(self.state.get("uploads", [])),
                 "extracted_clips": extracted_clips}
        with self.lock:
            self.state["config"] = c
            self.state["stats"] = stats
            self.state["step"] = 5
            self.state["data_ready"] = images > 0
            self.log(f"Label generation finished: {images} positive images, {boxes_n} bag boxes; skipped {skipped} empty frames.")

    def analyze(self) -> dict:
        counts = {"images": 0, "positive_images": 0, "bag_annotations": 0, "class_distribution": {"bag": 0}, "balance": "single-class"}
        box_buckets = {"1":0,"2-3":0,"4+":0}
        for split in ("train", "val"):
            ld = self.data / "labels" / split
            if ld.exists():
                for label in ld.glob("*.txt"):
                    lines = [x for x in label.read_text().splitlines() if x.strip()]
                    counts["images"] += 1; counts["positive_images"] += bool(lines)
                    counts["bag_annotations"] += len(lines); counts["class_distribution"]["bag"] += len(lines)
                    if lines: box_buckets["1" if len(lines)==1 else "2-3" if len(lines)<=3 else "4+"] += 1
        if not counts["images"]:
            for label in (self.data / "labels").glob("*.txt") if (self.data / "labels").exists() else []:
                lines = [x for x in label.read_text().splitlines() if x.strip()]
                counts["images"] += 1; counts["positive_images"] += bool(lines); counts["bag_annotations"] += len(lines)
                counts["class_distribution"]["bag"] += len(lines)
                if lines: box_buckets["1" if len(lines)==1 else "2-3" if len(lines)<=3 else "4+"] += 1
        preview_dir = self.data / "images"
        candidates = []
        for split in ("train", "val"):
            d = preview_dir / split
            if d.exists(): candidates.extend(sorted(d.glob("*")))
        if not candidates and preview_dir.exists():
            candidates = sorted(p for p in preview_dir.glob("*") if p.suffix.lower() in self.IMAGE_EXTS)
        counts["preview"] = [f"images/{p.parent.name}/{p.name}"
                             if p.parent.name in ("train", "val")
                             else f"images/{p.name}" for p in candidates[:12]]
        counts["images_by_bag_count"] = box_buckets
        counts["balance"] = "single class; balance by scene/scale/lighting coverage"
        with self.lock:
            self.state["stats"].update(counts)
            self.state["step"] = max(self.state.get("step", 1), 6)
            self.save()
        return counts

    def split(self, config: dict) -> None:
        seed = int(config.get("seed", 42))
        ratios = [float(config.get(k, d)) for k, d in (("train", .8), ("val", .2))]
        if any(x <= 0 for x in ratios) or not np.isclose(sum(ratios), 1.0):
            raise ValueError("Train and validation ratios must be positive and sum to 1")
        src_i, src_l = self.data / "images", self.data / "labels"
        imgs = sorted([p for p in src_i.glob("*") if p.suffix.lower() in self.IMAGE_EXTS])
        if len(imgs) < 2:
            raise ValueError("At least two bag-positive images are required for train/validation splits")
        random.Random(seed).shuffle(imgs)
        ntrain = max(1, min(len(imgs) - 1, int(round(len(imgs) * ratios[0]))))
        groups = {"train": imgs[:ntrain], "val": imgs[ntrain:]}
        for split, files in groups.items():
            (src_i / split).mkdir(exist_ok=True); (src_l / split).mkdir(exist_ok=True)
            for image in files:
                label = src_l / f"{image.stem}.txt"
                shutil.move(str(image), str(src_i / split / image.name))
                shutil.move(str(label), str(src_l / split / label.name))
        with self.lock:
            self.state["config"].update({"split_seed": seed, "split_ratios": ratios})
            self.state["step"] = 8
            self.state["stats"]["splits"] = {k: len(v) for k,v in groups.items()}
            self.save()

    def augment(self, config: dict, preview: Path | None = None) -> None:
        """Apply selected Albumentations transforms to train only, with YOLO boxes.

        If ``preview`` is given, a contact sheet of the first three augmented
        images is written there.
        """
        import albumentations as A
        specs = config.get("selected", [])
        if not specs:
            with self.lock:
                self.state["config"]["augmentation"] = config
                self.state["step"] = max(self.state.get("step", 1), 9)
                self.state["stats"]["augmented_images"] = 0
                self.save()
            return
        constructors = {
            "horizontal_flip": lambda p, q: A.HorizontalFlip(p=p),
            "vertical_flip": lambda p, q: A.VerticalFlip(p=p),
            "rotation": lambda p, q: A.Rotate(limit=q.get("limit", 15), p=p),
            "shift": lambda p, q: A.Affine(translate_percent={"x": tuple(q.get("range",[-.05,.05])), "y": tuple(q.get("range",[-.05,.05]))}, p=p),
            "scale": lambda p, q: A.Affine(scale=tuple(q.get("range",[.9,1.1])), p=p),
            "shift_scale_rotate": lambda p, q: A.ShiftScaleRotate(shift_limit=q.get("shift", .05), scale_limit=q.get("scale", .1), rotate_limit=q.get("rotate", 10), p=p),
            "shear": lambda p, q: A.Affine(shear=q.get("shear", [-10, 10]), p=p),
            "random_crop": lambda p, q: A.RandomResizedCrop(size=tuple(q.get("size", [384,384])), scale=tuple(q.get("scale", [.75,1.0])), p=p),
            "resize": lambda p, q: A.Resize(height=int(q.get("height",384)), width=int(q.get("width",384)), p=p),
            "perspective": lambda p, q: A.Perspective(scale=tuple(q.get("scale", [.02,.08])), p=p),
            "affine": lambda p, q: A.Affine(scale=tuple(q.get("scale", [.9,1.1])), translate_percent=tuple(q.get("shift", [-.05,.05])), rotate=tuple(q.get("rotate", [-10,10])), shear=tuple(q.get("shear", [-5,5])), p=p),
            "brightness": lambda p, q: A.RandomBrightnessContrast(brightness_limit=q.get("limit", .2), contrast_limit=0, p=p),
            "contrast": lambda p, q: A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=q.get("limit", .2), p=p),
            "saturation_hue": lambda p, q: A.HueSaturationValue(hue_shift_limit=q.get("hue", 10), sat_shift_limit=q.get("saturation", 15), val_shift_limit=0, p=p),
            "saturation": lambda p, q: A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=q.get("limit", 20), val_shift_limit=0, p=p),
            "hue": lambda p, q: A.HueSaturationValue(hue_shift_limit=q.get("limit", 10), sat_shift_limit=0, val_shift_limit=0, p=p),
            "gamma": lambda p, q: A.RandomGamma(gamma_limit=tuple(q.get("range", [80,120])), p=p),
            "rgb_shift": lambda p, q: A.RGBShift(p=p),
            "color_jitter": lambda p, q: A.ColorJitter(p=p),
            "gaussian_blur": lambda p, q: A.GaussianBlur(blur_limit=tuple(q.get("limit", [3,7])), p=p),
            "motion_blur": lambda p, q: A.MotionBlur(blur_limit=tuple(q.get("limit", [3,7])), p=p),
            "median_blur": lambda p, q: A.MedianBlur(blur_limit=tuple(q.get("limit", [3,7])), p=p),
            "gaussian_noise": lambda p, q: A.GaussNoise(p=p),
            "iso_noise": lambda p, q: A.ISONoise(p=p),
            "clahe": lambda p, q: A.CLAHE(p=p),
            "grayscale": lambda p, q: A.ToGray(p=p),
            "channel_shuffle": lambda p, q: A.ChannelShuffle(p=p),
            "coarse_dropout": lambda p, q: A.CoarseDropout(p=p),
            "random_shadow": lambda p, q: A.RandomShadow(p=p),
            "random_fog": lambda p, q: A.RandomFog(p=p),
        }
        transforms = []
        for spec in specs:
            key = spec["name"]
            if key not in constructors: raise ValueError(f"Unsupported augmentation: {key}")
            p = float(spec.get("p", .5))
            if not 0 <= p <= 1: raise ValueError(f"Probability for {key} must be in [0, 1]")
            transforms.append(constructors[key](p, spec.get("params", {})))
        pipeline = A.Compose(transforms, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=.1))
        train_i, train_l = self.data / "images/train", self.data / "labels/train"
        if not train_i.exists(): raise ValueError("Split the dataset before applying augmentations")
        seed = int(config.get("seed", self.state.get("config", {}).get("split_seed", 42)))
        random.seed(seed); np.random.seed(seed)
        previews = []
        preview_images = []
        originals = sorted(train_i.glob("*"))
        augmented_count = 0
        for idx, image_path in enumerate(originals):
            image = cv2.imread(str(image_path))
            label_path = train_l / f"{image_path.stem}.txt"
            rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
            boxes = [list(map(float, row[1:5])) for row in rows]
            labels = [0 for _ in rows]
            result = pipeline(image=image, bboxes=boxes, class_labels=labels)
            if not result["bboxes"]: continue
            aug_name = f"{image_path.stem}_aug{idx:04d}"
            cv2.imwrite(str(train_i / f"{aug_name}{image_path.suffix}"), result["image"])
            label_path = train_l / f"{aug_name}.txt"
            label_path.write_text("\n".join("0 " + " ".join(f"{v:.6f}" for v in b) for b in result["bboxes"]) + "\n")
            augmented_count += 1
            if len(previews) < 3:
                previews.append({"before": f"images/train/{image_path.name}",
                                 "after": f"images/train/{aug_name}{image_path.suffix}"})
                tile = cv2.resize(result["image"], (480, 300))
                cv2.putText(tile, aug_name[:55], (8, 290), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                preview_images.append(tile)
        preview_sheet = None
        if preview is not None and preview_images:
            preview_path = Path(preview)
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            tiles = list(preview_images)
            while len(tiles) < 3:
                tiles.append(tiles[0].copy() * 0)
            cv2.imwrite(str(preview_path), cv2.hconcat(tiles))
            preview_sheet = str(preview_path)
        with self.lock:
            self.state["config"]["augmentation"] = config
            self.state["step"] = max(self.state.get("step", 1), 9)
            self.state["stats"].update({"augmented_images": augmented_count,
                                        "augmentation_source_images": len(originals),
                                        "augmentation_preview": previews,
                                        "augmentation_preview_sheet": preview_sheet})
            self.log(f"Applied selected augmentations to training data only: {augmented_count}/{len(originals)} images retained.")

    def train(self, config: dict) -> None:
        from rfdetr import RFDETR, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

        from bagcount.label_check import read_class_names, resolve_class

        variants = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}
        variant = config.get("variant", "nano")
        if variant not in variants: raise ValueError("Unknown RF-DETR model variant")
        # --data makes the dataset config explicit. Without it, keep using the
        # workflow's own generated dataset so existing commands are unchanged.
        spec = config.get("data_yaml")
        if spec:
            data_yaml = Path(spec).expanduser().resolve()
            if not data_yaml.is_file():
                raise ValueError(f"data.yaml not found: {data_yaml}")
            dataset_dir = data_yaml.parent  # never rewrite a user-supplied config
        else:
            dataset_dir = self.data
            data_yaml = dataset_dir / "data.yaml"
            data_yaml.write_text(f"path: {dataset_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: bag\n")
        bag_class = resolve_class(read_class_names(data_yaml), "bag")
        cls = variants[variant]
        model = RFDETR.from_checkpoint(config["checkpoint"], device=config.get("device", "cpu")) if config.get("checkpoint") else cls(device=config.get("device", "cpu"))
        output = Path("runs/detect") / str(config.get("name", "bag_workflow"))
        output.mkdir(parents=True, exist_ok=True)
        self.log(f"Training RF-DETR {variant} on {data_yaml}: {config.get('epochs',40)} epochs, output {output}")
        model.train(dataset_dir=str(dataset_dir), epochs=int(config.get("epochs",40)),
                    resolution=int(config.get("resolution",384)), batch_size=int(config.get("batch_size",4)),
                    grad_accum_steps=int(config.get("grad_accum_steps",4)), lr=float(config.get("lr",2e-5)),
                    num_workers=int(config.get("workers",2)), device=config.get("device","cpu"),
                    output_dir=str(output),
                    **({"resume":config["checkpoint"]} if config.get("checkpoint") else {}))
        ckpt = output / "checkpoint_best_ema.pth"
        metrics = (self.evaluate(str(ckpt), config.get("device", "cpu"), dataset_dir=dataset_dir, bag_class=bag_class)
                   if ckpt.exists() else {"error":"Best checkpoint was not created"})
        with self.lock:
            self.state["training"] = {"output": str(output), "checkpoint": str(ckpt),
                                      "data_yaml": str(data_yaml), "metrics": metrics}
            self.state["step"] = 11
            self.log(f"Training finished; held-out validation metrics: {metrics}")

    def evaluate(self, checkpoint: str, device: str, dataset_dir: Path | None = None, bag_class: int = 0) -> dict:
        """Compute bag precision/recall/F1 and AP50 on the held-out validation set."""
        from rfdetr import RFDETR
        model = RFDETR.from_checkpoint(checkpoint, device=device)
        root = Path(dataset_dir) if dataset_dir is not None else self.data
        scores, tp, fp, fn = [], 0, 0, 0
        for image_path in sorted((root / "images/val").glob("*")):
            image = cv2.imread(str(image_path))
            if image is None: continue
            h, w = image.shape[:2]
            pred = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), threshold=.001)
            detections = []
            if pred is not None and len(pred):
                confidence = getattr(pred, "confidence", getattr(pred, "scores", np.ones(len(pred))))
                for box, score, cls in zip(np.asarray(pred.xyxy), np.asarray(confidence), np.asarray(pred.class_id)):
                    if int(cls) == bag_class: detections.append((box, float(score)))
            gtfile = root / "labels/val" / f"{image_path.stem}.txt"
            gt = []
            for row in gtfile.read_text().splitlines() if gtfile.exists() else []:
                parts = row.split()
                if not parts or int(parts[0]) != bag_class:
                    continue
                v = list(map(float, parts[1:5])); cx,cy,bw,bh=v
                gt.append(np.array([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
            matched=set()
            for box, score in sorted(detections,key=lambda x:-x[1]):
                ious=[]
                for j,g in enumerate(gt):
                    ix=max(0,min(box[2],g[2])-max(box[0],g[0])); iy=max(0,min(box[3],g[3])-max(box[1],g[1]))
                    inter=ix*iy; union=(box[2]-box[0])*(box[3]-box[1])+(g[2]-g[0])*(g[3]-g[1])-inter
                    ious.append(inter/union if union else 0)
                best=int(np.argmax(ious)) if ious else -1
                ok=best>=0 and ious[best]>=.5 and best not in matched
                if ok: tp+=1; matched.add(best)
                else: fp+=1
                scores.append((score, int(ok)))
            fn += len(gt)-len(matched)
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        ranked=sorted(scores,key=lambda x:-x[0]); cumtp=cumfp=ap=0
        for _,ok in ranked:
            cumtp+=ok; cumfp+=1-ok
            ap += (cumtp/(cumtp+cumfp)) * ok
        ap50=ap/max(1,tp+fn)
        return {"precision":round(precision,4),"recall":round(recall,4),"f1":round(2*precision*recall/(precision+recall),4) if precision+recall else 0,
                "ap50":round(ap50,4),"true_positives":tp,"false_positives":fp,"false_negatives":fn}


