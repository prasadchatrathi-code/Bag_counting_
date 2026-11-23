"""Fine-tune an RF-DETR detector on the bags dataset.

Usage: mise run train -- [--data datasets/bags_full/data.yaml] [--epochs 40] [--name bags_full]

Label counts and class balance are verified before training starts (see
``bagcount.label_check``); pass ``--no-verify`` only to bypass that gate.

RF-DETR reads our YOLO-format dataset directly (images/{train,val},
labels/{train,val}, data.yaml with names 0=person, 1=bag). Checkpoints land in
runs/detect/<name>/; use checkpoint_best_ema.pth (or checkpoint_best_regular.pth)
as the counter's --weights.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from rfdetr import RFDETR, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

VARIANTS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}


def prepare_training_data(data_yaml: Path, target_class: str, temp_root: Path) -> Path:
    """Make a temporary dataset excluding training images without the target class.

    Images are linked, labels are copied, and validation is kept unchanged so
    negative-only validation examples can still measure false positives.
    """
    import yaml

    from bagcount.label_check import read_class_names, resolve_class

    source = data_yaml.parent
    names = read_class_names(data_yaml)
    target_id = resolve_class(names, target_class)
    prepared = temp_root / "dataset"
    prepared.mkdir(parents=True)
    for split in ("train", "val"):
        src_images, src_labels = source / "images" / split, source / "labels" / split
        dst_images, dst_labels = prepared / "images" / split, prepared / "labels" / split
        dst_images.mkdir(parents=True)
        dst_labels.mkdir(parents=True)
        removed = 0
        for label in sorted(src_labels.glob("*.txt")):
            class_ids = {
                int(line.split()[0])
                for line in label.read_text().splitlines()
                if line.split()
            }
            if split == "train" and target_id not in class_ids:
                removed += 1
                continue
            image = next((p for p in src_images.glob(label.stem + ".*") if p.is_file()), None)
            if image is None:
                raise SystemExit(f"image missing for label: {label}")
            (dst_images / image.name).symlink_to(image.resolve())
            shutil.copy2(label, dst_labels / label.name)
        if split == "train":
            print(f"Training preparation: removed {removed} images without "
                  f"{names[target_id]} labels; validation unchanged", flush=True)
    config = yaml.safe_load(data_yaml.read_text())
    config["path"] = str(prepared)
    config["train"] = "images/train"
    config["val"] = "images/val"
    output_yaml = prepared / "data.yaml"
    output_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    return output_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("datasets/bags_full/data.yaml"))
    ap.add_argument("--model", default=None,
                    help="resume training from an RF-DETR checkpoint (.pt/.pth) "
                         "instead of starting from the pretrained variant")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="nano")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--resolution", type=int, default=384,
                    help="square input size (Nano default 384, divisible by 32)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, mps")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="bags_full")
    ap.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                    help="verify label counts/class balance before training (default: on)")
    ap.add_argument("--verify-class", default="bag",
                    help="class id or name to verify (default: bag)")
    ap.add_argument("--min-train", type=int, default=100)
    ap.add_argument("--min-val", type=int, default=10)
    ap.add_argument("--min-class-fraction", type=float, default=0.05)
    ap.add_argument("--max-split-ratio", type=float, default=3.0)
    ap.add_argument("--max-class-ratio", type=float, default=15.0)
    a = ap.parse_args()

    data = a.data.resolve()
    if not data.exists():
        raise SystemExit(f"dataset config not found: {data}")
    temp = tempfile.TemporaryDirectory(prefix="bagcount-train-")
    train_data = prepare_training_data(data, a.verify_class, Path(temp.name))
    if a.verify:
        from bagcount.label_check import verify_labels

        report = verify_labels(
            train_data,
            class_spec=a.verify_class,
            min_train=a.min_train,
            min_val=a.min_val,
            min_class_fraction=a.min_class_fraction,
            max_split_ratio=a.max_split_ratio,
            max_class_ratio=a.max_class_ratio,
        )
        print(report.render(), flush=True)
        if not report.ok:
            temp.cleanup()
            raise SystemExit(
                "label verification failed — fix the labels or pass --no-verify "
                "to train anyway"
            )

    if a.model:
        model = RFDETR.from_checkpoint(a.model, device=a.device)
    else:
        model = VARIANTS[a.variant](device=a.device)
    train_kwargs = {}
    if a.model:
        train_kwargs["resume"] = a.model  # actually continue weights + optimizer
    try:
        model.train(
            dataset_dir=str(train_data.parent),
            epochs=a.epochs,
            resolution=a.resolution,
            batch_size=a.batch,
            grad_accum_steps=a.grad_accum,
            lr=a.lr,
            num_workers=a.workers,
            device=a.device,
            output_dir=str(Path("runs") / "detect" / a.name),
            **train_kwargs,
        )
    finally:
        temp.cleanup()
    out = Path("runs") / "detect" / a.name
    print(f"checkpoints in {out}: checkpoint_best_ema.pth / checkpoint_best_regular.pth")


if __name__ == "__main__":
    main()
