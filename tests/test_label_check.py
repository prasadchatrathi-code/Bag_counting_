"""Tests for the pre-training label count / class balance verification."""
from __future__ import annotations

from pathlib import Path

import pytest

from bagcount.label_check import (
    count_split,
    read_class_names,
    resolve_class,
    verify_labels,
)


def make_dataset(
    root: Path,
    train: list[tuple[int, int]],
    val: list[tuple[int, int]],
) -> Path:
    """Build a tiny YOLO-format dataset: each tuple is (person boxes, bag boxes)."""
    data = root / "ds" / "data.yaml"
    data.parent.mkdir(parents=True)
    data.write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: bag\n")
    for split, frames in (("train", train), ("val", val)):
        label_dir = data.parent / "labels" / split
        label_dir.mkdir(parents=True)
        for i, (people, bags) in enumerate(frames):
            lines = ["0 0.5 0.5 0.1 0.1"] * people + ["1 0.5 0.5 0.1 0.1"] * bags
            (label_dir / f"f{i:03d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return data


def test_read_names_and_resolve_class(tmp_path: Path) -> None:
    data = make_dataset(tmp_path, [(1, 1)], [(1, 1)])
    names = read_class_names(data)
    assert names == {0: "person", 1: "bag"}
    assert resolve_class(names, "bag") == 1
    assert resolve_class(names, "1") == 1
    with pytest.raises(ValueError, match="unknown class"):
        resolve_class(names, "truck")


def test_count_split_counts_boxes_and_labelled_images(tmp_path: Path) -> None:
    data = make_dataset(tmp_path, [(2, 3), (0, 0)], [(1, 1)])
    names = read_class_names(data)
    s = count_split(data.parent, "train", names)
    assert s.label_files == 2
    assert s.labelled_images == 1
    assert s.count(0) == 2
    assert s.count(1) == 3
    assert s.total == 5
    assert s.fraction(1) == pytest.approx(0.6)
    assert s.density(1) == pytest.approx(1.5)


def test_balanced_dataset_passes(tmp_path: Path) -> None:
    data = make_dataset(tmp_path, [(10, 5)] * 20, [(7, 3)] * 10)
    report = verify_labels(data)
    assert report.ok, report.render()
    assert report.class_name == "bag"
    assert len(report.checks) >= 7
    assert "RESULT: PASS" in report.render()


def test_too_few_target_boxes_fails(tmp_path: Path) -> None:
    data = make_dataset(tmp_path, [(10, 5)] * 2, [(7, 3)] * 10)
    report = verify_labels(data)
    assert not report.ok
    assert any(c.name == "train bag boxes" and not c.ok for c in report.checks)
    assert "RESULT: FAIL" in report.render()


def test_class_share_fails(tmp_path: Path) -> None:
    data = make_dataset(tmp_path, [(50, 1)] * 20, [(7, 3)] * 10)
    report = verify_labels(data)
    assert not report.ok
    assert any(c.name == "train bag share" and not c.ok for c in report.checks)


def test_split_density_balance_fails(tmp_path: Path) -> None:
    data = make_dataset(tmp_path, [(0, 5)] * 20, [(30, 1)] * 10)
    report = verify_labels(data)
    assert not report.ok
    check = next(c for c in report.checks if c.name == "bag train/val density balance")
    assert not check.ok
    assert "n/a" not in check.detail


def test_custom_class_id_and_thresholds(tmp_path: Path) -> None:
    data = make_dataset(tmp_path, [(20, 1)] * 10, [(20, 1)] * 5)
    # With bag as the target the share check fails; person passes.
    assert not verify_labels(data, class_spec=1).ok
    report = verify_labels(data, class_spec=0, min_class_fraction=0.5,
                           max_class_ratio=25)
    assert report.ok, report.render()
    assert report.class_name == "person"


def test_report_shows_class_count_every_class_and_differences(tmp_path: Path) -> None:
    data = make_dataset(tmp_path, [(10, 5)] * 20, [(7, 3)] * 10)
    report = verify_labels(data)

    assert report.num_classes == 2
    assert report.class_ids == [0, 1]
    assert report.class_counts() == {
        0: {"train": 200, "val": 70},
        1: {"train": 100, "val": 30},
    }
    text = report.render()
    assert "Classes: 2 (0=person, 1=bag)" in text
    assert "Per-class counts:" in text
    assert "+70" in text  # train - val difference for bag (100 - 30)
    assert "Class differences within each split:" in text
    assert "most=person 200, least=bag 100, diff=100, ratio=2.00x" in text
    assert "most=person 70, least=bag 30, diff=40, ratio=2.33x" in text


def test_class_imbalance_fails(tmp_path: Path) -> None:
    data = make_dataset(tmp_path, [(100, 1)] * 20, [(7, 3)] * 10)
    report = verify_labels(data)
    assert not report.ok
    check = next(c for c in report.checks if c.name == "train class balance")
    assert not check.ok
    assert "diff=1980" in check.detail  # 2000 person vs 20 bag


def test_declared_class_with_no_boxes_is_a_warning(tmp_path: Path) -> None:
    root = tmp_path / "three"
    root.mkdir()
    data = root / "data.yaml"
    data.write_text("names:\n  0: person\n  1: bag\n  2: truck\n")
    for split, frames in (("train", [(10, 5)] * 20), ("val", [(7, 3)] * 10)):
        label_dir = root / "labels" / split
        label_dir.mkdir(parents=True)
        for i, (people, bags) in enumerate(frames):
            lines = ["0 0.5 0.5 0.1 0.1"] * people + ["1 0.5 0.5 0.1 0.1"] * bags
            (label_dir / f"f{i:03d}.txt").write_text("\n".join(lines))

    report = verify_labels(data)
    assert report.num_classes == 3
    assert report.ok  # missing class is a warning, not a hard failure
    assert any(c.name == "all declared classes labelled in both splits"
               and not c.ok and not c.fatal for c in report.checks)
    assert "2=truck" in report.render()
