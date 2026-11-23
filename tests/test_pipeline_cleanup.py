"""Cleanup guarantees for Pipeline.run() (stop request and exceptions)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bagcount import pipeline as P
from bagcount.config import Config


class FakeDetections:
    def __len__(self) -> int:
        return 0


class FakeModel:
    class_names = {0: "person", 1: "bag"}

    def __init__(self) -> None:
        self.fail = False

    def predict(self, frame: np.ndarray, **kwargs):
        if self.fail:
            raise RuntimeError("model failed")
        return FakeDetections()


class FakeRFDETR:
    model = FakeModel()

    @staticmethod
    def from_checkpoint(path, device=None):
        return FakeRFDETR.model


def make_video(path: Path, frames: int = 2) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 64))
    assert w.isOpened()
    try:
        for _ in range(frames):
            w.write(np.zeros((64, 64, 3), np.uint8))
    finally:
        w.release()


@pytest.fixture
def pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> P.Pipeline:
    monkeypatch.setattr(P, "RFDETR", FakeRFDETR)
    make_video(tmp_path / "in.mp4")
    return P.Pipeline(Config(), tmp_path / "out", save_video=True)


def test_stop_releases_resources_and_writes_summary(pipeline: P.Pipeline, tmp_path: Path) -> None:
    pipeline.stop = True
    summary = pipeline.run(str(tmp_path / "in.mp4"))

    out = tmp_path / "out"
    assert summary.frames == 0
    assert (out / "events.csv").exists()
    assert (out / "summary.json").exists()
    assert json.loads((out / "summary.json").read_text())["frames"] == 0


def test_exception_still_releases_resources_and_writes_summary(
    pipeline: P.Pipeline, tmp_path: Path
) -> None:
    pipeline.model.fail = True
    with pytest.raises(RuntimeError, match="model failed"):
        pipeline.run(str(tmp_path / "in.mp4"))

    out = tmp_path / "out"
    assert (out / "events.csv").exists()
    assert (out / "summary.json").exists()
