"""Time-window inference: seek to --start and stop after --duration."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bagcount import cli, pipeline as P
from bagcount.config import Config


class FakeDetections:
    def __len__(self) -> int:
        return 0


class FakeModel:
    class_names = {0: "person", 1: "bag"}

    def predict(self, frame: np.ndarray, **kwargs):
        return FakeDetections()


class FakeRFDETR:
    model = FakeModel()

    @staticmethod
    def from_checkpoint(path, device=None):
        return FakeRFDETR.model


@pytest.mark.parametrize(
    ("value", "expected"),
    [("90", 90.0), ("1.5", 1.5), ("90s", 90.0), ("2m", 120.0), ("1h", 3600.0),
     ("1hr26m30", 5190.0), ("1hr27m50", 5270.0), ("1h26m30s", 5190.0),
     ("1h 26m 30s", 5190.0), ("2m30", 150.0), ("1hour30minutes", 5400.0),
     ("01:30", 90.0), ("00:02:00", 120.0), (120, 120.0)],
)
def test_parse_time(value, expected):
    assert P.parse_time(value) == expected


@pytest.mark.parametrize("value", ["", "abc", "2x", "01:90", "1:2:3:4", "-5", "2mm", "1hrabc"])
def test_parse_time_rejects_invalid(value):
    with pytest.raises(ValueError):
        P.parse_time(value)


def test_parse_line_accepts_normalised_coordinates():
    assert P.parse_line("0.28,0.68,0.78,0.68") == ((0.28, 0.68), (0.78, 0.68))
    assert P.parse_line("0,0 1,1") == ((0.0, 0.0), (1.0, 1.0))


@pytest.mark.parametrize("value", ["", "a,b,c,d", "0.1,0.2,0.3", "0,0,1,2", "0.5,0.5,0.5,0.5"])
def test_parse_line_rejects_invalid(value):
    with pytest.raises(ValueError):
        P.parse_line(value)


def make_video(path: Path, frames: int = 80) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 64))
    assert w.isOpened()
    try:
        for _ in range(frames):
            w.write(np.zeros((64, 64, 3), np.uint8))
    finally:
        w.release()


def test_window_seeks_and_stops_with_source_relative_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(P, "RFDETR", FakeRFDETR)
    video = tmp_path / "in.mp4"
    make_video(video)
    out = tmp_path / "out"
    times: list[float] = []
    p = P.Pipeline(Config(), out, on_frame=lambda fi, t, frame: times.append(t))

    summary = p.run(str(video), start_s=2.0, duration_s=3.0)

    # 8 fps: start frame 16, process 24 frames (frames 16..39 = 2s..5s).
    assert summary.frames == 24
    assert summary.start_s == 2.0
    assert summary.duration_s == 3.0
    assert times[0] == pytest.approx(2.0)
    assert times[-1] == pytest.approx(2.0 + 23 / 8)
    with (out / "frame_times.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["frame"]) for r in rows] == list(range(16, 40))
    report = json.loads((out / "summary.json").read_text())
    assert report["window"] == {"start_frame": 16, "end_frame": 40, "source_relative_times": True}
    assert report["counting_line"]["enabled"] is True
    assert report["counting_line"]["source"] == "derived"
    assert len(report["counting_line"]["a"]) == len(report["counting_line"]["b"]) == 2


def test_start_beyond_source_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "RFDETR", FakeRFDETR)
    video = tmp_path / "in.mp4"
    make_video(video, frames=16)
    p = P.Pipeline(Config(), tmp_path / "out")
    with pytest.raises(RuntimeError, match="beyond the source"):
        p.run(str(video), start_s=10.0)


def test_cli_passes_window_to_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, cfg, out_dir, **kwargs) -> None:
            pass

        def run(self, source, max_frames=0, start_s=0.0, duration_s=0.0):
            calls.update(source=source, max_frames=max_frames, start_s=start_s, duration_s=duration_s)
            return SimpleNamespace(frames=0, loaded=0, unloaded=0, on_truck=0,
                                   events_csv="", start_s=start_s, duration_s=duration_s)

    monkeypatch.setattr(cli, "Pipeline", FakePipeline)
    monkeypatch.setattr(sys, "argv", ["bagcount", "in.mp4", "--out", str(tmp_path / "out"),
                                      "--start", "2m", "--end", "3m"])
    cli.main()
    assert calls == {"source": "in.mp4", "max_frames": 0, "start_s": 120.0, "duration_s": 60.0}
    assert "window=120s..180s" in capsys.readouterr().out


def test_cli_accepts_compound_hour_minute_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, cfg, out_dir, **kwargs) -> None:
            pass

        def run(self, source, max_frames=0, start_s=0.0, duration_s=0.0):
            calls.update(start_s=start_s, duration_s=duration_s)
            return SimpleNamespace(frames=0, loaded=0, unloaded=0, on_truck=0,
                                   events_csv="", start_s=start_s, duration_s=duration_s)

    monkeypatch.setattr(cli, "Pipeline", FakePipeline)
    monkeypatch.setattr(sys, "argv", ["bagcount", "video.mp4", "--out", str(tmp_path / "out"),
                                      "--start", "1hr26m30", "--end", "1hr27m50"])
    cli.main()
    assert calls == {"start_s": 5190.0, "duration_s": 80.0}


def test_cli_applies_count_line_and_tracker_age(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, cfg, out_dir, **kwargs) -> None:
            seen["line"] = cfg.zones.cross_line
            seen["tracker_age"] = cfg.detector.tracker_max_age

        def run(self, source, max_frames=0, start_s=0.0, duration_s=0.0):
            return SimpleNamespace(frames=0, loaded=0, unloaded=0, on_truck=0,
                                   events_csv="", start_s=0.0, duration_s=0.0)

    monkeypatch.setattr(cli, "Pipeline", FakePipeline)
    monkeypatch.setattr(sys, "argv", ["bagcount", "v.mp4", "--out", "out/x",
                                      "--line", "0.28,0.68,0.78,0.68",
                                      "--tracker-max-age", "30"])
    cli.main()
    assert seen == {"line": ((0.28, 0.68), (0.78, 0.68)), "tracker_age": 30}


def test_cli_rejects_conflicting_length_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["bagcount", "in.mp4", "--max-frames", "10", "--duration", "5"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
