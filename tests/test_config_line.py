import json

import pytest

from bagcount.config import Config


def test_defaults_have_thirty_frame_tracker_and_line_gating():
    cfg = Config()
    assert cfg.detector.tracker_max_age == 30
    assert cfg.counter.require_line_crossing is True


def test_config_loads_explicit_count_line(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({
        "zones": {"cross_line": [[0.1, 0.66], [0.9, 0.66]]},
        "detector": {"tracker_max_age": 30},
        "counter": {"require_line_crossing": True, "cross_line_tolerance": 0.02},
    }))
    cfg = Config.load(path)
    assert cfg.zones.cross_line == ((0.1, 0.66), (0.9, 0.66))
    assert cfg.detector.tracker_max_age == 30
    assert cfg.counter.cross_line_tolerance == 0.02


def test_config_rejects_invalid_count_line(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"zones": {"cross_line": [[0.1, 0.6]]}}))
    with pytest.raises(ValueError, match="two points"):
        Config.load(path)
