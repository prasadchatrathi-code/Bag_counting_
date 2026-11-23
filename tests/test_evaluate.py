from pathlib import Path

from bagcount.evaluate import evaluate


def _csv(path: Path, rows):
    path.write_text("time_s,direction\n" + "".join(f"{t},{d}\n" for t, d in rows))


def test_matching_within_tolerance(tmp_path):
    p, t = tmp_path / "p.csv", tmp_path / "t.csv"
    _csv(p, [(10.0, "loaded"), (20.0, "loaded"), (31.0, "unloaded"), (50.0, "loaded")])
    _csv(t, [(12.0, "loaded"), (20.5, "loaded"), (30.0, "unloaded"), (70.0, "loaded")])
    r = evaluate(p, t, tolerance_s=4.0)
    assert (r.tp, r.fp, r.fn) == (3, 1, 1)
    assert r.unmatched_pred == [(50.0, "loaded")]
    assert r.unmatched_true == [(70.0, "loaded")]


def test_direction_must_match(tmp_path):
    p, t = tmp_path / "p.csv", tmp_path / "t.csv"
    _csv(p, [(10.0, "unloaded")])
    _csv(t, [(10.0, "loaded")])
    r = evaluate(p, t)
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)


def test_window_filters_both_sides(tmp_path):
    p, t = tmp_path / "p.csv", tmp_path / "t.csv"
    _csv(p, [(5.0, "loaded"), (100.0, "loaded")])
    _csv(t, [(5.0, "loaded"), (200.0, "loaded")])
    r = evaluate(p, t, window=(0, 50))
    assert (r.tp, r.fp, r.fn) == (1, 0, 0)
