from bagcount.tracker import IoUTracker, box_iou


def test_same_box_keeps_its_id():
    t = IoUTracker(match_iou=0.5, max_age=2)
    box = (10.0, 10.0, 50.0, 50.0)
    first = t.update([box])
    second = t.update([box])
    assert set(first) == set(second) and len(second) == 1


def test_moved_box_is_matched_by_iou():
    t = IoUTracker(match_iou=0.3, max_age=2)
    a = (0.0, 0.0, 10.0, 10.0)
    b = (2.0, 0.0, 12.0, 10.0)
    first = t.update([a])
    second = t.update([b])
    assert set(first) == set(second)
    assert next(iter(second.values())) == b


def test_far_box_gets_a_new_id():
    t = IoUTracker(match_iou=0.5, max_age=2)
    first = t.update([(0.0, 0.0, 10.0, 10.0)])
    second = t.update([(100.0, 100.0, 110.0, 110.0)])
    # No stale prediction is returned to the counter, and the far detection is
    # not close enough to the predicted position to reuse the old id.
    assert len(second) == 1
    assert next(iter(second)) not in first


def test_missing_box_is_pruned_after_max_age():
    t = IoUTracker(match_iou=0.5, max_age=2)
    old_id = next(iter(t.update([(0.0, 0.0, 10.0, 10.0)])))
    assert t.update([]) == {} and old_id in t.pending_ids()
    assert t.update([]) == {} and old_id in t.pending_ids()
    assert t.update([]) == {} and old_id not in t.pending_ids()


def test_track_keeps_its_id_after_thirty_missing_frames():
    """A 30-frame CCTV dropout must not spawn a new bag identity."""
    t = IoUTracker(match_iou=0.5, max_age=30)
    original_id = next(iter(t.update([(0.0, 0.0, 10.0, 10.0)])))
    # Establish a steady +2 px/frame motion so the predictor is useful.
    for x in (2.0, 4.0):
        assert next(iter(t.update([(x, 0.0, x + 10.0, 10.0)]))) == original_id

    for _ in range(30):
        assert t.update([]) == {}
        assert original_id in t.pending_ids()

    # At frame 33 the bag is at x=66 (4 + 2*31); the predictor lands on it.
    resumed = t.update([(66.0, 0.0, 76.0, 10.0)])
    assert resumed == {original_id: (66.0, 0.0, 76.0, 10.0)}


def test_unrelated_box_does_not_reuse_a_pending_id():
    t = IoUTracker(match_iou=0.5, max_age=30)
    old_id = next(iter(t.update([(0.0, 0.0, 10.0, 10.0)])))
    for _ in range(30):
        t.update([])
    resumed = t.update([(400.0, 400.0, 410.0, 410.0)])
    assert old_id not in resumed


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
