from bagcount.config import CounterConfig, Zones
from bagcount.counter import Direction, TransitionCounter


def walk(counter, tid, ys, x=0.5, start=0):
    evs = []
    for i, y in enumerate(ys):
        evs += counter.update(start + i, {tid: (x, y)})
    return evs


def test_dock_to_truck_counts_as_loaded():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2))
    evs = walk(c, 1, [0.9, 0.85, 0.8, 0.7, 0.66, 0.6, 0.55, 0.5])
    assert [e.direction for e in evs] == [Direction.LOADED]
    assert c.loaded == 1 and c.unloaded == 0 and c.on_truck == 1


def test_truck_to_dock_counts_as_unloaded():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2))
    evs = walk(c, 7, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert [e.direction for e in evs] == [Direction.UNLOADED]
    assert c.on_truck == -1


def test_hover_at_boundary_does_not_double_count():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2))
    ys = [0.9, 0.85, 0.75, 0.66, 0.62, 0.66, 0.62, 0.6, 0.55, 0.5, 0.45]
    evs = walk(c, 2, ys)
    assert len(evs) == 1 and c.loaded == 1


def test_single_frame_flicker_is_ignored():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2))
    # One frame in the truck zone, then back on the dock: no event.
    evs = walk(c, 3, [0.9, 0.85, 0.5, 0.85, 0.9])
    assert evs == [] and c.loaded == 0


def test_track_that_starts_in_truck_and_stays_is_not_counted():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2))
    evs = walk(c, 4, [0.3, 0.3, 0.35, 0.4])
    assert evs == []


def test_outside_both_polygons_is_transit():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2))
    # Far left of frame: the other, static truck. Never counted.
    evs = walk(c, 5, [0.9, 0.8, 0.5, 0.3], x=0.1)
    assert evs == []


def test_round_trip_nets_to_zero():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2))
    walk(c, 6, [0.9, 0.85, 0.5, 0.45, 0.4, 0.5, 0.85, 0.9])
    assert c.loaded == 1 and c.unloaded == 1 and c.on_truck == 0


def test_id_switch_near_tailgate_is_stitched():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2))
    # Track 1 settles on the dock, then vanishes for 3 frames.
    walk(c, 1, [0.9, 0.85, 0.78], start=0)
    # A new id appears just across the tailgate and carries on into the truck.
    evs = walk(c, 9, [0.66, 0.62, 0.58, 0.5], start=6)
    assert [e.direction for e in evs] == [Direction.LOADED]
    assert c.loaded == 1


def test_new_track_far_from_lost_one_is_not_stitched():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2))
    walk(c, 1, [0.9, 0.85, 0.8], start=0)
    evs = walk(c, 9, [0.3, 0.3, 0.3], x=0.5, start=6)  # appears deep in the truck
    assert evs == [] and c.loaded == 0


def test_stale_lost_track_is_not_stitched():
    # Short time-based stitching window: a 37-frame gap is stale here.
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=100, stitch_seconds=0.5))
    walk(c, 1, [0.9, 0.85, 0.8], start=0)
    evs = walk(c, 9, [0.66, 0.62, 0.58, 0.5], start=40)  # 37 frames later
    # Not stitched to the stale track (own root)...
    assert c.tracks[9].root == 9
    # ...but it was born in transit and carried into the truck: a handover, counted on its own.
    assert [e.direction for e in evs] == [Direction.LOADED] and c.loaded == 1


def test_dock_track_lost_in_transit_heading_up_is_inferred_loaded():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=4))
    # Settles on the dock, seen twice in the transit band near the truck, then gone.
    evs = walk(c, 1, [0.9, 0.85, 0.8, 0.70, 0.66], start=0)
    assert evs == []
    for f in range(5, 12):
        evs += c.update(f, {})
    assert len(evs) == 1 and evs[0].direction is Direction.LOADED and evs[0].inferred
    assert evs[0].frame_idx == 4


def test_dock_track_lost_in_transit_heading_back_is_not_counted():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=4))
    # Goes to the tailgate, turns back toward the dock, then is lost: no event.
    evs = walk(c, 1, [0.9, 0.85, 0.8, 0.66, 0.71], start=0)
    for f in range(5, 12):
        evs += c.update(f, {})
    assert evs == []


def test_truck_track_lost_in_transit_heading_down_is_inferred_unloaded():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=4))
    evs = walk(c, 1, [0.3, 0.4, 0.5, 0.66, 0.70], start=0)
    for f in range(5, 12):
        evs += c.update(f, {})
    assert [(e.direction, e.inferred) for e in evs] == [(Direction.UNLOADED, True)]


def test_inference_can_be_disabled():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=4, infer_lost_in_transit=False))
    evs = walk(c, 1, [0.9, 0.85, 0.8, 0.70, 0.66], start=0)
    for f in range(5, 12):
        evs += c.update(f, {})
    assert evs == []


def test_fragmented_track_counts_once():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=4))
    walk(c, 8, [0.94, 0.9, 0.85, 0.8], start=0)           # settles on the dock, lost after f3
    evs = c.update(5, {11: (0.5, 0.76), 12: (0.5, 0.70)})   # two fragments born together
    evs += c.update(6, {12: (0.5, 0.64)})                   # the real one crosses the line
    for f in range(7, 14):                                  # ...and is lost mid-carry
        evs += c.update(f, {})
    assert [(e.direction, e.inferred) for e in evs] == [(Direction.LOADED, True)]
    assert c.loaded == 1


def test_two_successors_both_reaching_truck_count_once():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=4))
    walk(c, 1, [0.9, 0.85, 0.78], start=0)
    evs = []
    for i, y in enumerate([0.66, 0.6, 0.55, 0.5]):
        evs += c.update(5 + i, {21: (0.5, y), 22: (0.52, y - 0.02)})
    assert c.loaded == 1 and len(evs) == 1


def test_next_bag_from_same_spot_is_a_new_bag():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=30))
    # Bag A: dock fragment lost, continuation adopts it and is loaded.
    walk(c, 1, [0.9, 0.85, 0.78], start=0)
    walk(c, 2, [0.68, 0.62, 0.55, 0.45], start=4)
    assert c.loaded == 1
    # Bag B picked up from the same spot 6 frames later must count separately.
    walk(c, 3, [0.80, 0.78, 0.70, 0.62, 0.55, 0.45], start=9)
    assert c.loaded == 2


def test_duplicate_box_copies_live_track_not_a_lost_one():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=30))
    # Bag X settles on the dock at x=0.43 and is lost at f4.
    walk(c, 2, [0.94, 0.93, 0.91, 0.90, 0.87], x=0.43, start=0)
    # Bag Y is tracked on the dock at x=0.60; at f5 a duplicate box (id 6) appears on it.
    walk(c, 4, [0.95, 0.94, 0.92, 0.90, 0.88], x=0.60, start=1)
    c.update(6, {4: (0.60, 0.86), 6: (0.60, 0.83)})
    # Bag X continues as id 5 and both bags are carried into the truck.
    evs = []
    for i, (yx, yy) in enumerate(zip([0.78, 0.72, 0.65, 0.61, 0.5, 0.46], [0.81, 0.78, 0.74, 0.68, 0.62, 0.55])):
        evs += c.update(7 + i, {5: (0.44, yx), 6: (0.60, yy)})
    assert c.loaded == 2, [(e.track_id, e.direction) for e in evs]


def test_stale_fragment_of_a_counted_bag_is_not_inherited():
    """Seen with bags_v5 on the 13:00 clip: bag A is tracked as id 1 and a
    duplicate box (id 2) copies it on the dock. Id 2 is lost, id 1 carries A
    into the truck and is counted. Bag B then appears on the dock as id 3 next
    to where id 2 vanished. B must not inherit A's root, or its own crossing
    is suppressed as a duplicate of A's."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=30, dedup_frames=80))
    walk(c, 1, [0.95, 0.93, 0.91], x=0.5, start=0)
    c.update(3, {1: (0.5, 0.90), 2: (0.5, 0.88)})  # id 2 duplicates live id 1
    c.update(4, {1: (0.5, 0.85), 2: (0.5, 0.84)})
    # id 2 lost; id 1 crosses into the truck.
    walk(c, 1, [0.75, 0.68, 0.62, 0.55, 0.5], x=0.5, start=5)
    assert c.loaded == 1
    # Bag B picked up from the same spot as id 3, two frames later.
    evs = walk(c, 3, [0.86, 0.85, 0.80, 0.72, 0.66, 0.60, 0.55, 0.5], x=0.5, start=10)
    assert [e.direction for e in evs] == [Direction.LOADED]
    assert c.loaded == 2


def test_carrier_box_around_tight_box_is_the_same_bag():
    """bags_v5 often emits a tight box on the bag and, a few frames later, a
    narrow box covering bag + carrier: same top edge, same x-range, much
    taller. Centres are too far apart for the radius test. Count once."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, duplicate_radius=0.02))
    ys = [0.9, 0.85, 0.8, 0.75, 0.7, 0.64, 0.6, 0.55, 0.5, 0.45]
    evs = []
    for i, y in enumerate(ys):
        obs = {1: (0.43, y)}
        boxes = {1: (0.38, y - 0.1, 0.47, y + 0.1)}
        if i >= 3:  # carrier box: same top, extends 0.2 further down (centre 0.1 lower)
            obs[2] = (0.435, y + 0.1)
            boxes[2] = (0.39, y - 0.09, 0.48, y + 0.3)
        evs += c.update(i, obs, boxes)
    assert c.loaded == 1, [(e.track_id, e.frame_idx) for e in evs]


def test_bag_beneath_a_lifted_bag_is_a_different_bag():
    """Seen with bags_v3: when the top bag of the pile is lifted, the bag under
    it appears as a new box of similar size just below. Both get loaded."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, duplicate_radius=0.06))
    evs = []
    for i in range(14):
        y1 = 0.86 - 0.03 * i  # bag A lifted and carried in
        obs = {1: (0.45, y1)}
        boxes = {1: (0.40, y1 - 0.08, 0.50, y1 + 0.08)}
        if i >= 2:  # bag B visible beneath from the second frame of the lift
            y2 = 0.86 if i < 8 else 0.86 - 0.04 * (i - 8)
            obs[2] = (0.45, y2 + 0.07)
            boxes[2] = (0.40, y2 - 0.01, 0.50, y2 + 0.15)
        evs += c.update(i, obs, boxes)
    for i in range(14, 24):  # bag B carried in too
        y2 = 0.62 - 0.03 * (i - 14)
        evs += c.update(i, {2: (0.45, y2)}, {2: (0.40, y2 - 0.08, 0.50, y2 + 0.08)})
    assert c.loaded == 2, [(e.track_id, e.frame_idx) for e in evs]


def test_handover_track_born_in_truck_beside_pile_is_counted_once_it_moves():
    """Hand-up phase: the pile bag's track stays on the pile (the next bag
    beneath looks the same) while the lifted bag's own track starts inside the
    truck zone at the tailgate. It must still count, once it has moved."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, handover_edge_dist=0.25, handover_min_travel=0.06))
    pile = {1: (0.45, 0.80)}
    for f in range(5):
        c.update(f, dict(pile))
    evs = []
    for i, y in enumerate([0.58, 0.57, 0.56, 0.55, 0.50, 0.45, 0.40]):  # born at tailgate, then carried up
        evs += c.update(5 + i, {**pile, 2: (0.47, y)})
    assert [e.direction for e in evs] == [Direction.LOADED]
    assert evs[0].track_id == 2 and c.loaded == 1


def test_static_bag_redetected_near_tailgate_is_not_a_handover():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=10, handover_min_travel=0.06))
    pile = {1: (0.45, 0.80)}
    for f in range(5):
        c.update(f, dict(pile))
    for f in range(5, 20):  # bag on the bed near the tailgate flickers in, never moves
        c.update(f, {**pile, 3: (0.50, 0.58 + 0.003 * (f % 2))})
    for f in range(20, 40):
        c.update(f, dict(pile))
    assert c.loaded == 0


def test_revived_id_on_a_new_bag_at_the_pickup_spot_counts():
    """ByteTrack can revive a lost id on the next bag picked from the same
    spot. If that id's bag was already counted, the revived track is a new bag."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, revive_gap_frames=4, max_lost_frames=30))
    walk(c, 1, [0.95, 0.93, 0.91], x=0.5, start=0)
    c.update(3, {1: (0.5, 0.90), 2: (0.5, 0.88)})  # id 2 duplicates live id 1 (root 1)
    # id 1 lost at f4; id 2 carries the bag in and is counted (root 1).
    walk(c, 2, [0.8, 0.72, 0.66, 0.6, 0.55, 0.5], x=0.5, start=4)
    assert c.loaded == 1
    # Tracker revives id 1 on the next bag, 8 frames after it vanished, on the dock.
    evs = walk(c, 1, [0.9, 0.88, 0.8, 0.72, 0.66, 0.6, 0.55, 0.5], x=0.5, start=12)
    assert [e.direction for e in evs] == [Direction.LOADED]
    assert c.loaded == 2


def test_same_bag_seen_again_on_the_truck_is_not_recounted():
    """The revival rule must not fire when the id reappears outside its settled
    zone: that is the same bag, re-found after the carry."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, revive_gap_frames=4, max_lost_frames=30))
    walk(c, 1, [0.95, 0.93, 0.91, 0.8], x=0.5, start=0)       # settled dock, lost in transit at f3
    walk(c, 2, [0.75, 0.68, 0.62, 0.55, 0.5], x=0.5, start=5)  # successor adopts, counted (root 1)
    assert c.loaded == 1
    walk(c, 1, [0.5, 0.45, 0.4, 0.35], x=0.5, start=12)       # id 1 revived on the truck bed
    assert c.loaded == 1


def test_track_born_deep_in_truck_zone_is_not_a_handover():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, handover_edge_dist=0.25))
    # Born at the front of the bed (far from the tailgate) and moved around: no event.
    walk(c, 5, [0.2, 0.22, 0.25, 0.3, 0.35, 0.4], x=0.5)
    assert c.loaded == 0


def test_lost_duplicate_of_a_live_track_is_not_adopted():
    """v3 on the 13:00 clip: a carrier box (id 2) copies the lifted bag (id 1),
    vanishes, and the next pile bag (id 3) appears beside where it vanished
    while id 1 is still tracked and about to be counted. Id 3 must start fresh."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, duplicate_radius=0.02))
    ys1 = [0.9, 0.86, 0.82, 0.78, 0.74, 0.70, 0.66, 0.62, 0.58, 0.54, 0.5]
    evs = []
    for i, y in enumerate(ys1):
        obs = {1: (0.5, y)}
        boxes = {1: (0.46, y - 0.09, 0.55, y + 0.09)}
        if i in (2, 3):  # carrier box for two frames
            obs[2] = (0.5, y + 0.12); boxes[2] = (0.46, y - 0.08, 0.56, y + 0.32)
        if i >= 6:  # next bag on the pile, near where id 2 vanished
            obs[3] = (0.52, 0.93); boxes[3] = (0.47, 0.86, 0.57, 1.0)
        evs += c.update(i, obs, boxes)
    assert c.loaded == 1
    for i, y in enumerate([0.9, 0.84, 0.78, 0.72, 0.66, 0.6, 0.55, 0.5]):
        evs += c.update(len(ys1) + i, {3: (0.52, y)}, {3: (0.47, y - 0.07, 0.57, y + 0.07)})
    assert c.loaded == 2, [(e.track_id, e.frame_idx) for e in evs]


def test_carrier_box_recognised_after_birth():
    """The carrier box can be born on the dock before the lift, when its
    geometry does not yet match; once it does, the two tracks share a root."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, duplicate_radius=0.02, carrier_top_tol=0.05))
    evs = []
    ys = [0.9, 0.88, 0.86, 0.82, 0.76, 0.70, 0.64, 0.58, 0.52, 0.46]
    for i, y in enumerate(ys):
        obs = {1: (0.43, y), 2: (0.435, y + 0.12)}
        boxes = {1: (0.38, y - 0.1, 0.47, y + 0.1)}
        top_off = 0.10 if i < 3 else 0.02  # tops align only once the bag is lifted
        boxes[2] = (0.39, y - 0.1 + top_off, 0.48, y + 0.3)
        evs += c.update(i, obs, boxes)
    assert c.loaded == 1, [(e.track_id, e.frame_idx) for e in evs]


def test_false_bag_drifting_along_the_tailgate_is_not_counted():
    """A worker's head at the tailgate is sometimes detected as a bag. Such a
    track is born in the handover band and may wander, but never gets further
    from the tailgate: no event."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, handover_min_travel=0.06))
    walk(c, 7, [0.45, 0.5, 0.55, 0.6, 0.62, 0.63, 0.62, 0.6], x=0.35)  # drifts down to the tailgate
    for i, (x, y) in enumerate([(0.36, 0.63), (0.38, 0.63), (0.40, 0.63), (0.42, 0.62), (0.44, 0.62)]):
        c.update(8 + i, {7: (x, y)})  # ...then along it
    assert c.loaded == 0


def test_bag_handed_down_over_the_tailgate_counts_as_unloaded():
    """Stack-top phase: the worker on top lowers a bag to a carrier below. Its
    track is born inside the truck zone at the tailgate (so it looks like a
    handover going up) but then travels into the dock: unloaded."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, handover_edge_dist=0.25, allow_handover_unload=True, handover_min_travel=0.06,
                                                 handover_unload_frames=4))
    evs = walk(c, 5, [0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94])
    assert [e.direction for e in evs] == [Direction.UNLOADED]
    assert evs[0].frame_idx == 7  # fourth frame on the dock
    assert evs[0].track_id == 5 and c.unloaded == 1 and c.on_truck == -1


def test_handover_bag_hovering_just_below_the_tailgate_is_not_unloaded():
    """A lifted bag seen first over the tailgate, then held on the dock side
    right at the tailgate (carrier waiting), is not an unload until it clearly
    moves away; and if it then goes into the truck it is one load."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, handover_edge_dist=0.25, allow_handover_unload=True, handover_min_travel=0.06,
                                                 handover_unload_frames=4))
    walk(c, 6, [0.60, 0.62, 0.72, 0.73, 0.72, 0.73, 0.72])  # dock edge is y=0.72
    assert c.unloaded == 0 and c.loaded == 0
    evs = walk(c, 6, [0.62, 0.55, 0.50, 0.45], start=7)
    assert [e.direction for e in evs] == [Direction.LOADED] and c.unloaded == 0


def test_bag_dipping_below_the_tailgate_and_back_is_neither_unloaded_nor_reloaded():
    """Seen in the hand-up window: a bag box right on the tailgate line drops
    into the dock for two frames and comes back up. Before the dwell rule this
    gave an unload and a load three seconds apart."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, handover_min_travel=0.06, handover_unload_frames=4))
    walk(c, 50, [0.64, 0.65, 0.72, 0.72, 0.71, 0.66, 0.64, 0.64, 0.64], x=0.40)
    assert c.unloaded == 0 and c.loaded == 0


def test_track_born_in_transit_then_on_dock_is_not_unloaded():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, handover_min_travel=0.06))
    evs = walk(c, 8, [0.66, 0.68, 0.75, 0.8, 0.85, 0.9])  # picked up at the tailgate band, carried back
    assert evs == [] and c.unloaded == 0


def test_id_switch_right_after_count_does_not_double_count():
    """Observed on the Paramesu clip: the bag is counted, then the tracker
    swaps its id one frame later with a spatial jump larger than the duplicate
    radius. The old id is still 'live', so the newcomer must inherit its root
    instead of becoming a second, provisionally-docked bag."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=30))
    walk(c, 1, [0.90, 0.85, 0.80, 0.75, 0.70], start=0)
    walk(c, 1, [0.68, 0.66, 0.65], start=5)
    evs = walk(c, 1, [0.62, 0.60], start=8)
    assert [e.direction for e in evs] == [Direction.LOADED]
    # Both boxes are present on the switch frame; the old one is stale by 0.10.
    evs = c.update(10, {1: (0.5, 0.60), 2: (0.5, 0.50)})
    assert evs == []
    assert c.tracks[2].root == 1 and c.tracks[2].zone.value == "truck"
    evs = []
    for i, y in enumerate([0.48, 0.45, 0.40, 0.35, 0.30]):
        evs += c.update(11 + i, {2: (0.5, y)})
    assert evs == []
    assert c.loaded == 1


def test_lost_in_transit_then_reidentified_two_seconds_later_counts():
    """At 30 fps a 2 s detection gap is 60 frames — far beyond the old 16-frame
    stitching window. The lost history must survive long enough for the new id
    to inherit it, otherwise the bag is never counted at all."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=30))
    walk(c, 1, [0.70, 0.68, 0.66], start=0)  # seen only in the transit band
    for f in range(3, 63):
        assert c.update(f, {}) == []
    assert c.loaded == 0 and 1 in c.tracks  # history retained for stitching
    evs = []
    for i, y in enumerate([0.35, 0.34, 0.33, 0.32, 0.31]):
        evs += c.update(63 + i, {9: (0.5, y)})
    assert [e.direction for e in evs] == [Direction.LOADED]
    assert c.tracks[9].root == 1
    assert c.loaded == 1


def test_count_line_blocks_zone_only_transition():
    """With a line deeper in the truck, settling in the truck zone is not
    enough. The count must wait until the bag physically passes the line."""
    zones = Zones(cross_line=((0.0, 0.5), (1.0, 0.5)))
    c = TransitionCounter(zones, CounterConfig(min_origin_frames=2))
    evs = walk(c, 1, [0.9, 0.85, 0.8, 0.75, 0.62, 0.60, 0.58])
    assert evs == [] and c.loaded == 0  # zone is truck, but line is not crossed

    evs = walk(c, 1, [0.48, 0.42, 0.38], start=7)
    assert [e.direction for e in evs] == [Direction.LOADED]
    assert evs[0].frame_idx == 7
    assert c.loaded == 1


def test_same_count_line_gates_loading_and_unloading():
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2))
    loaded = walk(c, 1, [0.9, 0.85, 0.8, 0.66, 0.6, 0.55], start=0)
    unloaded = walk(c, 1, [0.6, 0.66, 0.72, 0.8, 0.9], start=6)
    assert [e.direction for e in loaded] == [Direction.LOADED]
    assert [e.direction for e in unloaded] == [Direction.UNLOADED]
    assert (c.loaded, c.unloaded, c.on_truck) == (1, 1, 0)


def test_line_counting_can_be_disabled_for_legacy_zone_mode():
    zones = Zones(cross_line=((0.0, 0.5), (1.0, 0.5)))
    cfg = CounterConfig(min_origin_frames=2, require_line_crossing=False)
    c = TransitionCounter(zones, cfg)
    evs = walk(c, 1, [0.9, 0.85, 0.8, 0.62, 0.60, 0.58])
    assert [e.direction for e in evs] == [Direction.LOADED]


def test_bag_handed_down_then_lost_on_the_dock_is_unloaded_when_enabled():
    """Opt-in hand-down mode: the carrier walks away with the bag and the
    detector loses it before the dwell is reached. Still an unload, dated to
    the last sighting. Disabled by default because it is ambiguous (see below).
    """
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=4,
                                                 allow_handover_unload=True, handover_min_travel=0.06,
                                                 handover_unload_frames=8))
    evs = walk(c, 9, [0.50, 0.55, 0.60, 0.75, 0.85])
    assert evs == []
    for f in range(5, 12):
        evs += c.update(f, {})
    assert [(e.direction, e.frame_idx, e.inferred) for e in evs] == [(Direction.UNLOADED, 4, False)]


def test_tailgate_born_bag_falling_back_into_dock_is_not_unloaded():
    """run_6 regression: a bag first seen at the tailgate on the truck side
    and then carried back into the dock was counted as `unloaded`, even though
    it was really a dock pickup that was briefly misdetected at the tailgate.
    With the ambiguous hand-down heuristic off it settles as a normal dock bag
    and emits no event."""
    from bagcount.counter import Zone

    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2))
    evs = walk(c, 1, [0.62, 0.62, 0.66, 0.72, 0.80, 0.83, 0.83, 0.83, 0.83, 0.83])
    assert evs == []
    assert (c.loaded, c.unloaded, c.on_truck) == (0, 0, 0)
    assert c.tracks[1].zone is Zone.DOCK
    # Held in the provisional state: no unload, but no early `loaded` either.
    assert c.tracks[1].provisional is True


def test_tailgate_born_bag_can_still_be_counted_as_a_later_load():
    """The same false-unload bag must remain usable: having been normalised to
    the dock, a later pickup and crossing counts once as `loaded`."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2))
    walk(c, 1, [0.62, 0.62, 0.66, 0.72, 0.80, 0.83, 0.83, 0.83])
    evs = walk(c, 1, [0.70, 0.60, 0.52, 0.45, 0.40], start=8)
    assert [e.direction for e in evs] == [Direction.LOADED]
    assert c.on_truck == 1


def test_hand_down_sequence_is_not_unloaded_by_default():
    """Same trajectory as the opt-in hand-down test, but with the default
    config: no false unload, so `on_truck` never goes negative."""
    c = TransitionCounter(Zones(), CounterConfig(min_origin_frames=2, max_lost_frames=4))
    evs = walk(c, 9, [0.50, 0.55, 0.60, 0.75, 0.85])
    for f in range(5, 12):
        evs += c.update(f, {})
    assert evs == []
    assert c.unloaded == 0 and c.on_truck == 0
