# Counting logic — line gating, tracker continuity and change log

This is the single reference for how bag counting works now. It explains the
new count line, the 30-frame tracker hold, the two miscounts you observed on the
Paramesu video, the exact code changes, and how to verify a run.

---

## 1. Pipeline today

```text
video frame
  → RF-DETR / ONNX detections (class "bag")
  → predictive IoU tracker (stable ids, 30-frame hold)
  → zone classifier (truck / dock / transit polygons)
  → COUNT LINE gate (dock side ↔ truck side)
  → zone-dwell counter (dock→truck = loaded, truck→dock = unloaded)
  → events.csv + thumbnails + summary.json + annotated.mp4
```

### The important rule

**The counter does not count ID changes, and it does not count from polygon
entry alone. An event is emitted only after the bag crosses the explicit count
line in the matching direction.**

| Crossing | Event |
|---|---|
| dock side → truck side | `loaded` (+1) |
| truck side → dock side | `unloaded` (+1) |
| no crossing (hover, flicker, polygon entry only) | no event |

Zone dwell (`min_origin_frames = 2`) is still required before a zone change is
trusted, so a bag shaking on the line does not produce rapid double counts.
The line gate is the final check in `_emit()`: every event — observed,
handover, inferred or adopted after an ID switch — must have a valid crossing
latch.

`on_truck = loaded - unloaded`.

---

## 2. The count line

### Where it comes from

- Default: derived halfway between the truck tailgate edge and the nearest dock
  edge, parallel to those edges and wide enough to cover the transition band.
  For the bundled camera geometry this is approximately
  `(0.30, 0.68) → (0.80, 0.68)` in normalised coordinates.
- Explicit: set `zones.cross_line` in a JSON config, or pass
  `--line X1,Y1,X2,Y2` on the CLI (normalised 0..1 coordinates).
- The live API accepts the same override as `line: "x1,y1,x2,y2"`.

The line is stored in `summary.json` under `counting_line`:

```json
"counting_line": {
  "enabled": true,
  "a": [0.3025, 0.68],
  "b": [0.8025, 0.68],
  "source": "derived",
  "truck_side_sign": -1,
  "tolerance": 0.01,
  "max_frames": 90
}
```

It is drawn as a yellow **COUNT LINE** on every annotated frame and on every
event thumbnail, so the gate used for counting is visible for audit.

### Antiflicker and custom lines

- Points within `cross_line_tolerance = 0.01` (normalised) of the line keep
  their previous side; a bag sitting on the line cannot flip the direction
  every frame.
- `cross_max_frames = 90` bounds how old a crossing latch may be. A bag carried
  a long way after crossing still counts; a stale latch from a return trip does
  not.
- If a custom line sits *inside* a zone (so the bag settles in the destination
  zone before reaching the line), the event is emitted at the crossing itself,
  using the remembered settled zone change. This is why line placement does not
  silently create undercounts.

---

## 3. 30-frame tracker continuity

`src/bagcount/tracker.py` was upgraded from a plain last-box IoU tracker to a
small predictive tracker:

- `DetectorConfig.tracker_max_age` is now **30 frames** (about 1 s at 30 fps).
- A confirmed position keeps a smoothed centre velocity.
- Missing tracks stay pending for up to `max_age` frames and are matched by
  predicted-box IoU **or** a centre-distance gate
  (`tracker_max_center_dist = 1.0` box diagonal).
- When detection resumes, the original ID is reused. No new bag identity is
  created for a momentary dropout.
- Pending predictions are **not** returned to the counter. Only real
  detections become observations, so a bag cannot appear to cross the line
  because of a tracker guess.

Override with `--tracker-max-age N`, `detector.tracker_max_age` in JSON, or the
live API's `tracker_max_age` field.

The counter also keeps its own stitching/history windows time-based
(`stitch_seconds = 2.5`, `history_seconds = 3.0`), so it agrees with the
tracker across frame rates.

---

## 4. Problems you observed and how they were fixed

### Problem A — counted, ID changed immediately, count increased again

Before the fix:

```text
f9   EVENT loaded root 1
f10  new id 2, old id 1 still live, 0.10 away
     → new root, provisional dock history
f13  EVENT loaded root 2   ← second count
```

Root cause: `_adopt_lost_track()` refused to adopt a track whose root was still
live in the same frame, and the duplicate-box radius (0.06) was too small for
the spatial jump. The newcomer became a fresh bag.

Fix: `_counted_neighbour()` now allows adoption when the candidate root was
counted within `stitch_switch_frames` and is settled in the **destination**
zone. The newcomer inherits that root and zone, so no second event is emitted.
A stale fragment still in the origin zone is deliberately not inherited, so
the next bag picked up from the same spot still counts.

### Problem B — lost in transition, re-identified ~2 s later, never counted

Before the fix:

- stitch window 16 frames (~0.53 s at 30 fps);
- history pruned at 24 frames (~0.8 s);
- new ID born deep in the truck gave `transit → truck`, which emits no event.

Fix: time-based stitching/history, heading-aware re-matching, origin
reconstruction, and inference decoupled from pruning. A 2 s gap (60 frames at
30 fps) now inherits the old history and counts once. A confirmed line crossing
also allows an inferred event even if the bag vanished before the zone dwell
completed.

### Problem C — a bag was counted `unloaded` while still inside the dock box

`out/bags_custom/run_6/events.csv` had this as its first event:

```csv
1,2,158367,5284.31,,unloaded,0,0.51,0.833,0,1,-1
```

The point `(0.51, 0.833)` is deep inside the dock polygon, and `on_truck` went
to `-1`. The `debug.log` shows how it got there:

```text
f72  new 1 born at the tailgate (truck): provisional dock history
f72  track 1 crossed count line -> loaded
f93  track 1 crossed count line -> unloaded
f129 new 2 adopts lost 1 (root 1, zone dock, gap 34, score 0.21)
f129 EVENT unloaded track 2 root 1
```

Root cause: a bag that is picked up from the dock pile is briefly detected on
the truck side of the line at the tailgate (f72). `_handover_or_new()` gives any
track born there provisional *dock* history, so when the same bag falls back
into the dock (f93) the existing "hand-down" heuristic reads the truck→dock
line crossing as a genuine `unloaded` at f129. A `transit → dock` settle would
not emit, but the provisional `zone = DOCK` plus the adopt-from-lost path did.

The crucial point is that a real hand-down (a bag in the truck being lowered
over the tailgate, first detected at the tailgate) is **observationally
identical** to a dock bag misdetected at the tailgate and then falling back.
No zone, heading, distance or line-crossing rule can tell them apart from the
observations alone. In a loading scene the second reading dominates, so the
hand-down interpretation is now **off by default**.

Fix:

- `CounterConfig.allow_handover_unload` (new, default `false`) gates both
  provisional-unload paths — the observed settle and the inferred-on-loss
  path. With it off, the bag stays provisional and is normalised to a dock bag
  without emitting an event.
- The provisional state is deliberately **kept** (not released early). Releasing
  it made a bag that dips below the tailgate and comes straight back produce a
  false `loaded` on the return trip; that regression is now covered by a test.
- With the heuristic on, a genuine hand-down still counts exactly as before.
- Normal unloads are unaffected: a bag first detected *inside* the truck and
  then carried across the count line into the dock is a settled zone transition
  and never uses the provisional hand-down path.

Enable the legacy behaviour only if your scene genuinely hands bags down over
the tailgate and detection habitually starts at the tailgate:

```json
{ "counter": { "allow_handover_unload": true } }
```

---

## 5. Files changed for this work

| File | Change |
|---|---|
| `src/bagcount/config.py` | `Zones.cross_line`, 30-frame tracker defaults, line-gating knobs, `allow_handover_unload`, JSON parsing/validation. |
| `src/bagcount/counter.py` | Count-line classifier, crossing latches, final `_emit()` gate, inferred-crossing logic, opt-in hand-down unload, earlier ID-switch/history fixes. |
| `src/bagcount/tracker.py` | Predictive matching, centre-distance gate, 30-frame pending hold, no stale pseudo-detections. |
| `src/bagcount/pipeline.py` | Builds the counter with real fps, draws the line, records it in `summary.json`, passes tracker settings, parses `--line`. |
| `src/bagcount/cli.py` | `--line`, `--no-line`, `--tracker-max-age` overrides and validation. |
| `src/bagcount/server.py` | Live API parity (`line`, `no_line`, `tracker_max_age`) and line in recorded-run info. |
| `tests/test_counter.py` | Line-gating tests for both directions, custom line, legacy mode, ID-switch and 2 s re-ID regressions, false tailgate unload and dip-and-return regressions. |
| `tests/test_tracker.py` | 30-frame ID continuity, unrelated-box separation, no stale detections. |
| `tests/test_pipeline_window.py` | `parse_line`, CLI wiring, line in `summary.json`. |
| `tests/test_config_line.py` | Default/config validation for line and tracker settings. |
| `README.md`, `COMMANDS.md` | Operator documentation for the new flags and behaviour. |
| `COUNTING_LOGIC.md` | This change note. |

Earlier in the session the repo was also reorganised (`configs/`,
`scripts/workflow/`, `scripts/utils/`, `scripts/evaluation/`, `scripts/legacy/`),
`mise run data-train -- --data …` was added, and time-window inference
(`--start`, `--duration`, `--end`) was implemented. Those changes are recorded
in `COMMANDS.md` and `README.md`.

---

## 6. Config knobs

```json
{
  "zones": {
    "cross_line": [[0.28, 0.68], [0.78, 0.68]]
  },
  "detector": {
    "tracker_max_age": 30,
    "tracker_max_center_dist": 1.0
  },
  "counter": {
    "require_line_crossing": true,
    "cross_line_tolerance": 0.01,
    "cross_max_frames": 90,
    "allow_handover_unload": false,
    "stitch_seconds": 2.5,
    "history_seconds": 3.0,
    "stitch_switch_radius": 0.15,
    "stitch_heading_radius": 0.6
  }
}
```

| Symptom | Knob to adjust |
|---|---|
| Counts fire before the bag reaches the expected spot | move `cross_line` toward the truck |
| Bag crosses but is not counted | move `cross_line` toward the dock, or raise `cross_max_frames` |
| Line jitter causes flip-flops | raise `cross_line_tolerance` |
| ID changes after a 30-frame dropout | raise `tracker_max_age` / `tracker_max_center_dist` |
| Unrelated bags merged after a gap | lower `tracker_max_center_dist` / `stitch_heading_radius` |
| False `unloaded` for a bag picked up from the dock | leave `allow_handover_unload: false` (default) |
| Genuine hand-downs at the tailgate are missed | set `counter.allow_handover_unload: true` |
| Need legacy zone-only behaviour | set `require_line_crossing: false` or use `--no-line` |

---

## 7. Verification

```
.venv/bin/python -m pytest -q
109 passed, 2 warnings   (unrelated torch.jit deprecations)
```

Regression results:

- ID switch right after a count: `loaded = 1`, second ID inherits root.
- Lost in transition, re-ID 60 frames later: `loaded = 1`.
- Zone-only settle with a custom deep line: no event until the line is crossed.
- Same line, both directions: one `loaded`, one `unloaded`.
- 30-frame detector dropout: original ID reused; unrelated far box is not.
- Tailgate-born bag falling back into the dock: no `unloaded`, no negative
  `on_truck`, and a later genuine pickup can still be counted as `loaded`.
- Bag dipping below the tailgate and returning: neither `unloaded` nor `loaded`.
- Opt-in hand-down mode still counts a genuine hand-down when enabled.

In `out/.../debug.log` look for:

```text
f… new X adopts lost Y (root Z, zone truck, …, counted-switch)
f… track T crossed count line -> loaded
f… EVENT loaded track T root Z inferred=False
f… BLOCK loaded track T root Z (no line crossing)   ← correctly suppressed
```

---

## 8. Known limitations

- `max_lost_frames` (when an inferred event first fires) is still frame-based;
  stitching/history are time-based. A mixed-frame-rate config may need tuning.
- Zone polygons and the default line are camera-specific. Reframing requires a
  new `--line` / `zones.cross_line`.
- A windowed run starts with fresh tracker/counter state. A bag already across
  the line at `--start` has no origin side and is not counted.
- The predictive matcher has no appearance model. Two nearly identical bags
  crossing each other during a long dropout can still be swapped; the counter's
  root de-duplication limits this to at most one event per bag.
