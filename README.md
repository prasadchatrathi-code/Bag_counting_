# Bag counting

Counts bags loaded onto and unloaded from a truck from the fixed
"MDP Loading Point top view" CCTV camera, and keeps a cumulative on-truck count.
Built to verify dispatch counts, so every counted event is logged with a
timestamp and an annotated thumbnail for audit.

## Approach

1. **Detect** people and *carried* bags per frame with a fine-tuned RF-DETR Nano
   checkpoint (`runs/detect/bags_full/checkpoint_best_ema.pth`). Stacked bags are
   deliberately not a class.
2. **Track** carried bags with a predictive IoU tracker so each physical bag keeps
   one track id even after a 30-frame detection gap (`src/bagcount/tracker.py`).
3. **Count** on an explicit line across the transition band. The scene has two
   polygons plus a count line (`src/bagcount/config.py`); the yellow line is drawn
   on every annotated frame. A bag track that settles on the dock and then crosses
   the line into the truck is one *loaded* event; crossing back into the dock is one
   *unloaded* event. Zone dwell still gives hysteresis, so a bag shaking at the
   tailgate does not flip-flop, but no event is emitted without a line crossing.
   Three rules in `counter.py` make this robust to tracker
   fragmentation (all tunable in `CounterConfig`, all unit-tested):
   - *Stitching*: a new track id born near a track lost a few frames earlier inherits
     its zone history (ID switch after a detection gap).
   - *Duplicate boxes*: a new id next to a live track copies that track, and fragments
     of one bag share a root id so the bag counts once.
   - *Lost in transit*: a dock track last seen in the band between the zones, heading
     for the truck, that is never seen again is counted as loaded and flagged
     `inferred=1` in the log (and vice versa for unloading). Carried bags seen from
     behind are the detector's weak spot; this rule covers the gap.
   - *No ambiguous hand-down*: a bag first seen at the tailgate on the truck side
     that falls back into the dock is not counted as unloaded — it looks exactly
     like a dock pickup that was briefly misdetected at the tailgate. Genuine
     unloads from inside the truck still count. Scene-specific opt-in:
     `counter.allow_handover_unload: true` (see `COUNTING_LOGIC.md`).
4. **Log** `events.csv`, `summary.json`, one JPEG per event in `events/`,
   optionally `annotated.mp4`, and with `--debug` a `debug.log` of adoptions and
   suppressed events.

## The three loading phases

The 3.5 h recording (camera clock 11:35–15:10, 10 June) has three distinct
regimes, and a detector trained on one does not transfer to the others:

| phase | camera time | how bags move | notes |
|---|---|---|---|
| hand-up | 11:35–~12:40 | bag lifted from the dock pile over the tailgate onto a carrier's head; carrier walks up the empty blue bed facing away | hardest for the detector: bag seen from behind, bright empty bed, crowded dark dock |
| walk-in | ~12:40–~14:30 | carrier picks a bag off the tarp and walks it in through the tailgate | easiest |
| hand-pass | ~14:30–15:10 | stack above the side walls; bags handed up to a worker on top | short crossing, heavy occlusion |

## Results so far

Hand-counted windows in `data/truth/` (tolerance ±5 s), detector `bags_full`
(RF-DETR Nano fine-tuned on 2,528 pseudo-labelled frames: Grounding DINO for the
walk-in and hand-pass phases, OWLv2 + motion filter for the hand-up phase, plus
one self-labelling round):

| window | phase | true events | predicted | precision | recall |
|---|---|---|---|---|---|
| mid clip, 13:00:00–13:03:00 | walk-in | 19 loaded | 19 loaded | 1.00 | 1.00 |
| late clip, 14:55:00–14:57:00 | hand-pass | 7 loaded | 7 loaded | 1.00 | 1.00 |
| early clip, 11:40:00–11:42:00 | hand-up | 9 loaded | 8 loaded | 1.00 | 0.89 |
| last clip, 15:05:21–15:08:21 | stack-top, bags taken back off | 2 loaded, 7 unloaded | 1 loaded, 4 unloaded | 1.00 | 0.56 |

No window produces a false event. The last three minutes of the recording are
the only place bags come *off* the truck: the worker on top lifts bags from the
top layer and lowers them to carriers on the dock. See "stack-top" below.

The hand-up phase went from 1/9 with `bags_v3` to 8/9. Half of that came from
the detector (OWLv2 labels made it see bags on carriers' heads) and half from
counter rules found by tracing the misses frame by frame:

- **Handover.** When a bag is lifted off the pile, the pile's track usually
  stays on the next bag beneath and the lifted bag's own track starts inside the
  truck zone at the tailgate. A track born in the transit band or within
  `handover_edge_dist` of the tailgate gets provisional dock history and is
  counted once it has moved `handover_min_travel` further from the tailgate
  (so a static bag re-detected on the bed, or a worker's head detected as a bag
  at the tailgate, never counts).
- **Carrier boxes.** The detector sometimes boxes bag + carrier as a second
  "bag" with the same top edge and width as the tight bag box. Such pairs share
  one root, checked every frame, so they count once.
- **Stale fragments.** A lost duplicate is never inherited while its bag is
  still tracked by another fragment or has already been counted, and an id the
  tracker revives on the next bag at the pickup spot starts fresh. Without
  these, the second of two bags picked from the same spot a few seconds apart
  was dropped as a repeat.
- **Reverse handover (unloading).** A track born inside the truck zone at the
  tailgate keeps its origin open: if it travels `handover_min_travel` into the
  dock and stays there (`handover_unload_frames`, or is lost there) it was
  handed *down* and counts as unloaded. A bag box that dips below the tailgate
  line for a moment and comes back is neither an unload nor a reload.

**Stack-top phase (last 10 minutes).** With the stack a full layer above the
side walls, bags are detected on carriers' heads but vanish against the white
stack, carriers queue at the tailgate with bags on their heads for 5–10 s, and
the ground-plane zones no longer match the image: a bag held overhead on the
dock projects into the truck polygon. Three of the seven hand-downs are merged
into one track chain at the queue spot and lost. The next steps are labelling
frames from 15:00–15:10 so the detector keeps the bag in front of the stack,
and making the tailgate line follow the stack height.

The remaining miss (81.5 s) is two bags handed over 4 s apart at the same spot
where the second bag's box is absorbed into the first bag's track; the next
lever there is the tracker rather than the detector.

Throughput on an M4 Pro at 8 fps / 1280 px: about 4× real time when the GPU is
not shared.

The pipeline reads frames sequentially from any OpenCV source, so the same code
path will serve the eventual live RTSP deployment.

## Setup

```sh
mise install          # python + uv pinned in mise.toml
mise run install      # uv sync: .venv + rfdetr[onnx,train]
mise run verify       # check tools, imports, datasets and model artifacts
mise run test         # unit tests
```

See `COMMANDS.md` for the full four-phase command reference.

## Workflow

```sh
VIDEO=assets/bags-loading-clip-5min.mp4

# 1. Optional: downsampled clip (8 fps, 1280 px)
mise run clip -- 0 300 myclip --source "$VIDEO"

# 2. Bootstrap labels with Grounding DINO, verify count/balance, fine-tune RF-DETR Nano
mise run label -- "$VIDEO" datasets/bags_full \
  --every 10 --limit 200 --val-every 5 --device cpu --require-person --motion
mise run check-labels -- datasets/bags_full train
mise run review-all -- datasets/bags_full out/review_full --per-sheet 12
mise run check-balance -- datasets/bags_full --class bag
mise run train -- --data datasets/bags_full/data.yaml \
  --epochs 40 --batch 4 --device cpu --name bags_full

# 3. Optional: export to ONNX
mise run export -- --weights runs/detect/bags_full/checkpoint_best_ema.pth \
  --out exports/bags_full_onnx --device cpu

# 4. Count (full video by default; native checkpoint or ONNX)
mise run count -- "$VIDEO" --out out/full \
  --weights runs/detect/bags_full/checkpoint_best_ema.pth --device cpu --video --debug
mise run count -- "$VIDEO" --out out/full_onnx \
  --weights exports/bags_full_onnx/rfdetr-nano.onnx --backend onnx --device cpu --video

# 5. Score against a hand-counted window (make sheets with `mise run sheets`)
mise run eval -- out/full/events.csv data/truth/<clip>_first180s.csv --tolerance 5

# 6. Demo / operator UI
mise run serve        # http://127.0.0.1:8008
```

## Inference (native checkpoint or ONNX)

Inference supports both a native RF-DETR checkpoint (`.pt` / `.pth`, loaded with
`RFDETR.from_checkpoint`) and an exported ONNX artifact (`.onnx` file or its
export directory, loaded with `RFDETRInference`). `--backend auto` (the default)
picks from the `--weights` path; use `--backend checkpoint` or `--backend onnx`
to force one. By default the whole video is processed. To count only a segment,
use `--start TIME` with `--duration TIME` (or `--end TIME`); `TIME` accepts
seconds, compound durations (`2m`, `1hr26m30`) or `MM:SS`. Event timestamps stay source-relative,
so a window can be compared directly with full runs. `--max-frames` remains an
optional frame-based smoke-test limit and cannot be combined with `--duration`.

The count line is derived automatically halfway between the truck tailgate and
the dock edge, and is recorded in `summary.json` (`counting_line`). Override it
with `--line X1,Y1,X2,Y2` using normalised 0..1 coordinates when the camera view
changes. `--tracker-max-age FRAMES` (default `30`) controls how long a missing
bag keeps its id; `--no-line` falls back to legacy zone-only counting for A/B
comparison. All three options are also available in the live API and in JSON
configs.

```sh
# Full video with the default checkpoint
mise run count -- assets/bags-loading-clip-5min.mp4 --out out/full --device cpu --video

# Explicit native checkpoint
mise run count -- assets/bags-loading-clip-5min.mp4 --out out/full2 \
  --weights runs/detect/bags_full/checkpoint_best_ema.pth --device cpu --video

# Export that checkpoint, then run the same pipeline on ONNX
mise run export -- --weights runs/detect/bags_full/checkpoint_best_ema.pth \
                    --out exports/bags_full_onnx --device cpu
mise run count -- assets/bags-loading-clip-5min.mp4 --out out/full_onnx \
  --weights exports/bags_full_onnx/rfdetr-nano.onnx --backend onnx --device cpu --video

# Only minutes 2-4 of a longer source (no clip extraction needed)
mise run count -- assets/bags-loading-clip-6min.mp4 --out out/window_2to4 \
  --weights exports/bags_custom_onnx/rfdetr-nano.onnx --backend onnx --device cpu \
  --start 2m --duration 2m

# Use an explicit count line for a reframed camera
mise run count -- assets/bags-loading-clip-5min.mp4 --out out/custom_line \
  --line 0.28,0.68,0.78,0.68 --tracker-max-age 30 --video --debug
```

Ctrl+C (`SIGINT`) or `SIGTERM` stops after the current frame: the capture,
video writer and event CSV are closed and `summary.json` is still written.
The web UI's **Stop** button does the same for live runs and joins the worker
thread before returning.

See `COMMANDS.md` for an example of every command and flag.

## Demo UI

`mise run serve` starts a small FastAPI app (`src/bagcount/server.py`, page in
`src/bagcount/static/index.html`) with two views:

- **Recorded runs**: every `out/<run>/` with a `summary.json`. Plays the annotated
  video, lists the event log with the per-event snapshot and camera clock, and the
  counters follow the playhead. Where a hand count exists for the clip
  (`data/truth/<clip>_first<N>s.csv`) it shows matched / extra / missed events.
  Clicking an event seeks the video; clicking a snapshot enlarges it.
- **Live**: starts the same `Pipeline` on a clip or an `rtsp://` URL in a background
  thread and streams the annotated frames (MJPEG) and the event feed (server-sent
  events) as they are produced. Output lands in `out/live_<timestamp>/` and then
  appears under Recorded runs. This is the path that will be used on site.

`configs/clips.json` records the camera clock at frame 0 of each clip so runs can show
camera time. The runs used for the demo (`out/demo_*`) were made with:

```sh
mise run count -- clips/mid_1300.mp4 --out out/demo_mid_1300 --video --wall-start 2026-06-10T12:59:59
```

Data preparation and model training are available as `mise run data-prepare`,
`data-review`, `data-split`, `data-augment`, and `data-train` commands. See
`COMMANDS.md` for the complete setup-to-inference sequence. The web UI is for
recorded runs and live counting only.

## Layout

- `src/bagcount/config.py`  zones, detector and counter settings (JSON-overridable)
- `src/bagcount/counter.py` pure zone-transition counting logic, unit-tested
- `src/bagcount/pipeline.py` detect, track, count, log
- `src/bagcount/cli.py`     entry point (`mise run count`)
- `src/bagcount/server.py`  demo / operator web UI (`mise run serve`), static page in `src/bagcount/static/`
- `src/bagcount/evaluate.py` match predicted events against hand-counted truth
- `configs/`                augmentation, clip-clock and tracker configs
- `data/truth/`             hand-counted windows; `configs/clips.json` camera clock per clip
- `scripts/workflow/`       bag-only preparation, review, split, augment and train CLI
- `scripts/utils/`          clip extraction, sheets, ONNX export, setup verification
- `scripts/evaluation/`     multi-clip counter evaluation
- `scripts/legacy/`         earlier two-class (person + bag) labelling and training tools
- `assets/`                 source video and screenshots (video is git-ignored)
