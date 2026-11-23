# Bag Counting commands

All commands below run from the repository root. `mise.toml` pins Python 3.12.16 and uv 0.13.0. The recommended data/model path is the bag-only `mise run data-*` workflow; the existing counter and its web viewer remain available.

Every command is exposed as a `mise` task, so you do not need to remember where a script lives. The layout is:

```text
configs/             augmentation, clip-clock and tracker configs
data/truth/          hand-counted evaluation windows
scripts/workflow/    bag-only data preparation and training CLI
scripts/utils/       clip, sheets, ONNX export, verification utilities
scripts/evaluation/  multi-clip counter evaluation
scripts/legacy/      earlier two-class labelling/training tools
src/bagcount/        runtime package (pipeline, counter, server, workflow)
datasets/ out/ runs/ generated data and artifacts (left in place)
```

## 1. Install and verify

```bash
mise install
mise run install       # uv sync; installs this project and its dependencies
mise run verify
```

If `python -m bagcount.server` reports `ModuleNotFoundError`, run `mise run install` from this directory, then retry. Unlabeled images/videos use Grounding DINO and need internet access the first time so its model weights can download. A ZIP YOLO dataset with a `bag` or `sack` class can supply its existing labels.

## Quickstart: one new video → bag training → counting

This is the complete end-to-end path for a single new video: make bag-only labels, train RF-DETR, run inference/counting, and score the events. Replace `VIDEO` with your input path and `NAME` with a short tag.

```bash
VIDEO=/path/to/your/input.mp4
NAME=myvideo
```

### Q1. Install and verify (once)

```bash
mise install
mise run install
mise run verify
mise run test
```

### Q2. Smoke-test labels before the full video

CPU Grounding DINO is slow, so check a small sample first.

```bash
mise run data-prepare -- "$VIDEO" \
  --fps 1 --confidence 0.30 --device cpu \
  --max-samples 20

mise run data-review -- --preview out/review/${NAME}_smoke.jpg --limit 40 --cols 5
```

`data-prepare` shows a live progress bar with sampled frames, saved bag images, skipped frames, bag boxes, elapsed time and ETA. Open `out/review/${NAME}_smoke.jpg` and confirm the boxes are on bags before continuing.

### Q3. Prepare the full bag-only dataset

```bash
mise run data-prepare -- "$VIDEO" \
  --fps 1 --confidence 0.30 --device cpu
```

Useful alternatives:

```bash
# Sample every 30th source frame instead of 1 fps
mise run data-prepare -- "$VIDEO" --interval 30 --device cpu

# Only a time window: start at 120 s, run for 300 s
mise run data-prepare -- "$VIDEO" --start 120 --duration 300 --interval 20 --device cpu
```

This writes `datasets/workflow/dataset/images/`, `datasets/workflow/dataset/labels/`, and `datasets/workflow/dataset/data.yaml` with one class: `0 = bag`.

### Q4. Review, split, augment

```bash
mise run data-review -- --preview out/review/${NAME}_full.jpg --limit 60 --per-sheet 9
mise run data-split

cp configs/augmentations.example.json configs/augmentations.json   # edit if needed
mise run data-augment -- --config configs/augmentations.json \
  --preview out/review/augmentation.jpg
```

`--per-sheet 9` writes multiple numbered sheets (`${NAME}_full_001.jpg`, `${NAME}_full_002.jpg`, …) with 9 annotated images each in a 3×3 grid. Skip the two augmentation lines if you do not want augmentation. Splitting must happen before augmentation.

### Q5. Train RF-DETR

```bash
mise run data-train -- \
  --variant nano --epochs 40 --resolution 384 \
  --batch 4 --grad-accum 4 --lr 0.00002 \
  --device cpu --name bags_${NAME}
```

By default this trains from `datasets/workflow/dataset/data.yaml` (the workflow
workspace selected by `--root`). Use `--data` to state the config explicitly —
for example, to train from a prepared dataset elsewhere:

```bash
mise run data-train -- --data datasets/workflow/dataset/data.yaml --name bags_${NAME}
```

A supplied `data.yaml` is read as-is and is never rewritten; without `--data`
the workflow writes its own single-class config as before. The chosen path is
recorded as `data_yaml` in the run's `training` state.

The best checkpoint is:

```text
runs/detect/bags_${NAME}/checkpoint_best_ema.pth
```

### Q6. Optional ONNX export

```bash
mise run export -- \
  --weights runs/detect/bags_${NAME}/checkpoint_best_ema.pth \
  --out exports/bags_${NAME}_onnx \
  --resolution 384 --device cpu
```

### Q7. Counting smoke test, then the full run

```bash
# Quick check on 80 frames
mise run count -- "$VIDEO" \
  --out out/${NAME}_smoke \
  --weights runs/detect/bags_${NAME}/checkpoint_best_ema.pth \
  --backend checkpoint --device cpu --video --debug \
  --max-frames 80

# Full video
mise run count -- "$VIDEO" \
  --out out/${NAME}_count \
  --weights runs/detect/bags_${NAME}/checkpoint_best_ema.pth \
  --backend checkpoint --device cpu --video --debug
```

Add `--wall-start 2026-06-10T11:35:00` if you know the camera clock at frame 0. To run only a segment of the source, add `--start 2m --duration 2m` (see §7); a windowed run still reports source-relative event times.

Outputs under `out/${NAME}_count/`:

```text
events.csv
summary.json
events/          # one JPEG per counted event
annotated.mp4
debug.log
```

ONNX alternative:

```bash
mise run count -- "$VIDEO" \
  --out out/${NAME}_count_onnx \
  --weights exports/bags_${NAME}_onnx/rfdetr-nano.onnx \
  --backend onnx --device cpu --video
```

### Q8. Evaluate against a hand count

```bash
mise run sheets -- "$VIDEO" out/sheets_${NAME} --start 0 --duration 180
```

Hand-count the sheets and create `data/truth/${NAME}_first180s.csv`:

```csv
time_s,direction
11.0,loaded
12.0,loaded
45.5,unloaded
```

Then score the run:

```bash
mise run eval -- \
  out/${NAME}_count/events.csv \
  data/truth/${NAME}_first180s.csv \
  --tolerance 5
```

### Q9. View results in the UI

```bash
mise run serve
```

Open <http://127.0.0.1:8008>.

If your GPU is configured, replace `--device cpu` with `--device cuda` in the train, export and count commands.

## 2. Prepare bag-only YOLO data

The commands accept image files, full videos, or ZIP archives. Video reading is sequential and does not load the complete video into memory. Only sampled frames with at least one bag box are saved. Labels use one class (`0 = bag`). A labeled ZIP is filtered to that class; otherwise the bag-only detector creates labels.

### Full video or image inputs

```bash
mise run data-prepare -- path/to/video.mp4 --fps 1 --confidence 0.30 --device cpu
mise run data-prepare -- path/to/video.mp4 another/video.mov --fps 0.5 --confidence 0.25
mise run data-prepare -- path/to/photo.jpg path/to/image_dataset.zip --device cpu
```

`--fps` samples at the requested rate. To sample every Nth source frame instead, use `--interval N` (for example, `--interval 30`). By default the whole video is read. Use `--start SECONDS` and `--duration SECONDS` to select a time window; duration `0` means through the end. `--max-samples N` is an optional short smoke run. `--extract-clip` also writes a sampled MP4 under `datasets/workflow/clips/`; it is off by default.

```bash
mise run data-prepare -- path/to/video.mp4 \
  --start 120 --duration 300 --interval 20 --confidence 0.30 --device cpu
mise run data-prepare -- path/to/video.mp4 --fps 1 --max-samples 20
mise run data-prepare -- path/to/video.mp4 --fps 1 --extract-clip --clip-fps 8
```

A terminal progress bar shows sampled frames, saved bag images, skipped empty frames, bag boxes, elapsed time and ETA. Generated images and labels are stored in `datasets/workflow/dataset/images/` and `datasets/workflow/dataset/labels/`. Re-running `prepare` starts a fresh generated dataset from the registered inputs.

### Inspect labels and dataset balance

```bash
mise run data-review
mise run data-review -- --preview out/review/bag_dataset.jpg --limit 40 --cols 5

# Multiple sheets of 9 images each (3x3 grid)
mise run data-review -- --preview out/review/bag_dataset.jpg --limit 60 --per-sheet 9
```

This prints image counts, positive-image counts, bag-box counts, and boxes-per-image distribution. Without `--per-sheet`, it writes one annotated contact sheet at `out/workflow-review.jpg` by default. With `--per-sheet 9`, it writes `out/review/bag_dataset_001.jpg`, `bag_dataset_002.jpg`, and so on, each with 9 images. `--cols` overrides the automatically chosen near-square grid.

## 3. Split data without leakage

Split after preparing and reviewing. Splitting happens before augmentation so near-identical augmented copies cannot appear in validation. Only two splits are created: `train` and `val`.

```bash
mise run data-split
mise run data-split -- --train-ratio 0.80 --val-ratio 0.20 --seed 42
```

Both ratios must be positive and sum to 1. Each source image is assigned to exactly one split under `datasets/workflow/dataset/{images,labels}/{train,val}`. Held-out training metrics are computed on `val`.

## 4. Configure and apply training-only augmentation

Create an augmentation JSON file. Each selected transform has its own probability (`p`) and parameters. Boxes use Albumentations YOLO support; only the training split is augmented.

```json
{
  "seed": 42,
  "selected": [
    {"name": "horizontal_flip", "p": 0.5, "params": {}},
    {"name": "rotation", "p": 0.4, "params": {"limit": 12}},
    {"name": "brightness", "p": 0.25, "params": {"limit": 0.15}}
  ]
}
```

Use the checked-in example or copy it to customize: `configs/augmentations.example.json`.

```sh
cp configs/augmentations.example.json configs/augmentations.json
```

Then run:

```bash
mise run data-augment -- --config configs/augmentations.json \
  --preview out/review/augmentation.jpg
```

The command writes one contact sheet at `out/review/augmentation.jpg` containing the first three augmented training images, so you can quickly check the transforms. Omit `--preview` to use that path by default.

Run this from the repository root. `configs/augmentations.json` must exist; create it from `configs/augmentations.example.json` as shown above.

Available names: `horizontal_flip`, `vertical_flip`, `rotation`, `shift`, `scale`, `shift_scale_rotate`, `shear`, `random_crop`, `resize`, `perspective`, `affine`, `brightness`, `contrast`, `saturation`, `saturation_hue`, `hue`, `gamma`, `rgb_shift`, `color_jitter`, `gaussian_blur`, `motion_blur`, `median_blur`, `gaussian_noise`, `iso_noise`, `clahe`, `grayscale`, `channel_shuffle`, `coarse_dropout`, `random_shadow`, and `random_fog`. Parameter keys are shown in the previous examples/defaults in `src/bagcount/workflow.py`. Use an empty `selected` list to skip augmentation.

## 5. Train RF-DETR and evaluate

```bash
mise run data-train -- --variant nano --epochs 40 --resolution 384 \
  --batch 4 --grad-accum 4 --lr 0.00002 --device cpu --name bags_workflow
```

Available variants are `nano`, `small`, `medium`, and `large`. Resolution should be divisible by 32. To resume from a checkpoint, add `--checkpoint path/to/checkpoint.pth`. The command writes checkpoints under `runs/detect/<name>/` and reports precision, recall, F1, and AP50 on the held-out validation split. `--data path/to/data.yaml` selects the dataset config explicitly (default: `<root>/dataset/data.yaml`); a supplied file is not modified, and validation metrics are computed from that dataset's `images/val` and `labels/val` using the `bag` class id declared in its `names`.

## 6. Optional ONNX export

```bash
mise run export -- --weights runs/detect/bags_workflow/checkpoint_best_ema.pth \
  --out exports/bags_workflow_onnx --resolution 384 --device cpu
```

## 7. Run bag-counting inference

The inference pipeline accepts either a video path or a stream URL. By default it processes the whole source. Use `--start TIME` and `--duration TIME` (or `--end TIME`) to count only a segment, or `--max-frames N` for a short frame-based check. `TIME` accepts seconds, compound durations (`90s`, `2m`, `1hr26m30`, `1h 26m 30s`) or a clock string (`MM:SS` / `HH:MM:SS`).

### Native RF-DETR checkpoint

```bash
mise run count -- path/to/video.mp4 --out out/bags_native \
  --weights runs/detect/bags_workflow/checkpoint_best_ema.pth \
  --backend checkpoint --device cpu --video --debug
```

### ONNX model

```bash
mise run count -- path/to/video.mp4 --out out/bags_onnx \
  --weights exports/bags_workflow_onnx/rfdetr-nano.onnx \
  --backend onnx --device cpu --video
```

Outputs include `events.csv`, `summary.json` (model/settings, video and run statistics, timing summaries, the resolved `counting_line`), `frame_times.csv` (per-frame inference and processing milliseconds), event thumbnails under `events/`, and optional `annotated.mp4` / `debug.log` under the chosen output folder. Useful options include `--conf 0.25`, `--config configs/custom.json`, `--wall-start 2026-06-10T11:35:00`, and `--max-frames 80` for a short check.

### Count line and tracker continuity

A yellow **COUNT LINE** is drawn on every annotated frame. A bag is counted only
when it crosses that line: dock → truck is `loaded`, truck → dock is `unloaded`.
Zone dwell still filters flicker, but a bag that only enters a polygon without
crossing the line is not counted. The line is derived from the zone polygons
unless you set `--line X1,Y1,X2,Y2` (normalised 0..1 coordinates, e.g.
`0.28,0.68,0.78,0.68`) or `zones.cross_line` in a JSON config. The resolved
coordinates and side sign are written to `summary.json`.

Trackers hold a missing bag id for 30 frames by default, using a short
constant-velocity prediction for re-matching; predictions are never reported as
detections to the counter. Change this with `--tracker-max-age N` (or
`detector.tracker_max_age` in config). Use `--no-line` only for legacy
zone-only A/B comparison; it disables the crossing gate.

A bag first seen at the tailgate on the truck side and then falling back into
the dock is **not** counted as `unloaded` by default: that trajectory is
indistinguishable from a dock pickup that was briefly misdetected at the
tailgate. Normal unloads — first seen inside the truck, then carried across the
line — are unaffected. If your scene really does hand bags down over the
tailgate and detection starts at the tailgate, opt in with
`counter.allow_handover_unload: true` in a `--config` JSON file. See
`COUNTING_LOGIC.md` for the full root-cause analysis.

### Count only part of a video

For the six-minute asset, run just minutes 2–4 (120 s → 240 s) without extracting a clip first:

```bash
mise run count -- assets/bags-loading-clip-6min.mp4 --out out/bags_custom \
  --weights exports/bags_custom_onnx/rfdetr-nano.onnx \
  --backend onnx --device cpu --video --debug \
  --start 2m --duration 2m --conf-series 0.5
```

`--end` is the alternative when you know the absolute stop time: `--start 00:02:00 --end 00:04:00`. Omitting `--duration`/`--end` runs from `--start` to the end of the file. Event `time_s`/`clock` values stay relative to the original video (not the window), `frame_times.csv` uses source frame numbers, and `summary.json` records `start_s`, `duration_s` and the `window` frame range. `--max-frames` and `--duration` are mutually exclusive; `--start` may be combined with either.

Because `--conf-series` is used above, results go into numbered folders such as `out/bags_custom/run_1/`; use plain `--conf` instead to write straight to `out/bags_custom/`.

To compare several confidence thresholds in one invocation, pass `--conf-series` followed by the values. Each run gets a numbered folder under `--out`, and the terminal shows frame progress:

```bash
mise run count -- path/to/video.mp4 --project out --name bags_custom \
  --weights runs/detect/bags_custom/checkpoint_best_ema.pth \
  --backend checkpoint --device cpu --video --debug \
  --conf-series 0.25 0.35 0.50 0.65
```

`--project out --name bags_custom` writes inside `out/bags_custom/`. Confidence-series results go into new numbered folders there, in threshold order. Repeating the command continues with the next unused number, preserving previous run folders. `--out out/bags_custom` remains supported as an alternative for existing commands.

## 8. Evaluate bag-counting events against hand counts

```bash
mise run eval -- out/bags_native/events.csv data/truth/mid_1300_first180s.csv --tolerance 5
mise run eval-all -- runs/detect/bags_workflow/checkpoint_best_ema.pth bags_workflow
```

`mise run sheets -- VIDEO OUTPUT_DIR --start 0 --duration 180` creates contact sheets for manual event counting.

## 9. Existing demo UI

```bash
mise run serve
```

Open <http://127.0.0.1:8008>. The UI contains **Recorded runs** and **Live** views; data preparation and training are run through the `mise run data-*` CLI commands above. It reads runs from `out/` and camera-clock metadata from `configs/clips.json`; those paths are resolved from the package root, so the task works from any directory.

## Command reference

| Command | Purpose |
|---|---|
| `mise run install` | Install/sync project dependencies |
| `mise run verify` / `mise run check` | Verify tools, imports, and model artifacts |
| `mise run data-prepare` | Bag-only image/video/ZIP preparation and YOLO labels |
| `mise run data-review` | Dataset statistics and annotated contact sheet(s); `--per-sheet N` writes numbered sheets |
| `mise run data-split` | Reproducible train/validation split |
| `mise run data-augment` | Albumentations on training split only; writes a 3-image preview sheet with `--preview` |
| `mise run data` | Bag-only workflow CLI (all subcommands in one entry point) |
| `mise run data-train` | RF-DETR training and held-out validation metrics; `--data` sets an explicit `data.yaml` |
| `mise run clip` | Create a downsampled clip with the general clip utility |
| `mise run sheets` | Build time-stamped contact sheets for manual event counting |
| `mise run label` | Legacy Grounding DINO labeler (person + bag classes) |
| `mise run merge` | Merge multiple YOLO label sources (legacy) |
| `mise run propagate` | Propagate labels with a checkpoint (legacy) |
| `mise run review` | Legacy YOLO dataset contact sheets (one split) |
| `mise run review-all` | Legacy YOLO dataset contact sheets (train + val) |
| `mise run check-labels` | Legacy labelled-box counts |
| `mise run check-balance` | Legacy label-count/balance gate before training |
| `mise run train` | Legacy training entry point for the two-class dataset |
| `mise run export` | Export RF-DETR checkpoint to ONNX |
| `mise run count` | Count bags in video or stream; `--start`/`--duration`/`--end` select a window, `--line`/`--tracker-max-age` control line gating and id continuity |
| `mise run eval` | Score one event log against a hand count |
| `mise run eval-all` | Score all three hand-counted clips with one checkpoint |
| `mise run serve` | Start recorded/live operator UI |
| `mise run test` | Run unit tests |

The `data-*` commands share `datasets/workflow/` as their workspace, so run prepare → review → split → augment (optional) → train in that order.
