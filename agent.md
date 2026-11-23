# Change record

- Added an 11-step data preparation and RF-DETR training workflow to the existing FastAPI UI.
- Added streamed image/video/ZIP uploads, optional clip output, sequential FPS/frame-interval sampling, bag-only YOLO label generation, empty-frame filtering, and dataset previews/statistics.
- Added Albumentations configuration with YOLO bounding-box transforms, reproducible train/validation/test splitting, and train-only augmentation.
- Added RF-DETR training and held-out test precision, recall, F1, and AP50 reporting.
- Added `albumentations` and `python-multipart` dependencies and refreshed `uv.lock`.
- Python syntax compilation passed. Tests and model training were not run.
- Fixed workflow polling so it updates status without rebuilding forms and clearing selected files; added file selection feedback, upload progress, and visible upload errors. Removed duplicate dataset-stat updates.
- Removed data/training controls and APIs from the web UI, added `mise run data-*` preparation/review/split/augment/train commands, documented setup through inference in `COMMANDS.md`, and added an augmentation JSON example under `configs/`.
