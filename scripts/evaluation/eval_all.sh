#!/usr/bin/env bash
# Run the counter on every hand-counted window and score it.
# Usage: scripts/evaluation/eval_all.sh [weights] [out_tag] [extra cli args...]
set -euo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,3p' "$0"
  exit 0
fi
W=${1:-runs/detect/bags_full/checkpoint_best_ema.pth}
TAG=${2:-eval}
shift $(( $# < 2 ? $# : 2 ))
EXTRA=("$@")
for f in "$W" \
  clips/early_1140.mp4 clips/mid_1300.mp4 clips/late_1520.mp4 \
  data/truth/early_1140_first120s.csv data/truth/mid_1300_first180s.csv \
  data/truth/late_1520_first120s.csv; do
  if [[ ! -f "$f" ]]; then
    echo "missing $f — train a checkpoint and generate the clips first" >&2
    exit 1
  fi
done
run() {  # clip truth max_frames window_hi
  local clip=$1 truth=$2 frames=$3 hi=$4
  local name; name=$(basename "$clip" .mp4)
  rm -rf "out/${TAG}_${name}"
  python -m bagcount.cli "clips/$clip" --out "out/${TAG}_${name}" --weights "$W" --max-frames "$frames" --debug ${EXTRA[@]+"${EXTRA[@]}"} >/dev/null 2>&1
  echo "== $name ($truth)"
  python -m bagcount.evaluate "out/${TAG}_${name}/events.csv" "data/truth/$truth" --tolerance 5 --window 0 "$hi"
}
run early_1140.mp4 early_1140_first120s.csv 960 120
run mid_1300.mp4   mid_1300_first180s.csv  1440 180
run late_1520.mp4  late_1520_first120s.csv  960 120
