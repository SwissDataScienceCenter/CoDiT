#!/usr/bin/env bash
# Dispatch the baseline evaluations (shipped checkpoints) as Renku CPU jobs,
# one job per model:
#   HYCOM-Parcels, GLORYS-Parcels, Linear Gauss, ConvNet
# Results go to $CODIT_DATA_PATH/results/<dataset>/<split>/<model>/ on the store;
# patches that already have a result file are skipped, so a resubmitted job
# resumes where the previous one stopped.
#
# Usage:
#   bash scripts/dev/renku/dispatch_baselines.sh [--dataset open_ocean] [--split test] \
#       [--ref origin/dev] [--tag MMDD] [--dry-run]
#
# Job names (<=20 chars, Renku limit): bl-<oo|cl>-<model>-<tag>, e.g. bl-oo-hycom-1127

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SUBMIT="$REPO_ROOT/scripts/dev/renku/submit_job.sh"
DATASET="open_ocean"
SPLIT="test"
REF="HEAD"
TAG="$(date +%m%d)"
DRY=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --dry-run) DRY=(--dry-run); shift ;;
    -h | --help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "dispatch_baselines: unknown argument $1" >&2; exit 1 ;;
  esac
done

case "$DATASET" in
  open_ocean) DS_SHORT=oo ;;
  coastline) DS_SHORT=cl ;;
  *) echo "dispatch_baselines: unknown dataset $DATASET" >&2; exit 1 ;;
esac
COMMON=(--dataset "$DATASET" --split "$SPLIT")

submit() {
  local name="$1"; shift
  bash "$SUBMIT" --cpu --ref "$REF" "${DRY[@]+"${DRY[@]}"}" \
    --name "bl-$DS_SHORT-$name-$TAG" "$@"
}

for current in hycom glorys; do
  submit "$current" --extra parcels -- \
    python scripts/dev/baselines/hycom_parcels_eval.py "${COMMON[@]}" --current_type "$current"
done
submit lingauss -- python scripts/user/eval/lingauss_eval.py "${COMMON[@]}"
submit convnet -- python scripts/dev/baselines/convnet_eval.py "${COMMON[@]}"
