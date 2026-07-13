#!/bin/bash
# Usage: ./fig4_open_ocean.sh <gru|tf|lingauss> [split=test] [subdir=fig4_repro]
#
# Regenerates one row of Table 2 for the open_ocean dataset from a shipped
# trained checkpoint, writing .npz files to
#   data/results/open_ocean/<split>/<subdir>/<model_dir>/
# so existing data/results/open_ocean/<split>/<model_dir>/ is untouched.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <gru|tf|lingauss> [split=test] [subdir=fig4_repro]" >&2
  exit 1
fi

model=$1
split=${2:-test}
subdir=${3:-fig4_repro}

source ~/.bashrc
micromamba activate codit
cd /mydata/adopt/christian/CoDiT/my_scripts/model_fitting

input_vars="hycom_current copernicus_current era5_wind"

case "$model" in
  gru)
    python codit_eval.py --dataset open_ocean --split "$split" \
      --model_type GRU --context_encoder_type conv --optimizer adam --K 100 \
      --input_vars $input_vars --result_subdir "$subdir"
    ;;
  tf)
    python codit_eval.py --dataset open_ocean --split "$split" \
      --model_type Transformer --context_encoder_type conv --optimizer adam --K 100 \
      --input_vars $input_vars --result_subdir "$subdir"
    ;;
  lingauss)
    python lingauss_eval.py --dataset open_ocean --split "$split" \
      --result_subdir "$subdir"
    ;;
  *)
    echo "Unknown model: $model (expected: gru, tf, lingauss)" >&2
    exit 1
    ;;
esac
