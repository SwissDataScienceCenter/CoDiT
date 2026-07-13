#!/bin/bash
# Usage: ./fig4_summarize.sh <dataset> [split=test] [subdir=fig4_repro]
#
# Aggregates evaluation results across models for a given dataset,
# writing fig4_summary.csv to data/results/<dataset>/<split>/<subdir>/.

# set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <dataset> [split=test] [subdir=fig4_repro]" >&2
  exit 1
fi

dataset=$1
split=${2:-test}
subdir=${3:-fig4_repro}

source ~/.bashrc
micromamba activate codit
cd /mydata/adopt/christian/CoDiT/my_scripts/model_fitting

python summarize_fig4.py --dataset "$dataset" --split "$split" \
  --result_subdir "$subdir"
