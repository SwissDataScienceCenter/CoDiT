#!/usr/bin/env bash
set -euo pipefail

# Reproduce CoDiT paper results (Table 1).
# Prerequisites: environment set up, data downloaded via scripts/download_data.py.
# Usage: bash scripts/reproduce_results.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
EVAL_DIR="$REPO_ROOT/my_scripts/model_fitting"

echo "=== Evaluating Linear Gaussian — open_ocean ==="
python "$EVAL_DIR/lingauss_eval.py" \
    --dataset open_ocean --split test --evaluate_subset

echo ""
echo "=== Evaluating Linear Gaussian — coastline ==="
python "$EVAL_DIR/lingauss_eval.py" \
    --dataset coastline --split test --evaluate_subset

echo ""
echo "=== Evaluating CoDiT (GRU) — open_ocean ==="
python "$EVAL_DIR/codit_eval.py" \
    --dataset open_ocean --split test --model_type GRU \
    --model_name codit_GRU_open_ocean --evaluate_subset

echo ""
echo "=== Evaluating CoDiT (GRU) — coastline ==="
python "$EVAL_DIR/codit_eval.py" \
    --dataset coastline --split test --model_type GRU \
    --model_name codit_GRU_coastline --evaluate_subset

echo ""
echo "=== Evaluating CoDiT (Transformer) — open_ocean ==="
python "$EVAL_DIR/codit_eval.py" \
    --dataset open_ocean --split test --model_type Transformer \
    --model_name codit_Transformer_open_ocean --evaluate_subset

echo ""
echo "=== Evaluating CoDiT (Transformer) — coastline ==="
python "$EVAL_DIR/codit_eval.py" \
    --dataset coastline --split test --model_type Transformer \
    --model_name codit_Transformer_coastline --evaluate_subset

echo ""
echo "=== All evaluations complete ==="
echo "Results saved under: \$CODIT_DATA_PATH/results/ (or data/results/)"
