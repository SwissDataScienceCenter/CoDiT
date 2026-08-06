#!/usr/bin/env bash
set -euo pipefail

# Reproduce CoDiT paper results (Table 1).
# Prerequisites: environment set up, data downloaded via scripts/user/setup/download_data.py.
#
# Usage:
#   bash scripts/user/setup/reproduce_results.sh          # quick subset eval (default)
#   bash scripts/user/setup/reproduce_results.sh --full   # full test-set eval
#
# By default this evaluates a subset of the test set — enough to sanity-check
# the pipeline in a few minutes on a single GPU. To reproduce the exact numbers
# from Table 1 of the paper, pass --full. Warning: full evaluation over the
# entire test set can take many hours to a few days depending on hardware
# (single-GPU CoDiT evaluation is the dominant cost).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
EVAL_DIR="$REPO_ROOT/scripts/user/eval"

FULL=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL=1 ;;
        -h|--help)
            sed -n '4,13p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--full]" >&2
            exit 1
            ;;
    esac
done

if [[ "$FULL" == "1" ]]; then
    SUBSET_FLAG=""
    echo "=== Running FULL test-set evaluation (this can take hours to days) ==="
else
    SUBSET_FLAG="--evaluate_subset"
    echo "=== Running SUBSET evaluation (pass --full to reproduce full Table 1 numbers) ==="
fi
echo ""

echo "=== Evaluating Linear Gaussian — coastline ==="
python "$EVAL_DIR/lingauss_eval.py" \
    --dataset coastline --split test $SUBSET_FLAG

echo ""
echo "=== Evaluating Linear Gaussian — open_ocean ==="
python "$EVAL_DIR/lingauss_eval.py" \
    --dataset open_ocean --split test $SUBSET_FLAG

echo ""
echo "=== Evaluating CoDiT (GRU) — coastline ==="
python "$EVAL_DIR/codit_eval.py" \
    --dataset coastline --split test --model_type GRU \
    --model_name codit_GRU_coastline $SUBSET_FLAG \
    --prediction_length 24

echo ""
echo "=== Evaluating CoDiT (GRU) — open_ocean ==="
python "$EVAL_DIR/codit_eval.py" \
    --dataset open_ocean --split test --model_type GRU \
    --model_name codit_GRU_open_ocean $SUBSET_FLAG \
    --prediction_length 72

echo ""
echo "=== Evaluating CoDiT (Transformer) — coastline ==="
python "$EVAL_DIR/codit_eval.py" \
    --dataset coastline --split test --model_type Transformer \
    --model_name codit_Transformer_coastline $SUBSET_FLAG \
    --prediction_length 24

echo ""
echo "=== Evaluating CoDiT (Transformer) — open_ocean ==="
python "$EVAL_DIR/codit_eval.py" \
    --dataset open_ocean --split test --model_type Transformer \
    --model_name codit_Transformer_open_ocean $SUBSET_FLAG \
    --prediction_length 72

echo ""
echo "=== All evaluations complete ==="
echo "Results saved under: \$CODIT_DATA_PATH/results/ (or data/results/)"
