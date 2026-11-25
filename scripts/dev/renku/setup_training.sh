#!/usr/bin/env bash
# Prepare a Renku session for training new CoDiT checkpoints from the Zenodo data.
#
# The Zenodo record (doi:10.5281/zenodo.21644091) ships raw per-patch NetCDFs.
# Training needs a merged `training_data_<L>h.h5`, which this script builds:
#
#   1. install the `codit` package (editable) into the active Python env
#   2. extract <dataset>_configs.tar.gz       -> $DATA_PATH/configs/<dataset>/
#   3. extract <dataset>_<split>.tar.gz       -> $DATA_PATH/datasets/<dataset>/<split>/
#   4. download GEBCO bathymetry (Polybox)    -> $DATA_PATH/bathymetry/
#   5. precompute the merged training h5      -> $DATA_PATH/datasets/<dataset>/train/processed/
#   6. symlink datasets/<dataset>/datasets/processed -> ../train/processed
#      (the train scripts still read the legacy `datasets/processed` location)
#
# Every step is skipped if its output already exists, so the script can be
# re-run after an interruption.
#
# Usage:
#   bash scripts/dev/renku/setup_training.sh --dataset coastline
#   bash scripts/dev/renku/setup_training.sh --dataset open_ocean --splits train,val,test
#
# Then train, e.g.:
#   python scripts/user/train/codit_train.py --dataset coastline --prediction_length 24

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATASET="coastline"
SPLITS="train"
ZENODO_DIR=""
LENGTH=""
SKIP_INSTALL=0
SKIP_PRECOMPUTE=0
PYTHON="${PYTHON:-python}"

usage() {
    awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"
    cat <<EOF
Options:
  --dataset {coastline,open_ocean}   dataset to prepare (default: coastline)
  --splits LIST                      comma-separated splits to extract (default: train)
  --zenodo-dir DIR                   folder with the Zenodo tarballs
                                     (default: ../codit-ocean-* next to the repo)
  --length HOURS                     snippet length (default: 24 coastline, 72 open_ocean)
  --skip-install                     do not pip-install the codit package
  --skip-precompute                  only extract data, do not build the training h5
Environment:
  CODIT_DATA_PATH                    data root (default: <repo>/data)
  PYTHON                             python interpreter (default: python)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --splits) SPLITS="$2"; shift 2 ;;
        --zenodo-dir) ZENODO_DIR="$2"; shift 2 ;;
        --length) LENGTH="$2"; shift 2 ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --skip-precompute) SKIP_PRECOMPUTE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

case "$DATASET" in
    coastline) : "${LENGTH:=24}" ;;
    open_ocean) : "${LENGTH:=72}" ;;
    *) echo "Unsupported dataset: $DATASET" >&2; exit 1 ;;
esac

if [[ -z "$ZENODO_DIR" ]]; then
    shopt -s nullglob
    candidates=("$REPO_ROOT"/../codit-ocean-*)
    shopt -u nullglob
    if [[ ${#candidates[@]} -ne 1 ]]; then
        echo "Could not locate a unique ../codit-ocean-* folder; pass --zenodo-dir." >&2
        exit 1
    fi
    ZENODO_DIR="${candidates[0]}"
fi
ZENODO_DIR="$(cd "$ZENODO_DIR" && pwd)"

export CODIT_DATA_PATH="${CODIT_DATA_PATH:-$REPO_ROOT/data}"
DATA_PATH="$CODIT_DATA_PATH"
DS_ROOT="$DATA_PATH/datasets/$DATASET"

if command -v pigz >/dev/null 2>&1; then
    TAR_GZ=(tar -I pigz)
else
    TAR_GZ=(tar -z)
fi

log() { printf '\n==> %s\n' "$*"; }

check_space() {  # $1 = tarball, $2 = target dir; warn if free space < 2x tarball
    local need avail
    need=$(( $(stat -c %s "$1") * 2 / 1024 ))
    avail=$(df -Pk "$2" | awk 'NR==2 {print $4}')
    if (( avail < need )); then
        echo "WARNING: $(basename "$1") may need ~$((need / 1024 / 1024)) GB," \
             "only $((avail / 1024 / 1024)) GB free under $2." >&2
    fi
}

echo "Repo:      $REPO_ROOT"
echo "Zenodo:    $ZENODO_DIR"
echo "Data path: $DATA_PATH"
echo "Dataset:   $DATASET (splits: $SPLITS, length: ${LENGTH}h)"

# 1. Install package -----------------------------------------------------------
if (( SKIP_INSTALL )); then
    log "Skipping package install"
elif "$PYTHON" -c "import codit" 2>/dev/null; then
    log "codit already importable — skipping install"
else
    log "Installing codit (editable)"
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$(command -v "$PYTHON")" -e "$REPO_ROOT"
    else
        "$PYTHON" -m pip install -e "$REPO_ROOT"
    fi
fi

# 2. Configs ---------------------------------------------------------------------
mkdir -p "$DATA_PATH/configs"
if [[ -d "$DATA_PATH/configs/$DATASET" ]]; then
    log "Configs for $DATASET already present"
else
    log "Extracting ${DATASET}_configs.tar.gz"
    "${TAR_GZ[@]}" -xf "$ZENODO_DIR/${DATASET}_configs.tar.gz" \
        --no-same-owner -C "$DATA_PATH/configs"
fi

# 3. Dataset splits ----------------------------------------------------------------
IFS=',' read -r -a SPLIT_ARR <<< "$SPLITS"
for split in "${SPLIT_ARR[@]}"; do
    dest="$DS_ROOT/$split"
    marker="$dest/.extracted"
    tarball="$ZENODO_DIR/${DATASET}_${split}.tar.gz"
    if [[ -f "$marker" ]]; then
        log "$DATASET/$split already extracted"
        continue
    fi
    [[ -f "$tarball" ]] || { echo "Missing $tarball" >&2; exit 1; }
    mkdir -p "$dest"
    check_space "$tarball" "$dest"
    log "Extracting $(basename "$tarball") -> $dest (this takes a while)"
    "${TAR_GZ[@]}" -xf "$tarball" --no-same-owner -C "$dest"
    touch "$marker"
done

# 4. Bathymetry ------------------------------------------------------------------
if [[ -d "$DATA_PATH/bathymetry/GEBCO_06_Jan_2025_ba69ac5d6a3b" ]]; then
    log "Bathymetry already present"
else
    log "Downloading bathymetry"
    "$PYTHON" "$REPO_ROOT/scripts/user/setup/download_data.py" \
        --bathymetry --data-path "$DATA_PATH"
fi

# 5. Precompute merged training h5 ----------------------------------------------------
if (( SKIP_PRECOMPUTE )); then
    log "Skipping precompute"
else
    if [[ ! -f "$DS_ROOT/train/.extracted" ]]; then
        echo "The train split is not extracted; add 'train' to --splits." >&2
        exit 1
    fi
    h5="$DS_ROOT/train/processed/training_data_${LENGTH}h.h5"
    if [[ -f "$h5" ]]; then
        log "Training h5 already exists: $h5"
    else
        log "Precomputing training h5 (slow on CPU; per-patch results are cached)"
        "$PYTHON" "$REPO_ROOT/scripts/dev/data/precompute_trainset.py" \
            --dataset "$DATASET" --split train --length "$LENGTH"
    fi

    # 6. Legacy path expected by codit_train.py / lingauss_train.py
    legacy="$DS_ROOT/datasets/processed"
    if [[ -e "$legacy" && ! -L "$legacy" ]]; then
        log "$legacy exists and is not a symlink — leaving it alone"
    else
        mkdir -p "$DS_ROOT/datasets"
        ln -sfn ../train/processed "$legacy"
        log "Linked $legacy -> ../train/processed"
    fi
fi

log "Done. Train with:"
echo "  python scripts/user/train/codit_train.py --dataset $DATASET --prediction_length $LENGTH"
