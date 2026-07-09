"""
Reproducibility tests: run eval on 3 trajectories and check mean Haversine
error is within ±10% of a pre-established reference.

Requires:
  - CODIT_DATA_PATH pointing at a data directory with open_ocean configs and
    datasets reorganised to canonical layout (see 2026-07-09-dataset-reorganisation plan)
  - Trained model checkpoints in the repo under data/trained_models/

Skipped automatically when any of the above are absent.
"""

import os

import numpy as np
import pytest

from codit.data_module import config_paths

# ── Reference values ─────────────────────────────────────────────────────────
# LinearGauss: established by running the shipped linear_gauss_open_ocean checkpoint
# on indices [0, 50, 100] of open_ocean_patch_1_year_2021 with num_samples=100.
# Computed as nanmean over (trajectories, samples, timesteps) of per-sample distances.
LGAUSS_REF_M: float = 26_471.9

# CoDiT-GRU: established by running the shipped codit_GRU_open_ocean checkpoint
# once on these 3 trajectories (see 2026-07-09-reproducibility-test plan, Task 1).
# The shipped checkpoint differs from the paper checkpoint, so this verifies
# self-consistency of the shipped model, not exact paper numbers.
CODIT_REF_M: float = 23_169.6

TOLERANCE: float = 0.10  # ±10%

# ── Paths ─────────────────────────────────────────────────────────────────────
# Checkpoints are always in the repo's data/trained_models/, independent of
# CODIT_DATA_PATH (which points at the dataset, potentially on a separate volume).
_MODELS_DIR = os.path.join(config_paths._REPO_ROOT, "data", "trained_models")

_CONFIG_FILE = os.path.join(
    config_paths.CONFIG_PATH, "open_ocean", "test", "open_ocean_patch_1_year_2021.yml"
)
_LGAUSS_CKPT = os.path.join(_MODELS_DIR, "linear_gauss_open_ocean", "nnx_model.pkl")
_CODIT_CKPT = os.path.join(_MODELS_DIR, "codit_GRU_open_ocean", "nn_params.npz")

# ── Skip guards ───────────────────────────────────────────────────────────────
requires_config = pytest.mark.skipif(
    not os.path.exists(_CONFIG_FILE),
    reason="open_ocean test configs not found — set CODIT_DATA_PATH and run download_data.py",
)
requires_lgauss = pytest.mark.skipif(
    not os.path.exists(_LGAUSS_CKPT),
    reason="linear_gauss_open_ocean checkpoint not found",
)
requires_codit = pytest.mark.skipif(
    not os.path.exists(_CODIT_CKPT),
    reason="codit_GRU_open_ocean checkpoint not found",
)


# ── Shared dataset fixture ────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def open_ocean_dataset():
    """Load open_ocean_patch_1_year_2021 once for all tests in this module."""
    from codit.data_module import config_funcs
    from codit.data_module.dataset import OceanDataset

    config_path = os.path.join(config_paths.CONFIG_PATH, "open_ocean", "test")
    config = config_funcs.load_config("open_ocean_patch_1_year_2021", config_path)
    return OceanDataset(config, 72, 16, 10000, "linear")


# ── Tests ─────────────────────────────────────────────────────────────────────
@requires_config
@requires_lgauss
def test_linear_gauss_mean_error(open_ocean_dataset):
    """LinearGauss mean Haversine error on 3 trajectories must be within 10% of reference."""
    from codit.eval_module.eval_runner import run_eval
    from codit.model_module.linear_gauss_model import LinearGaussModel

    model = LinearGaussModel.import_trained_model(
        "linear_gauss_open_ocean", import_path=_MODELS_DIR
    )
    distances, _, _ = run_eval(
        model, open_ocean_dataset, indices=[0, 50, 100], length=72, num_samples=100
    )
    mean_err = float(np.nanmean(distances))
    assert abs(mean_err - LGAUSS_REF_M) / LGAUSS_REF_M < TOLERANCE, (
        f"LinearGauss mean error {mean_err:.0f} m is more than 10% "
        f"from reference {LGAUSS_REF_M:.0f} m"
    )


@requires_config
@requires_codit
def test_codit_gru_mean_error(open_ocean_dataset):
    """CoDiT-GRU mean Haversine error on 3 trajectories must be within 10% of reference."""
    from codit.eval_module.eval_runner import run_eval
    from codit.model_module.diffusion_model import CoDiTModel

    model = CoDiTModel.import_trained_model(
        "codit_GRU_open_ocean", import_path=_MODELS_DIR
    )
    distances, _, _ = run_eval(
        model, open_ocean_dataset, indices=[0, 50, 100], length=72, num_samples=100
    )
    mean_err = float(np.nanmean(distances))
    assert abs(mean_err - CODIT_REF_M) / CODIT_REF_M < TOLERANCE, (
        f"CoDiT-GRU mean error {mean_err:.0f} m is more than 10% "
        f"from reference {CODIT_REF_M:.0f} m"
    )
