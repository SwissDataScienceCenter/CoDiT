# Reproducibility Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an integration test that runs the eval pipeline on 3 trajectories for both LinearGauss and CoDiT-GRU on `open_ocean` and asserts mean Haversine error is within ±10% of a known reference.

**Architecture:** Establish the CoDiT reference value by running the shipped checkpoint once, then write and run the test file. Single commit at the end.

**Tech Stack:** pytest, JAX/Flax, codit (local package), micromamba `codit` env.

## Global Constraints

- **Prerequisite:** `2026-07-09-dataset-reorganisation` plan must be complete before starting
- `CODIT_DATA_PATH=/mydata/adopt/christian/adopt/data_final` for all steps
- Python: `micromamba run -n codit python ...`
- Tests: `micromamba run -n codit pytest ...`
- All commits on `dev` branch
- `num_samples=100`, `length=72`, indices `[0, 50, 100]`, patch `open_ocean_patch_1_year_2021`
- Follow the skip-guard pattern in `tests/integration/test_checkpoint_loading.py`

---

## Task 1: Establish CoDiT reference value

**Files:**
- No files created or modified

**Interfaces:**
- Produces: `CODIT_REF_M` float (printed to stdout, copied into Task 2)

- [ ] **Step 1: Run CoDiT eval on 3 trajectories**

```bash
CODIT_DATA_PATH=/mydata/adopt/christian/adopt/data_final \
  micromamba run -n codit python - <<'EOF'
import os, numpy as np
from codit.data_module import config_paths, config_funcs
from codit.data_module.dataset import OceanDataset
from codit.model_module.diffusion_model import CoDiTModel
from codit.eval_module.eval_runner import run_eval

config_path = os.path.join(config_paths.CONFIG_PATH, "open_ocean", "test")
config = config_funcs.load_config("open_ocean_patch_1_year_2021", config_path)
dataset = OceanDataset(config, 72, 16, 10000, "linear")

models_dir = os.path.join(config_paths.MODEL_PATH, "trained_models")
model = CoDiTModel.import_trained_model("codit_GRU_open_ocean", import_path=models_dir)

distances, _, _ = run_eval(model, dataset, indices=[0, 50, 100], length=72, num_samples=100)
print(f"CODIT_REF_M = {float(np.nanmean(distances)):.1f}")
EOF
```

Expected: a line like `CODIT_REF_M = 42000.0`. Record the printed value — it goes into Task 2.

---

## Task 2: Write and run the reproducibility test

**Files:**
- Create: `tests/integration/test_reproducibility.py`

**Interfaces:**
- Consumes:
  - `codit.data_module.config_paths.CONFIG_PATH` — base config directory
  - `codit.data_module.config_paths.MODEL_PATH` — base model directory
  - `codit.data_module.config_funcs.load_config(name: str, config_path: str) -> dict`
  - `codit.data_module.dataset.OceanDataset(config, length, num_grid_points, delta_metres, interp_method)`
  - `codit.model_module.linear_gauss_model.LinearGaussModel.import_trained_model(model_name: str, import_path: str) -> LinearGaussModel`
  - `codit.model_module.diffusion_model.CoDiTModel.import_trained_model(model_name: str, import_path: str) -> CoDiTModel`
  - `codit.eval_module.eval_runner.run_eval(model, dataset, indices, length, num_samples) -> (distances, pred_coords, true_coords)` where `distances.shape == (N, num_samples, length)`
  - `CODIT_REF_M` float from Task 1

- [ ] **Step 1: Write the test file**

Replace `<CODIT_REF_M>` with the float printed in Task 1.

Create `tests/integration/test_reproducibility.py`:

```python
"""
Reproducibility tests: run eval on 3 trajectories and check mean Haversine
error is within ±10% of a pre-established reference.

Requires:
  - CODIT_DATA_PATH pointing at a data directory with open_ocean configs and datasets
  - Datasets reorganised to canonical layout (see 2026-07-09-dataset-reorganisation plan)
  - Trained model checkpoints under $CODIT_DATA_PATH/trained_models/

Skipped automatically when any of the above are absent.
"""

import os

import numpy as np
import pytest

from codit.data_module import config_paths

# ── Reference values ─────────────────────────────────────────────────────────
# LinearGauss: recomputed from pred_coords/true_coords in the original
# open_ocean_patch_1_year_2021_result.npz.  The shipped checkpoint is
# byte-identical to the one used to produce those results.
LGAUSS_REF_M: float = 34_651.0

# CoDiT-GRU: established by running the shipped codit_GRU_open_ocean checkpoint
# once on these 3 trajectories (see Task 1 of this plan).
# The shipped checkpoint differs from the paper checkpoint, so this verifies
# self-consistency of the shipped model, not exact paper numbers.
CODIT_REF_M: float = <CODIT_REF_M>

TOLERANCE: float = 0.10  # ±10%

# ── Skip guards ───────────────────────────────────────────────────────────────
_MODELS_DIR = os.path.join(config_paths.MODEL_PATH, "trained_models")
_CONFIG_FILE = os.path.join(
    config_paths.CONFIG_PATH, "open_ocean", "test", "open_ocean_patch_1_year_2021.yml"
)
_LGAUSS_CKPT = os.path.join(_MODELS_DIR, "linear_gauss_open_ocean", "nnx_model.pkl")
_CODIT_CKPT  = os.path.join(_MODELS_DIR, "codit_GRU_open_ocean", "nn_params.npz")

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
```

- [ ] **Step 2: Run only the new tests**

```bash
CODIT_DATA_PATH=/mydata/adopt/christian/adopt/data_final \
  micromamba run -n codit pytest tests/integration/test_reproducibility.py -v
```

Expected:
```
tests/integration/test_reproducibility.py::test_linear_gauss_mean_error PASSED
tests/integration/test_reproducibility.py::test_codit_gru_mean_error PASSED
2 passed in ...s
```

If either test fails with `AssertionError`, re-run Task 1 to check `CODIT_REF_M` was copied correctly.

- [ ] **Step 3: Run the full test suite**

```bash
micromamba run -n codit pytest tests/ -q
```

Expected: 109 passed (107 existing + 2 new). Without `CODIT_DATA_PATH` set, the 2 new tests are skipped — correct behaviour for CI.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_reproducibility.py
git commit -m "test: add reproducibility integration tests for LinearGauss and CoDiT-GRU"
```
