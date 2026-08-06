# Figure 4 / Table 2 Reproduction Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship two RunAI launcher scripts per dataset (one job per model), plus a per-dataset summary script, that regenerate the CoDiT (GRU + TF) and LinearGauss (WDM) rows of Table 2 / Figure 4C+D from the shipped checkpoints, into a parallel result subdirectory, then aggregate into a CSV that compares fresh median metrics against paper values.

**Architecture:** Additive `--result_subdir` arg on the two existing eval entry points (`codit_eval.py`, `lingauss_eval.py`) that inserts one directory between `<split>` and `<model_name>` in the output path. Fresh runs write to `data/results/<dataset>/test/fig4_repro/<model>/`, leaving existing `data/results/<dataset>/test/<model>/` untouched. New `summarize_fig4.py` reads the three fresh dirs, computes final-horizon medians via `codit.eval_module.metrics.get_metrics`, and emits `fig4_summary.csv` alongside a stdout comparison against hardcoded paper values.

**Tech Stack:** Python 3.10+, JAX/Flax NNX, `codit.eval_module.metrics`, `codit.data_module.data_transform.haversine_distance`, pytest, RunAI/micromamba.

## Global Constraints

- Branch: `feat/fig4-reproduction-scripts` off `dev`.
- No existing `.npz` result files may be overwritten — all fresh output lives under `data/results/<dataset>/<split>/fig4_repro/`.
- No modifications to trained model checkpoints under `data/trained_models/`.
- Existing behavior of both eval scripts (when `--result_subdir` is omitted) must be byte-identical to today.
- Env: `micromamba run -n codit …` for local sanity checks; RunAI shell scripts do `source ~/.bashrc && micromamba activate codit`.
- Repo root: `/mydata/adopt/christian/CoDiT/`.
- Input vars: `hycom_current copernicus_current era5_wind` for `open_ocean`; append `bathymetry` for `coastline`.
- Naive-filter threshold: 100.0 metres (`haversine_distance` returns metres; do NOT change units).
- Paper values (from `paper/paper.md` Table 2) — used verbatim in `summarize_fig4.py`:

  | Dataset     | Model       | PE ↓    | ES ↓    | SS ↑  |
  |-------------|-------------|---------|---------|-------|
  | open_ocean  | CoDiT (GRU) | 26.135  | 18.092  | 0.466 |
  | open_ocean  | CoDiT (TF)  | 26.418  | 18.154  | 0.460 |
  | open_ocean  | WDM         | 27.196  | 18.666  | 0.406 |
  | coastline   | CoDiT (GRU) | 8.638   | 5.746   | 0.268 |
  | coastline   | CoDiT (TF)  | 8.032   | 5.339   | 0.305 |
  | coastline   | WDM         | 9.651   | 7.842   | 0.167 |

---

## File Structure

**New files:**
- `my_scripts/model_fitting/summarize_fig4.py` — aggregation + comparison CLI
- `runai_scripts/fig4_open_ocean.sh` — eval launcher (`gru`/`tf`/`lingauss`)
- `runai_scripts/fig4_coastline.sh` — same, with bathymetry
- `runai_scripts/fig4_summarize.sh` — summary launcher (per dataset)
- `tests/model_module/test_summarize_fig4.py` — unit test with synthetic `.npz` fixture

**Modified files:**
- `my_scripts/model_fitting/codit_eval.py` — add `--result_subdir`, update path construction in both dataset branches
- `my_scripts/model_fitting/lingauss_eval.py` — same

---

### Task 1: Create branch

**Files:** none — git only.

- [ ] **Step 1: Create branch off `dev`**

```bash
cd /mydata/adopt/christian/CoDiT
git checkout dev
git pull
git checkout -b feat/fig4-reproduction-scripts
```

Expected: `git status` shows clean working tree on `feat/fig4-reproduction-scripts`.

---

### Task 2: Add `--result_subdir` flag to `codit_eval.py`

**Files:**
- Modify: `my_scripts/model_fitting/codit_eval.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `codit_eval.py` accepts `--result_subdir <str>` (default `""`). When set, results land at `<RESULT_PATH>/<dataset>/<split>/<result_subdir>/<model_name>/`. When empty, path is byte-identical to before.

- [ ] **Step 1: Read current arg block and path construction to locate the exact lines**

Run:

```bash
grep -n "add_argument\|result_path = os.path.join" /mydata/adopt/christian/CoDiT/my_scripts/model_fitting/codit_eval.py
```

Confirm you see two `result_path = os.path.join(...)` calls (one in each dataset branch, around lines 66 and 77).

- [ ] **Step 2: Add the new argparse argument**

Insert this block immediately after the existing `--model_name` argument (near line 52):

```python
    parser.add_argument(
        "--result_subdir",
        type=str,
        default="",
        help="Optional subdirectory inserted between --split and the model name in the result path. Default '' preserves the original path.",
    )
```

- [ ] **Step 3: Update the `open_ocean` path construction**

Replace:

```python
        result_path = os.path.join(
            config_paths.RESULT_PATH,
            "open_ocean",
            args.split,
            model_name,
        )
```

with:

```python
        result_path = os.path.join(
            config_paths.RESULT_PATH,
            "open_ocean",
            args.split,
            args.result_subdir,
            model_name,
        )
```

- [ ] **Step 4: Update the `coastline` path construction**

Same change in the `elif args.dataset == "coastline":` branch — insert `args.result_subdir` between `args.split` and `model_name`.

- [ ] **Step 5: Verify default behavior is unchanged**

Run:

```bash
cd /mydata/adopt/christian/CoDiT
micromamba run -n codit python -c "
import os
print(os.path.join('a', 'b', '', 'c'))
print(os.path.join('a', 'b', 'sub', 'c'))
"
```

Expected output:

```
a/b/c
a/b/sub/c
```

Then dry-run help:

```bash
micromamba run -n codit python my_scripts/model_fitting/codit_eval.py --help 2>&1 | grep -A1 result_subdir
```

Expected: line containing `--result_subdir RESULT_SUBDIR` plus its help text.

- [ ] **Step 6: Commit**

```bash
git add my_scripts/model_fitting/codit_eval.py
git commit -m "feat(eval): add --result_subdir to codit_eval for parallel output paths"
```

---

### Task 3: Add `--result_subdir` flag to `lingauss_eval.py`

**Files:**
- Modify: `my_scripts/model_fitting/lingauss_eval.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `lingauss_eval.py` accepts `--result_subdir <str>` (default `""`), symmetric to `codit_eval.py`. Path structure: `<RESULT_PATH>/<dataset>/<split>/<result_subdir>/linear_gauss/`.

- [ ] **Step 1: Add the argparse argument**

Insert immediately after the existing `--evaluate_subset` argument (near line 25):

```python
    parser.add_argument(
        "--result_subdir",
        type=str,
        default="",
        help="Optional subdirectory inserted between --split and the model name in the result path. Default '' preserves the original path.",
    )
```

- [ ] **Step 2: Update the `open_ocean` path construction**

Replace:

```python
        result_path = os.path.join(
            config_paths.RESULT_PATH, "open_ocean", args.split, "linear_gauss"
        )
```

with:

```python
        result_path = os.path.join(
            config_paths.RESULT_PATH,
            "open_ocean",
            args.split,
            args.result_subdir,
            "linear_gauss",
        )
```

- [ ] **Step 3: Update the `coastline` path construction**

Same change in the `elif args.dataset == "coastline":` branch.

- [ ] **Step 4: Verify help output**

```bash
cd /mydata/adopt/christian/CoDiT
micromamba run -n codit python my_scripts/model_fitting/lingauss_eval.py --help 2>&1 | grep -A1 result_subdir
```

Expected: line containing `--result_subdir RESULT_SUBDIR` plus its help text.

- [ ] **Step 5: Commit**

```bash
git add my_scripts/model_fitting/lingauss_eval.py
git commit -m "feat(eval): add --result_subdir to lingauss_eval for parallel output paths"
```

---

### Task 4: Write failing test for `summarize_fig4.py`

**Files:**
- Create: `tests/model_module/test_summarize_fig4.py`

**Interfaces:**
- Consumes: `summarize_fig4.py` (to be created in Task 5). CLI: `python summarize_fig4.py --dataset <ds> --split <split> --result_subdir <sub>`. Reads `<RESULT_PATH_ENV>/<dataset>/<split>/<sub>/<model_dir>/*_result.npz`. Writes CSV to `<RESULT_PATH_ENV>/<dataset>/<split>/<sub>/fig4_summary.csv`.
- Produces: pytest-style failing test that a fresh Task-5 implementation must satisfy.

- [ ] **Step 1: Create the test file**

Write the following to `tests/model_module/test_summarize_fig4.py`:

```python
"""Smoke test for summarize_fig4.py using synthetic .npz fixtures.

The fixture places one 'patch' file per model in the expected directory
layout, with pred_coords == true_coords (perfect predictions) so PE/ES
are zero, SS is 1.0, and the naive-drift filter is satisfied (trajectories
drift > 100 m from start).
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path("/mydata/adopt/christian/CoDiT")
SUMMARIZE_SCRIPT = REPO_ROOT / "my_scripts" / "model_fitting" / "summarize_fig4.py"


def _make_synthetic_patch(model_dir: Path, n_traj: int = 2, n_samples: int = 3, pred_len: int = 5):
    """Write one <name>_result.npz with pred == true (perfect predictions).

    Shapes match what codit_eval / lingauss_eval save:
      true_coords: (N, T, 2)         — N trajectories, T = pred_len + 1 timesteps
      pred_coords: (N, S, T-1, 2)    — N × S samples of predicted (T-1) steps
      distance:    (N, S, T-1)       — filler, not read by summarize

    Coords are (lat, lon) in degrees. Start at (0, 0); drift eastward by
    0.01 deg (~1.1 km) per step so the >100 m naive-drift filter passes.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    # True coords: N trajectories, T timesteps (pred_len + 1 including start)
    T = pred_len + 1
    step_deg = 0.01
    traj = np.stack(
        [np.zeros(T), step_deg * np.arange(T)], axis=-1
    )  # (T, 2)
    true_coords = np.tile(traj, (n_traj, 1, 1)).astype(np.float32)  # (N, T, 2)
    # Perfect predictions: every sample equals the true post-start trajectory
    pred_traj = true_coords[:, 1:, :]  # (N, T-1, 2)
    pred_coords = np.broadcast_to(
        pred_traj[:, None, :, :], (n_traj, n_samples, pred_len, 2)
    ).astype(np.float32)
    distance = np.zeros((n_traj, n_samples, pred_len), dtype=np.float32)
    np.savez(
        model_dir / "synthetic_patch_result.npz",
        true_coords=true_coords,
        pred_coords=pred_coords,
        distance=distance,
    )


@pytest.mark.parametrize("dataset,model_dirs", [
    (
        "open_ocean",
        [
            "codit_GRU_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind",
            "codit_Transformer_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind",
            "linear_gauss",
        ],
    ),
])
def test_summarize_fig4_writes_csv_with_expected_rows(tmp_path, dataset, model_dirs):
    split = "test"
    subdir = "fig4_repro"
    # Layout: <tmp_path>/results/<dataset>/<split>/<subdir>/<model_dir>/synthetic_patch_result.npz
    base = tmp_path / "results" / dataset / split / subdir
    for m in model_dirs:
        _make_synthetic_patch(base / m, pred_len=5)  # small T for speed

    env = os.environ.copy()
    env["CODIT_DATA_PATH"] = str(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SUMMARIZE_SCRIPT),
            "--dataset", dataset,
            "--split", split,
            "--result_subdir", subdir,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"summarize_fig4.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    csv_path = base / "fig4_summary.csv"
    assert csv_path.exists(), f"CSV not written at {csv_path}"

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Expect exactly three rows, one per model
    assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"

    labels = {row["model"] for row in rows}
    assert labels == {"CoDiT (GRU)", "CoDiT (TF)", "WDM"}

    # Each row must have the 9 comparison columns
    expected_cols = {
        "model",
        "PE_new_km", "PE_paper_km", "dPE_km",
        "ES_new_km", "ES_paper_km", "dES_km",
        "SS_new", "SS_paper", "dSS",
    }
    assert expected_cols.issubset(rows[0].keys()), (
        f"missing columns: {expected_cols - set(rows[0].keys())}"
    )

    # With perfect predictions PE and ES must be ~0, SS must be ~1
    for row in rows:
        assert float(row["PE_new_km"]) < 0.05, row
        assert float(row["ES_new_km"]) < 0.05, row
        assert float(row["SS_new"]) > 0.99, row
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
cd /mydata/adopt/christian/CoDiT
micromamba run -n codit pytest tests/model_module/test_summarize_fig4.py -v 2>&1 | tail -25
```

Expected: FAILURES / errors because `summarize_fig4.py` does not exist yet. The failure message must reference the missing script or a non-zero return code.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/model_module/test_summarize_fig4.py
git commit -m "test: failing smoke test for summarize_fig4 with synthetic fixture"
```

---

### Task 5: Implement `summarize_fig4.py`

**Files:**
- Create: `my_scripts/model_fitting/summarize_fig4.py`

**Interfaces:**
- Consumes:
  - `codit.eval_module.metrics.get_metrics(true_coords, pred_coords)` → dict with keys `"mean_distance"`, `"energy_score"`, `"skill_score"`, each shape `(N, T-1)`, values in metres for the first two and unitless for skill.
  - `codit.data_module.data_transform.haversine_distance(pos1, pos2)` → scalar distance in metres.
  - `codit.data_module.config_paths.RESULT_PATH` → base directory for results.
  - `<RESULT_PATH>/<dataset>/<split>/<result_subdir>/<model_dir>/*_result.npz` files with keys `true_coords` (N, T, 2), `pred_coords` (N, S, T-1, 2).
- Produces:
  - `<RESULT_PATH>/<dataset>/<split>/<result_subdir>/fig4_summary.csv` with columns `model, PE_new_km, PE_paper_km, dPE_km, ES_new_km, ES_paper_km, dES_km, SS_new, SS_paper, dSS`.
  - stdout: same table pretty-printed.

- [ ] **Step 1: Write the script**

Create `my_scripts/model_fitting/summarize_fig4.py` with:

```python
"""Aggregate Fig 4 / Table 2 metrics for CoDiT (GRU/TF) + WDM (LinearGauss).

Reads .npz files from three model result directories under
    <RESULT_PATH>/<dataset>/<split>/<result_subdir>/<model_dir>/
computes final-horizon median PE / ES / SS across the valid trajectories,
prints a comparison table against Table 2 from the paper, and writes
fig4_summary.csv into the same subdirectory.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from jax import numpy as jnp
from jax import vmap

from codit.data_module import config_paths
from codit.data_module import data_transform
from codit.eval_module import metrics


PAPER_TABLE_2 = {
    "open_ocean": {
        "CoDiT (GRU)": {"PE": 26.135, "ES": 18.092, "SS": 0.466},
        "CoDiT (TF)":  {"PE": 26.418, "ES": 18.154, "SS": 0.460},
        "WDM":         {"PE": 27.196, "ES": 18.666, "SS": 0.406},
    },
    "coastline": {
        "CoDiT (GRU)": {"PE": 8.638, "ES": 5.746, "SS": 0.268},
        "CoDiT (TF)":  {"PE": 8.032, "ES": 5.339, "SS": 0.305},
        "WDM":         {"PE": 9.651, "ES": 7.842, "SS": 0.167},
    },
}


def _input_var_str(dataset: str) -> str:
    if dataset == "open_ocean":
        return "hycom_current_copernicus_current_era5_wind"
    if dataset == "coastline":
        return "hycom_current_copernicus_current_era5_wind_bathymetry"
    raise ValueError(f"unknown dataset: {dataset!r}")


def _model_dir_map(dataset: str) -> dict:
    ivs = _input_var_str(dataset)
    return {
        "CoDiT (GRU)": f"codit_GRU_{dataset}_conv_adam_K100_{ivs}",
        "CoDiT (TF)":  f"codit_Transformer_{dataset}_conv_adam_K100_{ivs}",
        "WDM":         "linear_gauss",
    }


def _load_model_npz(model_dir: Path, patch_files: list) -> tuple:
    """Concatenate true_coords and pred_coords across a list of patch files."""
    trues, preds = [], []
    for name in patch_files:
        data = np.load(model_dir / name, allow_pickle=True)
        trues.append(np.asarray(data["true_coords"]))
        preds.append(np.asarray(data["pred_coords"]))
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def _naive_distances(true_coords: np.ndarray) -> np.ndarray:
    """Distance (metres) between start position and each subsequent step.

    true_coords: (N, T, 2). Returns (N, T-1).
    """
    start = jnp.asarray(true_coords[:, :1, :])           # (N, 1, 2)
    rest = jnp.asarray(true_coords[:, 1:, :])            # (N, T-1, 2)
    hav = vmap(vmap(data_transform.haversine_distance, in_axes=(None, 0)))
    return np.asarray(hav(start[:, 0, :], rest))         # (N, T-1)


def compute_metrics_for_model(
    model_dir: Path, patch_files: list
) -> dict:
    """Return dict with (N_total, T-1) arrays for PE, ES, SS + the concatenated true_coords."""
    true_coords, pred_coords = _load_model_npz(model_dir, patch_files)
    m = metrics.get_metrics(true_coords, pred_coords)
    return {
        "mean_distance": np.asarray(m["mean_distance"]),
        "energy_score":  np.asarray(m["energy_score"]),
        "skill_score":   np.asarray(m["skill_score"]),
        "true_coords":   true_coords,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                   choices=["open_ocean", "coastline"])
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--result_subdir", type=str, default="fig4_repro")
    args = p.parse_args()

    if args.dataset not in PAPER_TABLE_2:
        print(f"error: no paper values for dataset {args.dataset!r}", file=sys.stderr)
        return 2

    base = Path(config_paths.RESULT_PATH) / args.dataset / args.split / args.result_subdir
    model_dirs = _model_dir_map(args.dataset)

    # Verify all three model dirs exist
    resolved = {}
    for label, dname in model_dirs.items():
        d = base / dname
        if not d.is_dir():
            print(f"error: missing model dir: {d}", file=sys.stderr)
            return 3
        resolved[label] = d

    # Intersection of patch filenames across the three model dirs
    file_sets = [
        {f for f in os.listdir(d) if f.endswith("_result.npz")}
        for d in resolved.values()
    ]
    common = sorted(set.intersection(*file_sets))
    if not common:
        print("error: no common patch files across model dirs", file=sys.stderr)
        return 4
    print(f"summarizing {len(common)} common patches for {args.dataset}")

    # Compute metrics per model
    per_model = {}
    for label, d in resolved.items():
        per_model[label] = compute_metrics_for_model(d, common)

    # Sanity: all three must have the same total N and T-1
    shapes = {label: v["mean_distance"].shape for label, v in per_model.items()}
    ref_shape = next(iter(shapes.values()))
    for label, s in shapes.items():
        if s != ref_shape:
            print(
                f"error: {label} mean_distance shape {s} != {ref_shape}",
                file=sys.stderr,
            )
            return 5

    # Validity mask: no-NaN across models AND naive drift > 100 m at any horizon
    valid_no_nan = np.ones(ref_shape[0], dtype=bool)
    for v in per_model.values():
        valid_no_nan &= ~np.any(np.isnan(v["mean_distance"]), axis=1)

    naive = _naive_distances(per_model[next(iter(resolved))]["true_coords"])
    valid_drift = np.any(naive > 100.0, axis=1)  # metres
    valid = valid_no_nan & valid_drift
    n_valid = int(np.sum(valid))
    print(f"valid trajectories after filter: {n_valid} / {ref_shape[0]}")
    if n_valid == 0:
        print("error: no trajectories pass validity filter", file=sys.stderr)
        return 6

    # Final-horizon medians (index -1). Convert metres -> km for PE and ES.
    rows = []
    header = [
        "model",
        "PE_new_km", "PE_paper_km", "dPE_km",
        "ES_new_km", "ES_paper_km", "dES_km",
        "SS_new",    "SS_paper",    "dSS",
    ]
    paper = PAPER_TABLE_2[args.dataset]
    for label in ["CoDiT (GRU)", "CoDiT (TF)", "WDM"]:
        v = per_model[label]
        pe_new = float(np.nanmedian(v["mean_distance"][valid, -1]) / 1000.0)
        es_new = float(np.nanmedian(v["energy_score"][valid, -1])  / 1000.0)
        ss_new = float(np.nanmedian(v["skill_score"][valid,  -1]))
        pe_p, es_p, ss_p = paper[label]["PE"], paper[label]["ES"], paper[label]["SS"]
        rows.append({
            "model":        label,
            "PE_new_km":    f"{pe_new:.3f}",
            "PE_paper_km":  f"{pe_p:.3f}",
            "dPE_km":       f"{pe_new - pe_p:+.3f}",
            "ES_new_km":    f"{es_new:.3f}",
            "ES_paper_km":  f"{es_p:.3f}",
            "dES_km":       f"{es_new - es_p:+.3f}",
            "SS_new":       f"{ss_new:.3f}",
            "SS_paper":     f"{ss_p:.3f}",
            "dSS":          f"{ss_new - ss_p:+.3f}",
        })

    # Print
    col_widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in header}
    line = " | ".join(c.ljust(col_widths[c]) for c in header)
    sep  = "-+-".join("-" * col_widths[c] for c in header)
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(r[c].ljust(col_widths[c]) for c in header))

    # Write CSV alongside the raw .npz outputs
    out_csv = base / "fig4_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the failing test — expect it to now pass**

```bash
cd /mydata/adopt/christian/CoDiT
micromamba run -n codit pytest tests/model_module/test_summarize_fig4.py -v 2>&1 | tail -20
```

Expected: `1 passed`. If it fails, read the diff between expected and actual carefully; common issues: shape mismatch in the fixture, or `data_transform.haversine_distance` signature mismatch. Fix `summarize_fig4.py` (never the fixture) until the test passes.

- [ ] **Step 3: Run the rest of the model_module test suite to check for regressions**

```bash
micromamba run -n codit pytest tests/model_module/ -v --ignore=tests/model_module/test_summarize_fig4.py 2>&1 | tail -20
```

Expected: same pass/fail count as before this task. Fix regressions before committing.

- [ ] **Step 4: Commit**

```bash
git add my_scripts/model_fitting/summarize_fig4.py
git commit -m "feat: add summarize_fig4.py — Table 2 aggregation and comparison"
```

---

### Task 6: Add `runai_scripts/fig4_open_ocean.sh`

**Files:**
- Create: `runai_scripts/fig4_open_ocean.sh`

**Interfaces:**
- Consumes: `codit_eval.py --result_subdir` (Task 2), `lingauss_eval.py --result_subdir` (Task 3).
- Produces: shell entry point that dispatches to one of three eval commands based on the `<model>` positional arg. Invocation: `./runai_scripts/fig4_open_ocean.sh <gru|tf|lingauss> [split] [subdir]`.

- [ ] **Step 1: Write the file**

Create `runai_scripts/fig4_open_ocean.sh`:

```bash
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
```

- [ ] **Step 2: Make executable and sanity-check the arg parsing**

```bash
cd /mydata/adopt/christian/CoDiT
chmod +x runai_scripts/fig4_open_ocean.sh
bash -n runai_scripts/fig4_open_ocean.sh              # syntax check only
./runai_scripts/fig4_open_ocean.sh 2>&1 | head -3     # should print usage and exit 1
./runai_scripts/fig4_open_ocean.sh bogus 2>&1 | tail -3
```

Expected: first call prints the `usage:` line; second call prints `Unknown model: bogus` and exits.

- [ ] **Step 3: Commit**

```bash
git add runai_scripts/fig4_open_ocean.sh
git commit -m "feat(runai): fig4_open_ocean.sh launches one eval per model"
```

---

### Task 7: Add `runai_scripts/fig4_coastline.sh`

**Files:**
- Create: `runai_scripts/fig4_coastline.sh`

**Interfaces:**
- Same interfaces as Task 6; only the `--dataset` and `input_vars` differ.

- [ ] **Step 1: Write the file**

Create `runai_scripts/fig4_coastline.sh`:

```bash
#!/bin/bash
# Usage: ./fig4_coastline.sh <gru|tf|lingauss> [split=test] [subdir=fig4_repro]
#
# Regenerates one row of Table 2 for the coastline dataset from a shipped
# trained checkpoint, writing .npz files to
#   data/results/coastline/<split>/<subdir>/<model_dir>/

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

input_vars="hycom_current copernicus_current era5_wind bathymetry"

case "$model" in
  gru)
    python codit_eval.py --dataset coastline --split "$split" \
      --model_type GRU --context_encoder_type conv --optimizer adam --K 100 \
      --input_vars $input_vars --result_subdir "$subdir"
    ;;
  tf)
    python codit_eval.py --dataset coastline --split "$split" \
      --model_type Transformer --context_encoder_type conv --optimizer adam --K 100 \
      --input_vars $input_vars --result_subdir "$subdir"
    ;;
  lingauss)
    python lingauss_eval.py --dataset coastline --split "$split" \
      --result_subdir "$subdir"
    ;;
  *)
    echo "Unknown model: $model (expected: gru, tf, lingauss)" >&2
    exit 1
    ;;
esac
```

- [ ] **Step 2: Make executable and sanity-check**

```bash
cd /mydata/adopt/christian/CoDiT
chmod +x runai_scripts/fig4_coastline.sh
bash -n runai_scripts/fig4_coastline.sh
./runai_scripts/fig4_coastline.sh 2>&1 | head -3
./runai_scripts/fig4_coastline.sh bogus 2>&1 | tail -3
```

Expected: same output pattern as Task 6.

- [ ] **Step 3: Commit**

```bash
git add runai_scripts/fig4_coastline.sh
git commit -m "feat(runai): fig4_coastline.sh launches one eval per model"
```

---

### Task 8: Add `runai_scripts/fig4_summarize.sh`

**Files:**
- Create: `runai_scripts/fig4_summarize.sh`

**Interfaces:**
- Consumes: `summarize_fig4.py` (Task 5).
- Produces: shell entry point for the per-dataset summary. Invocation: `./runai_scripts/fig4_summarize.sh <open_ocean|coastline> [split] [subdir]`.

- [ ] **Step 1: Write the file**

Create `runai_scripts/fig4_summarize.sh`:

```bash
#!/bin/bash
# Usage: ./fig4_summarize.sh <open_ocean|coastline> [split=test] [subdir=fig4_repro]
#
# Aggregates the three eval outputs for one dataset into fig4_summary.csv
# and prints a comparison against paper Table 2 to stdout. Launch after the
# three fig4_<dataset>.sh jobs (gru, tf, lingauss) have completed.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <open_ocean|coastline> [split=test] [subdir=fig4_repro]" >&2
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
```

- [ ] **Step 2: Make executable and sanity-check**

```bash
cd /mydata/adopt/christian/CoDiT
chmod +x runai_scripts/fig4_summarize.sh
bash -n runai_scripts/fig4_summarize.sh
./runai_scripts/fig4_summarize.sh 2>&1 | head -3
```

Expected: prints the `usage:` line and exits 1.

- [ ] **Step 3: Commit**

```bash
git add runai_scripts/fig4_summarize.sh
git commit -m "feat(runai): fig4_summarize.sh aggregates one dataset into CSV"
```

---

### Task 9: Full test suite + PR

**Files:** none — verification and PR only.

- [ ] **Step 1: Run the full test suite**

```bash
cd /mydata/adopt/christian/CoDiT
micromamba run -n codit pytest tests/ -v 2>&1 | tail -30
```

Expected: no regressions relative to `dev`; `test_summarize_fig4.py::test_summarize_fig4_writes_csv_with_expected_rows[open_ocean-model_dirs0]` passes.

- [ ] **Step 2: Verify no untracked or unintended changes**

```bash
git status
git log --oneline dev..HEAD
```

Expected commits on this branch:
1. `feat(eval): add --result_subdir to codit_eval …`
2. `feat(eval): add --result_subdir to lingauss_eval …`
3. `test: failing smoke test for summarize_fig4 …`
4. `feat: add summarize_fig4.py …`
5. `feat(runai): fig4_open_ocean.sh …`
6. `feat(runai): fig4_coastline.sh …`
7. `feat(runai): fig4_summarize.sh …`

No other file changes.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin feat/fig4-reproduction-scripts

gh pr create \
  --base dev \
  --title "feat: Fig 4 / Table 2 reproduction scripts (CoDiT + WDM)" \
  --body "$(cat <<'EOF'
## Summary

- Adds two RunAI launcher scripts per dataset (`fig4_open_ocean.sh`, `fig4_coastline.sh`), each parameterized by `<gru|tf|lingauss>` so six independent eval jobs can be submitted in parallel from the shipped trained checkpoints.
- Adds `fig4_summarize.sh <dataset>` and `my_scripts/model_fitting/summarize_fig4.py` that aggregate the three fresh eval outputs per dataset into `fig4_summary.csv` and print a side-by-side comparison with paper Table 2 values.
- Additive-only edit to `codit_eval.py` and `lingauss_eval.py`: new `--result_subdir` arg with default `""` so existing output paths remain byte-identical.

## No-overwrite guarantee

- Fresh eval outputs land at `data/results/<dataset>/<split>/fig4_repro/<model>/`; existing `data/results/<dataset>/<split>/<model>/*.npz` files are not touched.
- Trained checkpoints under `data/trained_models/` are only read.

## How to run

```
# submit six eval jobs (in parallel via runai submit)
./runai_scripts/fig4_open_ocean.sh gru
./runai_scripts/fig4_open_ocean.sh tf
./runai_scripts/fig4_open_ocean.sh lingauss
./runai_scripts/fig4_coastline.sh gru
./runai_scripts/fig4_coastline.sh tf
./runai_scripts/fig4_coastline.sh lingauss

# after each triple completes, aggregate
./runai_scripts/fig4_summarize.sh open_ocean
./runai_scripts/fig4_summarize.sh coastline
```

## Test plan

- [ ] `pytest tests/model_module/test_summarize_fig4.py -v` → 1 passed
- [ ] `pytest tests/` → no regressions
- [ ] Smoke run of one eval job on the cluster confirms output lands in the `fig4_repro/` subdir
EOF
)"
```

Expected: PR URL printed.

---

## Self-review notes

- **Spec coverage:**
  - `--result_subdir` on both eval scripts → Tasks 2 & 3.
  - `summarize_fig4.py` behavior (dir resolution, intersection, metrics, filter, CSV, stdout) → Tasks 4 & 5.
  - Shell scripts (open_ocean, coastline, summarize) → Tasks 6, 7, 8.
  - No-overwrite guarantee via default-`""` argparse arg → verified in Tasks 2 & 3.
  - Paper Table 2 values → included verbatim in `Global Constraints` and re-used in Task 5's script.
- **Placeholder scan:** no TBDs, no "add appropriate", no missing code.
- **Type consistency:** CSV columns `PE_new_km / ES_new_km / SS_new / …` used identically in Task 4 test and Task 5 script. Model labels `CoDiT (GRU)`, `CoDiT (TF)`, `WDM` used identically. `input_vars` string matches shipped checkpoint dir naming.
