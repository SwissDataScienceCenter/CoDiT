# Figure 4 / Table 2 Reproduction Scripts — Design

**Date:** 2026-07-10
**Branch:** `feat/fig4-reproduction-scripts` (off `dev`)

## Goal

Provide two runai-launchable shell scripts (one per dataset) that regenerate the
CoDiT (GRU + TF) and WDM (LinearGauss) rows of the paper's **Table 2 / Figure
4C+D** from the shipped trained checkpoints, then aggregate the per-patch
`.npz` outputs into a single CSV that compares fresh median metrics at the
final horizon against the paper's published values.

## Non-goals

- Does **not** regenerate HYCOM, GLORYS, CNN, or Naive baselines.
- Does **not** produce the Figure 4A/B example trajectories or the Figure
  4C/D curves — only the final-horizon numbers.
- Does **not** modify existing `.npz` result files at
  `data/results/<dataset>/test/<model>/`. Fresh results are written to a
  parallel subdirectory (see "No-overwrite constraint" below).

## Constraints

- **No existing files may be overwritten.** Fresh evals write into a new
  subdirectory, `data/results/<dataset>/test/fig4_repro/<model>/`, alongside
  (not replacing) the existing `data/results/<dataset>/test/<model>/` files.
- **No skipping.** All patches are re-evaluated from scratch on every run,
  so `codit_eval.py`'s existing skip-if-exists check must be effectively
  disabled in the repro subdir (achieved automatically: nothing exists there
  on first run; if the user re-runs, the destination should be cleared or the
  subdir renamed — see runbook note in the plan).
- Runs must launch on the RunAI cluster using the same `micromamba activate
  codit && cd my_scripts/model_fitting && python …` pattern as the existing
  `runai_scripts/*.sh`.
- No modifications to trained model checkpoints or their config files.

## Design

### 1. Additive edit: `--result_subdir` flag

Both eval entry points get one new argparse argument, with default `""` so
existing behavior is unchanged:

```python
parser.add_argument(
    "--result_subdir",
    type=str,
    default="",
    help="Optional subdirectory inserted between the split and model name in the result path",
)
...
result_path = os.path.join(
    config_paths.RESULT_PATH, dataset, args.split, args.result_subdir, model_name
)
```

`os.path.join("a", "b", "", "c")` yields `"a/b/c"`, so the default value keeps
paths byte-identical to today's behavior — no existing caller is affected.

**Files touched:**

- `my_scripts/model_fitting/codit_eval.py` — one new arg, one line of
  `os.path.join` change.
- `my_scripts/model_fitting/lingauss_eval.py` — same, symmetric change.

### 2. New: `my_scripts/model_fitting/summarize_fig4.py`

Purpose: read the three fresh model result dirs, compute Table 2 metrics,
compare against the paper.

**CLI:**

```
python summarize_fig4.py --dataset {open_ocean,coastline} \
                         --split test \
                         --result_subdir fig4_repro
```

**Behavior:**

1. Resolve the three model dirs under
   `<RESULT_PATH>/<dataset>/<split>/<result_subdir>/`:
   - `codit_GRU_<dataset>_conv_adam_K100_<input_var_str>`
   - `codit_Transformer_<dataset>_conv_adam_K100_<input_var_str>`
   - `linear_gauss`

   where `input_var_str` is `hycom_current_copernicus_current_era5_wind` for
   `open_ocean` and appends `_bathymetry` for `coastline` — matching the
   shipped checkpoint names and the paper's §6.2/§6.3 methodology.

2. Take the intersection of `<data_name>_result.npz` filenames across all
   three dirs (a patch is dropped if any model failed on it).

3. For each patch and each model, load `true_coords`, `pred_coords`, call
   `codit.eval_module.metrics.get_metrics(true_coords, pred_coords)`, and
   concatenate the resulting `(n_traj, T-1)` arrays over patches.

4. Build a validity mask:

   ```python
   valid = ~any_nan_across_models(mean_distance) & \
           np.any(naive_mean_distance > 100.0, axis=1)   # metres
   ```

   where `naive_mean_distance` is computed from any model's `true_coords` as
   the haversine distance between the tiled start position and each true
   position (the "stuck buoy" filter from `result_plots.py`; threshold is 100
   metres, since `haversine_distance` returns metres).

5. Compute median across `valid` trajectories, at the final horizon column
   (`[:, -1]`), for `mean_distance`, `energy_score`, `skill_score`. Divide
   `mean_distance` and `energy_score` by 1000 to report km.

6. Compare against hardcoded paper values (Table 2):

   | Dataset     | Model       | PE ↓ | ES ↓ | SS ↑ |
   |-------------|-------------|------|------|------|
   | open_ocean  | CoDiT (GRU) | 26.135 | 18.092 | 0.466 |
   | open_ocean  | CoDiT (TF)  | 26.418 | 18.154 | 0.460 |
   | open_ocean  | WDM         | 27.196 | 18.666 | 0.406 |
   | coastline   | CoDiT (GRU) | 8.638  | 5.746  | 0.268 |
   | coastline   | CoDiT (TF)  | 8.032  | 5.339  | 0.305 |
   | coastline   | WDM         | 9.651  | 7.842  | 0.167 |

7. Print a table to stdout with columns
   `model, PE_new, PE_paper, ΔPE, ES_new, ES_paper, ΔES, SS_new, SS_paper, ΔSS`.

8. Write the same table to
   `data/results/<dataset>/<split>/<result_subdir>/fig4_summary.csv`. This
   path is inside the same new subdir as the raw `.npz` outputs, so no
   existing file is overwritten.

### 3. New shell scripts

Each dataset gets one eval-launcher script parameterized by model, so the
user can submit **6 independent RunAI jobs in parallel** (2 datasets × 3
models). A seventh (per-dataset) summary script is launched after the three
eval jobs for a dataset finish.

**`runai_scripts/fig4_open_ocean.sh <model> [split] [subdir]`**

Positional args:
- `<model>` — one of `gru`, `tf`, `lingauss`
- `[split]` — default `test`
- `[subdir]` — default `fig4_repro`

```bash
#!/bin/bash
set -euo pipefail
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

**`runai_scripts/fig4_coastline.sh <model> [split] [subdir]`** — identical
structure, `--dataset coastline` and
`input_vars="hycom_current copernicus_current era5_wind bathymetry"`.

**`runai_scripts/fig4_summarize.sh <dataset> [split] [subdir]`** — runs
`summarize_fig4.py` for one dataset. Launched separately after the three
eval jobs for that dataset finish:

```bash
#!/bin/bash
set -euo pipefail
dataset=$1
split=${2:-test}
subdir=${3:-fig4_repro}

source ~/.bashrc
micromamba activate codit
cd /mydata/adopt/christian/CoDiT/my_scripts/model_fitting

python summarize_fig4.py --dataset "$dataset" --split "$split" \
  --result_subdir "$subdir"
```

Total: **6 eval jobs** (parallelizable) + **2 summary jobs** (one per
dataset, launched after the corresponding evals complete).

## No-overwrite guarantee

- Existing `data/results/<dataset>/test/<model>/*.npz` is never touched
  because the new `--result_subdir fig4_repro` inserts a directory between
  the split and the model name.
- The summary CSV lives at
  `data/results/<dataset>/test/fig4_repro/fig4_summary.csv` — a new path.
- Trained checkpoints under `data/trained_models/` are only read.
- Shell scripts are new files; existing runai scripts unchanged.
- Only edits to existing files: two additive argparse args in
  `codit_eval.py` and `lingauss_eval.py`, both with a default that
  preserves current output paths byte-for-byte.

## Testing

- **Unit test (optional but recommended):** in
  `tests/model_module/test_summarize_fig4.py`, feed
  `summarize_fig4.py` a temp dir with synthetic `.npz` files and assert the
  produced CSV matches the expected medians for that fixture.
- **Smoke run:** launch `fig4_open_ocean.sh` on the cluster; expect fresh
  `.npz` files under `data/results/open_ocean/test/fig4_repro/…` and a CSV
  showing new numbers within a few percent of the paper values for CoDiT
  (GRU/TF) and WDM.

## Open risks

- If the RunAI job is re-launched without clearing
  `data/results/<dataset>/test/fig4_repro/`, `codit_eval.py`'s existing
  skip-if-exists check will skip patches that were already computed.
  Mitigation: document in the runbook that fresh reruns require
  `rm -rf data/results/<dataset>/test/fig4_repro/` first. Not a code
  change.
- Numerical determinism depends on JAX PRNG seeding inside `run_eval` (the
  eval scripts set `np.random.seed(0)` but the JAX key derivation should be
  verified). Small deviations (< 1%) from paper values are acceptable given
  the goal of qualitative reproduction.
