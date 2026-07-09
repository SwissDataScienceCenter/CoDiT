# Reproducibility Test Design

**Date:** 2026-07-09  
**Status:** Approved  
**Scope:** Verify that the refactored eval pipeline produces numerically consistent results for LinearGaussModel and CoDiT-GRU on `open_ocean`, checked against reference values from the original eval outputs.

---

## Goal

Run the evaluation pipeline on a small subset (3 trajectories from one patch) for both shipped models and assert that the mean Haversine error is within ±10% of a pre-established reference value. Confirms the refactored code path (`load_config` → `OceanDataset` → `run_eval`) is end-to-end correct.

---

## Intended data layout (canonical, for tar archives)

The canonical layout — already used by coastline — is:

```
DATA_PATH/
  datasets/
    {dataset}/          e.g. open_ocean, coastline
      {split}/          train, val, test
        {name}/         e.g. open_ocean_patch_1_year_2021
          datasets/
            {name}/     contains copernicus/, era5/, hycom/, trajectories/
  configs/
    {dataset}/
      {split}/
        {name}.yml      contains stale absolute paths — load_config rewrites dataset_path
```

`load_config` reconstructs `dataset_path` as `DATA_PATH/datasets/{rel}/{name}/datasets/{name}` so it is independent of the absolute paths stored in the YAML. The YAML files intentionally carry stale absolute paths from the original machine; they are overwritten at load time and can be ignored when authoring tarballs.

**When creating the tar archives**, each tarball must extract into this layout. The config YAML `dataset_path` field will be ignored at runtime, so its value does not need to be updated in the archive.

---

## Prerequisite: `load_config` is correct — no changes needed

The existing `load_config` reconstruction (`{rel}/{name}/datasets/{name}`) already matches the canonical layout above. No code change is required.

---

## Prerequisite: Symlinks for existing open_ocean data

The current `open_ocean` data at `data_final/` has an inconsistent layout — there is an extra `datasets/` directory at the split level instead of the patch level:

```
data_final/datasets/open_ocean/test/datasets/{name}/hycom/...   ← actual (wrong layout)
data_final/datasets/open_ocean/test/{name}/datasets/{name}/...  ← expected (canonical)
```

The coastline data already has the correct layout.

**Workaround (local only):** for each open_ocean patch, create a real directory `{name}/` and a symlink `{name}/datasets` → `../datasets` so that `{name}/datasets/{name}/` resolves to the actual data:

```bash
for split in train val test; do
  base="$CODIT_DATA_PATH/datasets/open_ocean/$split"
  [ -d "$base/datasets" ] || continue
  for patch in "$base/datasets"/*/; do
    name=$(basename "$patch")
    target="$base/$name"
    [ -e "$target" ] && continue
    mkdir -p "$target"
    ln -s ../datasets "$target/datasets"
  done
done
```

After this, `$base/{name}/datasets/{name}/hycom/...` resolves correctly.

**When creating the open_ocean tar archive**, reorganise the data to match the canonical layout (drop the extra `datasets/` level at the split level) so symlinks are no longer needed.

---

## New file: `tests/integration/test_reproducibility.py`

### Skip guards

Uses the same `pytest.mark.skipif` pattern as `test_checkpoint_loading.py`:

- Skip if `CONFIG_PATH/open_ocean/test/open_ocean_patch_1_year_2021.yml` not found
- Skip `test_linear_gauss_*` if `linear_gauss_open_ocean` checkpoint not found
- Skip `test_codit_gru_*` if `codit_GRU_open_ocean` checkpoint not found

### Shared fixture

A module-scoped pytest fixture loads `open_ocean_patch_1_year_2021` via `config_funcs.load_config` and constructs an `OceanDataset(config, length=72, ...)`. Shared between both tests to avoid loading the 1 GB NetCDF twice.

### Test parameters

| Parameter | Value |
|-----------|-------|
| Patch | `open_ocean_patch_1_year_2021` |
| Trajectory indices | `[0, 50, 100]` |
| `length` | 72 |
| `num_samples` | 100 |
| Tolerance | ±10% of reference mean Haversine error |

### Reference values

**LinearGauss:** `34 651 m`  
Source: recomputed from `pred_coords` / `true_coords` in `open_ocean_patch_1_year_2021_result.npz` from the original results directory. The shipped `linear_gauss_open_ocean` checkpoint is byte-identical to the one used to generate those results.

**CoDiT-GRU:** TBD — established by running the shipped `codit_GRU_open_ocean` checkpoint once on the same 3 trajectories after the `load_config` fix is applied. The value is then hardcoded as a module-level constant in the test file.

Note: the shipped CoDiT checkpoint has different parameters from the one used in the original results directory, so the CoDiT reference does not verify paper numbers — it verifies self-consistency of the shipped checkpoint.

### Metric

`numpy.nanmean(distances)` where `distances` has shape `(3, 100, 72)` — mean over all trajectories, samples, and time steps.

### Test structure (pseudocode)

```python
LGAUSS_REF_M = 34_651.0
CODIT_REF_M  = <value from one-time run>
TOLERANCE    = 0.10

def test_linear_gauss_mean_error(ocean_dataset):
    model = LinearGaussModel.import_trained_model("linear_gauss_open_ocean", ...)
    distances, _, _ = run_eval(model, ocean_dataset, [0, 50, 100], 72, 100)
    mean_err = float(np.nanmean(distances))
    assert abs(mean_err - LGAUSS_REF_M) / LGAUSS_REF_M < TOLERANCE

def test_codit_gru_mean_error(ocean_dataset):
    model = CoDiTModel.import_trained_model("codit_GRU_open_ocean", ...)
    distances, _, _ = run_eval(model, ocean_dataset, [0, 50, 100], 72, 100)
    mean_err = float(np.nanmean(distances))
    assert abs(mean_err - CODIT_REF_M) / CODIT_REF_M < TOLERANCE
```

---

## What this does NOT cover

- Transformer variant (future work)
- Coastline dataset (future work)
- Exact match to paper Table 1 for CoDiT (requires shipping the original paper checkpoint)
- Full-dataset eval (covered by `reproduce_results.sh`)
