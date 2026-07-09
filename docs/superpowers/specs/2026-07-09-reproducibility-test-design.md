# Reproducibility Verification Design

**Date:** 2026-07-09  
**Status:** Approved  
**Scope:** Two sequential plans — first reorganise the dataset layout and fix path loading, then add an integration test that verifies the eval pipeline produces numerically consistent results.

---

## Canonical data layout

The intended layout for both datasets is:

```
DATA_PATH/
  datasets/
    {dataset}/          e.g. open_ocean, coastline
      {split}/          train, val, test
        {name}/         e.g. open_ocean_patch_1_year_2021
          copernicus/
          era5/
          hycom/
          trajectories/
  configs/
    {dataset}/
      {split}/
        {name}.yml      stale absolute paths — load_config rewrites dataset_path at runtime
```

`load_config` reconstructs `dataset_path` as `DATA_PATH/datasets/{rel}/{name}` where
`rel = os.path.relpath(config_path, CONFIG_PATH)` (e.g. `open_ocean/test`). The absolute
`dataset_path` stored in the YAML is ignored at runtime and does not need to be updated.

---

## Current vs target layout

| Dataset | Current layout | Target layout |
|---------|---------------|---------------|
| open_ocean | `{split}/datasets/{name}/copernicus/...` | `{split}/{name}/copernicus/...` |
| coastline | `{split}/{name}/datasets/{name}/copernicus/...` | `{split}/{name}/copernicus/...` |

---

## Plan 1: Dataset reorganisation

### 1a. Fix `load_config` path reconstruction

`src/codit/data_module/config_funcs.py` lines 88–92 currently reconstruct `dataset_path`
with an incorrect extra `{name}/datasets/` nesting:

```python
# wrong — produces DATA_PATH/datasets/{rel}/{name}/datasets/{name}
config["dataset_path"] = os.path.join(
    config_paths.DATA_PATH, "datasets", rel,
    config["name"], "datasets", config["name"],
)
```

Fix:

```python
# correct — produces DATA_PATH/datasets/{rel}/{name}
config["dataset_path"] = os.path.join(
    config_paths.DATA_PATH, "datasets", rel, config["name"]
)
```

Also update the comment above to match:

```python
# data lives at: DATA_PATH/datasets/{dataset}/{split}/{name}/
```

### 1b. Smoke test before full reorganisation

Before moving all data, verify the fix works by constructing a minimal temp directory
tree with symlinks pointing to real data files (no copying), setting `CODIT_DATA_PATH`
to the temp dir, and confirming `OceanDataset` loads cleanly for one open_ocean patch
and one coastline patch.

### 1c. Full reorganisation

Move all patches on both datasets to the target layout using `mv` (same-filesystem
rename — fast, atomic per directory, no data duplication):

**open_ocean** (drop the split-level `datasets/` prefix):

```
{split}/datasets/{name}/  →  {split}/{name}/
```

```bash
for split in train val test; do
  base="$CODIT_DATA_PATH/datasets/open_ocean/$split"
  [ -d "$base/datasets" ] || continue
  for patch in "$base/datasets"/*/; do
    mv "$patch" "$base/$(basename "$patch")"
  done
  rmdir "$base/datasets"
done
```

**coastline** (collapse the inner `{name}/datasets/{name}` wrapper):

```
{split}/{name}/datasets/{name}/  →  {split}/{name}/
```

```bash
for split in train val test; do
  base="$CODIT_DATA_PATH/datasets/coastline/$split"
  for patch_dir in "$base"/*/; do
    name=$(basename "$patch_dir")
    inner="$patch_dir/datasets/$name"
    [ -d "$inner" ] || continue
    mv "$inner"/* "$patch_dir/"
    rmdir "$patch_dir/datasets/$name" "$patch_dir/datasets"
  done
done
```

### 1d. Verify and update tests

Run the existing test suite to confirm nothing regresses. Update `download_data.py`
comment to document the canonical layout.

---

## Plan 2: Reproducibility integration test

### Setup

Requires Plan 1 complete and `CODIT_DATA_PATH` pointing at the reorganised data.

### Test file: `tests/integration/test_reproducibility.py`

**Skip guards** — same pattern as `test_checkpoint_loading.py`:
- Skip if `CONFIG_PATH/open_ocean/test/open_ocean_patch_1_year_2021.yml` not found
- Skip LinearGauss test if `linear_gauss_open_ocean` checkpoint not found
- Skip CoDiT test if `codit_GRU_open_ocean` checkpoint not found

**Shared fixture** — module-scoped: loads `open_ocean_patch_1_year_2021` via
`config_funcs.load_config` and constructs `OceanDataset(config, length=72, ...)`.
Shared between both tests to avoid loading the NetCDF files twice.

**Test parameters:**

| Parameter | Value |
|-----------|-------|
| Patch | `open_ocean_patch_1_year_2021` |
| Trajectory indices | `[0, 50, 100]` |
| `length` | 72 |
| `num_samples` | 100 |
| Tolerance | ±10% of reference mean Haversine error |

**Reference values:**

- **LinearGauss:** `34 651 m` — recomputed from `pred_coords`/`true_coords` in the
  original `open_ocean_patch_1_year_2021_result.npz`. The shipped `linear_gauss_open_ocean`
  checkpoint is byte-identical to the one used to generate those results.
- **CoDiT-GRU:** established by running the shipped `codit_GRU_open_ocean` checkpoint once
  on the same 3 trajectories after Plan 1 is complete. Hardcoded as a module-level constant.
  Note: the shipped CoDiT checkpoint has different parameters from the paper checkpoint, so
  this verifies self-consistency of the shipped model, not exact paper numbers.

**Metric:** `numpy.nanmean(distances)` where `distances.shape == (3, 100, 72)`.

---

## What this does NOT cover

- Transformer variant (future work)
- Coastline dataset reproducibility test (future work)
- Exact match to paper Table 1 for CoDiT (requires shipping the original paper checkpoint)
- Full-dataset eval (covered by `reproduce_results.sh`)
