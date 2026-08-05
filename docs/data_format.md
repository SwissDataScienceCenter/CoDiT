# CoDiT data format

This document describes the processed-data layout used by every CoDiT training
and evaluation script. It complements the two notebooks under `notebooks/`.

## Layers of the pipeline

Raw NetCDFs (per patch, per data source)
  → per-patch training h5 (`training_data_<length>h.h5` under the patch's
    `dataset_path`) built by `OceanDataset` + `OceanTrainingDataset.from_dataset`
  → merged training h5 (`processed/training_data_<length>h.h5`) written by
    `preprocess_configs` after concatenating and NaN-stripping every patch

`codit_train.py` and `codit_eval.py` load the merged h5 via
`OceanTrainingDataset.from_disk`.

## Merged training h5

Each `processed/training_data_<L>h.h5` contains two datasets:

- `trajectories`: `(N, L+1, 2)` — lat/lon at hourly steps, one row per snippet
- `context`: `(N, L+1, G, G, 8)` — the 8-channel context tensor at each step

Here `N` is the number of trajectory snippets, `L` is `--prediction_length` in
hours, and `G` is `--num_grid_points`. Each snippet spans `L+1` hourly
observations (start position + `L` future positions).

## Context tensor: 8-channel layout

The channel order is frozen at
[`src/codit/model_module/base_model.py:83-90`](../src/codit/model_module/base_model.py):

| Index | Channel               | Source     |
|-------|-----------------------|------------|
| 0     | HYCOM u-current       | HYCOM      |
| 1     | HYCOM v-current       | HYCOM      |
| 2     | Copernicus u-current  | Copernicus |
| 3     | Copernicus v-current  | Copernicus |
| 4     | ERA5 u-wind           | ERA5       |
| 5     | ERA5 v-wind           | ERA5       |
| 6     | Bathymetry (elevation)| GEBCO      |
| 7     | Bathymetry (land=1)   | derived    |

Channel 7 is a binary land/ocean mask built from channel 6 in
[`src/codit/data_module/dataset.py:175-176`](../src/codit/data_module/dataset.py):

```python
bin_bathy = jnp.asarray(vals[..., -1] > 0, dtype=float)
vals = jnp.concatenate([vals, bin_bathy[..., None]], axis=-1)
```

`--input_vars` on `codit_train.py`/`codit_eval.py` selects sets of two channels
each (`hycom_current` → 0,1; `copernicus_current` → 2,3; `era5_wind` → 4,5;
`bathymetry` → 6,7). No mechanism exists for selecting a single channel from a
pair.

## Grid geometry: `num_grid_points` and `delta_metres`

At every trajectory step a `G × G` context tile is centred on the drifter, with
side length `2 × delta_metres` (metres). `delta_metres=10000` and
`num_grid_points=16` are the shipped values — the model was trained with these
and expects them at eval time. Changing them silently mismatches the training
distribution; retrain the model if you deviate.

## Config YAML schema

Per-patch YAMLs live under `configs/<dataset>/<split>/*.yml`. Required keys:

| Key              | Type        | Meaning                                            |
|------------------|-------------|----------------------------------------------------|
| `name`           | str         | Patch identifier (used for file names)             |
| `latitudes`      | [south, north] | Bounding-box latitudes (degrees)                |
| `longitudes`     | [west, east]   | Bounding-box longitudes (degrees, ±180 convention)|
| `start_date`     | ISO datetime | Inclusive start                                    |
| `end_date`       | ISO datetime | Inclusive end                                      |
| `context_delta`  | float        | Extra degrees added around the bbox for context    |
| `description`    | str          | Free-form (may be empty)                           |

`dataset_path`, `trajectories_file`, `hycom_file`, `era5_file`,
`copernicus_hourly_file`, `copernicus_daily_file` are **reconstructed by
`config_funcs.load_config`** from `CODIT_DATA_PATH`; do not commit them into
the YAML.

## Directory layout

```
$CODIT_DATA_PATH/
├── configs/
│   ├── open_ocean/{train,val,test}/*.yml
│   ├── coastline/{train,val,test}/*.yml
│   └── <your_dataset>/{train,val,test}/*.yml
├── datasets/
│   ├── open_ocean/datasets/processed/training_data_72h.h5   # shipped
│   ├── coastline/datasets/processed/training_data_24h.h5    # shipped
│   └── <your_dataset>/<split>/processed/training_data_<L>h.h5
├── trained_models/{ckpts,configs,params}/
├── results/<dataset>/<split>/<model_name>/*.npz
└── bathymetry/GEBCO_06_Jan_2025_.../*.nc
```

Note the historical `datasets/processed/` nesting used by the shipped
`open_ocean` and `coastline` datasets; the CLI in
`scripts/precompute_trainset.py` writes new datasets to
`<dataset>/<split>/processed/` instead.

## Adding a new context feature

Phase 4 does **not** provide a config-driven registry. To add a new context
channel (e.g. sea-surface temperature), you must edit:

- `src/codit/data_module/context_functions.py:76-230` — extend `load_data`
  with the new source and append it to the returned xarray Dataset.
- `src/codit/data_module/dataset.py:175-176` — decide whether the new channel
  needs a derived binary companion.
- `src/codit/model_module/base_model.py:83-90` — extend `var_idx` and the
  `input_vars` mapping.

Every model that reads `var_idx` (`diffusion_model.py`, `linear_gauss_model.py`,
`conv_net.py`, `direct_prediction_model.py`) must be re-checked to confirm it
tolerates the new channel count. This is Phase 5 work.

## Cross-references

- [`notebooks/01_train_quickstart.ipynb`](../notebooks/01_train_quickstart.ipynb) — end-to-end training on shipped `open_ocean` subset
- [`notebooks/02_prepare_your_own_region.ipynb`](../notebooks/02_prepare_your_own_region.ipynb) — full pipeline for a new region
- [`README.md`](../README.md#train-on-your-own-region) — "Train on your own region" section
