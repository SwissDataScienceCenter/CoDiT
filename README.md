# CoDiT: Conditional Diffusion Trajectory Predictions

CoDiT is a conditional denoising diffusion framework for multi-day ocean drifter
trajectory prediction. It forecasts the drift of floating objects on the ocean
surface, conditioned on ocean currents (HYCOM, Copernicus/GLORYS), winds (ERA5),
bathymetry (GEBCO), and initial GPS position. For more details check our [paper](https://iopscience.iop.org/article/10.1088/3049-4753/ae9bc5/meta).

## Quick Start

### 1. Set up the environment

**Option A — pip (recommended):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Option B — conda / mamba:**

```bash
conda create -n codit "python>=3.10"   # or: mamba create ...
conda activate codit
pip install -e .
```

**Option C — VS Code dev container:**

Open the repo in VS Code with the "Dev Containers" extension and choose
*Reopen in Container*. Builds from [`.devcontainer/Dockerfile`](.devcontainer/Dockerfile)
(base image `chdonner/msml`) and installs the package editable.

**GPU support (optional):**

Install after the base environment is set up (JAX ships CPU-only by default):

```bash
pip install --upgrade "jax[cuda12]"
```

**Hardware:** training on CPU works but is slow — a GPU is strongly
recommended. Evaluation of the shipped checkpoints runs on CPU. Plan for
~40 GB of disk after extracting both datasets and bathymetry.

### 2. Path configuration

Paths resolve from the `CODIT_DATA_PATH` environment variable. If unset,
defaults to `<repo_root>/data`.

```bash
export CODIT_DATA_PATH=/path/to/your/data
```

The on-disk layout expected under `CODIT_DATA_PATH` (datasets, configs,
trained_models, results, bathymetry) is documented in
[`docs/data_format.md`](docs/data_format.md).

### 3. Download data

The `open_ocean` and `coastline` datasets are hosted on Zenodo:
[10.5281/zenodo.21644091](https://doi.org/10.5281/zenodo.21644091).

```bash
# Download both datasets and bathymetry (~35 GB compressed)
python scripts/user/setup/download_data.py --dataset all --bathymetry
```

To download a single dataset:

```bash
python scripts/user/setup/download_data.py --dataset open_ocean
python scripts/user/setup/download_data.py --dataset coastline
```

### 4. Reproduce paper results

```bash
bash scripts/user/setup/reproduce_results.sh
```

This evaluates all models (Linear Gaussian, CoDiT-GRU, CoDiT-Transformer)
on both datasets using the `--evaluate_subset` flag for faster runs. Results
are saved as `.npz` files under `data/results/`.

To run a single evaluation:

```bash
python scripts/user/eval/codit_eval.py \
    --dataset open_ocean --split test \
    --model_name codit_GRU_open_ocean --evaluate_subset

python scripts/user/eval/lingauss_eval.py \
    --dataset coastline --split test
```

## Project Structure

```
CoDiT/
├── src/codit/                  # Source code
│   ├── codit_module/           # Diffusion logic, scheduler, noise models
│   ├── data_module/            # Data loading, transforms, context extraction
│   ├── model_module/           # CoDiT, Linear Gaussian, CNN, physics baselines
│   └── eval_module/            # Evaluation metrics
├── scripts/
│   ├── user/                   # User-facing entry points
│   │   ├── setup/              # download_data.py, reproduce_results.sh
│   │   ├── train/              # codit_train.py, lingauss_train.py
│   │   └── eval/               # codit_eval.py, lingauss_eval.py
│   └── dev/                    # Author-only scripts (baselines, figures, cluster, ...)
├── data/
│   ├── trained_models/         # Shipped model checkpoints (~4 MB total)
│   ├── datasets/               # Downloaded NetCDF datasets (not in git)
│   ├── configs/                # Downloaded YAML configs (not in git)
│   ├── ckpts/                  # Training scratch space (not in git)
│   └── bathymetry/             # Downloaded GEBCO data (not in git)
├── notebooks/
│   ├── user/                   # Tutorial notebooks
│   └── dev/                    # Author exploration notebooks
└── paper/                      # CoDiT paper (PDF, TeX)
```

## Training

### CoDiT (main model)

```bash
python scripts/user/train/codit_train.py \
    --dataset open_ocean \
    --model_type GRU \
    --context_encoder_type conv \
    --K 100 \
    --num_epochs 100 \
    --num_batches_per_epoch 100 \
    --batch_size 1024 \
    --learning_rate 1e-3 \
    --optimizer adam \
    --scheduler constant \
    --input_vars hycom_current copernicus_current era5_wind \
    --run_name my_experiment
```

The shipped `open_ocean` checkpoints were trained on the three variables shown
above (no bathymetry). The shipped `coastline` checkpoints add
`bathymetry` — the argparse default for `--input_vars` includes all four.

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `open_ocean` | `open_ocean` (72 h horizon) or `coastline` (24 h) |
| `--model_type` | `GRU` | Denoiser: `GRU` or `Transformer` |
| `--context_encoder_type` | `conv` | Context encoder: `conv`, `mlp`, or `gap` |
| `--K` | `100` | Number of diffusion steps |
| `--input_vars` | all four | Space-separated context variable names |
| `--optimizer` | `adam` | `adam`, `adamw`, or `sgd` |
| `--scheduler` | `constant` | LR schedule: `constant` or `cosine` |
| `--early_stopping_patience` | `10` | Epochs without val improvement before stopping |
| `--run_name` | `""` | Optional suffix appended to the model checkpoint name |

During training, intermediate checkpoints are written under
`data/ckpts/<model_name>/`; on completion, the final model is exported to
`data/trained_models/<model_name>/` (which is where `import_trained_model()`
and the eval scripts look for it).

### Linear Gaussian baseline

```bash
python scripts/user/train/lingauss_train.py \
    --dataset open_ocean \
    --num_epochs 1000 \
    --num_batches_per_epoch 100 \
    --batch_size 128
```

### WandB experiment tracking

CoDiT is set up to use [Weights & Biases](https://wandb.ai) for experiment tracking.
Before training, authenticate:

```bash
wandb login
```

To disable WandB logging entirely:

```bash
export WANDB_MODE=disabled
```

## Notebooks

CoDiT supports four data sources: **HYCOM currents, Copernicus/GLORYS currents,
ERA5 winds, and GEBCO bathymetry**. Users needing other context features (e.g.
sea-surface temperature, salinity) must adapt code — see the last section of
[`docs/data_format.md`](docs/data_format.md).

Three notebooks walk through the workflow:

- [`notebooks/user/01_predict_quickstart.ipynb`](notebooks/user/01_predict_quickstart.ipynb) —
  loads the shipped `codit_GRU_open_ocean` checkpoint and runs a 72 h forecast on a
  small downloaded region. Start here to see predictions end-to-end.
  *Requires `config.ini` (see [Credentials](#credentials-for-fresh-region-downloads)).*
- [`notebooks/user/02_train_quickstart.ipynb`](notebooks/user/02_train_quickstart.ipynb) — trains a
  small CoDiT-GRU on a subset of the shipped `open_ocean` data (≈10 min on CPU).
- [`notebooks/user/03_prepare_your_own_region.ipynb`](notebooks/user/03_prepare_your_own_region.ipynb) —
  full pipeline for a new region: pick lat/lon + date range, download the four
  sources, preprocess into a merged h5, train, evaluate.
  *Requires `config.ini` (see [Credentials](#credentials-for-fresh-region-downloads)).*

The processed-data schema and the 8-channel context layout are documented in
[`docs/data_format.md`](docs/data_format.md).

### Credentials for fresh-region downloads

The Zenodo download in Quick Start and `reproduce_results.sh` need
**no credentials**. You only need `config.ini` for notebooks 01 and 03,
which download fresh data for a new region.

Two independent services are involved:

- **Copernicus Climate Data Store (CDS)** — supplies ERA5 wind. Register at
  <https://cds.climate.copernicus.eu/>, then copy your personal API key from
  <https://cds.climate.copernicus.eu/how-to-api>.
- **Copernicus Marine Service (CMS)** — supplies GLORYS ocean currents.
  Register at <https://data.marine.copernicus.eu/register>.

HYCOM currents and NOAA Global Drifter Program trajectories are pulled from
public endpoints and need no credentials.

Copy the example file at the repo root and fill in your values:

```bash
cp config.ini.example config.ini
```

`config.ini` layout:

```ini
[copernicus]              # CDS — ERA5 wind
url = https://cds.climate.copernicus.eu/api
key = <your-CDS-API-key>

[copernicusmarine]        # CMS — GLORYS currents
username = <your-CMS-username>
password = <your-CMS-password>
```

By default the code looks for `config.ini` at the repository root. To keep
credentials outside the checkout, set `CODIT_CREDENTIALS_PATH` to an
absolute path:

```bash
export CODIT_CREDENTIALS_PATH=/path/to/your/config.ini
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CODIT_DATA_PATH` | `<repo_root>/data` | Root for datasets, configs, trained_models, results, bathymetry |
| `CODIT_CREDENTIALS_PATH` | `<repo_root>/config.ini` | Location of the credentials file for fresh-region downloads |
| `WANDB_MODE` | *(unset)* | Set to `disabled` to skip WandB logging during training |

## Troubleshooting

- **`config.ini` not found / `KeyError: 'copernicus'`** — the fresh-region
  downloads need credentials. Copy `config.ini.example` to `config.ini`
  at the repo root (or point `CODIT_CREDENTIALS_PATH` at your file) and fill
  in your CDS and CMS values. Not needed for the Zenodo download or
  `reproduce_results.sh`.
- **JAX still on CPU after installing `jax[cuda12]`** — reinstall JAX
  *after* `pip install -e .`. The editable install pulls the CPU wheel and
  will shadow the CUDA one if the order is reversed.

## Cite

If you find this helpful, please cite:

```latex
@article{Donner_2026,
doi = {10.1088/3049-4753/ae9bc5},
year = {2026},
publisher = {IOP Publishing},
volume = {2},
number = {2},
author = {Donner, Christian and Goshtasbpour, Shirin and Dalsasso, Emanuele and Romero, Mattia and Béchaz, Manon and Rußwurm, Marc and de Vries, Robin and Tuia, Devis and Volpi, Michele},
title = {CoDiT: conditional diffusion models for multi-day ocean drifter trajectory prediction},
journal = {Machine Learning: Earth},
}
```

The datasets are archived on Zenodo:
[10.5281/zenodo.21644091](https://doi.org/10.5281/zenodo.21644091).

## Acknowledgements

Global Drifter Program, HYCOM, Copernicus Marine Service, CDS, GEBCO,
Swiss Data Science Center (SDSC)

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for the full text.
