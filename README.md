# CoDiT: Conditional Diffusion Trajectory Predictions

CoDiT is a conditional denoising diffusion framework for multi-day ocean drifter
trajectory prediction. It forecasts the drift of floating objects on the ocean
surface, conditioned on ocean currents (HYCOM, Copernicus/GLORYS), winds (ERA5),
bathymetry (GEBCO), and initial GPS position.

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
conda env create -f environment.yml   # or: mamba env create -f environment.yml
conda activate codit
```

**GPU support (optional):**

```bash
pip install --upgrade "jax[cuda12]"
```

### 2. Download data

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

### 3. Reproduce paper results

```bash
bash scripts/user/setup/reproduce_results.sh
```

This evaluates all models (Linear Gaussian, CoDiT-GRU, CoDiT-Transformer)
on both datasets using the `--evaluate_subset` flag for faster runs. Results
are saved as `.npz` files under `data/results/`.

To run a single evaluation:

```bash
python my_scripts/model_fitting/codit_eval.py \
    --dataset open_ocean --split test \
    --model_name codit_GRU_open_ocean --evaluate_subset

python my_scripts/model_fitting/lingauss_eval.py \
    --dataset coastline --split test
```

### 4. Path configuration

Paths resolve from the `CODIT_DATA_PATH` environment variable. If unset,
defaults to `<repo_root>/data`.

```bash
export CODIT_DATA_PATH=/path/to/your/data
```

## Project Structure

```
CoDiT/
├── src/codit/                  # Source code
│   ├── codit_module/           # Diffusion logic, scheduler, noise models
│   ├── data_module/            # Data loading, transforms, context extraction
│   ├── model_module/           # CoDiT, Linear Gaussian, CNN, physics baselines
│   └── eval_module/            # Evaluation metrics
├── my_scripts/model_fitting/   # Training and evaluation scripts
├── scripts/                    # Download and reproduction scripts
├── data/
│   ├── trained_models/         # Shipped model checkpoints (~4 MB total)
│   ├── datasets/               # Downloaded NetCDF datasets (not in git)
│   ├── configs/                # Downloaded YAML configs (not in git)
│   └── bathymetry/             # Downloaded GEBCO data (not in git)
├── notebooks/                  # Tutorial notebooks
└── paper/                      # CoDiT paper (PDF, TeX)
```

## Training

### CoDiT (main model)

```bash
python my_scripts/model_fitting/codit_train.py \
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
    --input_vars hycom_current copernicus_current era5_wind bathymetry \
    --run_name my_experiment
```

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

The trained model is saved to `data/trained_models/ckpts/<model_name>/`.

### Linear Gaussian baseline

```bash
python my_scripts/model_fitting/lingauss_train.py \
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

## Train on your own region

CoDiT supports four data sources: **HYCOM currents, Copernicus/GLORYS currents,
ERA5 winds, and GEBCO bathymetry**. Users needing other context features (e.g.
sea-surface temperature, salinity) must adapt code — see the last section of
[`docs/data_format.md`](docs/data_format.md).

Two notebooks walk through the workflow:

- [`notebooks/01_train_quickstart.ipynb`](notebooks/01_train_quickstart.ipynb) — trains a
  small CoDiT-GRU on a subset of the shipped `open_ocean` data (≈10 min on CPU).
  Start here if you want to see the training loop end-to-end.
- [`notebooks/02_prepare_your_own_region.ipynb`](notebooks/02_prepare_your_own_region.ipynb) —
  full pipeline for a new region: pick lat/lon + date range, download the four
  sources, preprocess into a merged h5, train, evaluate.

The processed-data schema and the 8-channel context layout are documented in
[`docs/data_format.md`](docs/data_format.md).

## Cite

If you find this helpful, please cite:

> [Citation to be added upon publication]

The datasets are archived on Zenodo:
[10.5281/zenodo.21644091](https://doi.org/10.5281/zenodo.21644091).

## Acknowledgements

Global Drifter Program, HYCOM, Copernicus Marine Service, CDS, GEBCO,
Swiss Data Science Center (SDSC)
