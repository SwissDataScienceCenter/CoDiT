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

```bash
# Download both datasets and bathymetry (~101 GB total)
python scripts/download_data.py --dataset all --bathymetry
```

To download a single dataset:

```bash
python scripts/download_data.py --dataset open_ocean
python scripts/download_data.py --dataset coastline
```

Trained model checkpoints are already included in the repository under
`data/trained_models/`.

### 3. Reproduce paper results

```bash
bash scripts/reproduce_results.sh
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

For API credentials (only needed for downloading new ocean data, not for
reproducing results):

```bash
export CODIT_CREDENTIALS_PATH=/path/to/config.ini
```

The `config.ini` should contain Copernicus CDS and Marine Service credentials:

```ini
[copernicus]
user = <cds-email>
password = <cds-password>
url = https://cds.climate.copernicus.eu/api
key = <cds-key>

[copernicusmarine]
username = <username>
password = <password>
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

**Expected resources:** ~2 h on a single GPU (NVIDIA A100) for 100 epochs on `open_ocean`.
CPU training is supported but will be ~10× slower.

### Linear Gaussian baseline

```bash
python my_scripts/model_fitting/lingauss_train.py \
    --dataset open_ocean \
    --num_epochs 1000 \
    --num_batches_per_epoch 100 \
    --batch_size 128
```

Trains in under 10 minutes on CPU.

### WandB experiment tracking

CoDiT is set up to use [Weights & Biases](https://wandb.ai) for experiment tracking.
Before training, authenticate:

```bash
wandb login
```

Run names follow the convention:

```
codit_{model_type}_{dataset}_{context_encoder}_{optimizer}_K{K}_{vars}[_{run_name}]
```

Example: `codit_GRU_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry`

To disable WandB logging entirely:

```bash
export WANDB_MODE=disabled
```

## Cite

If you find this helpful, please cite:

> [Citation to be added upon publication]

## Acknowledgements

Global Drifter Program, HYCOM, Copernicus Marine Service, CDS, GEBCO,
Swiss Data Science Center (SDSC)

## Oceancleanup dataset (private)

The oceancleanup application dataset is not publicly distributed. It uses data provided
by [The Ocean Cleanup](https://theoceancleanup.com/) under a separate data agreement.

To use it, place the dataset under `$CODIT_DATA_PATH/datasets/oceancleanup/` following
the same directory structure as the public datasets. Evaluation scripts are in `oceancleanup/`.
