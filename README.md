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

### 4. Reproduce Table 2 / Figure 4 numbers

Table 2 in the paper reports final-horizon (72 h open_ocean, 24 h coastline)
median Position Error (PE), Energy Score (ES), and Skill Score (SS) for
CoDiT-GRU, CoDiT-Transformer, and the WDM (Linear Gaussian) baseline. To
reproduce these numbers from the shipped checkpoints, use the launcher
scripts in `runai_scripts/`:

- `fig4_open_ocean.sh <gru|tf|lingauss>` — evaluates one model on open_ocean.
- `fig4_coastline.sh <gru|tf|lingauss>` — evaluates one model on coastline.
- `fig4_summarize.sh <dataset>` — aggregates the three models' outputs into
  `fig4_summary.csv` alongside the published paper values.

Eval outputs are written under `data/results/<dataset>/test/fig4_repro/`,
so nothing under `data/results/<dataset>/test/` is overwritten.

**Recommended: 6 evals in parallel, then 2 summaries.**

```bash
# 6 eval jobs (submit in parallel on the cluster)
runai_scripts/fig4_open_ocean.sh gru
runai_scripts/fig4_open_ocean.sh tf
runai_scripts/fig4_open_ocean.sh lingauss
runai_scripts/fig4_coastline.sh  gru
runai_scripts/fig4_coastline.sh  tf
runai_scripts/fig4_coastline.sh  lingauss

# After the three evals for a dataset finish, aggregate:
runai_scripts/fig4_summarize.sh open_ocean
runai_scripts/fig4_summarize.sh coastline
```

`fig4_summarize.sh` prints and writes a comparison table with columns
`PE_new_km / PE_paper_km / dPE_km` (and analogous for ES/SS) to
`data/results/<dataset>/test/fig4_repro/fig4_summary.csv`.

To rerun from scratch, delete `data/results/<dataset>/test/fig4_repro/`
before relaunching — otherwise `codit_eval.py` skips patches whose `.npz`
files already exist.

### 5. Path configuration

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

The datasets are archived on Zenodo:
[10.5281/zenodo.21644091](https://doi.org/10.5281/zenodo.21644091).

## Acknowledgements

Global Drifter Program, HYCOM, Copernicus Marine Service, CDS, GEBCO,
Swiss Data Science Center (SDSC)

## Oceancleanup dataset (private)

The oceancleanup application dataset is not publicly distributed. It uses data provided
by [The Ocean Cleanup](https://theoceancleanup.com/) under a separate data agreement.

To use it, place the dataset under `$CODIT_DATA_PATH/datasets/oceancleanup/` following
the same directory structure as the public datasets. Evaluation scripts are in `oceancleanup/`.
