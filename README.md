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

## Cite

If you find this helpful, please cite:

> [Citation to be added upon publication]

## Acknowledgements

Global Drifter Program, HYCOM, Copernicus Marine Service, CDS, GEBCO,
Swiss Data Science Center (SDSC)
