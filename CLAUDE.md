# CoDiT: Conditional Diffusion Models for Trajectories

## Project overview

CoDiT is a conditional denoising diffusion framework for multi-day ocean drifter trajectory prediction. It adapts denoising diffusion (originally for image synthesis) to forecast the drift of floating objects on the ocean surface, conditioned on heterogeneous context data: ocean currents (HYCOM, Copernicus/GLORYS), winds (ERA5), bathymetry (GEBCO), and initial position. Ground truth comes from GPS trajectories in the Global Drifter Program.

The paper is in `paper/` (PDF, TeX, and markdown versions).

## Repository status

This repo is being cleaned up from a working/experimental repo (`../adopt/`) into a reproducible, publishable form. The principal development branch is `dev`. The paper is under re-revision, so Phase 1 is time-sensitive.

The full refactor plan with phased scope, affected files, and tech debt inventory is in **[`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md)**. Each phase is developed on its own clean branch off `dev`.

### Key goals

- Make results fully reproducible by others (Phase 1)
- Ship the two main datasets (open_ocean, coastline) as tar archives (Phase 1)
- Provide clean environment setup (Dockerfile / conda / mamba) (Phase 1)
- Fix bugs, add meaningful tests, code quality cleanup (Phase 2)
- CI/CD, training docs, extensibility (Phase 3)
- Keep the oceancleanup dataset/application private for now

## Architecture

### Source code (`src/codit/`)

- **`codit_module/`** - Core diffusion logic: noising functions, scheduler, diffusion models, Brownian noise models, direct (non-diffusion) models
- **`data_module/`** - Data loading, transforms, dataset construction, context feature extraction, download utilities
  - `config_paths.py` - **Hardcoded paths** (currently points to `../adopt/data_final/`). Must be updated when data moves into this repo.
  - `config_funcs.py` - YAML config loading
  - `dataset.py` - Dataset class
  - `context_functions.py` - Context feature extraction (currents, wind, bathymetry)
- **`model_module/`** - All model implementations:
  - `diffusion_model.py` - **CoDiT** (main model, GRU/Transformer variants)
  - `linear_gauss_model.py` - **Linear Gaussian** baseline (main baseline)
  - `conv_net.py` - ConvNet baseline
  - `direct_prediction_model.py` - Non-diffusion autoregressive model
  - `gru_model.py` - GRU-only model
  - `hycom_model.py`, `glorys_model.py` - Physics-based models
  - `hycom_parcels_model.py`, `glorys_parcels_model.py` - OceanParcels integration
  - `base_model.py` - Base class
- **`eval_module/`** - Evaluation metrics

### Scripts (`my_scripts/`)

- **`model_fitting/`** - Training and evaluation scripts (`codit_train.py`, `codit_eval.py`, `lingauss_train.py`, etc.)
- **`figures/`** - Plot generation scripts for paper figures

### Data (`data/` in this repo, currently mostly in `../adopt/data_final/`)

Current data location: `/mydata/adopt/christian/adopt/data_final/`

- **`datasets/`** - Two main datasets, each with `train/`, `val/`, `test/` splits:
  - `open_ocean/` (~65 GB, ~34k files) - NetCDF files with Copernicus currents, ERA5 wind, HYCOM currents, trajectories per spatial patch and year
  - `coastline/` (~36 GB, ~123k files) - Same structure for coastal regions
- **`configs/`** - YAML configs per patch (spatial bounds, dates, paths)
- **`models/trained_models/`** - Trained model checkpoints (small, ~1 MB each)
- **`results/`** - Evaluation outputs and summary CSVs
- **`bathymetry/`** - Static GEBCO bathymetry data (separate download)

### Other directories

- `notebooks/` - Tutorial and analysis notebooks (`get_started.ipynb` is the entry point)
- `runai_scripts/` - Shell scripts for RunAI cluster job submission
- `paper/` - The CoDiT paper (PDF, TeX, markdown)
- `plans/` - Implementation plan documents
- `.devcontainer/` - Docker setup (base image: `chdonner/msml`)

## Skills

When working on JAX-related code (models, transforms, PRNG, JIT, vmap), invoke the `/jax` skill (`.claude/skills/jax.md`) for patterns, critical rules, and anti-patterns specific to this project's stack.

## Tech stack

- **JAX / Flax** for model implementation (not PyTorch)
- **Python >= 3.8**, targeting 3.10+
- **xarray** for NetCDF data handling
- **NumPy, SciPy, Pandas** for data processing
- **Cartopy, Matplotlib** for geospatial plotting
- **WandB** for experiment tracking
- **micromamba** with `codit` environment (current setup), transitioning to reproducible env
- **Black, Ruff, Mypy** for code quality

## Running

### Environment

Currently: `micromamba run -n codit python <script>`. A Dockerfile exists in `.devcontainer/`. GPU support is optional (uncomment JAX CUDA line in Dockerfile). CPU works but is slow for training.

### Training (example)

```bash
python scripts/user/train/codit_train.py --dataset open_ocean
python scripts/user/train/lingauss_train.py --dataset open_ocean
```

### Evaluation (example)

```bash
python scripts/user/eval/codit_eval.py --dataset open_ocean
python scripts/user/eval/lingauss_eval.py --dataset open_ocean
```

## Data shipping

The datasets are large (101 GB total, ~157k NetCDF files). Strategy:
- **Tar archives per dataset per split** (e.g., `open_ocean_train.tar.gz`, `coastline_test.tar.gz`) to reduce file count for transfer
- Host on Zenodo (or similar) for a citable DOI
- Provide a download script that fetches and extracts into `data/datasets/`
- Git LFS for model checkpoints (small, ~1 MB each)
- Bathymetry is already hosted on Polybox

## Important notes

- `config.ini` contains API credentials - it is gitignored (`*.ini` pattern) but exists locally. Never commit credentials.
- `config_paths.py` is also gitignored. Each user must configure their own paths.
- The config YAMLs contain hardcoded absolute paths to dataset directories - these will need updating when data moves.
- When modifying model code: changes must be methodologically sound. Flag anything that looks like a correctness issue. Performance optimizations are welcome as long as qualitative behavior is preserved.
- Tests are currently minimal (`tests/model_module/`). Expanding test coverage is a priority.
