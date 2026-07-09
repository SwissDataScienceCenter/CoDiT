# CoDiT Architecture

## Overview

CoDiT is a **conditional denoising diffusion model** for probabilistic ocean drifter
trajectory prediction. It adapts the DDPM framework (Ho et al., 2020) from image synthesis
to 2-D trajectory sequences: instead of denoising images, CoDiT denoises displacement
increments conditioned on spatiotemporal ocean context.

## Prediction pipeline

```
Input:
  start_position  — (lat, lon) in decimal degrees
  start_time      — datetime.datetime (UTC)
  NetCDF files    — currents (HYCOM, Copernicus/GLORYS), ERA5 wind, GEBCO bathymetry

         ┌──────────────────────────────────┐
         │  OceanDataset.get_context()      │
         │  context_functions.py            │
         │  → tensor of shape (T × H × W × C)│
         └──────────────┬───────────────────┘
                        │  context tensor
                        ▼
         ┌──────────────────────────────────┐
         │  Context Encoder                 │
         │  (conv / mlp / gap)              │
         │  diffusion_models.py             │
         └──────────────┬───────────────────┘
                        │  context embedding
                        ▼
         ┌──────────────────────────────────┐
         │  Reverse Diffusion  (K steps)    │
         │  backward_denoising_step()       │
         │  diffusion_model.py              │
         │                                  │
         │  denoiser: GRU or Transformer    │
         │  (Flax/JAX)                      │
         └──────────────┬───────────────────┘
                        │  displacement increments  (T × 2), metres
                        ▼
         ┌──────────────────────────────────┐
         │  Coordinate transform            │
         │  data_transform.py               │
         └──────────────┬───────────────────┘
                        ▼
Output: trajectory samples — (num_samples × T × 2), decimal degrees
```

## Diffusion process

**Forward process** (training only): given a ground-truth trajectory-increment sequence
**y**₀, add Brownian-motion noise over K steps to obtain **y**_K ≈ 𝒩(0, I).

**Reverse process** (inference): start from **y**_K drawn from the Brownian-motion prior
and iteratively denoise back to **y**₀ using the learned denoiser, conditioned on the
ocean context embedding.

The noise schedule is linear in β_t from β_min = 10⁻⁴ to β_max = 0.02 over K = 100
steps by default. See `src/codit/codit_module/scheduler.py` and `noising_functions.py`.

## Source layout

```
src/codit/
├── codit_module/
│   ├── diffusion_models.py       # Flax modules: GRU/Transformer denoiser, context encoders
│   ├── noising_functions.py      # forward_diffusion(), forward_process()
│   ├── scheduler.py              # linear_schedule() → {beta, alpha, alpha_bar, K}
│   └── brownian_noise_models.py  # WalkNoiseModel: Brownian-motion noise sampler
│
├── data_module/
│   ├── dataset.py                # OceanDataset, OceanTrainingDataset
│   ├── context_functions.py      # extract_context(): currents + wind + bathymetry → tensor
│   ├── data_transform.py         # haversine(), transform_traj_to_metre(), coordinate utils
│   ├── config_paths.py           # DATA_PATH, MODEL_PATH, RESULT_PATH (from CODIT_DATA_PATH)
│   ├── config_funcs.py           # load_config(): reads per-patch YAML configs
│   └── dataset_download.py       # download utilities for ocean reanalysis products
│
├── model_module/
│   ├── base_model.py             # BaseTrajectoryModel ABC: fit(), predict(), export/import
│   ├── diffusion_model.py        # CoDiTModel          ← main model
│   ├── linear_gauss_model.py     # LinearGaussModel    ← main baseline
│   ├── currents_baseline_model.py # physics-only baseline (no learning)
│   ├── currents_parcels_model.py  # OceanParcels Lagrangian integration (optional dep)
│   ├── conv_net.py               # ConvNet regression baseline
│   ├── gru_model.py              # Autoregressive GRU without diffusion
│   └── direct_prediction_model.py # direct autoregressive baseline
│
└── eval_module/
    ├── metrics.py                # get_haversine_error(), get_energy_score(), get_skill_score()
    └── eval_runner.py            # run_eval(): shared evaluation loop for all models
```

## Data flow during training

```
OceanTrainingDataset (NetCDF files on disk)
    │  from_disk()          — lazy file list
    │  load_into_memory()   — materialise arrays in RAM
    │  shuffle() / split()  — train / val partition
    ▼
torch.utils.data.DataLoader (batching)
    ▼
CoDiTModel._fit()
    │  for each batch:
    │    forward_diffusion()     → noised trajectory y_K
    │    denoiser.apply(params)  → predicted noise ε_θ
    │    loss = MSE(ε_θ, ε_true)
    │    optax.adam update
    │  EarlyStopping on val loss
    ▼
Orbax checkpoint  →  data/trained_models/ckpts/<model_name>/
```

## Extending CoDiT

### Adding a new denoiser architecture

1. Implement a Flax `nn.Module` in `src/codit/codit_module/diffusion_models.py` with the
   same call signature as `Autoregressive_Denoiser`:
   `__call__(self, y, y_history, x, k, train=True)` → `jnp.ndarray` of shape `(T-1, 2)`.
2. Register it in `CoDiTModel.__init__` under a new `model_type` string.
3. Pass `--model_type your_type` to `my_scripts/model_fitting/codit_train.py`.

### Adding a new context variable

1. Add extraction logic to `src/codit/data_module/context_functions.py`.
2. Register the variable name in the dataset config YAML under `data/configs/`.
3. Pass the variable name via `--input_vars` when training.

## Baselines

| Model | Class | Description |
|---|---|---|
| Linear Gaussian | `LinearGaussModel` | Learns current weight α and Gaussian noise σ |
| ConvNet | `ConvNetModel` | CNN regression on context patch |
| GRU direct | `AutoregressiveModel` | Autoregressive GRU without diffusion |
| Currents | `CurrentsBaselineModel` | Advects drifter with raw current velocity |
| OceanParcels | `CurrentsParcelsModel` | Lagrangian integration via `parcels` library |

## Evaluation metrics

All models are evaluated with the same metrics (see `src/codit/eval_module/metrics.py`):

- **Haversine error** — great-circle distance between predicted and observed positions (km).
- **Energy Score** — multivariate probabilistic accuracy metric; lower is better.
- **Skill Score** — normalised improvement over the Linear Gaussian baseline; positive means better.
