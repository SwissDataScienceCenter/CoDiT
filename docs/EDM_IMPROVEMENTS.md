# Diffusion improvements suggested by EDM

Notes from reviewing the CoDiT diffusion code against Karras et al., *Elucidating the
Design Space of Diffusion-Based Generative Models* (EDM, NeurIPS 2022).
Review date: 2026-11-25. Item 2 (Heun sampler) is implemented on `feature/heun-sampler`
(see its status section); nothing else has been implemented yet.

The paper is published, so these are improvements to the codebase. The published
results must stay reproducible:

- Keep the shipped checkpoints and `sampler="ddpm"` as the default.
- Ship any retrained model as a new, separately named checkpoint next to the published
  ones, not as a replacement.

Items 1, 3, 4 and 6 (EMA) require retraining. Item 2 runs on the shipped checkpoints.

## Current setup (shipped defaults)

| Setting | Value | Where |
|---|---|---|
| Diffusion steps K | 100 | `scripts/user/train/codit_train.py` (`--K`) |
| β schedule | linear, 1e-4 → 0.02 | `src/codit/model_module/diffusion_model.py:268-269`, `scheduler.linear_schedule` |
| Noise | Brownian motion, σ = 1, dt = 1/prediction_length | `codit_train.py`, `brownian_noise_models.WalkNoiseModel` |
| Data scaling | endpoint displacement std = 1 | `diffusion_model.py:406` |
| Parameterisation | ε-prediction of the white increments `dW`; path = `noise_model.transform(dW)` | `diffusion_models.denoising_loss`, `diffusion_model.backward_denoising_step` |
| Training σ sampling / weighting | k ~ Uniform{0..K-1}, unweighted ε-MSE | `diffusion_models.train_step` |
| Sampler | ancestral DDPM, K steps, rerun for each forecast step | `diffusion_model.denoise`, `CoDiTModel.predict` |

For a 72 h open-ocean forecast, `predict` costs 72 × 100 = 7,200 network calls per sample.

## 1. The noise schedule never reaches pure noise (most important)

`CoDiTModel` overrides `beta_max` to 0.02 (the scheduler's own default is 0.2).
With K = 100 the forward process ends far from pure noise:

| Schedule | ᾱ at last step | Signal left (√ᾱ) | EDM-equivalent σ_max = √((1−ᾱ)/ᾱ) |
|---|---|---|---|
| **Shipped: K=100, β_max=0.02** | **0.36** | **0.60** | **1.3** |
| K=100, β_max=0.2 | 2e-5 | 0.005 | 216 |
| Original DDPM: K=1000, β_max=0.02 | 4e-5 | 0.006 | 157 |

- At the noisiest training step, 60 % of the trajectory signal is still present.
- `predict` starts sampling from pure Brownian noise (`initial_y`), which the network
  never saw during training.
- EDM recommends σ_max ≫ σ_data (they use 80 against 0.5). Here σ_max ≈ 1.3 against
  σ_data ≈ 1.
- The 0.02 value looks copied from DDPM, which uses it with K = 1000.
- Data and noise both have roughly unit variance, so overall input variance still
  roughly matches. That is probably why the model works, but the first denoising steps
  run on inputs the network never saw.

**Fix:** move to a schedule that reaches noise, and retrain. Plain β_max = 0.2 with
K = 100 removes the bias but makes the steps near zero noise too coarse: in the Gaussian
test below, confident forecasts (true spread 0.1) come out 13 % too wide. Prefer a
cosine schedule, K = 1000 together with the Heun sampler (item 2), or the continuous-σ
formulation (items 3–4).

**Measured bias with a perfect denoiser.** DDPM sampling with an exact Gaussian
denoiser, mean drift 1.0 (normalised units; 1 unit ≈ 38–51 km open ocean, ≈ 12 km
coastline):

| True spread | Shipped (β_max = 0.02): drift / spread | β_max = 0.2: drift / spread |
|---|---|---|
| 0.1 | 99.5 % / 105 % | 99.9 % / 113 % |
| 0.3 | 95.4 % / 102 % | 100.2 % / 105 % |
| 0.5 | 87.9 % / 102 % | 100.4 % / 103 % |
| 1.0 | 64.4 % / 99 % | 100.9 % / 100 % |

The shipped open-ocean ensembles have spreads around 0.25–0.6 units at 72 h, which
suggests a mean drift roughly 3–12 % too short. This estimate comes from an idealised
model.

**Measuring it on the real network:** `scripts/dev/debug/diagnose_prior_mismatch.py`
runs the production `predict()` twice per test trajectory with identical noise and keys:
once from pure noise, and once from an exact draw of the trained forward marginal,
√ᾱ_K·y_true + √(1−ᾱ_K)·B. That second start uses the truth, so it is not a forecast. The
paired difference (noise − oracle) in error, energy score, spread and drift ratio is the
cost of the mismatch.

- Gap within the ±2 s.e. band or under ~1 km: the schedule is not a practical problem
  for the shipped models; fix it only as part of a larger retrain (items 3–4).
- Gap of several km: retraining with a corrected schedule is worthwhile on its own.

## 2. Deterministic Heun sampler (EDM Algorithm 1)

- **Cost:** a 2nd-order Heun ODE sampler with ρ = 7 step spacing typically needs about
  18 steps (35 network calls) instead of 100. Since the chain is rerun for each forecast
  step, sampling compute drops roughly 3–5×.
- **Fit with the rollout:** `predict` already freezes `W` and `initial_y` across forecast
  steps so the rolled-out trajectory stays consistent. A deterministic ODE sampler gives
  that by construction: the trajectory is a fixed function of the initial noise and
  the context.
- **The correlated noise is not an obstacle.** The noise path is a linear transform of
  white increments (path = L·dW, `noise_model.transform`), and the network predicts `dW`.
  The EDM formulation carries over to increment space, with the denoiser
  D(y; σ) = y − σ·L·ε̂.
- **Can be tried without retraining:** convert the trained ε-model to EDM form with
  σ_k = √((1−ᾱ_k)/ᾱ_k), x_EDM = x_VP / √ᾱ_k, and run Heun on a subset of the 100
  trained noise levels. This is still limited by item 1 (σ_max ≈ 1.3).

**Validation:** check that CRPS and errors on the test split match the published
numbers at lower cost.

### Status (2026-11-25, branch `feature/heun-sampler`)

Implemented as `heun_denoise` / `heun_step_indices` in
`src/codit/model_module/diffusion_model.py`, selected with
`CoDiTModel.set_sampler("heun", num_sampling_steps)` or
`codit_eval.py --sampler heun`. The default stays `ddpm`.

- **The sampler is correct.** Against an exact Gaussian denoiser it matches the closed-form
  ODE solution (about 0.2–0.5 % error at 18 levels) and shows 2nd-order convergence
  (`tests/model_module/test_heun_sampler.py`).
- **It is not a drop-in for the shipped checkpoints, because of item 1.** An ODE
  sampler carries its starting distribution through unchanged. The shipped schedule ends
  at ᾱ = 0.36 but sampling starts from pure noise, so Heun output is biased. Ancestral
  DDPM injects fresh noise at every step and partly corrects this.

Exact Gaussian denoiser, endpoint of the first coordinate (truth: mean 2.0, std 0.30):

| Schedule | DDPM (100 NFE) | Heun, 18 levels (≈35 NFE) |
|---|---|---|
| shipped, β_max = 0.02 | mean 1.90, std 0.30 | mean 1.56, std 0.37 |
| β_max = 0.2 | mean 2.00, std 0.31 | mean 2.00, std 0.33 |

The shipped `codit_GRU_open_ocean` network with synthetic context (400 samples, 72 h
endpoint) shows the same pattern. Heun has converged by 18 levels (33 NFE, about 3× faster
than DDPM), but its ensembles are 1.3–1.6× wider. Under strong context its mean drift is
about 30 % shorter (e.g. 160 km vs 226 km).

Consequences:

- `set_sampler("heun")` warns when the schedule ends above ᾱ = 1e-3.
- Even with a perfect network, the shipped DDPM sampler undershoots the mean drift by
  about 5 % in the Gaussian test. Item 1 is therefore a measurable bias, not only a
  theoretical one.
- Heun becomes useful once a model is retrained with a schedule that reaches noise
  (item 1). Until then, keep `ddpm` for the shipped checkpoints.

## 3. Preconditioning and noise conditioning (EDM Sec. 5, Table 1)

- **Condition on noise level, not step index:** feed c_noise = ¼·ln σ instead of
  `TemporalEncoding(k/K)`. The network then no longer depends on K, so the schedule or
  sampler can change without retraining.
- **Add c_skip / c_out / c_in:** ε-prediction amplifies network error at high σ. Minor
  with the current σ_max, but important once item 1 is fixed.
- **Measure σ_data in increment space; don't assume it.** Endpoint std is normalised
  to 1, but the per-increment std in whitened units ranges from about 1/√72 ≈ 0.12
  (straight-line drift) to 1 (pure random walk). The preconditioning depends on this
  number, so estimate it from the training set.

Requires retraining.

## 4. Training noise distribution and loss weighting (EDM Sec. 5)

Replace uniform k with ln σ ~ N(P_mean, P_std²) (EDM: −1.2, 1.2) and weight the loss by
λ(σ) = (σ² + σ_data²) / (σ·σ_data)². This puts training effort on the mid-range noise
levels that matter most for sample quality. Retune P_mean / P_std for the measured
σ_data. Implement together with item 3.

## 5. Stochastic churn (EDM Algorithm 2) for calibration

EDM found that re-injecting some noise during sampling (S_churn, S_tmin, S_tmax,
S_noise) corrects errors from an imperfect model. Ensemble spread and calibration are
central for a probabilistic forecaster, so tune S_churn on validation CRPS or coverage.
The churn noise must be pre-sampled like `W` to keep rollout consistency.

## 6. Smaller items

- **Weight EMA:** there is no EMA of the weights anywhere in `src/`. EDM and most
  diffusion work evaluate EMA weights. Cheap to add with `optax.ema`.
- **Shared noise key (harmless today):** `denoising_loss`
  (`src/codit/codit_module/diffusion_models.py:732-743`) noises the prediction and the
  history with the same subkey, while the sampler uses a fresh key for the history. No
  effect at present: the history is one point, so its noise is identically zero. Revisit
  if the history gets longer.

## Suggested order

1. ~~Item 2 on the existing checkpoints.~~ Done, but biased on the shipped checkpoints
   because of item 1 (see item 2 status).
2. Fix item 1 (β_max or continuous σ), retrain, then evaluate both `ddpm` and `heun`
   on that model.
3. Items 3 + 4 together as one EDM rewrite on a separate branch off `dev`, then item 5
   and EMA. Benchmark against the published numbers and ship as new checkpoints; the
   published models and DDPM code path stay available for reproduction.
