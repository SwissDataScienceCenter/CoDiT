# Diffusion improvements suggested by EDM

Notes from reviewing the CoDiT diffusion code against Karras et al., *Elucidating the
Design Space of Diffusion-Based Generative Models* (EDM, NeurIPS 2022).
Review date: 2026-11-25. Status 2026-11-26: item 1 is implemented (cosine schedule) and
EMA (item 6) is in the training loop. Item 2 (Heun sampler) was implemented, tested on a
retrained model and removed again (see its section).

The paper is published, so these are improvements to the codebase. The published
results must stay reproducible:

- Keep the shipped checkpoints and the DDPM sampler.
- Ship any retrained model as a new, separately named checkpoint next to the published
  ones, not as a replacement.

Items 1, 3, 4 and 6 (EMA) require retraining.

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
cosine schedule (implemented, see Status), K = 1000, or the continuous-σ
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

**Status:** `CoDiTModel(noise_schedule="cosine")` (β from 1e-4 to 0.2 on a cosine ramp,
ᾱ_K ≈ 2e-5) and `codit_train.py --noise_schedule cosine`. The default stays `linear`, so
the shipped checkpoints are unchanged. A cosine retrain of `codit_GRU_open_ocean` beats a
linear retrain with otherwise identical settings by 2.85 ± 0.75 km energy score at 72 h
(15 test patches, 273 paired trajectories, DDPM sampling).

**Measuring it on the real network:** `scripts/dev/debug/diagnose_prior_mismatch.py`
runs the production `predict()` twice per test trajectory with identical noise and keys:
once from pure noise, and once from an exact draw of the trained forward marginal,
√ᾱ_K·y_true + √(1−ᾱ_K)·B. That second start uses the truth, so it is not a forecast. The
paired difference (noise − oracle) in error, energy score, spread and drift ratio is the
cost of the mismatch.

- Gap within the ±2 s.e. band or under ~1 km: the schedule is not a practical problem
  for the shipped models; fix it only as part of a larger retrain (items 3–4).
- Gap of several km: retraining with a corrected schedule is worthwhile on its own.

## 2. Deterministic Heun sampler (EDM Algorithm 1): tried and removed

The idea: run EDM's 2nd-order Heun ODE sampler on a subset of the trained noise levels
(σ_k = √((1−ᾱ_k)/ᾱ_k), x_EDM = x_VP / √ᾱ_k, path = L·dW as in the network output).
About 18 levels (35 network calls) instead of 100, with no retraining. It was added in
commit `2187150` and removed on 2026-11-26. The diagnosis is on the branch
`debug/heun-jumps` (`scripts/dev/debug/check_heun_jumps.py`).

- **The sampler itself was correct.** Against an exact Gaussian denoiser it matched the
  closed-form ODE solution and converged at 2nd order.
- **On the shipped checkpoints it is biased** because of item 1: an ODE sampler carries
  its starting distribution through unchanged, and the shipped schedule ends at ᾱ = 0.36
  while sampling starts from pure noise. Ensembles came out 1.3–1.6× too wide and the
  mean drift up to 30 % too short.
- **On the cosine retrain (item 1) it produces non-physical jumps.** Heun with 18 levels
  gave the best energy score on 15 test patches (−0.9 km vs DDPM at 72 h), but 5.8 % of
  its samples contain a one-hour jump of 10–300 km that returns the next hour (true
  drifters stay below ~10 km/h; DDPM models: 0.4–0.8 % of samples above 10 km/h).
  - Each forecast hour re-solves the whole trajectory; a jump is one position of one solve
    where the solver leaves part of a large local kink of the starting noise in place.
  - More levels do not help: 18, 35 and 50 give the same jumps in the same samples.
  - Re-noising the finished Heun sample and finishing with DDPM removes the jumps only
    from σ ≈ 10 (k = 69); at σ ≤ 3.5 the network keeps a 200 km kink as signal. That
    costs more network calls than DDPM and gave the same energy score.
  - This matches the literature: deterministic samplers pass on errors of the learned
    network unchanged, stochastic ones contract them (Karras et al. 2022; Xu et al. 2023,
    *Restart Sampling*).

Worth revisiting only for a better-trained model (or with items 3–5), and only if the
share of samples with jumps drops to the DDPM level.

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
The churn noise must be pre-sampled like `W` to keep rollout consistency. Only relevant
together with a deterministic sampler (item 2).

## 6. Smaller items

- **Weight EMA:** done. Training keeps an EMA of the weights (decay 0.999 with warm-up);
  model selection and the exported model use it, the best raw weights are exported
  alongside (`codit_eval.py --weights raw`).
- **Shared noise key (harmless today):** `denoising_loss`
  (`src/codit/codit_module/diffusion_models.py:732-743`) noises the prediction and the
  history with the same subkey, while the sampler uses a fresh key for the history. No
  effect at present: the history is one point, so its noise is identically zero. Revisit
  if the history gets longer.

## Suggested order

1. ~~Item 1~~: done (cosine schedule).
2. ~~Item 2~~: removed, see above.
3. Train the cosine model longer: both retrains selected epoch 97–98 of 100.
4. Items 3 + 4 together as one EDM rewrite on a separate branch off `dev`. Benchmark
   against the published numbers and ship as new checkpoints; the published models and
   the DDPM code path stay available for reproduction.
