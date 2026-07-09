"""Quantify predict() speedup from JIT caching and dynamic-index optimisations.

Measures two scenarios in one run so results are directly comparable:

  A — no-cache  : reset _denoise_func before every call (pre-optimisation baseline)
  B — with-cache: cache persists across calls (post-optimisation)

Run with:
    micromamba run -n codit python my_scripts/model_fitting/benchmark_predict.py
"""
import os
import time

import jax
import numpy as np

from codit.data_module import config_funcs, config_paths
from codit.data_module.dataset import OceanDataset
from codit.model_module.diffusion_model import CoDiTModel

MODEL_NAME = "codit_GRU_open_ocean"
DATASET = "open_ocean"
NUM_TRAJ = 5        # trajectories per scenario
NUM_SAMPLES = 10    # samples per predict() call

model = CoDiTModel.import_trained_model(MODEL_NAME)
config_path = os.path.join(config_paths.CONFIG_PATH, DATASET, "test")
files = sorted(f for f in os.listdir(config_path) if f.endswith(".yml"))[:1]
assert files, f"No .yml configs in {config_path}"
config = config_funcs.load_config(files[0].split(".")[0], config_path)
ds = OceanDataset(config, 72, 16, 10000, "linear")

samples = [ds[idx] for idx in range(min(NUM_TRAJ, len(ds)))]
inputs = [(sp, st, interp) for sp, st, _t, interp, _ in samples]


def _time_calls(force_recompile_each: bool) -> list[float]:
    """Time NUM_TRAJ predict() calls.

    force_recompile_each=True  → reset _denoise_func before every call
                                 (simulates pre-optimisation behaviour)
    force_recompile_each=False → cache persists across calls
                                 (simulates post-optimisation behaviour)
    """
    model._denoise_func = None  # always start fresh
    times = []
    for sp, st, interp in inputs:
        if force_recompile_each:
            model._denoise_func = None
        t0 = time.perf_counter()
        result = model.predict(sp, st, 72, interp, ds, num_samples=NUM_SAMPLES)
        jax.block_until_ready(result)  # wait for async dispatch to finish
        times.append(time.perf_counter() - t0)
    return times


# ── Scenario A: no-cache (pre-optimisation baseline) ─────────────────────────
print("Scenario A — no-cache (pre-optimisation baseline):")
times_a = _time_calls(force_recompile_each=True)
for i, t in enumerate(times_a):
    print(f"  call {i}: {t:.2f}s")

# ── Scenario B: with-cache (post-optimisation) ────────────────────────────────
print("\nScenario B — with-cache (post-optimisation):")
times_b = _time_calls(force_recompile_each=False)
for i, t in enumerate(times_b):
    print(f"  call {i}: {t:.2f}s")

# ── Summary ──────────────────────────────────────────────────────────────────
mean_a = np.mean(times_a)
mean_b_steady = np.mean(times_b[1:]) if len(times_b) > 1 else times_b[0]
print("\n── Summary ──────────────────────────────────────────────")
print(f"  A  no-cache avg:            {mean_a:.2f}s  (all calls recompile)")
print(f"  B  cache, first call:       {times_b[0]:.2f}s  (compile once)")
print(f"  B  cache, steady-state avg: {mean_b_steady:.2f}s")
print(f"  A→B steady-state speedup:   {mean_a / mean_b_steady:.1f}x")
