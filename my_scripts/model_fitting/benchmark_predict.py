"""Benchmark predict() call time to verify JIT caching speedup.

Run with:
    micromamba run -n codit python my_scripts/model_fitting/benchmark_predict.py
"""
import time
import os
import numpy as np
from codit.model_module.diffusion_model import CoDiTModel
from codit.data_module import config_paths, config_funcs
from codit.data_module.dataset import OceanDataset

MODEL_NAME = "codit_GRU_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry"
DATASET = "open_ocean"
NUM_TRAJ = 5
NUM_SAMPLES = 10

model = CoDiTModel.import_trained_model(MODEL_NAME)

config_path = os.path.join(config_paths.CONFIG_PATH, DATASET, "val")
files = sorted(f for f in os.listdir(config_path) if f.endswith(".yml"))[:3]
dataset_name = files[0].split(".")[0]
config = config_funcs.load_config(dataset_name, config_path)
ds = OceanDataset(config, 72, 16, 10000, "linear")

times_per_call = []
for idx in range(min(NUM_TRAJ, len(ds))):
    start_position, start_time, times, interpolator, coordinates = ds[idx]
    t0 = time.perf_counter()
    model.predict(start_position, start_time, 72, interpolator, ds, num_samples=NUM_SAMPLES)
    elapsed = time.perf_counter() - t0
    times_per_call.append(elapsed)
    print(f"  predict call {idx}: {elapsed:.2f}s")

print(f"\nFirst call (incl. JIT): {times_per_call[0]:.2f}s")
if len(times_per_call) > 1:
    print(f"Subsequent calls avg:  {np.mean(times_per_call[1:]):.2f}s")
    print(f"Speed-up ratio (1st/subsequent): {times_per_call[0]/np.mean(times_per_call[1:]):.1f}x")
