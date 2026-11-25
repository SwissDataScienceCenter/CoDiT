"""Time the parts of one CoDiT training step: data loading, batch transform, the
jitted gradient step, and the EMA update.

Uses the first ``--num_samples`` samples of the merged training h5 (or synthetic data
with ``--synthetic``) and the same model, DataLoader and step functions as
``codit_train.py`` / ``CoDiTModel._fit``.

    python scripts/dev/debug/time_train_step.py --batch_size 1024
    python scripts/dev/debug/time_train_step.py --synthetic --batch_size 64   # no data needed
"""

import argparse
import os
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random, vmap
from torch.utils.data import DataLoader

from codit.codit_module import diffusion_models
from codit.data_module import config_paths
from codit.data_module.data_transform import transform_traj_to_metre
from codit.data_module.dataset import OceanTrainingDataset
from codit.model_module.diffusion_model import CoDiTModel


def timed(fn, repeats):
    out, times = None, []
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t)
    return out, times


def fmt(times):
    t = np.asarray(times)
    return f"first {t[0]:7.3f} s | later median {np.median(t[1:]):7.3f} s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="open_ocean")
    p.add_argument("--length", type=int, default=72)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_samples", type=int, default=3000)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    L = args.length
    print("devices:", jax.devices())

    t = time.perf_counter()
    if args.synthetic:
        rng = np.random.default_rng(0)
        n = max(args.num_samples, 2 * args.batch_size)
        start = np.c_[rng.uniform(-10, 10, n), rng.uniform(-150, -140, n)]
        traj = start[:, None] + rng.normal(0, 0.02, (n, L + 1, 2)).cumsum(1)
        traj = traj.astype(np.float32)
        ctx = rng.normal(size=(n, L + 1, 16, 16, 8)).astype(np.float32)
    else:
        path = os.path.join(
            config_paths.DATA_PATH, "datasets", args.dataset, "datasets", "processed",
            f"training_data_{L}h.h5",
        )
        with h5py.File(path, "r") as f:
            traj = f["trajectories"][: args.num_samples]
            ctx = f["context"][: args.num_samples]
    ds = OceanTrainingDataset(traj, ctx)
    print(f"load {len(ds)} samples: {time.perf_counter() - t:.1f} s")

    model = CoDiTModel("timing", {
        "prediction_length": L, "dt": 1.0 / L, "noise_schedule": "cosine",
        "vars": ["hycom_current", "copernicus_current", "era5_wind"],
        "aligned_context": True,
    })
    ds.set_context_var_idx(model.var_idx)
    std = jnp.std(vmap(transform_traj_to_metre)(jnp.asarray(ds.trajectories))[:, -1], axis=0)
    model.std = std
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    batches = diffusion_models._cycle(loader)

    sample = ds[0]
    traj0 = model.center_and_scale_data(jnp.asarray(sample[0]))
    params = model.model.init(
        random.PRNGKey(0), y=traj0[1:], y_history=traj0[:1],
        x={"history_target_vals": traj0[:1], "history_context": sample[1][:1],
           "prediction_context": sample[1][1:]},
        k=jnp.ones((1,), dtype=jnp.int32), train=True,
    )
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    sched, K = model.schedule_dict, model.schedule_dict["K"]

    @jax.jit
    def step(carry, batch):
        return diffusion_models.train_step(
            carry, batch, model.model, model.noise_model, opt, sched, K, args.batch_size
        )

    torch_batch, t_load = timed(lambda: next(batches), args.repeats)
    print(f"DataLoader batch       : {fmt(t_load)}")
    jb, t_tr = timed(lambda: diffusion_models.transform_batch(torch_batch, std), args.repeats)
    print(f"transform_batch        : {fmt(t_tr)}")
    t = time.perf_counter()
    arrays = diffusion_models._dataset_arrays(loader)
    jax.block_until_ready(arrays)
    print(f"copy data to device    : {time.perf_counter() - t:.1f} s (once per run)")
    gather = jax.jit(lambda tr, cx, idx: diffusion_models.transform_batch((tr[idx], cx[idx]), std))
    idx_iter = diffusion_models._device_batches(len(ds), args.batch_size, random.PRNGKey(2))
    _, t_dev = timed(lambda: gather(*arrays, next(idx_iter)), args.repeats)
    print(f"device batch (gather)  : {fmt(t_dev)}   <- replaces DataLoader + transform_batch")
    carry = (params, opt_state, random.PRNGKey(1), 0.0)
    times = []
    for _ in range(args.repeats):
        t = time.perf_counter()
        carry, _ = step(carry, jb)
        jax.block_until_ready(carry)
        times.append(time.perf_counter() - t)
    print(f"jitted train step      : {fmt(times)}   (first = compile + autotune)")
    ema, decay = params, diffusion_models.ema_decay_at(5, 0.999)
    _, t_ema = timed(lambda: diffusion_models.ema_update(ema, carry[0], decay), args.repeats)
    print(f"EMA update             : {fmt(t_ema)}")
    for name, data_time in [
        ("DataLoader", np.median(t_load[1:]) + np.median(t_tr[1:])),
        ("device data", np.median(t_dev[1:])),
    ]:
        per_step = data_time + np.median(times[1:]) + np.median(t_ema[1:])
        print(f"=> {name:11s}: ~{per_step:.2f} s per step, "
              f"~{per_step * 100 / 60:.1f} min per 100-step epoch")


if __name__ == "__main__":
    main()
