"""Measure what the diffusion schedule's prior mismatch costs a trained CoDiT model.

Background (docs/EDM_IMPROVEMENTS.md, item 1): the shipped schedule (K=100,
beta_max=0.02) ends at alpha_bar ~ 0.36, i.e. the last training step still holds
~60% signal, but sampling starts from pure noise.

This script runs the production ``predict()`` rollout twice per trajectory, with
the same noise and PRNG keys, changing only the starting state of the reverse
process:

  noise   standard start, ``y_K = B`` (pure noise path)
  oracle  ``y_K = sqrt(ab_K) * y_true + sqrt(1 - ab_K) * B``, an exact draw from the
          forward-process marginal the network was trained on (``y_true`` is one
          draw of the conditional data distribution)

``oracle`` uses the ground truth, so it is NOT a usable forecast. It measures how
much better the model would be if sampling started from the right distribution.
The paired difference (noise - oracle) is the cost of the schedule mismatch for
the real network. A gap near zero means the schedule fix (cosine schedule /
retraining) can wait; a gap of several km means it is worth a retrain.

Reported per (sampler, start), over trajectories:
  err_{h}h      mean haversine error of samples vs truth at hour h (km)
  es_{h}h       energy score at hour h (km)
  drift_ratio   along-track ensemble-mean displacement / true displacement at the
                final hour (1.0 = right drift length; <1 = drift too short).
                Median over trajectories with true displacement > --min_drift_km.
  spread_km     ensemble spread at the final hour (km)
  nan_frac      share of samples that contain NaN (left the context grid); they
                are excluded from all metrics above, as in get_energy_score

Run on a machine with the test data, e.g.:
    python scripts/dev/debug/diagnose_prior_mismatch.py --dataset open_ocean
    python scripts/dev/debug/diagnose_prior_mismatch.py --dataset coastline \\
        --model_name codit_GRU_coastline --prediction_length 24
"""

import argparse
import os
import warnings

import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp
from jax import random, vmap
from tqdm import tqdm

from codit.data_module import config_funcs, config_paths, data_transform
from codit.data_module.data_transform import transform_traj_to_metre
from codit.data_module.dataset import OceanDataset
from codit.eval_module.metrics import get_energy_score
from codit.model_module.diffusion_model import CoDiTModel

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default="open_ocean", choices=["open_ocean", "coastline"])
    p.add_argument("--split", default="test")
    p.add_argument("--model_name", default="codit_GRU_open_ocean")
    p.add_argument("--prediction_length", type=int, default=72)
    p.add_argument("--num_patches", type=int, default=20, help="patches, evenly spaced")
    p.add_argument("--per_patch", type=int, default=10, help="trajectories per patch")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--min_drift_km", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out_dir",
        default=os.path.join(config_paths.RESULT_PATH, "diagnostics", "prior_mismatch"),
    )
    return p.parse_args()


def to_model_units(coords: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    """(L+1, 2) lat/lon -> centred, scaled trajectory, exactly as in training."""
    return transform_traj_to_metre(coords) / std


def check_round_trip(coords: jnp.ndarray, std: jnp.ndarray) -> float:
    """Max error (m) of model units -> lat/lon with predict()'s inverse mapping."""
    y = to_model_units(coords, std) * std
    dist = jnp.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
    bearing = jnp.degrees(jnp.arctan2(y[:, 1], y[:, 0]))
    back = vmap(data_transform.inverse_haversine, in_axes=(None, 0, 0))(
        coords[0], dist, bearing
    )
    return float(jnp.max(vmap(data_transform.haversine_distance)(back, coords)))


def trajectory_metrics(
    pred: jnp.ndarray, coords: jnp.ndarray, hours: list[int]
) -> dict:
    """pred: (S, L, 2) lat/lon; coords: (L+1, 2) truth including the start.

    Samples containing NaN (the rollout left the context grid, where the
    interpolator returns NaN) are excluded from every metric, as
    ``get_energy_score`` already does; their share is reported as ``nan_frac``.
    """
    true_m = transform_traj_to_metre(coords)[-1]  # (2,)
    true_len2 = float(jnp.sum(true_m**2))
    valid = ~np.isnan(np.asarray(pred)).any(axis=(1, 2))  # (S,)
    out = {
        "true_drift_km": float(jnp.sqrt(true_len2)) / 1000,
        "nan_frac": float(1 - valid.mean()),
    }
    es = get_energy_score(coords[None, 1:], pred[None])[0]  # (L,), NaN-aware
    for h in hours:
        out[f"es_{h}h"] = float(es[h - 1]) / 1000
    if not valid.any():
        out.update({"drift_ratio": np.nan, "spread_km": np.nan})
        out.update({f"err_{h}h": np.nan for h in hours})
        return out

    pred = pred[valid]
    dist = vmap(vmap(data_transform.haversine_distance), in_axes=(None, 0))(
        coords[1:], pred
    )  # (S_valid, L)
    # displacement from the start in local metres (same transform as training)
    full = jnp.concatenate(
        [jnp.broadcast_to(coords[:1], (pred.shape[0], 1, 2)), pred], axis=1
    )
    pred_m = vmap(transform_traj_to_metre)(full)[:, -1]  # (S_valid, 2)
    out["drift_ratio"] = (
        float(jnp.dot(pred_m.mean(0), true_m)) / true_len2 if true_len2 > 0 else np.nan
    )
    out["spread_km"] = float(jnp.sqrt(jnp.sum(pred_m.var(0)))) / 1000
    for h in hours:
        out[f"err_{h}h"] = float(dist[:, h - 1].mean()) / 1000
    return out


def main() -> None:
    args = parse_args()
    L = args.prediction_length
    hours = sorted({h for h in (24, 48, 72) if h <= L} | {L})

    # keyed by sampler so the output keeps the format of earlier runs, which
    # also compared the since-removed Heun sampler
    models = {"ddpm": CoDiTModel.import_trained_model(args.model_name)}
    ref = next(iter(models.values()))
    ab_end = float(ref.schedule_dict["alpha_bar"][-1])
    print(
        f"{args.model_name}: K={ref.schedule_dict['K']}, alpha_bar_end={ab_end:.3g} "
        f"(signal left sqrt={np.sqrt(ab_end):.2f}), std={np.asarray(ref.std)} m"
    )

    config_path = os.path.join(config_paths.CONFIG_PATH, args.dataset, args.split)
    names = sorted(f[:-4] for f in os.listdir(config_path) if f.endswith(".yml"))
    configs = [config_funcs.load_config(n, config_path) for n in names]
    configs = [
        c for c in configs
        if c["start_date"].year >= 2020 and os.path.exists(c["dataset_path"])
    ]
    if not configs:
        raise SystemExit(f"No usable configs with data in {config_path}")
    pick = np.unique(np.linspace(0, len(configs) - 1, args.num_patches, dtype=int))
    configs = [configs[i] for i in pick]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    key = random.PRNGKey(args.seed)
    checked = False
    patch_bar = tqdm(configs, desc="patches", position=0)
    for config in patch_bar:
        patch_bar.set_postfix_str(f"loading {config['name']}")
        try:
            ds = OceanDataset(config, L, 16, 10000, "linear")
        except Exception as e:  # noqa: BLE001 - skip unreadable patches, like codit_eval
            tqdm.write(f"skip {config['name']}: {e}")
            continue
        if len(ds) == 0:
            continue
        patch_bar.set_postfix_str(config["name"])
        indices = np.unique(np.linspace(0, len(ds) - 1, args.per_patch, dtype=int))
        # One tick per predict() call: trajectories x samplers x {noise, oracle}.
        pred_bar = tqdm(
            total=len(indices) * len(models) * 2,
            desc="  predict",
            position=1,
            leave=False,
        )
        for idx in indices:
            try:
                start_pos, start_time, _times, interp, coords = ds[int(idx)]
            except ValueError as e:
                tqdm.write(f"skip {config['name']}[{idx}]: {e}")
                pred_bar.update(len(models) * 2)
                continue
            if not checked:
                err = check_round_trip(coords, ref.std)
                print(f"round-trip check (model units -> lat/lon): max error {err:.2f} m")
                if err > 100:
                    raise SystemExit("Coordinate transforms disagree; oracle start invalid")
                checked = True

            key, key_b, key_pred = random.split(key, 3)
            B = vmap(ref.noise_model, in_axes=(0, None))(
                random.split(key_b, args.num_samples), L
            )[0]
            y_true = to_model_units(coords, ref.std)
            starts = {
                "noise": B,
                "oracle": jnp.sqrt(ab_end) * y_true[None] + jnp.sqrt(1 - ab_end) * B,
            }
            for sampler, model in models.items():
                for start, y0 in starts.items():
                    pred_bar.set_postfix_str(f"traj {idx} {sampler}/{start}")
                    try:
                        pred = model.predict(
                            start_pos, start_time, L, interp, ds,
                            num_samples=args.num_samples, key=key_pred, initial_y=y0,
                        )
                    except ValueError as e:
                        tqdm.write(f"skip {config['name']}[{idx}] {sampler}/{start}: {e}")
                        pred_bar.update(1)
                        continue
                    pred_bar.update(1)
                    rows.append(
                        {"patch": config["name"], "idx": int(idx), "sampler": sampler,
                         "start": start, **trajectory_metrics(pred, coords, hours)}
                    )
        pred_bar.close()
        pd.DataFrame(rows).to_csv(
            os.path.join(args.out_dir, f"{args.model_name}_{args.split}.csv"), index=False
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No trajectories evaluated")
    metric_cols = (
        [f"err_{h}h" for h in hours] + [f"es_{h}h" for h in hours] + ["spread_km", "nan_frac"]
    )
    moving = df[df["true_drift_km"] > args.min_drift_km]

    print(f"\n{df[['patch', 'idx']].drop_duplicates().shape[0]} trajectories, "
          f"{args.num_samples} samples each (km; drift_ratio: median, "
          f"true drift > {args.min_drift_km} km)")
    summary = df.groupby(["sampler", "start"])[metric_cols].mean()
    summary["drift_ratio"] = moving.groupby(["sampler", "start"])["drift_ratio"].median()
    print(summary.round(3).to_string())

    print("\nPaired cost of the mismatch, noise - oracle (mean ± 2 s.e.):")
    for sampler in models:
        sub = df[df["sampler"] == sampler].pivot_table(
            index=["patch", "idx"], columns="start", values=metric_cols
        )
        parts = []
        for col in metric_cols:
            d = (sub[(col, "noise")] - sub[(col, "oracle")]).dropna()
            parts.append(f"{col} {d.mean():+.2f}±{2 * d.std() / np.sqrt(len(d)):.2f}")
        print(f"  {sampler}: " + ", ".join(parts))
    print(f"\nper-trajectory results: {args.out_dir}")


if __name__ == "__main__":
    main()
