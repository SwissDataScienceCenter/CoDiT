"""Re-run one test trajectory with several samplers and count non-physical jumps.

Each hour of ``predict`` re-solves the whole trajectory and keeps position ``i``,
so a sample that jumps far for one hour and comes back is one diverged solve.
A step is flagged as a jump when it exceeds ``--jump_km`` (true drifters stay
below ~10 km/h).

    JAX_PLATFORMS=cuda python scripts/dev/debug/check_heun_jumps.py \\
        --patch open_ocean_patch_0_year_2021 --index 149 \\
        --samplers heun18 heun35 heun50 ddpm

``--trace 44:49`` follows sample 44 through the Heun solves at hours 48, 49 and 50
(only one context slot differs between them) and prints, per noise level, how far
apart the solves are at position 49, to show the level where hour 49 splits off.
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from codit.data_module import config_funcs, config_paths
from codit.data_module.dataset import OceanDataset
from codit.eval_module import metrics
from codit.model_module.diffusion_model import CoDiTModel, edm_sigmas

COSINE = (
    "codit_GRU_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind_cosine"
)


def hourly_steps_km(start: np.ndarray, pred: np.ndarray) -> np.ndarray:
    x = np.concatenate([np.broadcast_to(start, (pred.shape[0], 1, 2)), pred], axis=1)
    d = np.diff(x, axis=1)
    lat = np.deg2rad(x[:, :-1, 1])
    return 111.2 * np.hypot(d[..., 0] * np.cos(lat), d[..., 1])  # (S, T)


def km_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(0.5 * (a[..., 1] + b[..., 1]))
    return 111.2 * np.hypot((a[..., 0] - b[..., 0]) * np.cos(lat), a[..., 1] - b[..., 1])


def trace(model, name, start, start_time, interp, dataset, true, num_samples,
          sample, hour, stem):
    hours = [hour - 1, hour, hour + 1]
    _, steps = model.predict(
        start, start_time, 72, interp, dataset, num_samples=num_samples,
        return_denoising_steps=True, denoising_steps_hours=hours,
    )
    steps = np.asarray(steps)[sample]  # (3, S, T + 1, 2)
    ab = np.asarray(model.schedule_dict["alpha_bar"])
    idx = np.asarray(model._heun_indices)
    sig = np.r_[np.asarray(edm_sigmas(ab))[idx[1:]], 0.0]
    print(f"\n{name}: sample {sample}, solves at hours {hours}, position {hour}")
    print("  steps are in DDPM space (positions shrunk by sqrt(alpha_bar) at high noise)")
    print(f"  {'level':>5s} {'k':>4s} {'sigma':>9s} {'sqrt_ab':>8s} "
          f"{f'h{hour}-h{hour - 1}':>10s} {f'h{hour + 1}-h{hour - 1}':>10s} "
          f"{f'|h{hour} disp|':>11s}")
    for j in range(steps.shape[1]):
        k = idx[j + 1] if j + 1 < len(idx) else -1
        sab = np.sqrt(ab[k]) if k >= 0 else 1.0
        d_jump = km_between(steps[1, j, hour], steps[0, j, hour])
        d_ctrl = km_between(steps[2, j, hour], steps[0, j, hour])
        disp = km_between(steps[1, j, hour], np.asarray(start))
        print(f"  {j:5d} {k if k >= 0 else 'end':>4} {sig[j]:9.3g} {sab:8.3g} "
              f"{d_jump:10.1f} {d_ctrl:10.1f} {disp:11.1f}")

    fig, ax = plt.subplots(figsize=(6, 7))
    for h, c in zip(hours, ["tab:green", "tab:red", "tab:purple"]):
        traj = steps[hours.index(h), -1, 1:]
        ax.plot(traj[:, 0], traj[:, 1], c=c, lw=1.2, label=f"solve at hour {h}")
        ax.scatter(*traj[h - 1], c=c, s=40, zorder=6)
    ax.plot(true[:, 0], true[:, 1], c="k", lw=2, label="true")
    ax.scatter(*true[0], c="k", zorder=5)
    ax.set_aspect(1 / np.cos(np.deg2rad(true[0, 1])))
    ax.legend(fontsize=8)
    ax.set_title(f"{name} sample {sample}: full trajectories of the solves\n"
                 f"(dot = position kept from each solve)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{stem}_{name}_trace_s{sample}_h{hour}.png", dpi=100)
    np.savez(f"{stem}_{name}_trace_s{sample}_h{hour}.npz", steps=steps, hours=hours,
             sigma=sig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch", default="open_ocean_patch_0_year_2021")
    p.add_argument("--index", type=int, default=149)
    p.add_argument("--model_name", default=COSINE)
    p.add_argument("--samplers", nargs="+", default=["heun18", "heun35", "heun50", "ddpm"])
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--jump_km", type=float, default=10.0)
    p.add_argument("--trace", nargs="*", default=[], metavar="SAMPLE:HOUR",
                   help="Trace these jumps through the Heun solves (Heun samplers only)")
    p.add_argument(
        "--reference",
        default=os.path.join(
            config_paths.RESULT_PATH, "open_ocean", "test", "full_cmp",
            COSINE + "_heun18",
        ),
        help="Result dir of the full eval, to check that heun18 reproduces it",
    )
    p.add_argument(
        "--out",
        default=os.path.join(config_paths.RESULT_PATH, "diagnostics", "heun_jumps"),
    )
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    config_path = os.path.join(config_paths.CONFIG_PATH, "open_ocean", "test")
    config = config_funcs.load_config(args.patch, config_path)
    dataset = OceanDataset(config, 72, 16, 10000, "linear")
    start, start_time, _, interp, true = dataset[args.index]
    true = np.asarray(true)

    ref = None
    ref_file = os.path.join(args.reference, f"{args.patch}_result.npz")
    if os.path.exists(ref_file):
        r = np.load(ref_file)
        if np.allclose(r["true_coords"][args.index], true, equal_nan=True):
            ref = r["pred_coords"][args.index]
        else:
            print("WARNING: reference row does not match this trajectory, not compared")

    model = CoDiTModel.import_trained_model(model_name=args.model_name)
    preds = {}
    print(f"{args.patch} index {args.index}, {args.num_samples} samples, "
          f"jump = step > {args.jump_km} km/h\n")
    print(f"{'sampler':8s} {'s':>6s} {'jump smp':>8s} {'jumps':>6s} {'max km/h':>9s} "
          f"{'NaN smp':>7s} {'ES24':>6s} {'ES48':>6s} {'ES72':>6s} {'PE72':>6s}")
    for name in args.samplers:
        if name == "ddpm":
            model.set_sampler("ddpm")
        else:
            model.set_sampler("heun", int(name.removeprefix("heun")))
        t0 = time.time()
        pred = np.asarray(model.predict(
            start, start_time, 72, interp, dataset, num_samples=args.num_samples
        ))
        dt = time.time() - t0
        preds[name] = pred
        steps = hourly_steps_km(true[0], pred)
        jump = steps > args.jump_km
        m = metrics.get_metrics(true[None], pred[None])
        es = np.asarray(m["energy_score"])[0] / 1000
        pe = np.asarray(m["mean_distance"])[0] / 1000
        print(f"{name:8s} {dt:6.1f} {jump.any(1).sum():8d} {jump.sum():6d} "
              f"{np.nanmax(steps):9.1f} {np.isnan(pred).any((1, 2)).sum():7d} "
              f"{es[23]:6.2f} {es[47]:6.2f} {es[71]:6.2f} {pe[71]:6.2f}")
        if name == "heun18" and ref is not None:
            diff = np.nanmax(np.abs(pred - ref))
            print(f"         heun18 vs full eval: max |diff| {diff:.2e} deg")
        for s, h in np.argwhere(jump)[:5]:
            print(f"         sample {s:3d} hour {h + 1:2d}: {steps[s, h]:6.1f} km")

    fig, axs = plt.subplots(1, len(preds), figsize=(4.5 * len(preds), 6), squeeze=False)
    for ax, (name, pred) in zip(axs[0], preds.items()):
        jump = (hourly_steps_km(true[0], pred) > args.jump_km).any(1)
        for s in range(pred.shape[0]):
            ax.plot(np.r_[true[0, 0], pred[s, :, 0]], np.r_[true[0, 1], pred[s, :, 1]],
                    c="tab:red" if jump[s] else "tab:blue",
                    alpha=0.6 if jump[s] else 0.1, lw=0.8)
        ax.plot(true[:, 0], true[:, 1], c="k", lw=2)
        ax.scatter(*true[0], c="k", zorder=5)
        ax.set_title(f"{name}: {jump.sum()} samples with jumps (red)")
        ax.set_aspect(1 / np.cos(np.deg2rad(true[0, 1])))
    fig.suptitle(f"{args.patch} traj {args.index}")
    fig.tight_layout()
    stem = os.path.join(args.out, f"{args.patch}_traj{args.index}")
    fig.savefig(stem + ".png", dpi=100)
    np.savez(stem + ".npz", true=true, **preds)
    print(f"\nwritten: {stem}.png / .npz")

    traces = [tuple(int(v) for v in t.split(":")) for t in args.trace]
    for name in [n for n in args.samplers if n != "ddpm"] if traces else []:
        model.set_sampler("heun", int(name.removeprefix("heun")))
        for sample, hour in traces:
            if not 2 <= hour <= 71:
                raise SystemExit(f"--trace hour must be in 2..71, got {hour}")
            trace(model, name, start, start_time, interp, dataset, true,
                  args.num_samples, sample, hour, stem)
    if traces:
        print(f"written: {stem}_<sampler>_trace_s<sample>_h<hour>.png / .npz")


if __name__ == "__main__":
    main()
