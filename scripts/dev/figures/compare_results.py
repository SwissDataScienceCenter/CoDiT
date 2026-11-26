"""Compare CoDiT result directories trajectory by trajectory.

Reads the ``*_result.npz`` files written by ``codit_eval.py`` for several models,
keeps the patches present in all of them, checks that they contain the same
trajectories, and reports per model (km):

  - energy score (ES) and median-sample position error (PE) at the given hours,
    median and mean over trajectories (the paper reports the median at 72 h)
  - paired differences to a reference model: mean +- standard error over
    trajectories (negative = better than the reference)

Trajectories whose ensemble is entirely NaN (all samples left the context grid) in
any model are dropped from the paired comparison; the share of NaN samples per
model is reported.

    python scripts/dev/figures/compare_results.py --result_subdir retrain_cmp \\
        --models paper=codit_GRU_open_ocean \\
                 linear=codit_GRU_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind \\
                 cosine=codit_GRU_..._cosine cosine_heun=codit_GRU_..._cosine_heun18
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from codit.data_module import config_paths
from codit.eval_module import metrics


def load(model_dir: Path, patch_files: list) -> tuple[np.ndarray, np.ndarray]:
    trues, preds = [], []
    for name in patch_files:
        data = np.load(model_dir / name, allow_pickle=True)
        trues.append(np.asarray(data["true_coords"]))
        preds.append(np.asarray(data["pred_coords"]))
    return np.concatenate(trues), np.concatenate(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="open_ocean")
    p.add_argument("--split", default="test")
    p.add_argument("--result_subdir", default="")
    p.add_argument("--models", nargs="+", required=True,
                   help="label=result_dir_name pairs; the first is the reference")
    p.add_argument("--hours", nargs="+", type=int, default=[24, 48, 72])
    args = p.parse_args()

    base = Path(config_paths.RESULT_PATH) / args.dataset / args.split / args.result_subdir
    models = dict(m.split("=", 1) for m in args.models)
    dirs = {label: base / name for label, name in models.items()}
    for label, d in dirs.items():
        if not d.is_dir():
            raise SystemExit(f"{label}: no result directory {d}")
    common = sorted(set.intersection(*(
        {f for f in os.listdir(d) if f.endswith("_result.npz")} for d in dirs.values()
    )))
    if not common:
        raise SystemExit("No patch result files common to all models")
    print(f"{len(common)} patches common to all models ({base})")

    es, pe, nan_frac, true_ref = {}, {}, {}, None
    for label, d in dirs.items():
        true, pred = load(d, common)
        if true_ref is None:
            true_ref = true
        elif true.shape != true_ref.shape or not np.allclose(true, true_ref, equal_nan=True):
            raise SystemExit(f"{label}: trajectories differ from the reference model")
        m = metrics.get_metrics(true, pred)  # (N, T-1), metres
        es[label] = np.asarray(m["energy_score"]) / 1000
        pe[label] = np.asarray(m["mean_distance"]) / 1000
        nan_frac[label] = float(np.isnan(pred).any(axis=(-1, -2)).mean())

    labels = list(dirs)
    ref = labels[0]
    valid = np.all([np.isfinite(es[lab]).all(axis=1) for lab in labels], axis=0)
    n = int(valid.sum())
    print(f"{len(true_ref)} trajectories, {n} valid in all models\n")

    rows = []
    hdr = f"{'model':14s} {'NaN smp':>7s} " + " ".join(
        f"{'ES' + str(h) + 'h med/mean':>18s}" for h in args.hours
    ) + " " + f"{'PE' + str(args.hours[-1]) + 'h med':>10s}"
    print(hdr)
    for lab in labels:
        cells = []
        row = {"model": lab, "dir": models[lab], "n": n, "nan_sample_frac": nan_frac[lab]}
        for h in args.hours:
            v = es[lab][valid, h - 1]
            cells.append(f"{np.median(v):8.2f} /{np.mean(v):7.2f}  ")
            row[f"es_{h}h_median"], row[f"es_{h}h_mean"] = np.median(v), np.mean(v)
        pe_last = np.median(pe[lab][valid, args.hours[-1] - 1])
        row[f"pe_{args.hours[-1]}h_median"] = pe_last
        print(f"{lab:14s} {nan_frac[lab]:7.1%} " + " ".join(cells) + f" {pe_last:10.2f}")
        rows.append(row)

    print(f"\nPaired ES difference vs {ref} (mean +- s.e., km; negative = better):")
    for lab in labels[1:]:
        parts = []
        for h in args.hours:
            d = es[lab][valid, h - 1] - es[ref][valid, h - 1]
            se = d.std(ddof=1) / np.sqrt(len(d))
            parts.append(f"{h}h {d.mean():+6.2f} +- {se:4.2f}")
            next(r for r in rows if r["model"] == lab)[f"es_{h}h_diff_vs_{ref}"] = d.mean()
            next(r for r in rows if r["model"] == lab)[f"es_{h}h_diff_se"] = se
        print(f"  {lab:14s} " + " | ".join(parts))

    out = base / "compare_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=str))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
