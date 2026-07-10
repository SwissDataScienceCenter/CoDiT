"""Aggregate Fig 4 / Table 2 metrics for CoDiT (GRU/TF) + WDM (LinearGauss).

Reads .npz files from three model result directories under
    <RESULT_PATH>/<dataset>/<split>/<result_subdir>/<model_dir>/
computes final-horizon median PE / ES / SS across the valid trajectories,
prints a comparison table against Table 2 from the paper, and writes
fig4_summary.csv into the same subdirectory.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from jax import numpy as jnp
from jax import vmap

from codit.data_module import config_paths
from codit.data_module import data_transform
from codit.eval_module import metrics


PAPER_TABLE_2 = {
    "open_ocean": {
        "CoDiT (GRU)": {"PE": 26.135, "ES": 18.092, "SS": 0.466},
        "CoDiT (TF)":  {"PE": 26.418, "ES": 18.154, "SS": 0.460},
        "WDM":         {"PE": 27.196, "ES": 18.666, "SS": 0.406},
    },
    "coastline": {
        "CoDiT (GRU)": {"PE": 8.638, "ES": 5.746, "SS": 0.268},
        "CoDiT (TF)":  {"PE": 8.032, "ES": 5.339, "SS": 0.305},
        "WDM":         {"PE": 9.651, "ES": 7.842, "SS": 0.167},
    },
}


def _input_var_str(dataset: str) -> str:
    if dataset == "open_ocean":
        return "hycom_current_copernicus_current_era5_wind"
    if dataset == "coastline":
        return "hycom_current_copernicus_current_era5_wind_bathymetry"
    raise ValueError(f"unknown dataset: {dataset!r}")


def _model_dir_map(dataset: str) -> dict:
    ivs = _input_var_str(dataset)
    return {
        "CoDiT (GRU)": f"codit_GRU_{dataset}_conv_adam_K100_{ivs}",
        "CoDiT (TF)":  f"codit_Transformer_{dataset}_conv_adam_K100_{ivs}",
        "WDM":         "linear_gauss",
    }


def _load_model_npz(model_dir: Path, patch_files: list) -> tuple:
    """Concatenate true_coords and pred_coords across a list of patch files."""
    trues, preds = [], []
    for name in patch_files:
        data = np.load(model_dir / name, allow_pickle=True)
        trues.append(np.asarray(data["true_coords"]))
        preds.append(np.asarray(data["pred_coords"]))
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def _naive_distances(true_coords: np.ndarray) -> np.ndarray:
    """Distance (metres) between start position and each subsequent step.

    true_coords: (N, T, 2). Returns (N, T-1).
    """
    start = jnp.asarray(true_coords[:, :1, :])           # (N, 1, 2)
    rest = jnp.asarray(true_coords[:, 1:, :])            # (N, T-1, 2)
    hav = vmap(vmap(data_transform.haversine_distance, in_axes=(None, 0)))
    return np.asarray(hav(start[:, 0, :], rest))         # (N, T-1)


def compute_metrics_for_model(
    model_dir: Path, patch_files: list
) -> dict:
    """Return dict with (N_total, T-1) arrays for PE, ES, SS + the concatenated true_coords."""
    true_coords, pred_coords = _load_model_npz(model_dir, patch_files)
    m = metrics.get_metrics(true_coords, pred_coords)
    return {
        "mean_distance": np.asarray(m["mean_distance"]),
        "energy_score":  np.asarray(m["energy_score"]),
        "skill_score":   np.asarray(m["skill_score"]),
        "true_coords":   true_coords,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                   choices=["open_ocean", "coastline"])
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--result_subdir", type=str, default="fig4_repro")
    args = p.parse_args()

    if args.dataset not in PAPER_TABLE_2:
        print(f"error: no paper values for dataset {args.dataset!r}", file=sys.stderr)
        return 2

    base = Path(config_paths.RESULT_PATH) / args.dataset / args.split / args.result_subdir
    model_dirs = _model_dir_map(args.dataset)

    # Verify all three model dirs exist
    resolved = {}
    for label, dname in model_dirs.items():
        d = base / dname
        if not d.is_dir():
            print(f"error: missing model dir: {d}", file=sys.stderr)
            return 3
        resolved[label] = d

    # Intersection of patch filenames across the three model dirs
    file_sets = [
        {f for f in os.listdir(d) if f.endswith("_result.npz")}
        for d in resolved.values()
    ]
    common = sorted(set.intersection(*file_sets))
    if not common:
        print("error: no common patch files across model dirs", file=sys.stderr)
        return 4
    print(f"summarizing {len(common)} common patches for {args.dataset}")

    # Compute metrics per model
    per_model = {}
    for label, d in resolved.items():
        per_model[label] = compute_metrics_for_model(d, common)

    # Sanity: all three must have the same total N and T-1
    shapes = {label: v["mean_distance"].shape for label, v in per_model.items()}
    ref_shape = next(iter(shapes.values()))
    for label, s in shapes.items():
        if s != ref_shape:
            print(
                f"error: {label} mean_distance shape {s} != {ref_shape}",
                file=sys.stderr,
            )
            return 5

    # Validity mask: no-NaN across models AND naive drift > 100 m at any horizon
    valid_no_nan = np.ones(ref_shape[0], dtype=bool)
    for v in per_model.values():
        valid_no_nan &= ~np.any(np.isnan(v["mean_distance"]), axis=1)

    naive = _naive_distances(per_model[next(iter(resolved))]["true_coords"])
    valid_drift = np.any(naive > 100.0, axis=1)  # metres
    valid = valid_no_nan & valid_drift
    n_valid = int(np.sum(valid))
    print(f"valid trajectories after filter: {n_valid} / {ref_shape[0]}")
    if n_valid == 0:
        print("error: no trajectories pass validity filter", file=sys.stderr)
        return 6

    # Final-horizon medians (index -1). Convert metres -> km for PE and ES.
    rows = []
    header = [
        "model",
        "PE_new_km", "PE_paper_km", "dPE_km",
        "ES_new_km", "ES_paper_km", "dES_km",
        "SS_new",    "SS_paper",    "dSS",
    ]
    paper = PAPER_TABLE_2[args.dataset]
    for label in ["CoDiT (GRU)", "CoDiT (TF)", "WDM"]:
        v = per_model[label]
        pe_new = float(np.nanmedian(v["mean_distance"][valid, -1]) / 1000.0)
        es_new = float(np.nanmedian(v["energy_score"][valid, -1])  / 1000.0)
        ss_new = float(np.nanmedian(v["skill_score"][valid,  -1]))
        pe_p, es_p, ss_p = paper[label]["PE"], paper[label]["ES"], paper[label]["SS"]
        rows.append({
            "model":        label,
            "PE_new_km":    f"{pe_new:.3f}",
            "PE_paper_km":  f"{pe_p:.3f}",
            "dPE_km":       f"{pe_new - pe_p:+.3f}",
            "ES_new_km":    f"{es_new:.3f}",
            "ES_paper_km":  f"{es_p:.3f}",
            "dES_km":       f"{es_new - es_p:+.3f}",
            "SS_new":       f"{ss_new:.3f}",
            "SS_paper":     f"{ss_p:.3f}",
            "dSS":          f"{ss_new - ss_p:+.3f}",
        })

    # Print
    col_widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in header}
    line = " | ".join(c.ljust(col_widths[c]) for c in header)
    sep  = "-+-".join("-" * col_widths[c] for c in header)
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(r[c].ljust(col_widths[c]) for c in header))

    # Write CSV alongside the raw .npz outputs
    out_csv = base / "fig4_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
