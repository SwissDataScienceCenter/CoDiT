import os
import sys

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm

sys.path.append("/mydata/adopt/christian/adopt")
from adopt import config_paths


def get_dataset_config(dataset: str) -> dict:
    if dataset == "open_ocean":
        models = [
            "naive",
            "hycom_parcels",
            "glorys_parcels",
            "linear_gauss",
            "conv_net",
            "codit_GRU_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind",
            "codit_Transformer_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind",
        ]
        time_steps = [72,]
    elif dataset == "coastline":
        models = [
            "naive",
            "hycom_parcels",
            "glorys_parcels",
            "linear_gauss",
            "conv_net",
            "codit_GRU_coastline_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry",
            "codit_Transformer_coastline_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry",
        ]
        time_steps = [24,]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    label_dict = {
        "naive": "Naive",
        "hycom_parcels": "HYCOM",
        "glorys_parcels": "GLORYS",
        "conv_net": "CNN",
        "linear_gauss": "WDM",
        "codit_GRU_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind": "CoDiT (GRU)",
        "codit_Transformer_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind": "CoDiT (TF)",
        "codit_GRU_coastline_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry": "CoDiT (GRU)",
        "codit_Transformer_coastline_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry": "CoDiT (TF)",
    }
    color_dict = {
        "CNN": "C1",
        "WDM": "C2",
        "CoDiT (GRU)": "C3",
        "CoDiT (TF)": "C4",
    }

    return {
        "models": models,
        "time_steps": time_steps,
        "labels": label_dict,
        "colors": color_dict,
    }


def get_overlapping_result_files(result_path: str, required_models: list, truth_model: str = "hycom_parcels") -> list:
    model_sets = []
    for model in [truth_model] + required_models:
        model_path = os.path.join(result_path, model)
        if not os.path.isdir(model_path):
            print(f"Warning: model folder not found, skipping from overlap: {model_path}")
            continue
        model_sets.append(set(os.listdir(model_path)))

    if not model_sets:
        return []

    overlap = set.intersection(*model_sets)
    return sorted(overlap)


def compute_pit_values(result_path: str, result_file_names: list, prob_models: list, cache_path: str) -> dict:
    if os.path.exists(cache_path):
        print(f"Loading PIT values from {cache_path}")
        pit_data = np.load(cache_path, allow_pickle=True)
        return {
            m: {
                "lat": pit_data[f"{m}_lat"],
                "lon": pit_data[f"{m}_lon"],
            }
            for m in prob_models
        }

    pit_dict = {}
    for model in prob_models:
        model_path = os.path.join(result_path, model)
        if not os.path.isdir(model_path):
            print(f"Warning: model folder missing, skipping PIT for {model}")
            continue

        available_files = set(os.listdir(model_path))
        model_file_names = [f for f in result_file_names if f in available_files]
        if len(model_file_names) == 0:
            print(f"Warning: no overlapping files for {model}, skipping")
            continue

        pit_dict[model] = {"lat": [], "lon": []}
        for result_file_name in tqdm(model_file_names, desc=f"PIT {model}"):
            if "glorys" in model:
                true_file = np.load(
                    os.path.join(result_path, "glorys_parcels", result_file_name),
                    allow_pickle=True,
                )
            else:
                true_file = np.load(
                    os.path.join(result_path, "hycom_parcels", result_file_name),
                    allow_pickle=True,
                )

            pred_file = np.load(
                os.path.join(result_path, model, result_file_name),
                allow_pickle=True,
            )

            true_future = true_file["true_coords"][:, 1:, :]
            pred_coords = pred_file["pred_coords"]

            pit_lat = np.mean(pred_coords[:, :, :, 0] < true_future[:, None, :, 0], axis=1)
            pit_lon = np.mean(pred_coords[:, :, :, 1] < true_future[:, None, :, 1], axis=1)

            pit_dict[model]["lat"].append(pit_lat)
            pit_dict[model]["lon"].append(pit_lon)

        pit_dict[model]["lat"] = np.concatenate(pit_dict[model]["lat"], axis=0)
        pit_dict[model]["lon"] = np.concatenate(pit_dict[model]["lon"], axis=0)

    if len(pit_dict) == 0:
        raise RuntimeError("No PIT values were computed. Check result folders and model names.")

    flat = {}
    for model in pit_dict:
        flat[f"{model}_lat"] = pit_dict[model]["lat"]
        flat[f"{model}_lon"] = pit_dict[model]["lon"]
    np.savez(cache_path, **flat)
    print(f"Saved PIT cache to {cache_path}")

    return pit_dict


def plot_pit_histograms(
    pit_dict: dict,
    prob_models: list,
    pit_time_steps: list,
    label_dict: dict,
    color_dict: dict,
    dataset: str,
) -> None:
    prob_models = [m for m in prob_models if m in pit_dict]
    if len(prob_models) == 0:
        raise RuntimeError("No available probabilistic models to plot after PIT computation.")

    n_models = len(prob_models)
    n_rows = len(pit_time_steps)

    fig, axes = plt.subplots(
        n_rows,
        n_models,
        figsize=(3 * n_models, 2.4 * n_rows + .5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for j, horizon in enumerate(pit_time_steps):
        t_idx = horizon - 1
        for i, model in enumerate(prob_models):
            label = label_dict.get(model, model)
            color = color_dict.get(label, "C0")
            ax = axes[j, i]

            pit_lat = pit_dict[model]["lat"][:, t_idx]
            pit_lon = pit_dict[model]["lon"][:, t_idx]
            valid = ~(np.isnan(pit_lat) | np.isnan(pit_lon))

            ax.hist(
                pit_lat[valid],
                bins=10,
                range=(0, 1),
                density=True,
                alpha=0.7,
                color=color,
            )
            ax.hist(
                pit_lon[valid],
                bins=10,
                range=(0, 1),
                density=True,
                alpha=0.4,
                color="gray",
            )
            ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
            ax.set_xlim(0, 1)

            if j == 0:
                ax.set_title(label)
            if i == 0:
                ax.set_ylabel(f"{horizon}h\nDensity")
            if j == n_rows - 1:
                ax.set_xlabel("PIT")

    title = "Open ocean" if dataset == "open_ocean" else "Coastal"
    fig.suptitle(f"PIT Histograms ({title})", y=1.01)

    legend_handles = [
        Rectangle((0, 0), 1, 1, fc="k", alpha=0.7),
        Rectangle((0, 0), 1, 1, fc="gray", alpha=0.4),
        Line2D([0], [0], color="k", linestyle="--"),
    ]
    fig.legend(legend_handles, ["Lat", "Lon", "Uniform"], loc="upper right", frameon=True)

    plt.tight_layout()


def plot_hdr_calibration(
    ground_truth,
    samples,
    n_alphas=20,
    n_kde_samples=200,
    ax=None,
):
    """HDR calibration plot for 2D trajectory predictions."""
    n_samples, n_particles, _ = samples.shape
    alphas = np.linspace(0.05, 0.95, n_alphas)
    observed_coverage = np.zeros(n_alphas)
    valid_count = 0

    for i in range(n_samples):
        s = samples[i]

        if n_particles > n_kde_samples:
            idx = np.random.choice(n_particles, n_kde_samples, replace=False)
            s_kde = s[idx]
        else:
            s_kde = s

        try:
            kde = gaussian_kde(s_kde.T)
            sample_densities = kde(s_kde.T)
            gt_density = float(kde(ground_truth[i][:, None])[0])
        except np.linalg.LinAlgError:
            # Degenerate sample clouds cannot be handled by KDE; skip this item.
            continue

        valid_count += 1
        for j, alpha in enumerate(alphas):
            threshold = np.quantile(sample_densities, 1 - alpha)
            observed_coverage[j] += float(gt_density >= threshold)

    if valid_count > 0:
        observed_coverage /= valid_count
    else:
        observed_coverage[:] = np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.plot(alphas, observed_coverage, "o-", lw=2, markersize=5, label="Model")
    ax.fill_between(
        alphas,
        alphas,
        observed_coverage,
        alpha=0.15,
        color="steelblue",
    )

    ax.set_xlabel("Nominal coverage (alpha)", fontsize=11)
    ax.set_ylabel("Observed coverage", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return fig, ax, (alphas, observed_coverage)


def compute_hdr_inputs(result_path: str, result_file_names: list, prob_models: list, cache_path: str) -> dict:
    if os.path.exists(cache_path):
        print(f"Loading HDR inputs from {cache_path}")
        hdr_data = np.load(cache_path, allow_pickle=True)
        out = {}
        for model in prob_models:
            gt_key = f"{model}_gt"
            pred_key = f"{model}_pred"
            if gt_key in hdr_data and pred_key in hdr_data:
                out[model] = {
                    "gt": hdr_data[gt_key],
                    "pred": hdr_data[pred_key],
                }
        return out

    hdr_inputs = {}
    for model in prob_models:
        model_path = os.path.join(result_path, model)
        if not os.path.isdir(model_path):
            print(f"Warning: model folder missing, skipping HDR for {model}")
            continue

        available_files = set(os.listdir(model_path))
        model_file_names = [f for f in result_file_names if f in available_files]
        if len(model_file_names) == 0:
            print(f"Warning: no overlapping files for HDR {model}, skipping")
            continue

        gt_parts = []
        pred_parts = []

        for result_file_name in tqdm(model_file_names, desc=f"HDR {model}"):
            true_file = np.load(
                os.path.join(result_path, "hycom_parcels", result_file_name),
                allow_pickle=True,
            )
            pred_file = np.load(
                os.path.join(result_path, model, result_file_name),
                allow_pickle=True,
            )

            true_future = true_file["true_coords"][:, 1:, :]
            pred_coords = pred_file["pred_coords"]
            gt_parts.append(true_future)
            pred_parts.append(pred_coords)

        if len(gt_parts) == 0 or len(pred_parts) == 0:
            print(f"Warning: empty HDR data for {model}, skipping")
            continue

        hdr_inputs[model] = {
            "gt": np.concatenate(gt_parts, axis=0),
            "pred": np.concatenate(pred_parts, axis=0),
        }

    if len(hdr_inputs) == 0:
        raise RuntimeError("No HDR inputs were computed. Check result folders and model names.")

    flat = {}
    for model in hdr_inputs:
        flat[f"{model}_gt"] = hdr_inputs[model]["gt"]
        flat[f"{model}_pred"] = hdr_inputs[model]["pred"]
    np.savez(cache_path, **flat)
    print(f"Saved HDR cache to {cache_path}")

    return hdr_inputs


def plot_hdr_calibration_grid(
    hdr_inputs: dict,
    prob_models: list,
    pit_time_steps: list,
    label_dict: dict,
    dataset: str,
    cache_path: str = None,
    max_eval_samples: int = 50,
) -> None:
    prob_models = [m for m in prob_models if m in hdr_inputs]
    if len(prob_models) == 0:
        raise RuntimeError("No available probabilistic models to plot after HDR computation.")

    n_models = len(prob_models)
    n_cols = len(pit_time_steps)
    fig, axes = plt.subplots(
        n_models,
        n_cols,
        figsize=(3.2 * n_cols, 3.0 * n_models),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    curve_cache = {}
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Loading HDR curves from {cache_path}")
        cache_data = np.load(cache_path, allow_pickle=True)
        for key in cache_data.files:
            curve_cache[key] = cache_data[key]

    curve_updates = {}
    rng = np.random.default_rng(0)

    for i, model in enumerate(prob_models):
        gt = hdr_inputs[model]["gt"]
        pred = hdr_inputs[model]["pred"]
        label = label_dict.get(model, model)

        for j, horizon in enumerate(pit_time_steps):
            ax = axes[i, j]
            t_idx = horizon - 1
            if gt.shape[1] <= t_idx or pred.shape[2] <= t_idx:
                ax.set_axis_off()
                continue

            gt_h = gt[:, t_idx, :]
            pred_h = pred[:, :, t_idx, :]
            valid = ~(
                np.isnan(gt_h).any(axis=1)
                | np.isnan(pred_h).any(axis=(1, 2))
            )

            if np.sum(valid) == 0:
                ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                valid_idx = np.flatnonzero(valid)
                if max_eval_samples is not None and len(valid_idx) > max_eval_samples:
                    valid_idx = rng.choice(valid_idx, size=max_eval_samples, replace=False)

                cache_key_alpha = f"{model}_h{horizon}_alpha"
                cache_key_obs = f"{model}_h{horizon}_obs"

                if cache_key_alpha in curve_cache and cache_key_obs in curve_cache:
                    alphas = curve_cache[cache_key_alpha]
                    observed_coverage = curve_cache[cache_key_obs]
                    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
                    ax.plot(alphas, observed_coverage, "o-", lw=2, markersize=5, label="Model")
                    ax.fill_between(alphas, alphas, observed_coverage, alpha=0.15, color="steelblue")
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_aspect("equal")
                    ax.grid(True, alpha=0.3)
                else:
                    _, _, (alphas, observed_coverage) = plot_hdr_calibration(
                        ground_truth=gt_h[valid_idx],
                        samples=pred_h[valid_idx],
                        n_alphas=20,
                        n_kde_samples=200,
                        ax=ax,
                    )
                    curve_updates[cache_key_alpha] = alphas
                    curve_updates[cache_key_obs] = observed_coverage

                if i == 0 and j == n_cols - 1:
                    ax.legend(loc="lower right", fontsize=8)
                else:
                    leg = ax.get_legend()
                    if leg is not None:
                        leg.remove()

            if i == 0:
                ax.set_title(f"{horizon}h")
            if j == 0:
                ax.set_ylabel(f"{label}\nObserved")
            if i == n_models - 1:
                ax.set_xlabel("Nominal (alpha)")

    title = "Open ocean" if dataset == "open_ocean" else "Coastal"
    fig.suptitle(f"HDR Calibration ({title})", y=1.01)
    plt.tight_layout()

    if cache_path is not None and len(curve_updates) > 0:
        all_curves = {}
        all_curves.update(curve_cache)
        all_curves.update(curve_updates)
        np.savez(cache_path, **all_curves)
        print(f"Saved HDR curves cache to {cache_path}")


def main() -> None:
    dataset = "coastline"
    split = "test"

    cfg = get_dataset_config(dataset)
    models = cfg["models"]
    label_dict = cfg["labels"]
    color_dict = cfg["colors"]
    pit_time_steps = cfg["time_steps"]

    prob_models = [
        m
        for m in models
        if m in {"conv_net", "linear_gauss"} or "codit_" in m
    ]

    result_path = os.path.join(config_paths.RESULT_PATH, dataset, split)
    result_file_names = get_overlapping_result_files(result_path, prob_models, truth_model="hycom_parcels")
    if len(result_file_names) == 0:
        raise RuntimeError(
            f"No overlapping files found in {result_path} for models: {prob_models}"
        )
    print(f"Using {len(result_file_names)} overlapping result files")

    pit_cache_path = os.path.join(config_paths.RESULT_PATH, f"ddpm_{dataset}_pit.npz")
    pit_dict = compute_pit_values(
        result_path=result_path,
        result_file_names=result_file_names,
        prob_models=prob_models,
        cache_path=pit_cache_path,
    )

    hdr_cache_path = os.path.join(config_paths.RESULT_PATH, f"ddpm_{dataset}_hdr_inputs.npz")
    hdr_inputs = compute_hdr_inputs(
        result_path=result_path,
        result_file_names=result_file_names,
        prob_models=prob_models,
        cache_path=hdr_cache_path,
    )

    plot_pit_histograms(
        pit_dict=pit_dict,
        prob_models=prob_models,
        pit_time_steps=pit_time_steps,
        label_dict=label_dict,
        color_dict=color_dict,
        dataset=dataset,
    )

    out_pdf = os.path.join(config_paths.FIG_PATH, f"ddpm_{dataset}_pit_histograms.pdf")
    out_png = os.path.join(config_paths.FIG_PATH, f"ddpm_{dataset}_pit_histograms.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    plot_hdr_calibration_grid(
        hdr_inputs=hdr_inputs,
        prob_models=prob_models,
        pit_time_steps=pit_time_steps,
        label_dict=label_dict,
        dataset=dataset,
        cache_path=os.path.join(config_paths.RESULT_PATH, f"ddpm_{dataset}_hdr_curves.npz"),
        max_eval_samples=50,
    )

    out_hdr_pdf = os.path.join(config_paths.FIG_PATH, f"ddpm_{dataset}_hdr_calibration.pdf")
    out_hdr_png = os.path.join(config_paths.FIG_PATH, f"ddpm_{dataset}_hdr_calibration.png")
    plt.savefig(out_hdr_pdf, bbox_inches="tight")
    plt.savefig(out_hdr_png, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved: {out_hdr_pdf}")
    print(f"Saved: {out_hdr_png}")


if __name__ == "__main__":
    main()
