# %%
# %load_ext autoreload
# %autoreload 2
import sys
sys.path.append("/mydata/adopt/christian/adopt")
import numpy as np
from matplotlib import pyplot as plt
from adopt import config_paths
from codit.data_module import config_funcs, data_transform
from codit.data_module.dataset import OceanDataset
from codit.eval_module import metrics
from jax import numpy as jnp
from jax import vmap
import xarray as xr
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# %%
# ─── Configuration ────────────────────────────────────────────────────────────
dataset = "open_ocean"   # "open_ocean" or "coastline"
split = "test"
#metric = "energy_score"  # "mean_distance", "energy_score", or "skill_score"

if dataset == "open_ocean":
    primary_model = (
        f"codit_GRU_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind"
    )
    length = 72
elif dataset == "coastline":
    primary_model = (
        f"codit_Transformer_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry"
    )
    length = 24

# Add extra models here to overlay them in the stratified plots.
# Example: extra_models = [f"codit_Transformer_{dataset}_conv_adam_K100_..."]
extra_models = ["glorys_parcels"]
models_to_plot = [primary_model] + extra_models

label_dict = {
    "codit_GRU_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind": "CoDiT (GRU)",
    "codit_Transformer_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind": "CoDiT (TF)",
    "codit_GRU_coastline_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry": "CoDiT (GRU)",
    "codit_Transformer_coastline_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry": "CoDiT (TF)",
    "naive": "Naive",
    "hycom_parcels": "HYCOM",
    "glorys_parcels": "GLORYS",
    "conv_net": "CNN",
    "linear_gauss": "WDM",
}
color_dict = {
    "Naive": "black",
    "HYCOM": "gray",
    "GLORYS": "C8",
    "CNN": "C1",
    "WDM": "C2",
    "CoDiT (GRU)": "C3",
    "CoDiT (TF)": "C4",
}

metric_labels = {
    "mean_distance": "Mean distance [km]",
    "energy_score": "Energy score",
    "skill_score": "Skill score",
}
result_path = os.path.join(config_paths.RESULT_PATH, dataset, split)
config_path_dir = os.path.join(config_paths.CONFIG_PATH, dataset, split)

# %%
# ─── Helper: load trajectory metadata from a config ──────────────────────────
def load_trajectory_metadata(config: dict, true_start_positions: np.ndarray, length: int) -> dict:
    """Match result trajectories to their source snippets and return per-trajectory metadata.

    Recreates the same non-overlapping snippets as OceanDataset._load_trajectories, then
    matches each result trajectory's start (lat, lon) to the nearest snippet.

    Returns arrays ``start_month`` (1–12), ``drogue_status`` (1.0 = drogue, 0.0 = no
    drogue, NaN = unknown), and ``start_times`` (np.datetime64, NaT = unknown),
    aligned to ``true_start_positions``.
    """
    n = len(true_start_positions)
    start_month = np.full(n, np.nan)
    drogue_status = np.full(n, np.nan)
    start_times = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

    traj_file = config.get("trajectories_file", "")
    if not os.path.exists(traj_file):
        return {"start_month": start_month, "drogue_status": drogue_status, "start_times": start_times}

    traj_ds = xr.open_dataset(traj_file)
    traj_df = pd.DataFrame(
        traj_ds[["ID", "time", "latitude", "longitude", "DrogueBallast"]].to_array().T,
        columns=["ID", "time", "latitude", "longitude", "DrogueBallast"],
        index=traj_ds["time"].values,
    )

    grouped = traj_df.groupby("ID")
    length_to_extract = length + 1
    snippet_starts = []
    snippet_months = []
    snippet_drogues = []
    snippet_start_times = []

    for _, group in grouped:
        for i in range(0, len(group) - length_to_extract + 1, length_to_extract):
            snippet = group.iloc[i : i + length_to_extract]
            if np.all(snippet["time"].diff()[1:] * 1e-9 == 3600):
                snippet_starts.append(
                    (snippet["latitude"].iloc[0], snippet["longitude"].iloc[0])
                )
                snippet_months.append(snippet.index[0].month)
                val = 0 if snippet["DrogueBallast"].iloc[0] == "0 kg" else 1
                snippet_drogues.append(np.nan if pd.isna(val) else float(val > 0))
                snippet_start_times.append(np.datetime64(snippet.index[0], "ns"))

    traj_ds.close()

    if len(snippet_starts) == 0:
        return {"start_month": start_month, "drogue_status": drogue_status, "start_times": start_times}

    snippet_starts = np.array(snippet_starts)       # (M, 2)
    snippet_months = np.array(snippet_months)
    snippet_drogues = np.array(snippet_drogues)
    snippet_start_times = np.array(snippet_start_times, dtype="datetime64[ns]")

    for i, pos in enumerate(true_start_positions):
        dists = np.sqrt(
            (snippet_starts[:, 0] - pos[0]) ** 2 + (snippet_starts[:, 1] - pos[1]) ** 2
        )
        idx = int(np.argmin(dists))
        if dists[idx] < 1e-3:
            start_month[i] = snippet_months[idx]
            drogue_status[i] = snippet_drogues[idx]
            start_times[i] = snippet_start_times[idx]

    return {"start_month": start_month, "drogue_status": drogue_status, "start_times": start_times}


def load_glorys_eke_from_config(
    config: dict,
    start_positions: np.ndarray,   # (N, 2)  lat, lon
    start_times: np.ndarray,        # (N,)    np.datetime64[ns]
    length: int,
) -> np.ndarray:
    """Lookup GLORYS EKE = 0.5*(uo²+vo²) for each trajectory using OceanDataset.

    Instantiates an OceanDataset from the config to obtain the merged context
    (``context_xarray``), which includes GLORYS ``uo``/``vo`` fields.  For every
    trajectory the time range [start_time, start_time + length hours] is interpolated
    at the trajectory's start lat/lon.  Mean 0.5*(uo²+vo²) over the available time
    steps is returned as a scalar EKE per trajectory.

    If a trajectory's start_time is NaT (metadata not found) or outside the dataset's
    time range, NaN is returned for that trajectory.

    Args:
        config: Completed config dictionary (will be deep-copied internally to avoid
            mutation by OceanDataset.__init__).
        start_positions: Array of shape (N, 2) with (lat, lon) for each trajectory start.
        start_times: Array of shape (N,) with np.datetime64 start timestamps.
        length: Trajectory length in hours.

    Returns:
        eke: np.ndarray of shape (N,) with mean GLORYS EKE [m²/s²] per trajectory.
    """
    n = len(start_positions)
    eke_out = np.full(n, np.nan)

    valid_mask = ~np.isnat(start_times)
    if not valid_mask.any():
        return eke_out

    dataset = OceanDataset(config, length)
    ds = dataset.context_xarray[["uo", "vo"]]

    # Ensure latitude is ascending for xarray.interp
    if float(ds["latitude"][0]) > float(ds["latitude"][-1]):
        ds = ds.isel(latitude=slice(None, None, -1))

    ds.load()  # pull into memory once — much faster than per-trajectory round trips

    for i in np.where(valid_mask)[0]:
        t0 = pd.Timestamp(start_times[i])
        t_end = t0 + pd.Timedelta(hours=length)
        lat_i = float(start_positions[i, 0])
        lon_i = float(start_positions[i, 1])
        try:
            ds_pt = ds.sel(time=slice(t0, t_end)).interp(
                latitude=lat_i,
                longitude=lon_i,
                method="linear",
            )
            uo_vals = ds_pt["uo"].values
            vo_vals = ds_pt["vo"].values
            valid = ~np.isnan(uo_vals) & ~np.isnan(vo_vals)
            if valid.any():
                eke_out[i] = float(np.mean(0.5 * (uo_vals[valid] ** 2 + vo_vals[valid] ** 2)))
        except Exception:
            pass  # leave as NaN

    return eke_out


# %%
# ─── Load result files: errors + metadata ────────────────────────────────────
np.random.seed(0)

conv_result_file_names = os.listdir(os.path.join(result_path, "conv_net"))
lingauss_result_file_names = os.listdir(os.path.join(result_path, "linear_gauss"))
result_file_names = sorted(
    set(conv_result_file_names).intersection(set(lingauss_result_file_names))
)
print(f"Result files: {len(result_file_names)}")

# Models needed for the validity filter (same logic as result_plots.py)
filter_models = ["conv_net", "hycom_parcels", "glorys_parcels"]
all_models = models_to_plot + filter_models + ["naive"]
metrics_dict = {m: {"mean_distance": [], "energy_score": [], "skill_score": []} for m in all_models}

all_true_coords = []
all_start_months = []
all_drogue_status = []
all_snippet_start_times = []
all_glorys_eke = []

for result_file_name in tqdm(result_file_names):
    # true_coords come from hycom_parcels (as in result_plots.py)
    hycom_file = np.load(
        os.path.join(result_path, "hycom_parcels", result_file_name), allow_pickle=True
    )
    true_coords = hycom_file["true_coords"]  # (N, T+1, 2)
    N = true_coords.shape[0]
    T = true_coords.shape[1] - 1

    # naive: predict constant position (start repeated)
    pred_naive = np.tile(true_coords[:, None, :1], (1, 1, T, 1))
    m_naive = metrics.get_metrics(true_coords, pred_naive)
    for key in metrics_dict["naive"]:
        metrics_dict["naive"][key].append(m_naive[key])

    for model in all_models:
        if model == "naive":
            continue
        model_dir = os.path.join(result_path, model)
        model_file = os.path.join(model_dir, result_file_name)
        if not os.path.exists(model_file):
            for key in metrics_dict[model]:
                metrics_dict[model][key].append(np.full((N, T), np.nan))
            continue
        if model == "glorys_parcels":
            glorys_file = np.load(model_file, allow_pickle=True)
            tc = glorys_file["true_coords"]
            pred = glorys_file["pred_coords"]
            print(tc.shape, pred.shape)
            m = metrics.get_metrics(tc, pred)
            print(m['mean_distance'].shape, m['energy_score'].shape, m['skill_score'].shape)
        else:
            result_file = np.load(model_file, allow_pickle=True)
            pred = result_file["pred_coords"]
            print(true_coords.shape, pred.shape)
            m = metrics.get_metrics(true_coords, pred)
            print(m['mean_distance'].shape, m['energy_score'].shape, m['skill_score'].shape)
        for key in metrics_dict[model]:
            metrics_dict[model][key].append(m[key])

    all_true_coords.append(true_coords)

    # Load trajectory metadata (dates, drogue) via the matching config
    data_name = result_file_name.replace("_result.npz", "")
    #try:
    config = config_funcs.load_config(data_name, config_path_dir)
    meta = load_trajectory_metadata(config, true_coords[:, 0, :], length)
    # except Exception:
    #     meta = {"start_month": np.full(N, np.nan), "drogue_status": np.full(N, np.nan)}
    all_start_months.append(meta["start_month"])
    all_drogue_status.append(meta["drogue_status"])
    all_snippet_start_times.append(meta["start_times"])
    glorys_eke = load_glorys_eke_from_config(
        config, true_coords[:, 0, :], meta["start_times"], length
    )
    all_glorys_eke.append(glorys_eke)

for model in all_models:
    for key in metrics_dict[model]:
        metrics_dict[model][key] = np.concatenate(metrics_dict[model][key], axis=0)

all_true_coords = np.concatenate(all_true_coords, axis=0)  # (N_total, T+1, 2)
all_start_months = np.concatenate(all_start_months)
all_drogue_status = np.concatenate(all_drogue_status)
all_snippet_start_times = np.concatenate(all_snippet_start_times)
all_glorys_eke = np.concatenate(all_glorys_eke)  # (N_total,)
print(f"Total trajectories: {len(all_true_coords)}")

# %%
# ─── Compute EKE = 0.5*(U²+V²) from first displacement in true_coords ────────
# velocity_vector_from_pos_time returns (vx, vy) [m/s] via haversine distance + bearing
vel = np.asarray(
    vmap(vmap(data_transform.velocity_vector_from_pos_time, in_axes=(0, 0, None)), in_axes=(0, 0, None))(
        jnp.asarray(all_true_coords[:, :-1, :]),  # pos_start  (lat, lon)
        jnp.asarray(all_true_coords[:, 1:, :]),  # pos_end
        3600.0,                                  # Δt = 1 hour in seconds
    )
)  # (N, 2)
all_eke = 0.5 * (vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)  # [m²/s²]
all_eke = np.mean(all_eke, axis=1)  # average EKE over trajectory duration → (N,)
# %%
# ─── Validity filter (identical to result_plots.py) ──────────────────────────
for metric in ["mean_distance", "energy_score", "skill_score"]:
    print(metrics_dict["glorys_parcels"][metric].shape)
    valid_mask = (
        ~np.any(np.isnan(metrics_dict["conv_net"][metric]), axis=1)
        & ~np.any(np.isnan(metrics_dict["hycom_parcels"][metric]), axis=1)
        & ~np.any(np.isnan(metrics_dict["glorys_parcels"][metric]), axis=1)
        & (metrics_dict["naive"]["mean_distance"][:, -1] > 500)
    )
    valid_idx = np.where(valid_mask)[0]
    print(f"Valid trajectories: {len(valid_idx)}")

    # %%
    # ─── Stratification setup ─────────────────────────────────────────────────────
    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    LAT_BINS = [-60, -30, 0, 15, 30, 60]
    LAT_LABELS = [f"{a}° to {b}°" for a, b in zip(LAT_BINS[:-1], LAT_BINS[1:])]
    N_EKE_BINS = 3

    # Metadata for valid trajectories only
    lat_valid = all_true_coords[valid_idx, 0, 0]
    month_valid = all_start_months[valid_idx]
    drogue_valid = all_drogue_status[valid_idx]
    eke_valid = all_eke[valid_idx]

    # EKE bin edges (quantile-based so each bin has ~equal count)
    eke_edges = np.nanquantile(eke_valid, np.linspace(0, 1, N_EKE_BINS + 1))
    eke_edges[-1] += 1e-9  # make last edge inclusive

    def bin_masks(data, edges):
        return [(data >= lo) & (data < hi) for lo, hi in zip(edges[:-1], edges[1:])]

    lat_masks = bin_masks(lat_valid, LAT_BINS)
    eke_masks = bin_masks(eke_valid, eke_edges)
    eke_bin_labels = [
        f"Q{i+1}\n{eke_edges[i]:.2f}–{eke_edges[i+1]:.2f}" if i < N_EKE_BINS - 1
        else f"Q{i+1}\n>{eke_edges[i]:.2f}"
        for i in range(N_EKE_BINS)
    ]

    # %%
    # ─── Stratified box plots (2 × 2) ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_suffix = "open ocean" if dataset == "open_ocean" else "coastal"
    fig.suptitle(f"CoDiT prediction error stratification — {title_suffix} dataset", fontsize=13)

    n_models = len(models_to_plot)
    box_width = 0.6 / max(n_models, 1)
    scale = 1.0 if metric == "skill_score" else 1e-3  # scale distances to km for plotting

    bp_style = dict(patch_artist=True, manage_ticks=False, showfliers=False)

    for imodel, model in enumerate(models_to_plot):
        label = label_dict.get(model, model)
        color = color_dict.get(label, "C0")
        err = metrics_dict[model][metric][valid_idx, -1] * scale
        offset = (imodel - (n_models - 1) / 2) * box_width

        def safe_bins(masks, data):
            return [data[m] if m.any() else np.array([np.nan]) for m in masks]

        bp_kwargs = dict(
            boxprops=dict(color=color, facecolor=color),
            medianprops=dict(color="w", linewidth=2),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            **bp_style,
        )

        # ── Latitude ──────────────────────────────────────────────────────────────
        ax = axes[0, 0]
        pos = np.arange(len(LAT_LABELS)) + offset
        bp = ax.boxplot(safe_bins(lat_masks, err), positions=pos, widths=box_width, **bp_kwargs)
        bp["boxes"][0].set_label(label)

        # ── Month ──────────────────────────────────────────────────────────────────
        ax = axes[0, 1]
        pos = np.arange(12) + offset
        month_data = [err[month_valid == m + 1] for m in range(12)]
        month_data = [d if len(d) > 0 else np.array([np.nan]) for d in month_data]
        bp = ax.boxplot(month_data, positions=pos, widths=box_width, **bp_kwargs)
        bp["boxes"][0].set_label(label)

        # ── Drogue ─────────────────────────────────────────────────────────────────
        ax = axes[1, 0]
        drogue_data = [
            err[drogue_valid == 1.0] if (drogue_valid == 1.0).any() else np.array([np.nan]),
            err[drogue_valid == 0.0] if (drogue_valid == 0.0).any() else np.array([np.nan]),
        ]
        pos = np.array([0, 1]) + offset
        bp = ax.boxplot(drogue_data, positions=pos, widths=box_width, **bp_kwargs)
        bp["boxes"][0].set_label(label)

        # ── EKE ────────────────────────────────────────────────────────────────────
        ax = axes[1, 1]
        pos = np.arange(N_EKE_BINS) + offset
        bp = ax.boxplot(safe_bins(eke_masks, err), positions=pos, widths=box_width, **bp_kwargs)
        bp["boxes"][0].set_label(label)

    # ── Axis formatting ────────────────────────────────────────────────────────────
    ax_lat, ax_month = axes[0, 0], axes[0, 1]
    ax_drogue, ax_eke = axes[1, 0], axes[1, 1]

    ax_lat.set_xticks(range(len(LAT_LABELS)))
    ax_lat.set_xticklabels(LAT_LABELS, rotation=45, ha="right")
    ax_lat.set_xlabel("Latitude (bin centre)")
    ax_lat.set_ylabel("Final " + metric_labels.get(metric, metric))
    ax_lat.set_title("Latitude")
    ax_lat.set_ylim(bottom=0)
    ax_lat.grid(True, axis="y", alpha=0.3)

    ax_month.set_xticks(range(12))
    ax_month.set_xticklabels(MONTH_NAMES, rotation=45, ha="right")
    ax_month.set_xlabel("Month")
    ax_month.set_ylabel("Final " + metric_labels.get(metric, metric))
    ax_month.set_title("Month")
    ax_month.set_ylim(bottom=0)
    ax_month.grid(True, axis="y", alpha=0.3)

    ax_drogue.set_xticks([0, 1])
    ax_drogue.set_xticklabels(["Drogue", "No drogue"])
    ax_drogue.set_xlabel("Drogue status")
    ax_drogue.set_ylabel("Final " + metric_labels.get(metric, metric))
    ax_drogue.set_title("Drogue vs no drogue")
    ax_drogue.set_ylim(bottom=0)
    ax_drogue.grid(True, axis="y", alpha=0.3)

    ax_eke.set_xticks(range(N_EKE_BINS))
    ax_eke.set_xticklabels(eke_bin_labels)
    ax_eke.set_xlabel("EKE quantile bin  (Q1 = low, Q5 = high)")
    ax_eke.set_ylabel("Final " + metric_labels.get(metric, metric))
    ax_eke.set_title(r"Eddy kinetic energy  $\frac{1}{2}(U^2+V^2)$")
    ax_eke.set_ylim(bottom=0)
    ax_eke.grid(True, axis="y", alpha=0.3)

    if n_models > 1:
        for ax in axes.flat:
            ax.legend(fontsize=8)

    plt.tight_layout()

    out_base = os.path.join(config_paths.FIG_PATH, f"stratified_errors_{dataset}_{metric}")
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    print(f"Saved to {out_base}.png / .pdf")
    plt.show()

# %%
