# %%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("/mydata/adopt/christian/adopt")
import numpy as np
from matplotlib import pyplot as plt
from adopt import config_paths
from codit.data_module import config_funcs
import os
from tqdm import tqdm
from codit.eval_module import metrics

# %%
figs_path = os.path.join(config_paths.DATA_PATH, "figs")
dataset = "coastline"  #'coastline'#
split = 'test'
if dataset == "open_ocean":
    models = ["naive",
            "hycom_parcels", 
            "glorys_parcels",
            "linear_gauss",
            "conv_net", 
            f"codit_GRU_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind",
            f"codit_Transformer_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind", ] 
    label_dict = {
        "naive": "Naive",
        "hycom_parcels": "HYCOM",
        "glorys_parcels": "GLORYS",
        "conv_net": "CNN",
        "linear_gauss": "Linear-Gaussian",
        f"codit_GRU_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind": "CoDiT (GRU)",
        f"codit_Transformer_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind": "CoDiT (TF)",
    }
    length = 72
elif dataset == "coastline":
    models = ["naive",
            "hycom_parcels", 
            "glorys_parcels",
            "linear_gauss",
            "conv_net",
            f"codit_GRU_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry",
            f"codit_Transformer_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry", ]
    label_dict = {
        "naive": "Naive",
        "hycom_parcels": "HYCOM",
        "glorys_parcels": "GLORYS",
        "conv_net": "CNN",
        "linear_gauss": "Linear-Gaussian",
        f"codit_GRU_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry": "CoDiT (GRU)",
        f"codit_Transformer_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry": "CoDiT (TF)",
    }
    length = 24
#colors = ["black", "gray", "C1", "C3", "C4"]
#color_dict = {models[i]: colors[i] for i in range(len(models))}

if dataset == "open_ocean":
    time_steps = [24, 48, 72]
    x_vec = np.linspace(0, 200, 50)
elif dataset == "coastline":
    time_steps = [6, 12, 24]
    x_vec = np.linspace(0, 100, 50)


# def get_energy_score(true_coords, pred_coords):
#     num_samples = jnp.sum(jnp.prod(~jnp.isnan(pred_coords), axis=-1), axis=1)
#     expected_error = (
#         jnp.nansum(
#             vmap(vmap(vmap(jax_transform.haversine_distance), in_axes=(None, 0)))(
#                 true_coords, pred_coords
#             ),
#             axis=1,
#         )
#         / num_samples
#     )
#     num_traj = true_coords.shape[0]
#     expected_cross_distance = []
#     for itraj in range(num_traj):
#         expected_distance_pred = vmap(
#             vmap(vmap(jax_transform.haversine_distance), in_axes=(0, None)),
#             in_axes=(None, 0),
#         )(pred_coords[itraj], pred_coords[itraj])
#         expected_cross_distance.append(
#             jnp.nansum(expected_distance_pred, axis=(0, 1)) / num_samples[itraj] ** 2
#         )
#     expected_cross_distance = jnp.array(expected_cross_distance)
#     return expected_error - 0.5 * expected_cross_distance
#     # expected_distance_pred = jnpvmap(vmap(vmap(jax_transform.haversine_distance), in_axes=(0, None)), in_axes=(None,0))(pred_coords[0], pred_coords[0])

#     # return vmap(vmap(vmap(jax_transform.haversine_distance)), in_axes=(None,0), out_axes=0)(true_coords, pred_coords)
#     # return expected_distance_pred


# def get_haversine_error(true_coords, pred_coords, aggregation="median"):
#     if aggregation == "median":
#         aggr_coords = np.nanmedian(pred_coords, axis=1)
#     elif aggregation == "mean":
#         aggr_coords = np.nanmean(pred_coords, axis=1)
#     return vmap(vmap(jax_transform.haversine_distance))(true_coords, aggr_coords)


# def get_metrics(true_coords, pred_coords) -> dict:
#     # calculate the mean distance
#     mean_distance = get_haversine_error(true_coords, pred_coords)
#     energy_score = get_energy_score(true_coords, pred_coords)
#     return {"mean_distance": mean_distance, "energy_score": energy_score}


# %%

result_path = os.path.join(config_paths.RESULT_PATH, dataset, split)
# result_file_names = os.listdir(
#     os.path.join(result_path,  f"codit_Transformer_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry",
# )
# )
conv_result_file_names = os.listdir(
    os.path.join(result_path,  f"conv_net",
)
)
lingauss_result_file_names = os.listdir(
    os.path.join(result_path,  f"linear_gauss",
)
)

# get the overlap of the two sets of file names
result_file_names = set(conv_result_file_names).intersection(set(lingauss_result_file_names))

print(len(result_file_names))
# %%
metrics_dict = {m: {"mean_distance": [], "energy_score": [], "skill_score": []} for m in models}
for model in models:
    distance = []
    for result_file_name in tqdm(result_file_names):
        if not "glorys" in model:
            hycom_file = np.load(
                os.path.join(result_path, "hycom_parcels", result_file_name), allow_pickle=True
            )
            true_coords = hycom_file["true_coords"]
        else:
            hycom_file = np.load(
                os.path.join(result_path, "glorys_parcels", result_file_name), allow_pickle=True
            )
            true_coords = hycom_file["true_coords"]
        if model != "naive":
            result_file = np.load(
                os.path.join(result_path, model, result_file_name), allow_pickle=True
            )
            pred_coords = result_file["pred_coords"]
        else:
            pred_coords = np.tile(true_coords[:,None,:1], (1,1,true_coords.shape[1]-1,1))
        # distances = haversine_distance_func(result_file['true_coords'][:,:], result_file['pred_coords'][:,:,:])
        # distance.append(np.nanmedian(distances, axis=1))
        # distances = haversine_distance_func(result_file['true_coords'][:,:], np.nanmedian(result_file['pred_coords'][:,:,:], axis=1))
        # distance.append(distances)
        true_coords = hycom_file["true_coords"]
        metrics_batch = metrics.get_metrics(true_coords[:,:], pred_coords)

        for key in metrics_dict[model].keys():
            metrics_dict[model][key].append(metrics_batch[key])
    for key in metrics_dict[model].keys():
        metrics_dict[model][key] = np.concatenate(metrics_dict[model][key], axis=0)
        
# %%
colors = ["gray", "C1", "C8", "C3", "C7", "C4", "C2", "C5"]
if dataset == "open_ocean":
    x_ticks = [12, 24, 36, 48, 60, 72]
elif dataset == "coastline":
    x_ticks = [6, 12, 18, 24]
time_steps = np.arange(1, result_file["pred_coords"].shape[2] + 1)
fig = plt.figure(figsize=(6, 8))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax3 = plt.subplot(3, 1, 3, sharex=ax1)

# label_dict = {
#     "naive": "Naive",
#     "hycom": "HYCOM",
#     "conv_net": "CNN",
#     f"codit_GRU_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry": "CODIT GRU",
#     f"codit_Transformer_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry": "CODIT Transformer",
# }
color_dict = {
    "Naive": "black",
    "HYCOM": "gray",
    "GLORYS": "C8",
    "CNN": "C1",
    "Linear-Gaussian": "C2",
    "CoDiT (GRU)": "C3",
    "CoDiT (TF)": "C4",
}
valid_idx1 = np.asarray(
    np.logical_not(
        np.any(np.isnan(metrics_dict[f"conv_net"]["mean_distance"]), axis=1)
    ) *
    np.logical_not(
        np.any(np.isnan(metrics_dict[f"hycom_parcels"]["mean_distance"]), axis=1)
    ) *
    np.logical_not(
        np.any(np.isnan(metrics_dict[f"glorys_parcels"]["mean_distance"]), axis=1)
    )
    , dtype=bool
)
valid_idx2 = np.asarray(metrics_dict['naive']['mean_distance'][:,-1] > 500)
valid_idx = np.where(np.logical_and(
    valid_idx1,
    valid_idx2
))[0]
for imodel, model in enumerate(models):
    label = label_dict.get(model, model)
    # ax1.plot(
    #     time_steps,
    #     np.nanmedian(metrics_dict[model]["mean_distance"][valid_idx] * 1e-3, axis=0),
    #     color="w",
    #     linewidth=3,
    # )
    ax1.plot(
        time_steps,
        np.nanmedian(metrics_dict[model]["mean_distance"][valid_idx] * 1e-3, axis=0),
        label=label,
        color=color_dict.get(label, "C0"),
        linewidth=2,
        alpha=0.8,
    )

    # quantiles = np.nanquantile(
    #     metrics_dict[model]["mean_distance"][valid_idx] * 1e-3, [0.25, 0.75], axis=0
    # )
    # ax1.fill_between(
    #     time_steps, quantiles[0], quantiles[1], alpha=0.5, color=color_dict.get(label, "C0")
    # )
    # ax2.plot(
    #     time_steps,
    #     np.nanmedian(metrics_dict[model]["energy_score"][valid_idx] * 1e-3, axis=0),
    #     color="w",
    #     linewidth=3,
    # )
    ax2.plot(
        time_steps,
        np.nanmedian(metrics_dict[model]["energy_score"][valid_idx] * 1e-3, axis=0),
        label=label,
        color=color_dict.get(label, "C0"),
        linewidth=2,
    )
    # quantiles = np.nanquantile(
    #     metrics_dict[model]["energy_score"][valid_idx] * 1e-3, [0.25, 0.75], axis=0
    # )
    # ax2.fill_between(
    #     time_steps, quantiles[0], quantiles[1], alpha=0.5, color=color_dict.get(label, "C0")
    # )
    # ax3.plot(
    #     time_steps,
    #     np.nanmedian(metrics_dict[model]["skill_score"][valid_idx], axis=0),
    #     color="w",
    #     linewidth=3,
    # )
    ax3.plot(
        time_steps,
        np.nanmedian(metrics_dict[model]["skill_score"][valid_idx], axis=0),
        label=label,
        color=color_dict.get(label, "C0"),
        linewidth=2,
    )
    # quantiles = np.nanquantile(
    #     metrics_dict[model]["skill_score"][valid_idx], [0.25, 0.75], axis=0
    # )
    # ax3.fill_between(
    #     time_steps, quantiles[0], quantiles[1], alpha=0.5, color=color_dict.get(label, "C0")
    # )
    #ax3.hlines(.45, time_steps[0], time_steps[-1], color="k", linestyle="--", label="Baseline skill score")

ax1.set_xlim([1, time_steps[-1]])
ax1.set_xticks(x_ticks)
ax3.set_xlabel("Prediction horizon [h]")
ax2.set_ylabel("Energy score")
#ax2.set_xticks(x_ticks)
ax1.set_ylabel("Prediction Error [km]")
ax3.set_ylabel("Skill score")

ax1.set_ylim([0, None])
ax2.set_ylim([0, None])
if dataset == "open_ocean":
    ax1.set_title("Open ocean dataset")
    ax1.set_yticks([0, 20, 40])
    ax2.set_yticks([0, 20, 40])
    ax3.set_yticks([0, 0.2, 0.4, 0.6])
    ax3.set_ylim([0, 0.6])
elif dataset == "coastline":
    ax1.set_title("Coastal dataset")
    ax1.set_yticks([0, 5, 10])
    ax2.set_yticks([0, 5, 10])
    ax3.set_yticks([0, 0.2, 0.4])
    ax3.set_ylim([0, 0.4])

#ax3.set_xticks(x_ticks)
# set grids for all axes
ax1.grid()
ax2.grid()
ax3.grid()
ax1.legend(frameon=True)
plt.tight_layout()
plt.savefig(
    os.path.join(
        config_paths.FIG_PATH, f"ddpm_{dataset}_metrics.pdf"
    ),
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(
        config_paths.FIG_PATH, f"ddpm_{dataset}_metrics.png"
    ),
    bbox_inches="tight",
    dpi=300,
)
plt.show()

# %%
colors = ["gray", "C1", "C8", "C3", "C7", "C4", "C2", "C5"]
box_delta = 1. * length / 24.
if dataset == "open_ocean":
    x_ticks = np.array([12, 24, 36, 48, 60, 72])
elif dataset == "coastline":
    x_ticks = np.array([6, 12, 18, 24])
time_steps = np.arange(1, result_file["pred_coords"].shape[2] + 1)
plot_indices = np.where(np.isin(time_steps, x_ticks))[0]
fig = plt.figure(figsize=(6, 8))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax3 = plt.subplot(3, 1, 3, sharex=ax1)

# label_dict = {
#     "naive": "Naive",
#     "hycom": "HYCOM",
#     "conv_net": "CNN",
#     f"codit_GRU_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry": "CODIT GRU",
#     f"codit_Transformer_{dataset}_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry": "CODIT Transformer",
# }
# color_dict = {
#     "Naive": "black",
#     "HYCOM": "gray",
#     "CNN": "C1",
#     "Linear-Gaussian": "C2",
#     "CoDiT (GRU)": "C3",
#     "CoDiT (TF)": "C4",
# }
# valid_idx1 = np.asarray(
#     np.logical_not(
#         np.any(np.isnan(metrics_dict[f"conv_net"]["mean_distance"]), axis=1)
#     )
# , dtype=bool)
# valid_idx2 = np.asarray(metrics_dict['naive']['mean_distance'][:,-1] > 100)
# valid_idx = np.where(np.logical_and(
#     valid_idx1,
#     valid_idx2
# ))[0]
for imodel, model in enumerate(models):
    label = label_dict.get(model, model)
    bp = ax1.boxplot(
        metrics_dict[model]["mean_distance"][valid_idx][:, plot_indices] * 1e-3,
        positions=x_ticks - box_delta + imodel * box_delta * 2 / (len(models) - 1),
        widths=box_delta / (len(models) - 1),
        boxprops=dict(color=color_dict.get(label, "C0"),
                      facecolor=color_dict.get(label, "C0")),
        medianprops=dict(color='w', linewidth=2),
        whiskerprops=dict(color=color_dict.get(label, "C0")),
        capprops=dict(color=color_dict.get(label, "C0")),
        flierprops=dict(markeredgecolor=color_dict.get(label, "C0")),
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
    )
    for ipatch, patch in enumerate(bp['boxes']):
        if ipatch == 0:
            patch.set_label(label)
    bp = ax2.boxplot(
        metrics_dict[model]["energy_score"][valid_idx][:, plot_indices] * 1e-3,
        positions=x_ticks - box_delta + imodel * box_delta * 2 / (len(models) - 1),
        widths=box_delta / (len(models) - 1),
        boxprops=dict(color=color_dict.get(label, "C0"),
                      facecolor=color_dict.get(label, "C0")),
        medianprops=dict(color='w', linewidth=2),
        whiskerprops=dict(color=color_dict.get(label, "C0")),
        capprops=dict(color=color_dict.get(label, "C0")),
        flierprops=dict(markeredgecolor=color_dict.get(label, "C0")),
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
    )
    bp = ax3.boxplot(
        metrics_dict[model]["skill_score"][valid_idx][:, plot_indices],
        positions=x_ticks - box_delta + imodel * box_delta * 2 / (len(models) - 1),
        widths=box_delta / (len(models) - 1),
        boxprops=dict(color=color_dict.get(label, "C0"),
                      facecolor=color_dict.get(label, "C0")),
        medianprops=dict(color='w', linewidth=2),
        whiskerprops=dict(color=color_dict.get(label, "C0")),
        capprops=dict(color=color_dict.get(label, "C0")),
        flierprops=dict(markeredgecolor=color_dict.get(label, "C0")),
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
    )
    # for patch in bp['boxes']:
    #     patch.set_facecolor(color_dict.get(label, "C0"))
    
    # ax1.plot(
    #     time_steps,
    #     np.nanmedian(metrics_dict[model]["mean_distance"][valid_idx] * 1e-3, axis=0),
    #     color="w",
    #     linewidth=3,
    # )
    # ax1.plot(
    #     time_steps,
    #     np.nanmedian(metrics_dict[model]["mean_distance"][valid_idx] * 1e-3, axis=0),
    #     label=label,
    #     color=color_dict.get(label, "C0"),
    # )

    # quantiles = np.nanquantile(
    #     metrics_dict[model]["mean_distance"][valid_idx] * 1e-3, [0.25, 0.75], axis=0
    # )
    # ax1.fill_between(
    #     time_steps, quantiles[0], quantiles[1], alpha=0.5, color=color_dict.get(label, "C0")
    # )
    # ax2.plot(
    #     time_steps,
    #     np.nanmedian(metrics_dict[model]["energy_score"][valid_idx] * 1e-3, axis=0),
    #     color="w",
    #     linewidth=3,
    # )
    # ax2.plot(
    #     time_steps,
    #     np.nanmedian(metrics_dict[model]["energy_score"][valid_idx] * 1e-3, axis=0),
    #     label=label,
    #     color=color_dict.get(label, "C0"),
    # )
    # quantiles = np.nanquantile(
    #     metrics_dict[model]["energy_score"][valid_idx] * 1e-3, [0.25, 0.75], axis=0
    # )
    # ax2.fill_between(
    #     time_steps, quantiles[0], quantiles[1], alpha=0.5, color=color_dict.get(label, "C0")
    # )
    # ax3.plot(
    #     time_steps,
    #     np.nanmedian(metrics_dict[model]["skill_score"][valid_idx], axis=0),
    #     color="w",
    #     linewidth=3,
    # )
    # ax3.plot(
    #     time_steps,
    #     np.nanmedian(metrics_dict[model]["skill_score"][valid_idx], axis=0),
    #     label=label,
    #     color=color_dict.get(label, "C0"),
    # )
    # quantiles = np.nanquantile(
    #     metrics_dict[model]["skill_score"][valid_idx], [0.25, 0.75], axis=0
    # )
    # ax3.fill_between(
    #     time_steps, quantiles[0], quantiles[1], alpha=0.5, color=color_dict.get(label, "C0")
    # )
    #ax3.hlines(.45, time_steps[0], time_steps[-1], color="k", linestyle="--", label="Baseline skill score")

ax1.set_xlim([1, time_steps[-1] + box_delta * 2])
ax1.set_xticks(x_ticks)
ax3.set_xlabel("Prediction horizon [h]")
ax2.set_ylabel("Energy score")
#ax2.set_xticks(x_ticks)
ax1.set_ylabel("Prediction Error [km]")
ax3.set_ylabel("Skill score")

ax1.set_ylim([0, None])
ax2.set_ylim([0, None])
if dataset == "open_ocean":
    ax1.set_title("Open ocean dataset")
    ax1.set_yticks([0, 40, 80, 120])
    ax2.set_yticks([0, 40, 80, 120])
    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_ylim([0, 1])
elif dataset == "coastline":
    ax1.set_title("Coastal dataset")
    ax1.set_yticks([0, 20, 40])
    ax2.set_yticks([0, 20, 40])
    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_ylim([0, 1])

#ax3.set_xticks(x_ticks)
# set grids for all axes
ax1.grid()
ax2.grid()
ax3.grid()
ax1.legend(frameon=True)
plt.tight_layout()
plt.savefig(
    os.path.join(
        config_paths.FIG_PATH, f"ddpm_{dataset}_boxplots.pdf"
    ),
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(
        config_paths.FIG_PATH, f"ddpm_{dataset}_boxplots.png"
    ),
    bbox_inches="tight",
    dpi=300,
)
plt.show()


# %%
len(valid_idx)
# %%

import pandas as pd
results_df = {}
for model in models:
    label = label_dict.get(model, model)
    results_df[label] = {
        'Pred Error [km]': np.nanmean(metrics_dict[model]['mean_distance'][valid_idx,-1]) * 1e-3,
        'Energy Score [km]': np.nanmean(metrics_dict[model]['energy_score'][valid_idx,-1]) * 1e-3,
        'Skill Score': np.nanmean(metrics_dict[model]['skill_score'][valid_idx,-1]),
    }

results_df = pd.DataFrame(results_df).T
results_df



# %%

metric = "energy_score"  # 'energy_score'
label_dict = {"mean_distance": "Prediction error [km]", "energy_score": "Energy score"}

plt.figure(figsize=(12, 8))
for i, time_idx in enumerate(time_steps):
    ax1 = plt.subplot(2, 3, i + 1)
    for model in models:
        label = label_dict.get(model, model)
        bins = plt.hist(
            metrics_dict[model][metric][:, time_idx - 1] * 1e-3,
            label=label,
            alpha=0.5,
            bins=x_vec,
            density=True,
            color=color_dict.get(label, "C0")
        )[1]
    ax1.set_title(f"{time_idx}h")
    ax1.set_yticks([])
    if i == 1:
        ax1.set_xlabel(label_dict[metric])
    if i == 0:
        ax1.set_ylabel("Density")
    ax1.set_xlim([0, x_vec[-1]])
    ax2 = plt.subplot(2, 3, i + 4)
    for model in models:
        label = label_dict.get(model, model)
        metrics_hycom = metrics_dict["hycom_parcels"][metric][:, time_idx - 1] * 1e-3
        if not model == "hycom_parcels":
            metrics_model = metrics_dict[model][metric][:, time_idx - 1] * 1e-3
            metrics_model = metrics_model
            valid_points = np.logical_and(
                ~np.isnan(metrics_model), ~np.isnan(metrics_hycom)
            )
            metrics_model = metrics_model[valid_points]
            metrics_hycom = metrics_hycom[valid_points]
            ax2.plot(
                metrics_dict["hycom_parcels"][metric][:, time_idx - 1] * 1e-3,
                metrics_dict[model][metric][:, time_idx - 1] * 1e-3,
                "o",
                alpha=0.5,
                label=label,
                color=color_dict.get(label, "C0")
            )
            # fit a line through the points
            fit_res = np.polyfit(metrics_hycom, metrics_model, 1)
            ax2.plot(
                x_vec,
                fit_res[0] * x_vec + fit_res[1],
                color="k",
                alpha=1,
                linewidth=2,
                zorder=10,
            )
            ax2.plot(
                x_vec,
                fit_res[0] * x_vec + fit_res[1],
                color=color_dict.get(label, "C0"),
                alpha=1,
                zorder=10,
            )

            ax2.plot([0, x_vec[-1]], [0, x_vec[-1]], "k--")
    ax2.set_xlim([0, x_vec[-1]])
    ax2.set_ylim([0, x_vec[-1]])
    if i == 0:
        ax2.set_ylabel(f"Model {label_dict[metric]}")
    if i == 1:
        ax2.set_xlabel(f"HYCOM {label_dict[metric]}")
ax1.legend()

plt.tight_layout()
plt.savefig(
    os.path.join(figs_path, f"{dataset}_{metric}_error_histograms.png"), dpi=300
)
plt.savefig(
    os.path.join(figs_path, f"{dataset}_{metric}_error_histograms.pdf"), dpi=300
)
# %%

# %%
# visualize the results
result_dict = {}
result_file_names = os.listdir(os.path.join(result_path, f"ddpm_Transformer_{dataset}_conv_adam_constant"))
# %%
dataset_idx = 1
for model in models:
    result_file = np.load(
        os.path.join(result_path, model, result_file_names[dataset_idx]),
        allow_pickle=True,
    )
    result_dict[model] = result_file
# %%
from cartopy import crs
import cartopy.geodesic as cgeo
import seaborn as sns
import cartopy.feature as cfeature

# plt.figure()

#ax = plt.axes(projection=crs.PlateCarree())
#ax.coastlines()

delta = 0.3

scale_length_km = 10  # km
scale_length_deg = scale_length_km / 111  # 1 degree ≈ 111 k
idx = 215
fig = plt.figure(figsize=(12, 4))
for subplot_idx, t_idx in enumerate(time_steps):
    ax = plt.subplot(1, 3, subplot_idx + 1, projection=crs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND, color="lightgray")
    for i, model in enumerate(models):
        # plt.plot(result_dict[model]['pred_coords'][idx,:,:t_idx,1].T, result_dict[model]['pred_coords'][idx,:,:t_idx,0].T, label=model, color=colors[i], alpha=0.2)
        if model != "hycom":
            sns.kdeplot(
                x=result_dict[model]["pred_coords"][idx, :, t_idx - 1, 1].flatten(),
                y=result_dict[model]["pred_coords"][idx, :, t_idx - 1, 0].flatten(),
                levels=1,
                color=colors[i],
                alpha=0.5,
                zorder=8,
            )
        plt.plot(
            np.median(result_dict[model]["pred_coords"][idx, :, :t_idx, 1], axis=0),
            np.median(result_dict[model]["pred_coords"][idx, :, :t_idx, 0], axis=0),
            label=model,
            color="w",
            zorder=9,
            linewidth=2,
        )
        plt.plot(
            np.median(result_dict[model]["pred_coords"][idx, :, :t_idx, 1], axis=0),
            np.median(result_dict[model]["pred_coords"][idx, :, :t_idx, 0], axis=0),
            label=model,
            color=colors[i],
            zorder=10,
        )
    plt.plot(
        result_dict[model]["true_coords"][idx, :t_idx, 1],
        result_dict[model]["true_coords"][idx, :t_idx, 0],
        label="True",
        color="black",
    )
    plt.plot(
        result_dict[model]["true_coords"][idx, 0, 1
        ],
        result_dict[model]["true_coords"][idx, 0, 0],
        marker="o",
        color="black",
        markersize=5,
    )
    plt.plot(
        result_dict[model]["true_coords"][idx, t_idx - 1, 1],
        result_dict[model]["true_coords"][idx, t_idx - 1, 0],
        marker="s",
        color="black",
        markersize=5,
    )
    plt.xlim(
        [
            result_dict[model]["true_coords"][idx, :, 1].min() - delta,
            result_dict[model]["true_coords"][idx, :, 1].max() + delta,
        ]
    )
    plt.ylim(
        [
            result_dict[model]["true_coords"][idx, :, 0].min() - delta,
            result_dict[model]["true_coords"][idx, :, 0].max() + delta,
        ]
    )
    offset_x = result_dict[model]["true_coords"][idx, :, 1].min() - delta + 0.05
    offset_y = result_dict[model]["true_coords"][idx, :, 0].min() - delta + 0.05
    ax.set_title(f"{t_idx}h")
plt.plot(
    [offset_x, offset_x + scale_length_deg],
    [offset_y, offset_y],
    color="black",
    linewidth=2,
    transform=crs.PlateCarree(),
)
plt.text(
    offset_x + scale_length_deg / 2,
    offset_y,
    "{} km".format(scale_length_km),
    transform=crs.PlateCarree(),
    ha="center",
    va="bottom",
)
plt.savefig(
    os.path.join(figs_path, f"{dataset}_{dataset_idx}_{idx}_predictions.png"), dpi=300
)
plt.savefig(
    os.path.join(figs_path, f"{dataset}_{dataset_idx}_{idx}_predictions.pdf"), dpi=300
)
# %%
plt.plot(
    result_file["pred_coords"][0, 0, :, 0].T,
)
# %%
