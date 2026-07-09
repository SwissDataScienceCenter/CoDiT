# %%
import os
from codit.data_module.dataset import OceanDataset
from codit.data_module import config_funcs
from codit.data_module import config_paths, data_transform
from cartopy import crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from jax import numpy as jnp
from codit.model_module.diffusion_model import CoDiTModel

from codit.codit_module import noising_functions
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib as mpl

figs_path = os.path.join(config_paths.DATA_PATH, "figs")
dataset_name = "open_ocean"
model_type = "Transformer"
if dataset_name == "open_ocean":
    length = 72
elif dataset_name == "coastline":
    length = 24
delta_metres = 10000
num_grid_points = 16
interp_method = "linear"
config_path = os.path.join(
    config_paths.CONFIG_PATH,
    dataset_name,
)
files = os.listdir(config_path)
dataset_files = [f for f in files if f.endswith(".yml")]

config = config_funcs.load_config(dataset_files[1].split(".")[0], config_path)
dataset = OceanDataset(config, length, num_grid_points, delta_metres, interp_method)
# %%
# train_dataset = AdoptTrainingDataset.from_dataset(dataset)

model = CoDiTModel.import_trained_model('codit_Transformer_open_ocean', '/mydata/adopt/christian/CoDiT/data/trained_models')
# %%
from codit.codit_module import scheduler
model.schedule_dict = scheduler.linear_schedule(beta_min=1e-4, beta_max=.04, K=100)
# %%

start_position, start_time, times, interpolator, coordinates = dataset[10]

# %%
pred_coord, pred_coords_steps = model.predict(
    start_position,
    start_time,
    prediction_length=72,
    interpolator=interpolator,
    test_dataset=dataset,
    num_samples=100,
    return_denoising_steps=True,
)

# %%

# Add colorbars for time on the right hand side
# fig = plt.gcf()
# cbar_ax_viridis = fig.add_axes([0.98, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
# cbar_viridis = plt.colorbar(
#     scatters[0],
#     cax=cbar_ax_viridis,
#     orientation="vertical",
#     label="Timestep [h]",
#     ticks=[0, 24, 48, 72, 96],
# )

# # Add a colorbar for the gray colorscale (for the LineCollection)
# # sm_gray = mpl.cm.ScalarMappable(cmap="gray", norm=plt.Normalize(vmin=0, vmax=num_time_steps-1))
# # sm_gray.set_array([])
# # cbar_ax_gray = fig.add_axes([0.96, 0.15, 0.015, 0.7])
# # cbar_gray = plt.colorbar(
# #     sm_gray,
# #     cax=cbar_ax_gray,
# #     orientation="vertical",
# #     label="Timestep (gray)"
# # )
# plt.savefig(os.path.join(figs_path, f"backward_process.png"), dpi=300, transparent=True)
# plt.savefig(os.path.join(figs_path, f"backward_process.pdf"), dpi=300, transparent=True)

# %%
from jax import vmap

key = random.PRNGKey(1)
keys = random.split(key, 100)
noise, _ = vmap(model.noise_model, in_axes=(0, None))(keys, length)

plot_steps = jnp.array([0, 25, 50, 75, 100])
num_time_steps = pred_coords_steps.shape[2]
time_colors = plt.cm.viridis(np.linspace(0, 1, num_time_steps+1))

def transform_traj_to_metre(traj):
    velocity_vectors = vmap(
        data_transform.velocity_vector_from_pos_time, in_axes=(None, 0, None)
    )(traj[0], traj, 3600)
    traj_in_metres = velocity_vectors * 3600
    return traj_in_metres


traj_metres = transform_traj_to_metre(coordinates)
traj_scaled = traj_metres / model.std

plt.figure(figsize=(11, 2))
plot_delta = 2
scatters = []

for ax_idx, k in enumerate(plot_steps):

    if k == 0:
        alpha = 1.
    else:
        alpha = model.schedule_dict["alpha_bar"][k - 1]
    traj_noised = jnp.sqrt(alpha) * traj_scaled[None] + jnp.sqrt(1 - alpha) * noise
    margin = 0.01  # Small margin between plots
    total_width = 0.9 - (len(plot_steps) - 1) * margin
    width = total_width / len(plot_steps)
    left = 0.05 + ax_idx * (width + margin) - 0.03
    ax = plt.axes([left, 0.15, width, 0.7])
    ax.set_title(f"Step {k}")
    # Plot observed trajectory with color-coded time
    points = traj_scaled[:, [1, 0]]
    segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)
    # lc_obs = LineCollection(
    #     segments,
    #     colors=time_colors[:-1],
    #     linewidths=2,
    #     alpha=1.0,
    #     label="Observed Trajectory",
    # )
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=np.arange(points.shape[0]),
        cmap="viridis",
        s=5,
        edgecolor="k",
        linewidths=0.2,
        zorder=3,
        label="Timesteps",
        marker='v'
    )
    #ax.add_collection(lc_obs)
    ax.plot(
        0, 0, "rx", markersize=5, label="Start Position", zorder=10
    )
    scatter = ax.scatter(
        traj_scaled[:, 1],
        traj_scaled[:, 0],
        c=np.arange(traj_scaled.shape[0]),
        cmap="viridis",
        s=5,
        edgecolor="k",
        linewidths=0.2,
        zorder=3,
        label="Timesteps",
        marker='v'
    )
    scatters.append(scatter)

    # Plot noised trajectory with color-coded time
    for i in range(pred_coords_steps.shape[0]):
        points_noised = traj_noised[i, :, [1, 0]].T
        segments_noised = np.concatenate([points_noised[:-1, None, :], points_noised[1:, None, :]], axis=1)
        lc_noised = LineCollection(
            segments_noised,
            colors=time_colors[:-1],
            linewidths=1,
            alpha=.3,
            linestyle="--",
            label="Noised Trajectory",
            rasterized=True
        )
        ax.add_collection(lc_noised)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(
        - plot_delta,
        plot_delta,
    )
    ax.set_ylim(
        - plot_delta,
        plot_delta,
    )
#ax.legend()
fig = plt.gcf()
cbar_ax_viridis = fig.add_axes([0.93, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
cbar_viridis = plt.colorbar(
    scatters[0],
    cax=cbar_ax_viridis,
    orientation="vertical",
    label="Timestep [h]",
    ticks=[0, 24, 48, 72, ],
)
plt.tight_layout()
plt.savefig(os.path.join(figs_path, f"forward_process.png"), dpi=300, transparent=True)
plt.savefig(os.path.join(figs_path, f"forward_process.pdf"), dpi=300, transparent=True)
# %%
plt.plot(model.schedule_dict["beta"])
plt.xlabel("Diffusion Step")
plt.ylabel("Beta")
# %%
plt.figure(figsize=(11, 2))


delta_coord = 2
axes = []
scatters = []

pred_coords_metres = vmap(vmap(transform_traj_to_metre))(pred_coords_steps)
pred_coords_scaled = pred_coords_metres / model.std

for ax_idx, k in enumerate(plot_steps):
    margin = 0.01  # Small margin between plots
    total_width = 0.9 - (len(plot_steps) - 1) * margin
    width = total_width / len(plot_steps)
    left = 0.05 + (len(plot_steps) - 1 - ax_idx) * (width + margin) - 0.03
    ax = plt.axes([left, 0.15, width, 0.7])
    axes.append(ax)
    #ax.add_feature(cfeature.LAND, zorder=0)
    #ax.add_feature(cfeature.COASTLINE, zorder=1)
    ax.set_title(f"Step {100-k}")

    for i in range(pred_coords_scaled.shape[0]):
        segs = np.array(pred_coords_scaled[i, k])
        points = segs[:, [1, 0]]
        segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)
        lc = LineCollection(
            segments,
            colors=time_colors[:-1],
            linewidths=1,
            alpha=0.3,
            zorder=2,
            rasterized=True
        )
        ax.add_collection(lc)

    ax.plot(
        0, 0, "rx", markersize=5, label="Start Position", zorder=10
    )
    ax.plot(traj_scaled[:, 1], traj_scaled[:, 0], "k")
    scatter = ax.scatter(
        traj_scaled[:, 1],
        traj_scaled[:, 0],
        c=np.arange(traj_scaled.shape[0]),
        cmap="viridis",
        s=5,
        edgecolor="k",
        linewidths=0.2,
        zorder=3,
        label="Timesteps",
        marker='v'
    )
    scatters.append(scatter)
    ax.set_xlim(
        -plot_delta,
        plot_delta,
    )
    ax.set_ylim(
        -plot_delta,
        plot_delta,
    )
    
    ax.set_xticks([])
    ax.set_yticks([])

fig = plt.gcf()
cbar_ax_viridis = fig.add_axes([0.93, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
cbar_viridis = plt.colorbar(
    scatters[0],
    cax=cbar_ax_viridis,
    orientation="vertical",
    label="Timestep [h]",
    ticks=[0, 24, 48, 72],
)
plt.tight_layout()
plt.savefig(os.path.join(figs_path, f"backward_process.png"), dpi=300, transparent=True)
plt.savefig(os.path.join(figs_path, f"backward_process.pdf"), dpi=300, transparent=True)
# %%
