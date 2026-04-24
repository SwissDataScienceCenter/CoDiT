# %%
%load_ext autoreload
%autoreload 2
import os
from codit.data_module.dataset import (
    OceanDataset,
    OceanTrainingDataset
)
from codit.data_module import config_funcs
from codit.data_module import config_paths
from cartopy import crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from jax import numpy as jnp

data_path_train = config_paths.DATA_PATH[:-5] + 'test'
config_path_train = os.path.join(data_path_train, 'configs')

data_name = 'coastline'  # 'open_ocean' or 'coastline'
figs_path = os.path.join(config_paths.DATA_PATH, "figs")
if data_name == 'open_ocean':
    length = 72
elif data_name == 'coastline':
    length = 24

delta_metres = 10000
num_grid_points = 16
interp_method = "linear"
config_path_train = os.path.join(
    config_path_train,
    data_name,
)
config_path_test = os.path.join(
    config_paths.CONFIG_PATH,
    data_name,
    'test',
)
files_train = os.listdir(config_path_train)
dataset_files_train = [f for f in files_train  if f.endswith(".yml")]
files_test = os.listdir(config_path_test)
dataset_files_test = [f for f in files_test  if f.endswith(".yml")]
config_path = config_path_train

train_regions_of_interest = []
test_regions_of_interest = []
train_coords = []
test_coords = []
dataset_plot = None

for dataset_name in dataset_files_train:
    try:
        config = config_funcs.load_config(dataset_name.split(".")[0], config_path)
        dataset = OceanDataset(
            config,
            length,
            num_grid_points,
            delta_metres,
            interp_method,
            no_context=True,
        )
        if config["end_date"].year < 2020 and os.path.exists(config["dataset_path"]):
            longitudes = config["longitudes"]
            longitudes[0] += config["context_delta"]
            longitudes[1] -= config["context_delta"]
            latitudes = config["latitudes"]
            latitudes[0] += config["context_delta"]
            latitudes[1] -= config["context_delta"]
            train_regions_of_interest.append(
                [longitudes, latitudes]
            )
            if len(train_regions_of_interest) == 21:
                dataset_plot = OceanDataset(
                    config,
                    length,
                    num_grid_points,
                    delta_metres,
                    interp_method,
                    no_context=False,
                )
            for traj in dataset.trajectories:
                train_coords.append(traj["coords"][None])
        # elif config["end_date"].year >= 2020 and os.path.exists(config["dataset_path"]):
        #     test_regions_of_interest.append([config["longitudes"], config["latitudes"]])
        #     for traj in dataset.trajectories:
        #         test_coords.append(traj["coords"][None])
    except Exception as e:
        print(f"Error creating dataset for {dataset_name}: {e}")
        continue
for dataset_name in dataset_files_test:
    try:
        config = config_funcs.load_config(dataset_name.split(".")[0], config_path_test)
        dataset = OceanDataset(
            config,
            length,
            num_grid_points,
            delta_metres,
            interp_method,
            no_context=True,
        )
        if os.path.exists(config["dataset_path"]):
            longitudes = config["longitudes"]
            longitudes[0] += config["context_delta"]
            longitudes[1] -= config["context_delta"]
            latitudes = config["latitudes"]
            latitudes[0] += config["context_delta"]
            latitudes[1] -= config["context_delta"]
            test_regions_of_interest.append(
                [longitudes, latitudes]
            )
            for traj in dataset.trajectories:
                test_coords.append(traj["coords"][None])
    except Exception as e:
        print(f"Error creating dataset for {dataset_name}: {e}")
        continue

train_coords = jnp.concatenate(train_coords, axis=0)
test_coords = jnp.concatenate(test_coords, axis=0)

# %%
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.LAND, color="lightgray")
# ax.add_feature(cfeature.OCEAN, color='lightblue')
ax.set_extent([-180, 180, -80, 80], crs=ccrs.PlateCarree())

for i, region in enumerate(train_regions_of_interest):
    lon, lat = region
    patch = ax.add_patch(
        plt.Rectangle(
            (lon[0], lat[0]),
            lon[1] - lon[0],
            lat[1] - lat[0],
            linewidth=2,
            transform=ccrs.PlateCarree(),
            color="green",
            alpha=0.5,
            label="Training Region ($\leq$ 2020)" if i == 0 else None,
        )
    )
    if i == 20 and data_name == 'open_ocean':
        patch = ax.add_patch(
        plt.Rectangle(
                (lon[0], lat[0]),
                lon[1] - lon[0],
                lat[1] - lat[0],
                transform=ccrs.PlateCarree(),
                facecolor="none",
                linewidth=2,
                edgecolor="k",
                linestyle='--',
            )
        )
for idx, region in enumerate(test_regions_of_interest):
    lon, lat = region
    patch = ax.add_patch(
        plt.Rectangle(
            (lon[0], lat[0]),
            lon[1] - lon[0],
            lat[1] - lat[0],
            transform=ccrs.PlateCarree(),
            linewidth=2,
            color="orange",
            alpha=0.5,
            label="Test Region (2022)" if idx == 0 else None,
        )   
    )
ax.legend()
ax.set_title("Open ocean dataset")
plt.savefig(os.path.join(figs_path, f"{data_name}_dataset.png"), dpi=300, bbox_inches='tight', transparent=True)
plt.savefig(os.path.join(figs_path, f"{data_name}_dataset.pdf"), dpi=300)
plt.show()
# %%
import numpy as np
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.LAND, color="lightgray")

for i, t in enumerate(dataset_plot.trajectories[:]):

    if i == 107:
        ax.plot(
            t["coords"][:, 1],
            t["coords"][:, 0],
            color="k",
            alpha=1,)
    else:
        ax.plot(
            t["coords"][:, 1],
            t["coords"][:, 0],
            color=np.random.randint(0, 256, size=3) / 256.0,
            alpha=0.5,
        )
plt.savefig(os.path.join(figs_path, f"{data_name}_trajectories.png"), dpi=300)
plt.savefig(os.path.join(figs_path, f"{data_name}_trajectories.pdf"), dpi=300)

# %%
dataset_train_plot = OceanTrainingDataset.from_dataset(dataset_plot)

# %%

# def lonlat_to_figcoords(extent, lon, lat, width_deg=.1, height_deg=.1):
#     """
#     Convert lon/lat to figure coordinates for placing an inset axes.
#     width_deg, height_deg: size of the inset in degrees.
#     Returns: [left, bottom, width, height] for fig.add_axes
#     """
#     min_lon, max_lon, min_lat, max_lat = extent
#     # Convert lon/lat to axis fraction
#     left = (lon - min_lon) / (max_lon - min_lon)
#     bottom = (lat - min_lat) / (max_lat - min_lat)
#     width = width_deg / (max_lon - min_lon)
#     height = height_deg / (max_lat - min_lat)
#     return [left, bottom, width, height]

def lonlat_to_figcoords(ax, lon, lat, width=0.1, height=0.1):
    """
    Convert lon/lat to figure coordinates for placing an inset axes.
    width, height: size of the inset in figure fraction.
    Returns: [left, bottom, width, height] for fig.add_axes
    """
    # Transform lon/lat to display (pixel) coordinates
    display_coords = ax.transData.transform((lon, lat))
    # Transform display coords to figure fraction
    fig = ax.figure
    fig_coords = fig.transFigure.inverted().transform(display_coords)
    left, bottom = fig_coords
    return [left, bottom+.1, width, height]

traj, context = dataset_train_plot[107]
cmap = plt.get_cmap("Greens")
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.coastlines()
ax.plot(traj[:, 1], traj[:, 0], color="k", label="Trajectory", marker="o", markersize=5)
ax.set_extent(
    [
        jnp.min(traj[:, 1]) - 0.2,
        jnp.max(traj[:, 1]) + 0.2,
        jnp.min(traj[:, 0]) - 0.1,
        jnp.max(traj[:, 0]) + 0.3,
    ]
)
extent = ax.get_extent(crs=ccrs.PlateCarree())

for i in range(0, context.shape[0], 12):
    pos = traj[i]
    for j in range(8):
        # put inset at position
        color = cmap(i / context.shape[0])
        inset = fig.add_axes(lonlat_to_figcoords(ax, pos[1] + j*.003, pos[0]+j*.002, .1, .1))
        inset.set_facecolor(color)
        inset.spines["top"].set_color(color)
        inset.spines["bottom"].set_color(color)
        inset.spines["left"].set_color(color)
        inset.spines["right"].set_color(color)
        inset.set_aspect("equal")
        inset.axes.set_xticks([])
        inset.axes.set_yticks([])
        inset.pcolor(context[i, ..., -j], cmap="coolwarm", rasterized=True)
    ax.scatter(
        pos[1],
        pos[0],
        color=color,
        zorder=10,
    )
plt.savefig(os.path.join(figs_path, f"{data_name}_trajectory_with_context.png"), dpi=300)
plt.savefig(os.path.join(figs_path, f"{data_name}_trajectory_with_context.pdf"), dpi=300)
# %%
from codit.data_module import data_transform
import jax
import numpy as np


def circular_hist(
    ax, x, bins=16, density=True, offset=0, gaps=True, color="C0", alpha=0.5
):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x + np.pi) % (2 * np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps and isinstance(bins, int):
        bins = np.linspace(-np.pi, np.pi, num=bins + 1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area / np.pi) ** 0.5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(
        bins[:-1],
        radius,
        zorder=1,
        align="edge",
        width=widths,
        edgecolor="C0",
        fill=True,
        linewidth=1,
        facecolor=color,
        alpha=alpha,
    )

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


# jax_transform.velocity_vector_from_pos_time(train_coords[0, 0], train_coords[0, -1], 3600)

train_distances = jax.vmap(data_transform.haversine_distance)(
    train_coords[:, 0], train_coords[:, -1]
)
test_distances = jax.vmap(data_transform.haversine_distance)(
    test_coords[:, 0], test_coords[:, -1]
)
train_velocity_vec = jax.vmap(
    jax.vmap(data_transform.velocity_vector_from_pos_time, in_axes=(0, 0, None)),
    in_axes=(0, 0, None),
)(train_coords[::, :-1], train_coords[::, 1:], 3600)
test_velocity_vec = jax.vmap(
    jax.vmap(data_transform.velocity_vector_from_pos_time, in_axes=(0, 0, None)),
    in_axes=(0, 0, None),
)(test_coords[::, :-1], test_coords[::, 1:], 3600)
train_velocities = jnp.linalg.norm(train_velocity_vec, axis=-1)
test_velocities = jnp.linalg.norm(test_velocity_vec, axis=-1)
train_angles = jnp.arctan2(train_velocity_vec[:, :, 1], train_velocity_vec[:, :, 0])
test_angles = jnp.arctan2(test_velocity_vec[:, :, 1], test_velocity_vec[:, :, 0])
# test_angles = jax.vmap(jax_transform.angle_from_velocity_vector)(test_velocity_vec)

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(131)
if data_name == 'coastline':
    max_travel_distance = 100
else:
    max_travel_distance = 450
bins = ax.hist(
    np.clip(train_distances * 1e-3, 0, max_travel_distance),
    bins=50,
    density=True,
    alpha=0.5,
    label="Training Region ($\leq$ 2020)",
    color="green",
)[1]
ax.hist(
    np.clip(test_distances * 1e-3, 0, max_travel_distance),
    bins=bins,
    density=True,
    alpha=0.5,
    label="Test Region (> 2020)",
    color="orange",
)
ax.set_yticks([])
# ax.set_xticks(jnp.arange(-1, 4), labels=10 ** jnp.arange(-1, 4))
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Density")
ax.set_title(f"Travel distance after {length} h")
ax.set_xlim([0, max_travel_distance])
ax.legend()
ax = fig.add_subplot(132)
bins = ax.hist(
    np.clip(train_velocities.flatten(), 0, 2),
    bins=50,
    density=True,
    alpha=0.5,
    label="Training Region ($\leq$ 2020)",
    color="green",
)[1]

ax.hist(
    np.clip(test_velocities.flatten(), 0, 2),
    bins=bins,
    density=True,
    alpha=0.5,
    label="Test Region (> 2020)",
    color="orange",
)
ax.set_yticks([])
ax.set_xlabel("Velocity (m/s)")
ax.set_title("Floating velocity")
ax.set_xlim([0, 2])
ax = fig.add_subplot(133, polar=True)
bins = circular_hist(
    ax,
    train_angles.flatten(),
    bins=50,
    density=True,
    offset=np.pi / 2.0,
    color="green",
    alpha=0.5,
    gaps=False,
)[1]
circular_hist(
    ax,
    test_angles.flatten(),
    bins=bins,
    density=True,
    offset=np.pi / 2,
    color="orange",
    alpha=0.5,
    gaps=False,
)

# ax.hist(train_angles.flatten(), bins=100, density=True, alpha=0.5, label='Training Region ($\leq$ 2020)', color='blue')
# ax.hist(test_angles.flatten(), bins=100, density=True, alpha=0.5, label='Test Region (> 2020)', color='red')

ax.set_title("Floating direction")
plt.savefig(os.path.join(figs_path, f"{data_name}_dataset_stats.png"), dpi=300)
plt.savefig(os.path.join(figs_path, f"{data_name}_dataset_stats.pdf"), dpi=300)

# %%
print("Dataset statistics:")
print(f"Number of trajectories: {train_coords.shape[0] + test_coords.shape[0]}")
print(f"Number of training trajectories: {train_coords.shape[0]}")
print(f"Number of test trajectories: {test_coords.shape[0]}")
print(f"Number of test trajectories: {test_coords.shape[0]}")
print(f"Mean travel distance: {jnp.mean(jnp.concatenate([train_distances, test_distances]))*1e-3:.1f} km")
print(f"Mean travel distance train: {jnp.mean(train_distances)*1e-3:.1f} km")
print(f"Mean travel distance test: {jnp.mean(test_distances)*1e-3:.1f} km")
print(f"Mean velocity: {jnp.mean(jnp.concatenate([train_velocities, test_velocities])):.2f} m/s")
print(f"Mean velocity train: {jnp.mean(train_velocities):.2f} m/s")
print(f"Mean velocity test: {jnp.mean(test_velocities):.2f} m/s")
# %%
config = config_funcs.load_config(dataset_files[1].split(".")[0], config_path)
dataset = OceanDataset(config, length, num_grid_points, delta_metres, interp_method)
# %%
train_dataset = OceanTrainingDataset.from_dataset(dataset)
# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3, color="k"):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    # Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    # Make tmc horizontally centred on the middle of the map,
    # vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    # Calculate a scale bar length if none has been given
    # (Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000  # in km
        ndim = int(np.floor(np.log10(length)))  # number of digits in number
        length = round(length, -ndim)  # round to 1sf

        # Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ["1", "2", "5"]:
                return int(x)
            else:
                return scale_number(x - 10**ndim)

        length = scale_number(length)

    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    # Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color=color, linewidth=linewidth)
    # Plot the scalebar label
    ax.text(
        sbx,
        sby,
        str(length) + " km",
        transform=tmc,
        horizontalalignment="center",
        verticalalignment="bottom",
        color=color,
    )


traj_idx = 20
trajectory_data = train_dataset.trajectories[traj_idx]
start_position, times, interpolator, coordinates = dataset[traj_idx]
fig = plt.figure(figsize=(10, 5))
min_lon, max_lon = (
    jnp.min(trajectory_data[:, 1]) - delta_deg,
    jnp.max(trajectory_data[:, 1]) + delta_deg,
)
min_lat, max_lat = (
    jnp.min(trajectory_data[:, 0]) - delta_deg,
    jnp.max(trajectory_data[:, 0]) + delta_deg,
)
lat_vec = jnp.linspace(min_lat, max_lat, 100)
lon_vec = jnp.linspace(min_lon, max_lon, 100)
grid = np.meshgrid(lon_vec, lat_vec)

bathy = jax.vmap(jax.vmap(interpolator, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(
    times[0], grid[1], grid[0]
)[..., -2]

ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
delta_deg = 0.1
ax.coastlines()
# ax.add_feature(cfeature.LAND, color='lightgray')

ax.set_extent(
    [
        jnp.min(trajectory_data[:, 1]) - delta_deg,
        jnp.max(trajectory_data[:, 1]) + delta_deg,
        jnp.min(trajectory_data[:, 0]) - delta_deg,
        jnp.max(trajectory_data[:, 0]) + delta_deg,
    ],
    crs=ccrs.PlateCarree(),
)
# ax.contourf(lon_vec, lat_vec, bathy, cmap='Greys_r', shading='auto',)
ax.plot(
    train_dataset.trajectories[traj_idx][:, 1],
    train_dataset.trajectories[traj_idx][:, 0],
    color="k",
    label="Trajectory",
    marker="o",
    markersize=2,
)
ax.scatter(
    start_position[1],
    start_position[0],
    color="r",
    label="Start position",
    marker="s",
    s=100,
    zorder=10,
)
time_idx = 4
context = train_dataset.context[traj_idx, time_idx, ...]
ax.scatter(
    trajectory_data[time_idx, 1],
    trajectory_data[time_idx, 0],
    color="C8",
    label="Context position",
    marker="o",
    s=100,
    zorder=10,
)
for i in range(context.shape[-1]):
    inset = fig.add_axes([0.28 + i * 0.003, 0.5 + i * 0.005, 0.1, 0.1])
    inset.spines["top"].set_color("C8")
    inset.spines["bottom"].set_color("C8")
    inset.spines["left"].set_color("C8")
    inset.spines["right"].set_color("C8")
    inset.set_aspect("equal")
    inset.axes.set_xticks([])
    inset.axes.set_yticks([])
    inset.pcolor(context[..., -i], cmap="coolwarm")

time_idx = 24
context = train_dataset.context[traj_idx, time_idx, ...]
ax.scatter(
    trajectory_data[time_idx, 1],
    trajectory_data[time_idx, 0],
    color="C9",
    label="Context position",
    marker="o",
    s=100,
    zorder=10,
)
for i in range(context.shape[-1]):
    inset = fig.add_axes([0.42 + i * 0.003, 0.5 + i * 0.005, 0.1, 0.1])
    inset.spines["top"].set_color("C9")
    inset.spines["bottom"].set_color("C9")
    inset.spines["left"].set_color("C9")
    inset.spines["right"].set_color("C9")
    inset.set_aspect("equal")
    inset.axes.set_xticks([])
    inset.axes.set_yticks([])
    inset.pcolor(context[..., -i], cmap="coolwarm")


time_idx = 48
context = train_dataset.context[traj_idx, time_idx, ...]
ax.scatter(
    trajectory_data[time_idx, 1],
    trajectory_data[time_idx, 0],
    color="C6",
    label="Context position",
    marker="o",
    s=100,
    zorder=10,
)
for i in range(context.shape[-1]):
    inset = fig.add_axes([0.6 + i * 0.003, 0.6 + i * 0.005, 0.1, 0.1])
    # make border green
    inset.spines["top"].set_color("C6")
    inset.spines["bottom"].set_color("C6")
    inset.spines["left"].set_color("C6")
    inset.spines["right"].set_color("C6")
    inset.set_aspect("equal")
    inset.axes.set_xticks([])
    inset.axes.set_yticks([])
    inset.pcolor(context[..., -i], cmap="coolwarm")
scale_bar(ax, length=10, location=(0.90, 0.05), linewidth=3, color="gray")
# plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', label='Bathymetry (km)', ticks=[-4, -3, -2, -1,])
# ax.(dataset.context_xarray['longitude'], dataset.context_xarray['latitude'], dataset.context_xarray['elevation'], shading='auto')

plt.savefig(os.path.join(figs_path, "open_ocean_dataset_example.png"), dpi=300)
plt.savefig(os.path.join(figs_path, "open_ocean_dataset_example.pdf"), dpi=300)

# %%
# plot context in an isolated plot
fig = plt.figure(figsize=(5, 5))
for i in range(context.shape[-1]):
    inset = fig.add_axes([0.55 + i * 0.003, 0.6 + i * 0.005, 0.1, 0.1])
    inset.set_aspect("equal")
    inset.axes.set_xticks([])
    inset.axes.set_yticks([])
    inset.pcolor(context[..., -i], cmap="coolwarm")
plt.savefig(os.path.join(figs_path, "open_ocean_dataset_context.png"), dpi=300)
plt.savefig(os.path.join(figs_path, "open_ocean_dataset_context.pdf"), dpi=300)
# %%
x0, t0, times, interpolator, traj = dataset_plot[107]

# %%
from jax import vmap
from jax import config
from codit.data_module import context_functions
from codit.data_module import data_transform
from datetime import datetime, timedelta
import numpy as np
from cartopy import crs as ccrs
config.update("jax_enable_x64", False)
from matplotlib.patches import FancyArrowPatch


fig = plt.figure(figsize=(20, 5))
index_range = [0, 24, 48, 72]
x_range = np.arange(-.8, .8, 0.083) + float(traj[36, 1])
y_range = np.arange(-.8, .8, 0.083) + float(traj[36, 0])
grid_x, grid_y = jnp.meshgrid(
    x_range,
    y_range,
)
proj = ccrs.AzimuthalEquidistant(central_longitude=traj[36, 1], central_latitude=traj[36, 0])
transform = ccrs.PlateCarree()
for i, idx in enumerate(index_range):
    ax = fig.add_subplot(1, len(index_range), i + 1, projection=proj)
    ax.coastlines()
    grid = context_functions.make_grid(num_grid_points=2)
    context_grid = context_functions.get_grid_around_coordinate(traj[idx:idx+1], grid[0], grid[1])[0]
    interp_x = vmap(vmap(interpolator, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(times[idx], grid_y, grid_x)
    ax.plot(traj[:, 1], traj[:, 0], c='k', transform=ccrs.PlateCarree(), zorder=15)
    ax.scatter(traj[idx, 1], traj[idx, 0], color='k', zorder=10, transform=ccrs.PlateCarree())
    ax.quiver(grid_x, grid_y, u=interp_x[..., 0], v=interp_x[..., 1], color='C1', scale=10, transform=ccrs.PlateCarree())
    ax.quiver(grid_x, grid_y, u=interp_x[..., 2], v=interp_x[..., 3], color='C2', scale=5, transform=ccrs.PlateCarree()
              )
    # plt.quiver(
    #     grid_x,
    #     grid_y,
    #     interp_x[..., 5],
    #     interp_x[..., 4],
    #     color='C3',
    #     scale=5,
    #     alpha=0.5,
    # # )
    plt.streamplot(
        x_range,
        y_range,
        interp_x[..., 4],
        interp_x[..., 5],
        color='C3',
        density=.5,
        linewidth=1,
        arrowsize=1,
        zorder=5,
        transform=ccrs.PlateCarree(),
    )
    im = plt.pcolor(grid_x, grid_y, interp_x[..., -2], cmap='Blues', zorder=0, transform=ccrs.PlateCarree())
    patch = plt.Rectangle(
        (context_grid[0, 0, 1], context_grid[0, 0, 0]),
        context_grid[-1, 0, 1] - context_grid[0, -1, 1],
        context_grid[0, -1, 0] - context_grid[0, 0, 0],
        facecolor="none",
        linewidth=1,
        edgecolor="k",
        transform=ccrs.PlateCarree(),
    )
    
    ax.add_patch(patch)
    ax.set_title(f'{t0 + timedelta(hours=idx):%Y-%m-%d %H:%M} UTC')
    ax.set_aspect('equal')
    #ax.set_xticks([])
    #ax.set_yticks([])
    
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='C1', lw=3, label='HYCOM current'),
    Line2D([0], [0], color='C2', lw=3, label='Copernicus marine current'),
    Line2D([0], [0], color='C3', lw=3, label='Wind field'),
]


ax.legend(handles=legend_elements, loc='upper left')

cax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
plt.colorbar(im, cax=cax, label='Bathymetry (km)')
plt.savefig(os.path.join(figs_path, "open_ocean_dataset_trajectory_example.png"), dpi=300)
plt.savefig(os.path.join(figs_path, "open_ocean_dataset_trajectory_example.pdf"), dpi=300)
# %%

# polar 