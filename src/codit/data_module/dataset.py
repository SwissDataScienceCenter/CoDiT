import os
from datetime import datetime, timedelta

import h5py
import jax
import numpy as np
import pandas as pd
import xarray as xr
from interpax import Interpolator3D
from jax import numpy as jnp
from jax import vmap
from torch.utils.data import Dataset
from tqdm import tqdm

from codit.data_module import context_functions, utils


class OceanDataset(Dataset):

    def __init__(
        self,
        config: dict,
        length: int,
        num_grid_points: int = 16,
        delta_metres: int = 10000,
        interp_method: str = "linear",
        no_context: bool = False,
    ):
        """Dataset for drifter data with conctext.

        It contains the trajectories of drifters and the context data that can be used for interpolation.

        Args:
            config: Configuration dictionary.
            length: Length of the trajectory snippets.
            num_grid_points: Number of grid points in the context data (per dimension).
            delta_metres: The size of the patch around the drifter that is considered for context.
            interp_method: The interpolation method to use. At the moment only 'linear' is supported.
        """
        self.dataset_name = config["name"]
        self.length = length
        self.num_grid_points = num_grid_points
        self.delta_metres = delta_metres
        if interp_method != "linear":
            raise ValueError(f"Interpolation method {interp_method} is not supported.")
        self.interp_method = interp_method
        self.config = config
        longitudes = config["longitudes"]
        longitudes[0] -= config["context_delta"]
        longitudes[1] += config["context_delta"]
        latitudes = config["latitudes"]
        latitudes[0] -= config["context_delta"]
        latitudes[1] += config["context_delta"]
        if not no_context:
            self.context_xarray = context_functions.load_data(
                self.config, latitudes=latitudes, longitudes=longitudes
            )
        try:
            self.trajectories = self._load_trajectories()
        except FileNotFoundError:
            self.trajectories = []
        self.context_var_idx = jnp.arange(0, 8, dtype=int)
        
    def set_context_var_idx(self, var_idx: jnp.ndarray):
        self.context_var_idx = var_idx

    def _load_trajectories(self) -> list:
        """Load the trajectories from the dataset.

        Returns:
            A list of dictionaries containing the trajectories.
        """
        trajectory = xr.open_dataset(self.config["trajectories_file"])
        traj_df = pd.DataFrame(
            trajectory[["ID", "time", "latitude", "longitude"]].to_array().T,
            columns=["ID", "time", "latitude", "longitude"],
            index=trajectory["time"].values,
        )
        grouped = traj_df.groupby("ID")
        trajectories = []
        length_to_extract = self.length + 1
        for name, group in grouped:
            for i in range(0, len(group) - length_to_extract + 1, length_to_extract):
                snippet = group.iloc[i : i + length_to_extract]
                if np.all(snippet["time"].diff()[1:] * 1e-9 == 3600):
                    time = np.asarray(snippet.index.to_numpy(), dtype=np.datetime64)
                    traj_dict = {
                        "time": utils.convert_date_to_hours(time),
                        "dates": snippet.index,
                        "coords": snippet[["latitude", "longitude"]].to_numpy(
                            dtype=float
                        ),
                        "ids": snippet["ID"].to_numpy(),
                    }
                    trajectories.append(traj_dict)

        return trajectories

    def _create_grid(self, delta_metres: float) -> list[jax.Array]:
        """Create a grid around the drifter.

        Returns:
            A grid that can be used for calulating the context data. The differences are in metres and 0 is the
            drifter position.
        """
        x_vec = jnp.linspace(-delta_metres, delta_metres, self.num_grid_points)
        y_vec = jnp.linspace(-delta_metres, delta_metres, self.num_grid_points)
        grid = jnp.meshgrid(x_vec, y_vec)
        return grid

    def get_context(
        self, times: jnp.ndarray, coordinates: jnp.ndarray, interpolator: Interpolator3D
    ) -> jnp.ndarray:
        """Get the context data for the given times and coordinates.

        Args:
            times: The times for which to get the context data.
            coordinates: The coordinates of the drifter.
            interpolator: The interpolator to use for the context data.

        Returns:
            The context data for the given times and coordinates.
        """
        distance_from_origin, bearing_deg = context_functions.make_grid(
            num_grid_points=self.num_grid_points, delta_metres=self.delta_metres
        )
        coord_grid = (
            vmap(context_functions.from_grid_to_coords, in_axes=(0, None, None))(
                coordinates, distance_from_origin, bearing_deg
            )
            + coordinates[:, None, None]
        )
        longitude = coord_grid[:, :, :, 1]
        latitude = coord_grid[:, :, :, 0]
        xq = jnp.asarray(times)
        yq = jnp.asarray(latitude)
        zq = jnp.asarray(longitude)
        context = vmap(vmap(interpolator, in_axes=(None, 0, 0)))(xq, yq, zq)
        return context

    def __len__(self) -> int:
        """Get the number of trajectory snippets.

        Returns:
            The number of trajectory snippets.
        """
        return len(self.trajectories)

    def get_interpolator(
        self, min_date: datetime, max_date: datetime
    ) -> Interpolator3D:
        """Get the interpolator for the given data.

        Args:
            min_data: The minimum data for the interpolator.
            max_data: The maximum data for the interpolator.

        Returns:
            The interpolator for the given data.
        """
        context_array = self.context_xarray.sel(time=slice(min_date, max_date))
        # get hourly time range between min_date and max date
        times = np.arange(
            min_date,
            max_date + timedelta(hours=1),
            timedelta(hours=1),
            dtype=np.datetime64,
        )
        traj_data = context_array.interp(time=times)
        vals = jnp.asarray(traj_data.to_array().to_numpy())
        vals = vals.transpose(1, 2, 3, 0)
        vals = jnp.nan_to_num(vals)
        bin_bathy = jnp.asarray(vals[..., -1] > 0, dtype=float)
        vals = jnp.concatenate([vals, bin_bathy[..., None]], axis=-1)
        vals = vals[..., self.context_var_idx]
        x = jnp.asarray(utils.convert_date_to_hours(times))
        y = jnp.asarray(traj_data.latitude)
        z = jnp.asarray(traj_data.longitude)
        interpolator = Interpolator3D(x, y, z, vals, method=self.interp_method)
        return interpolator

    def __getitem__(self, idx: int) -> tuple:
        """Get a trajectory snippet.

        Args:
            idx: The index of the trajectory snippet.

        Returns:
            A tuple containing the start position, times, interpolator and coordinates of the trajectory snippet.
        """
        traj = self.trajectories[idx]
        min_date = traj["dates"][0] - pd.Timedelta(self.length, "h")
        max_date = traj["dates"][-1] + pd.Timedelta(self.length, "h")
        context_array = self.context_xarray.sel(time=slice(min_date, max_date))
        traj_data = context_array.interp(time=traj["dates"])
        vals = jnp.asarray(traj_data.to_array().to_numpy())
        vals = vals.transpose(1, 2, 3, 0)
        vals = jnp.nan_to_num(vals)
        bin_bathy = jnp.asarray(vals[..., -1] > 0, dtype=float)
        vals = jnp.concatenate([vals, bin_bathy[..., None]], axis=-1)
        x = jnp.asarray(utils.convert_date_to_hours(traj_data.time))
        y = jnp.asarray(traj_data.latitude)
        z = jnp.asarray(traj_data.longitude)
        interpolator = Interpolator3D(x, y, z, vals, method=self.interp_method)
        coordinates = jnp.asarray(traj["coords"])
        start_position = coordinates[0]
        times = jnp.asarray(traj["time"])
        start_time = traj["dates"][0]
        return start_position, start_time, times, interpolator, coordinates


class OceanTrainingDataset(Dataset):

    def __init__(
        self,
        trajectories: h5py.Dataset | jax.Array,
        context: h5py.Dataset | jax.Array,
        indeces: jax.Array | None = None,
    ):
        """Dataset for training data.

        It contains the trajectories of drifters and the context data at the true positions.

        Args:
            trajectories: The trajectories of the drifters.
            context: The context data at the true positions.
        """
        self.trajectories = jnp.asarray(trajectories)
        self.context = jnp.asarray(context)
        if indeces is not None:
            self.indices = indeces
        else:
            self.indices = jnp.arange(len(self.trajectories))
        self.context_var_idx = jnp.arange(0, 8, dtype=int)
        
    def set_context_var_idx(self, var_idx: jnp.ndarray):
        self.context_var_idx = var_idx

    def __len__(self) -> int:
        """Get the number of training samples.

        Returns:
            The number of training samples.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get a trajectory and the context data.

        Args:
            idx: The index of the trajectory and context data.

        Returns:
            A tuple containing the trajectory and the context data.
        """
        return np.asarray(self.trajectories[self.indices[idx]]), np.asarray(
            self.context[self.indices[idx]][..., self.context_var_idx]
        )

    def save(self, dataset_path: str) -> None:
        """Save the training data to disk.

        Args:
            dataset_path: The path to save the training data to.
        """
        # save_path = os.path.join(dataset_path, f'training_data_{self.trajectories.shape[1]-1}h.npz')
        # np.savez_compressed(save_path, context=self.context, trajectories=self.trajectories)
        save_path = os.path.join(
            dataset_path, f"training_data_{self.trajectories.shape[1]-1}h.h5"
        )
        h5_data = h5py.File(save_path, "w")
        h5_data.create_dataset("trajectories", data=self.trajectories)
        h5_data.create_dataset("context", data=self.context)
        h5_data.close()

    @classmethod
    def from_disk(cls, dataset_path: str, length: int) -> "OceanTrainingDataset":
        """Load the training data from disk.

        Args:
            dataset_path: The path to the training data.
            length: The length of the trajectories.

        Returns:
            The training dataset.
        """
        # save_path = os.path.join(dataset_path, f'training_data_{length}h.npz')
        # data = np.load(save_path)
        save_path = os.path.join(dataset_path, f"training_data_{length}h.h5")
        data = h5py.File(save_path, "r")
        return cls(data["trajectories"], data["context"])  # type: ignore
        # return cls(data['trajectories'], data['context'])

    @classmethod
    def from_dataset(cls, dataset: OceanDataset) -> "OceanTrainingDataset":
        """Create a training dataset from a dataset.

        Args:
            dataset: The dataset to create the training dataset from.

        Returns:
            The training dataset.
        """
        trajectories = []
        context = []

        for sample in tqdm(iter(dataset)):
            #_, times, interpolator, coordinates = sample
            _, _, times, interpolator, coordinates = sample
            context.append(dataset.get_context(times, coordinates, interpolator)[None])
            trajectories.append(coordinates[None])

        context = jnp.concatenate(context, axis=0)
        trajectories = jnp.concatenate(trajectories, axis=0)
        return cls(trajectories, context)

    def shuffle(self) -> None:
        """Shuffle the training data."""
        self.indices = jnp.asarray(
            self.indices[np.random.permutation(len(self.trajectories))]
        )
        # self.trajectories = self.trajectories[idx]
        # self.context = self.context[idx]

    def split(self, split: float) -> tuple:
        """Split the training data into two parts.

        Args:
            split: The fraction of the data to put in the first part.

        Returns:
            Two training datasets.
        """
        split_idx = int(len(self.trajectories) * split)
        train_indices = jnp.asarray(self.indices[:split_idx], dtype=int)
        val_indices = jnp.asarray(self.indices[split_idx:], dtype=int)
        return OceanTrainingDataset(
            self.trajectories, self.context, train_indices
        ), OceanTrainingDataset(self.trajectories, self.context, val_indices)

    def remove_nans(self) -> None:
        """Remove samples with NaNs."""
        nan_idx = np.isnan(self.context).any(axis=(1, 2, 3, 4))
        self.trajectories = self.trajectories[~nan_idx]
        self.context = self.context[~nan_idx]
        self.indices = np.arange(len(self.trajectories))

    def load_into_memory(self) -> None:
        """Load the dataset into memory."""
        self.trajectories = self.trajectories[:]
        self.context = self.context[:]


def merge_training_sets(datasets: list) -> OceanTrainingDataset:
    """Merge multiple training datasets into one.

    Args:
        datasets: The datasets to merge.

    Returns:
        The merged training dataset.
    """
    trajectories = []
    context = []
    for dataset in datasets:
        trajectories.append(dataset.trajectories)
        context.append(dataset.context)
    trajectories = jnp.concatenate(trajectories, axis=0)
    context = jnp.concatenate(context, axis=0)
    return OceanTrainingDataset(trajectories, context)
