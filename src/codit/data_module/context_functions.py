"""
Functions to create context for trajectory prediction models.
"""

import logging
import os
import warnings
import zipfile
from functools import partial

import jax
import xarray as xr
from jax import numpy as jnp
from jax import vmap

from codit.data_module import config_paths, data_transform

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning, message=".*serialization.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*cannot be serialized.*"
)

from_grid_to_coords = vmap(
    vmap(data_transform.inverse_haversine_grad, in_axes=(None, 0, 0)),
    in_axes=(None, 0, 0),
)
get_velocity = vmap(
    data_transform.velocity_vector_from_pos_time, in_axes=(None, 0, None)
)


@partial(vmap, in_axes=(0, None, None))
def get_grid_around_coordinate(
    center_coord: jax.Array, distance_from_origin: jax.Array, bearing_deg: jax.Array
) -> jax.Array:
    """Get a grid of coordinates around a center coordinate based on distance and bearing.

    Args:
        center_coord: The center coordinate.
        distance_from_origin: The distance from the origin.
        bearing_deg: The bearing in degrees.

    Returns:
        A grid of coordinates around the center coordinate.
    """
    coords_grid = (
        from_grid_to_coords(center_coord, distance_from_origin, bearing_deg)
        + center_coord
    )
    return coords_grid


def make_grid(
    num_grid_points: int = 16, delta_metres: float = 10000
) -> tuple[jax.Array, jax.Array]:
    """Create a grid of distances and bearings.

    Args:
        num_grid_points: The number of grid points. Defaults to 16.
        delta_metres: The size of the grid in metres. Defaults to 10000.

    Returns:
        A tuple containing the distance from the origin and the bearing in degrees.
    """
    x_vec = jnp.linspace(-delta_metres, delta_metres, num_grid_points)
    y_vec = jnp.linspace(-delta_metres, delta_metres, num_grid_points)
    grid = jnp.meshgrid(x_vec, y_vec)
    distance_from_origin = jnp.sqrt(grid[0] ** 2 + grid[1] ** 2)
    bearing = jnp.arctan2(grid[1], grid[0])
    bearing_deg = jnp.degrees(bearing)
    return distance_from_origin, bearing_deg


def load_data(
    config: dict,
    latitudes: jax.Array | None = None,
    longitudes: jax.Array | None = None,
) -> xr.Dataset:
    """Load the context data for a specific dataset.

    Args:
        config: The configuration dictionary.
        latitudes: The latitude range for the context data. Defaults to None.
        longitudes: The longitude range for the context data. Defaults to None.

    Returns:
        The context data as an xarray Dataset.
    """
    if longitudes is None:
        longitudes = config["longitudes"]
    if latitudes is None:
        latitudes = config["latitudes"]

    assert longitudes is not None
    assert latitudes is not None
    if latitudes[0] < 10 and latitudes[1] < 10:
        if longitudes[0] < -60 and longitudes[1] < -60:
            bathy = xr.open_dataset(
                os.path.join(
                    config_paths.DATA_PATH,
                    "bathymetry",
                    "GEBCO_06_Jan_2025_ba69ac5d6a3b",
                    "gebco_2023_n10.0_s-90.0_w-180.0_e-60.0.nc",
                )
            )
        elif longitudes[0] < 70 and longitudes[1] < 70:
            bathy = xr.open_dataset(
                os.path.join(
                    config_paths.DATA_PATH,
                    "bathymetry",
                    "GEBCO_06_Jan_2025_ba69ac5d6a3b",
                    "gebco_2023_n10.0_s-90.0_w-70.0_e70.0.nc",
                )
            )
        elif longitudes[0] < 180 and longitudes[1] < 180:
            bathy = xr.open_dataset(
                os.path.join(
                    config_paths.DATA_PATH,
                    "bathymetry",
                    "GEBCO_06_Jan_2025_ba69ac5d6a3b",
                    "gebco_2023_n10.0_s-90.0_w60.0_e180.0.nc",
                )
            )
    elif latitudes[0] > -10 and latitudes[1] > -10:
        if longitudes[0] < -60 and longitudes[1] < -60:
            bathy = xr.open_dataset(
                os.path.join(
                    config_paths.DATA_PATH,
                    "bathymetry",
                    "GEBCO_06_Jan_2025_ba69ac5d6a3b",
                    "gebco_2023_n90.0_s-10.0_w-180.0_e-60.0.nc",
                )
            )
        elif longitudes[0] < 70 and longitudes[1] < 70:
            bathy = xr.open_dataset(
                os.path.join(
                    config_paths.DATA_PATH,
                    "bathymetry",
                    "GEBCO_06_Jan_2025_ba69ac5d6a3b",
                    "gebco_2023_n90.0_s-10.0_w-70.0_e70.0.nc",
                )
            )
        elif longitudes[0] < 180 and longitudes[1] < 180:
            bathy = xr.open_dataset(
                os.path.join(
                    config_paths.DATA_PATH,
                    "bathymetry",
                    "GEBCO_06_Jan_2025_ba69ac5d6a3b",
                    "gebco_2023_n90.0_s-10.0_w60.0_e180.0.nc",
                )
            )
    else:
        raise ValueError(
            "Latitude and longitude ranges not supported. Please check the config."
        )

    bathy = bathy.rename({"lat": "latitude", "lon": "longitude"})
    hycom_data = xr.open_dataset(config["hycom_file"]).isel(depth=0, drop=True)
    hycom_data = hycom_data.drop_duplicates(dim="lon").drop_duplicates(dim="lat")
    hycom_data = hycom_data.drop_duplicates(dim="time")
    hycom_data = hycom_data.rename({"lat": "latitude", "lon": "longitude"})

    hycom_data = hycom_data.sel(
        latitude=slice(latitudes[0], latitudes[1]),
        longitude=slice(longitudes[0], longitudes[1]),
    )
    bathy = bathy.sel(
        latitude=slice(hycom_data["latitude"].min(), hycom_data["latitude"].max()),
        longitude=slice(hycom_data["longitude"].min(), hycom_data["longitude"].max()),
    )
    try:
        era5_data = xr.open_dataset(config["era5_file"])
    except ValueError:
        # unzip file
        logging.info("Unzipping file")
        with zipfile.ZipFile(config["era5_file"], "r") as zip_ref:
            zip_ref.extractall(config["era5_file"].replace(".nc", "") + "/")
        # load parts
        logging.info("Loading parts")
        stream = xr.open_dataset(
            config["era5_file"].replace(".nc", "")
            + "/data_stream-oper_stepType-instant.nc"
        )
        wave = xr.open_dataset(
            config["era5_file"].replace(".nc", "")
            + "/data_stream-wave_stepType-instant.nc"
        )
        # join the two datasets
        logging.info("Merging")
        era5_data = xr.merge([stream, wave])
        logging.info("Filled NA")
        era5_data = era5_data.interpolate_na(
            dim="longitude", method="linear"
        ).interpolate_na(dim="latitude", method="linear")
        # save the merged dataset
        logging.info("Saving")
        era5_data.to_netcdf(config["era5_file"])
        era5_data = xr.open_dataset(config["era5_file"])

    if era5_data["latitude"][0] > era5_data["latitude"][-1]:
        era5_data = era5_data.sel(latitude=slice(None, None, -1))

    logging.info("Selecting data")
    era5_data = era5_data.sel(
        time=slice(hycom_data["time"].min(), hycom_data["time"].max())
    )
    era5_data = era5_data.sel(
        latitude=slice(latitudes[0], latitudes[1]),
        longitude=slice(longitudes[0], longitudes[1]),
    )
    copernicus_data = xr.open_dataset(config["copernicus_daily_file"]).isel(
        depth=0, drop=True
    )
    copernicus_data = copernicus_data.sel(
        latitude=slice(latitudes[0], latitudes[1]),
        longitude=slice(longitudes[0], longitudes[1]),
    )
    logging.info("Interpolating")
    hycom_longitudes = hycom_data["longitude"].to_numpy()
    hycom_latitudes = hycom_data["latitude"].to_numpy()
    hycom_interp = hycom_data.interp(
        latitude=hycom_latitudes, longitude=hycom_longitudes, time=era5_data["time"]
    )
    era5_interp = era5_data.interp(latitude=hycom_latitudes, longitude=hycom_longitudes)
    copernicus_interp = copernicus_data.interp(
        latitude=hycom_latitudes, longitude=hycom_longitudes, time=era5_data["time"]
    )
    bathy = bathy.interp(latitude=hycom_latitudes, longitude=hycom_longitudes)
    logging.info("Merging")
    merged = xr.merge([hycom_interp, era5_interp, copernicus_interp, bathy * 1e-3])
    return merged[["water_u", "water_v", "uo", "vo", "u10", "v10", "elevation"]]
