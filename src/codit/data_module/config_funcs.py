"""
Functions for config handling for datasets.

Author: Christian Donner
"""

import os

import yaml

from codit.data_module import config_paths


def complete_config(config: dict) -> dict:
    """Complete the config dictionary with paths and filenames based on the provided configuration.

    Args:
        config: The configuration dictionary.

    Returns:
        The completed configuration dictionary.
    """
    latitudes = config["latitudes"]
    longitudes = config["longitudes"]
    south, north = latitudes
    west, east = longitudes
    roi_str = f"n{north}_w{west}_s{south}_e{east}"
    roi_str = roi_str.replace(".", "dot")
    start_date = config["start_date"]
    end_date = config["end_date"]
    time_str = f"{start_date.strftime('y%Ym%md%d')}_{end_date.strftime('y%Ym%md%d')}"

    # hycom file
    hycom_path = os.path.join(config["dataset_path"], "hycom")
    hycom_file = os.path.join(hycom_path, f"hycom_time{time_str}_{roi_str}_d0.nc4")

    # era5 file
    era5_path = os.path.join(config["dataset_path"], "era5")
    era5_file = os.path.join(era5_path, f"era5_time{time_str}_{roi_str}.nc")

    # copernicus file
    copernicus_path = os.path.join(config["dataset_path"], "copernicus")
    copernicus_hourly_file = os.path.join(
        copernicus_path, f"copernicus_hourly_time{time_str}_{roi_str}.nc"
    )
    copernicus_daily_file = os.path.join(
        copernicus_path, f"copernicus_daily_time{time_str}_{roi_str}.nc"
    )
    # trajectories file
    trajectories_path = os.path.join(config["dataset_path"], "trajectories")
    trajectories_file = os.path.join(
        trajectories_path, f"trajectories_time{time_str}_{roi_str}.nc"
    )
    config["hycom_path"] = hycom_path
    config["hycom_file"] = hycom_file
    config["era5_path"] = era5_path
    config["era5_file"] = era5_file
    config["copernicus_path"] = copernicus_path
    config["copernicus_hourly_file"] = copernicus_hourly_file
    config["copernicus_daily_file"] = copernicus_daily_file
    config["trajectories_path"] = trajectories_path
    config["trajectories_file"] = trajectories_file
    return config


def load_config(data_name: str, config_path: str) -> dict:
    """Load the configuration for a specific dataset.
    This function reads a YAML configuration file for a dataset, completes the configuration
    with additional paths and filenames, and returns the completed configuration dictionary.

    Args:
        data_name: The name of the dataset.
        config_path: The path to the configuration files.

    Returns:
        The completed configuration dictionary.
    """

    path = os.path.join(config_path, data_name + ".yml")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Reconstruct dataset_path relative to current DATA_PATH.
    # YAML configs store stale absolute paths from the original environment.
    # Config directory structure mirrors dataset structure:
    #   config lives at:  CONFIG_PATH/{dataset}/{split}/{name}.yml
    #   data lives at:    DATA_PATH/datasets/{dataset}/{split}/{name}/datasets/{name}
    rel = os.path.relpath(config_path, config_paths.CONFIG_PATH)
    config["dataset_path"] = os.path.join(
        config_paths.DATA_PATH, "datasets", rel,
        config["name"], "datasets", config["name"],
    )

    config = complete_config(config)
    return config


def save_config(configs: dict, config_path: str, data_path: str) -> None:
    """Save the configuration for a specific dataset.

    Args:
        configs: The configuration dictionary.
        config_path: The path to the configuration files.
        data_path: The path to the data files.
    """
    data_name = configs["name"]
    dataset_path = os.path.join(data_path, "datasets", data_name)
    configs["dataset_path"] = dataset_path
    configs["type"] = "region_based"
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    config_file = os.path.join(config_path, data_name + ".yml")
    with open(config_file, "w") as f:
        yaml.dump(configs, f, default_flow_style=False)
