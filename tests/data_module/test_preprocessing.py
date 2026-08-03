"""Unit and CLI-smoke tests for preprocess_configs."""
import os
from datetime import datetime, timedelta

import h5py
import numpy as np
import pytest
import xarray as xr
import yaml

from codit.data_module import config_funcs


def _write_trajectory_nc(path: str, start_date: datetime, length_hours: int, lat: float, lon: float) -> None:
    """Write a minimal trajectories NetCDF matching the real Global Drifter Program schema.

    Real GDP files use dimension ``row`` with ``time``, ``ID``, ``latitude``, ``longitude``
    all as data variables (not coords); ``time`` is ``datetime64[ns]`` and ``ID`` is a
    string (object dtype). ``OceanDataset._load_trajectories`` requires this shape.
    """
    times = np.array(
        [start_date + timedelta(hours=h) for h in range(length_hours + 1)],
        dtype="datetime64[ns]",
    )
    n = len(times)
    ds = xr.Dataset(
        data_vars={
            "ID": ("row", np.full(n, "test_drifter", dtype=object)),
            "time": ("row", times),
            "latitude": ("row", np.full(n, lat, dtype=np.float32)),
            "longitude": ("row", np.full(n, lon, dtype=np.float32)),
        },
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_netcdf(path)


def _synthetic_context_xarray(start_date: datetime, length_hours: int, lat: float, lon: float) -> xr.Dataset:
    """Return an xarray Dataset shaped exactly like context_functions.load_data output.

    Seven raw channels (hycom u/v, copernicus u/v, era5 u/v, bathymetry_raw) with
    a small lat/lon/time cube. Dataset.py appends the binary bathymetry channel.
    """
    times = np.array(
        [start_date + timedelta(hours=h) for h in range(length_hours + 3)],
        dtype="datetime64[ns]",
    )
    lats = np.linspace(lat - 0.5, lat + 0.5, 8, dtype=np.float32)
    lons = np.linspace(lon - 0.5, lon + 0.5, 8, dtype=np.float32)
    shape = (len(times), len(lats), len(lons))
    dims = ("time", "latitude", "longitude")
    ones = np.ones(shape, dtype=np.float32) * 0.1  # small nonzero currents/winds
    bathy = np.full(shape, -1000.0, dtype=np.float32)  # negative → ocean
    return xr.Dataset(
        data_vars={
            "hycom_u": (dims, ones),
            "hycom_v": (dims, ones),
            "copernicus_u": (dims, ones),
            "copernicus_v": (dims, ones),
            "era5_u": (dims, ones),
            "era5_v": (dims, ones),
            "bathymetry": (dims, bathy),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


@pytest.fixture
def two_patch_fixture(tmp_path, monkeypatch):
    """Two synthetic patches, both preprocessable without any external data."""
    monkeypatch.setenv("CODIT_DATA_PATH", str(tmp_path))
    import importlib
    from codit.data_module import config_paths
    importlib.reload(config_paths)
    importlib.reload(config_funcs)

    from codit.data_module import context_functions

    start = datetime(2019, 6, 1, 0, 0, 0)
    length = 12  # short trajectory to keep the test fast
    configs = []
    for i, (lat, lon) in enumerate([(15.0, -60.0), (16.0, -59.0)]):
        name = f"patch_{i}"
        raw = {
            "name": name,
            "latitudes": [lat - 0.5, lat + 0.5],
            "longitudes": [lon - 0.5, lon + 0.5],
            "start_date": start,
            "end_date": start + timedelta(hours=length + 2),
            "context_delta": 0.1,
            "description": "",
        }
        cfg_dir = tmp_path / "configs" / "unit" / "train"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / f"{name}.yml", "w") as f:
            yaml.dump(raw, f)
        cfg = config_funcs.load_config(name, str(cfg_dir))
        os.makedirs(cfg["trajectories_path"], exist_ok=True)
        _write_trajectory_nc(cfg["trajectories_file"], start, length, lat, lon)
        configs.append(cfg)

    monkeypatch.setattr(
        context_functions,
        "load_data",
        lambda config, latitudes=None, longitudes=None: _synthetic_context_xarray(
            start, length, config["latitudes"][0] + 0.5, config["longitudes"][0] + 0.5
        ),
    )
    return configs, tmp_path


def test_preprocess_configs_writes_expected_h5(two_patch_fixture):
    from codit.data_module.preprocessing import preprocess_configs

    configs, tmp_path = two_patch_fixture
    out_dir = tmp_path / "processed"

    result_path = preprocess_configs(
        configs=configs,
        length=6,  # short snippet length
        num_grid_points=4,
        delta_metres=1000,
        output_path=str(out_dir),
    )

    assert result_path == str(out_dir / "training_data_6h.h5")
    assert os.path.exists(result_path)

    with h5py.File(result_path, "r") as f:
        trajectories = f["trajectories"][:]
        context = f["context"][:]
        # (N, 7, 2): 7 = snippet length + 1 endpoint; 2 = lat/lon
        assert trajectories.shape[1] == 7
        assert trajectories.shape[2] == 2
        # (N, 7, num_grid_points, num_grid_points, 8) — 8 channels after
        # dataset.py appends the binary bathymetry channel.
        assert context.shape[1] == 7
        assert context.shape[2] == 4
        assert context.shape[3] == 4
        assert context.shape[4] == 8
        # both patches contributed at least one snippet each
        assert trajectories.shape[0] >= 2
