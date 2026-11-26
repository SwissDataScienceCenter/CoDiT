"""
Utility functions for data processing and manipulation.

"""

from datetime import datetime, timedelta
from typing import Union

import numpy as np
import xarray as xr

_NETCDF3_MAGIC = b"CDF"
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def open_netcdf(path: str, **kwargs) -> xr.Dataset:
    """Open a NetCDF file with a backend chosen from its signature.

    NetCDF-3 files are read with ``scipy`` and NetCDF-4/HDF5 files with ``h5netcdf``,
    which is what xarray picks when the ``netCDF4`` package is absent. Naming the
    backend keeps the decoding independent of which optional backends are installed
    (``netCDF4``, pulled in by Parcels, fails on the trajectories' character arrays
    with xarray 2024.2 and NumPy 2).

    Raises:
        ValueError: If the file is neither NetCDF-3 nor NetCDF-4/HDF5 (e.g. a zip
            archive), as ``xr.open_dataset`` does when no backend matches.
    """
    with open(path, "rb") as f:
        magic = f.read(8)
    if magic.startswith(_NETCDF3_MAGIC):
        engine = "scipy"
    elif magic == _HDF5_MAGIC:
        engine = "h5netcdf"
    else:
        raise ValueError(f"{path} is neither a NetCDF-3 nor a NetCDF-4/HDF5 file")
    return xr.open_dataset(path, engine=engine, **kwargs)


def convert_date(nseconds: int) -> datetime:
    reference = datetime(1970, 1, 1, 0, 0, 0)
    time_delta = timedelta(seconds=nseconds)
    date = reference + time_delta
    return date


def update_time(nseconds: float) -> float:
    # nhours of 30 years
    diff = datetime(2000, 1, 1, 0, 0, 0) - datetime(1970, 1, 1, 0, 0, 0)
    # get the hours of diff
    delta_seconds = diff.total_seconds()
    new_nseconds = (nseconds - delta_seconds) / 3600.0
    return new_nseconds


def convert_date_to_hours(
    dates: Union[np.ndarray, np.datetime64],
) -> Union[np.ndarray, np.float64]:
    reference_data = np.datetime64("2000-01-01")
    return (dates - reference_data) / np.timedelta64(1, "h")
