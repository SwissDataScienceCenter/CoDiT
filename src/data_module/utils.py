"""
Utility functions for data processing and manipulation.

"""

import logging
from datetime import datetime, timedelta
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from jax import numpy as jnp

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