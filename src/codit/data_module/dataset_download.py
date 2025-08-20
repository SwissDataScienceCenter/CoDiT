"""
Download functions for various datasets used in the project.
"""

import configparser
import functools
import json
import os
import shutil
import warnings
from datetime import datetime, timedelta
from typing import Callable

import cdsapi
import copernicusmarine
import requests
import xarray as xr
from tqdm import tqdm

from codit.data_module import config_paths

os.environ["COPERNICUSMARINE_CACHE_DIRECTORY"] = "/tmp"
TMP_PATH = "/tmp"


def deprecated(func: Callable) -> Callable:
    @functools.wraps(func)
    def new_func(*args: list, **kwargs: dict) -> Callable:
        warnings.warn(
            f"Call to deprecated function {func.__name__}. Please use the recommended alternative function or approach instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return new_func


def download_drifter(config: dict) -> None:
    """Download drifter data

    Args:
        config (dict): Dataset dictionary with the following keys:
            - latitudes (tuple): A tuple of two floats specifying the latitude range (south, north).
            - longitudes (tuple): A tuple of two floats specifying the longitude range (west, east).
            - start_date (datetime): The start date for the data download.
            - end_date (datetime): The end date for the data download.
            - trajectories_path (str): Path to save the downloaded trajectories data.
    """

    latitudes = config["latitudes"]
    longitudes = config["longitudes"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    noaa_url = (
        "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/"
        + "drifter_hourly_qc.nc?ID%2CWMO%2Cexpno%2Ctime%2C"
        + "latitude%2Clongitude%2Clon360%2Cve%2Cvn%2Cerr_lat%2C"
        + "err_lon%2Cerr_ve%2Cerr_vn%2Ctypebuoy%2Cgap%2C"
        + "location_type%2CFloatDiameter%2CSubsfcFloatPresence%2C"
        + "DrogueType%2CDrogueLength%2CDrogueBallast%2C"
        + "DragAreaAboveDrogue%2CDragAreaOfDrogue%2CDragAreaRatio%2C"
        + "DrogueCenterDepth%2CDrogueDetectSensor&"
        + f"time%3E={start_date_str}&time%3C={end_date_str}&"
        + f"latitude%3E={latitudes[0]}&latitude%3C={latitudes[1]}"
        + f"&longitude%3E={longitudes[0]}&longitude%3C={longitudes[1]}"
    )

    trajectories_path = config["trajectories_path"]
    if not os.path.exists(trajectories_path):
        os.makedirs(trajectories_path, exist_ok=True)
    south, north = latitudes
    west, east = longitudes
    roi_str = f"n{north}_w{west}_s{south}_e{east}"
    # Replace periods with "dot" to ensure the region string is file-system friendly.
    roi_str = roi_str.replace(".", "dot")
    time_str = f"{start_date.strftime('y%Ym%md%d')}_{end_date.strftime('y%Ym%md%d')}"
    if not os.path.isdir(trajectories_path):
        raise ValueError(f"Invalid or inaccessible directory: {trajectories_path}")
    trajectories_file = os.path.join(
        trajectories_path, f"trajectories_time{time_str}_{roi_str}.nc"
    )
    trajectories_file = os.path.join(
        trajectories_path, f"trajectories_time{time_str}_{roi_str}.nc"
    )
    # Open the file in binary mode as the response content is binary data
    try:
        response = requests.get(noaa_url)
        if response.status_code == 200:
            import logging

            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )

            with open(trajectories_file, "wb") as f:
                f.write(response.content)
            logging.info("Drifter data downloaded successfully")
            print("Drifter data downloaded successfully")
        else:
            print(f"Failed to download drifter data: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download drifter data: {e}")


def _create_hycom_url(toi: tuple, roi: tuple, depth: float = 0) -> list[str]:
    """Create HYCOM URL

    Args:
        toi: Time of interest (start_date, end_date)
        roi: Region of interest (north, west, south, east)
        depth: Depth to download. Defaults to 0.

    Returns:
        Download URL
    """
    min_date, max_date = toi
    if min_date < datetime(year=2014, month=7, day=1):
        catalog = f"/GLBv0.08/expt_57.X/data/{min_date.year}"
    elif min_date <= datetime(year=2016, month=4, day=30):
        catalog = "/GLBv0.08/expt_56.3"
    elif min_date <= datetime(year=2017, month=1, day=31):
        catalog = "/GLBv0.08/expt_57.2"
    elif min_date < datetime(year=2017, month=5, day=31):
        catalog = "/GLBv0.08/expt_92.8"
    elif min_date < datetime(year=2017, month=9, day=30):
        catalog = "/GLBv0.08/expt_57.7"
    elif min_date < datetime(year=2017, month=12, day=31):
        catalog = "/GLBv0.08/expt_92.9"
    elif min_date < datetime(year=2020, month=2, day=18):
        catalog = "/GLBv0.08/expt_93.0"
    else:
        catalog = "/GLBy0.08/expt_93.0/uv3z"
    # catalog = f"/GLBv0.08/expt_93.0"
    north, west, south, east = roi
    if west < 0:
        west_hycom = west + 360
    else:
        west_hycom = west
    if east < 0:
        east_hycom = east + 360
    else:
        east_hycom = east

    data_str_min = min_date.strftime("%Y-%m-%d")
    data_str_max = max_date.strftime("%Y-%m-%d")

    if data_str_min == data_str_max:
        data_str_max = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")

    if west_hycom < east_hycom:
        url = (
            f"https://ncss.hycom.org/thredds/ncss{catalog}?var=water_u&var=water_v"
            + f"&north={north}&west={west_hycom}&east={east_hycom}&south={south}"
            + f"&disableProjSubset=on&horizStride=1&time_start={data_str_min}"
            + f"T12%3A00%3A00Z&time_end={data_str_max}T09%3A00%3A00Z&timeStride=1&vertCoord={depth}&accept=netcdf4"
        )
        return [url]
    else:
        url1 = (
            f"https://ncss.hycom.org/thredds/ncss{catalog}?var=water_u&var=water_v"
            + f"&north={north}&west={west_hycom}&east={360}&south={south}"
            + f"&disableProjSubset=on&horizStride=1&time_start={min_date.strftime('%Y-%m-%d')}T12%3A00%3A00Z"
            + f"&time_end={max_date.strftime('%Y-%m-%d')}T09%3A00%3A00Z&timeStride=1&vertCoord={depth}&accept=netcdf4"
        )
        url2 = (
            f"https://ncss.hycom.org/thredds/ncss{catalog}?var=water_u&var=water_v"
            + f"&north={north}&west={0}&east={east_hycom}&south={south}"
            + f"&disableProjSubset=on&horizStride=1&time_start={min_date.strftime('%Y-%m-%d')}T12%3A00%3A00Z"
            + f"&time_end={max_date.strftime('%Y-%m-%d')}T09%3A00%3A00Z&timeStride=1&vertCoord={depth}&accept=netcdf4"
        )
        return [url1, url2]


def _download_hycom_roi_data(
    latitudes: tuple,
    longitudes: tuple,
    context_delta: float,
    start_date: datetime,
    end_date: datetime,
    depth: int,
    path: str,
) -> str:
    """Download HYCOM data for a region of interest

    Args:
        latitudes: Min and max latitudes
        longitudes: Min and max longitudes
        start_date: Start date
        end_date: End date
        depth: Depth of the water column
        path: Path to save the data

    Returns:
        Path to the downloaded data
    """
    south, north = latitudes
    west, east = longitudes
    roi_str = f"n{north}_w{west}_s{south}_e{east}"
    roi_str = roi_str.replace(".", "dot")
    current_date = start_date
    all_days = []

    while current_date <= end_date:
        all_days.append(current_date)
        current_date += timedelta(days=1)

    for day in tqdm(all_days, desc="Downloading HYCOM data", leave=False, unit="days"):
        start_date = day  # datetime(year=year, month=month, day=1)
        # if month < 12:
        #    end_date = datetime(year=year, month=month + 1, day=1)
        # else:
        #    end_date = datetime(year=year + 1, month=1, day=1)
        end_date = day + timedelta(days=1)
        day_str = day.strftime("y%Ym%md%d")
        path_water = os.path.join(path, f"hycom_{day_str}_{roi_str}_d{depth}.nc4")
        if os.path.exists(path_water):
            continue
        url = _create_hycom_url(
            (start_date, end_date),
            (
                north + context_delta,
                west - context_delta,
                south - context_delta,
                east + context_delta,
            ),
            depth=depth,
        )
        if len(url) == 2:
            path_water_1 = os.path.join(
                TMP_PATH, f"hycom_{day_str}_{roi_str}_d{depth}_a.nc4"
            )
            path_water_2 = os.path.join(
                TMP_PATH, f"hycom_{day_str}_{roi_str}_d{depth}_b.nc4"
            )

            response1 = requests.get(url[0])
            response2 = requests.get(url[1])
            with open(path_water_1, "wb") as f1:
                f1.write(response1.content)
            with open(path_water_2, "wb") as f2:
                f2.write(response2.content)
            # concatenate two files
            try:
                hycom_data1 = xr.open_dataset(path_water_1)
                hycom_data2 = xr.open_dataset(path_water_2)
                hycom_data = xr.concat([hycom_data1, hycom_data2], dim="lon")
                hycom_data.to_netcdf(path_water)
            except ValueError:
                import logging

                logging.warning(
                    f"Temporary file not found: {path_water_1} or {path_water_2}"
                )
            # delete temporary files
            try:
                os.remove(path_water_1)
                os.remove(path_water_2)
            except FileNotFoundError:
                pass
        else:
            response = requests.get(url[0])
            tmp_path_water = os.path.join(
                TMP_PATH, f"hycom_{day_str}_{roi_str}_d{depth}.nc4"
            )
            # tmp_path_water = path_water#os.path.join(path_water, f"hycom_{day_str}_{roi_str}_d{depth}.nc4")
            with open(tmp_path_water, "wb") as f:
                f.write(response.content)

            try:
                shutil.move(tmp_path_water, path_water)
            except Exception as e:
                print(f"Failed to move file from {tmp_path_water} to {path_water}: {e}")

        try:
            hycom_data2 = xr.open_dataset(path_water)
        except (ValueError, FileNotFoundError):
            print(f"Could not find file for {day_str}")

    return path_water


def download_hycom(config: dict) -> None:
    """Download HYCOM data for a region of interest

    Args:
        config: Dataset dictionary
    """
    latitudes = config["latitudes"]
    longitudes = config["longitudes"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    context_delta = config["context_delta"]
    hycom_path = config["hycom_path"]
    if not os.path.exists(hycom_path):
        os.makedirs(hycom_path)
    hycom_tmp_path = os.path.join(hycom_path, "tmp")
    if not os.path.exists(hycom_tmp_path):
        os.makedirs(hycom_tmp_path)

    _download_hycom_roi_data(
        latitudes,
        longitudes,
        context_delta,
        start_date,
        end_date,
        depth=0,
        path=hycom_tmp_path,
    )

    current_date = start_date
    all_days = []

    while current_date < end_date:
        all_days.append(current_date)
        current_date += timedelta(days=1)
    south, north = latitudes
    west, east = longitudes
    roi_str = f"n{north}_w{west}_s{south}_e{east}"
    # Replace . by p
    roi_str = roi_str.replace(".", "dot")

    days_not_loaded = []
    for day in all_days:
        day_str = day.strftime("y%Ym%md%d")
        if day == start_date:
            # print(hycom_tmp_path, f"hycom_{day_str}_{roi_str}_d0.nc4")
            hycom_data = xr.open_dataset(
                os.path.join(hycom_tmp_path, f"hycom_{day_str}_{roi_str}_d0.nc4")
            )
            lon = hycom_data["lon"].to_numpy()
            lon[lon > 180] -= 360
            hycom_data["lon"] = lon
        else:
            try:
                # print(hycom_tmp_path, f"hycom_{day_str}_{roi_str}_d0.nc4")
                hycom_data2 = xr.open_dataset(
                    os.path.join(hycom_tmp_path, f"hycom_{day_str}_{roi_str}_d0.nc4")
                )
                lon = hycom_data2["lon"].to_numpy()
                lon[lon > 180] -= 360
                hycom_data2["lon"] = lon
                hycom_data = xr.concat([hycom_data, hycom_data2], dim="time")
            except ValueError:
                days_not_loaded.append(day_str)
                print(f"Could not find file for {day_str}")
                continue

    hycom_data.to_netcdf(
        os.path.join(
            hycom_path,
            f"hycom_time{start_date.strftime('y%Ym%md%d')}_{end_date.strftime('y%Ym%md%d')}_{roi_str}_d0.nc4",
        )
    )
    with open(os.path.join(hycom_path, "days_not_loaded.json"), "w") as f:
        json.dump(days_not_loaded, f)
    print("HYCOM data downloaded successfully")


def _download_era5_roi_data(
    c: cdsapi.Client,
    latitudes: tuple,
    longitudes: tuple,
    context_delta: float,
    start_date: datetime,
    end_date: datetime,
    path: str,
) -> str:

    south, north = latitudes
    west, east = longitudes
    roi_str = f"n{north}_w{west}_s{south}_e{east}"
    roi_str = roi_str.replace(".", "dot")
    path_wind = os.path.join(
        path,
        f"era5_time{start_date.strftime('y%Ym%md%d')}_{end_date.strftime('y%Ym%md%d')}_{roi_str}.grib",
    )
    tmp_file = os.path.join(
        TMP_PATH,
        f"era5_time{start_date.strftime('y%Ym%md%d')}_{end_date.strftime('y%Ym%md%d')}_{roi_str}_tmp.grib",
    )
    if os.path.exists(path_wind):
        return path_wind
    # datestring = get_datestr(start_date, end_date)
    # datestring = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    area = [
        north + context_delta,
        west - context_delta,
        south - context_delta,
        east + context_delta,
    ]
    # round to two decimal places
    area = [round(coord, 2) for coord in area]
    # get range of years as list of strings
    years = [str(year) for year in range(start_date.year, end_date.year + 1)]
    # get months
    months = [str(month).zfill(2) for month in range(1, 13)]
    if end_date.year == start_date.year:
        months = [
            str(month).zfill(2) for month in range(start_date.month, end_date.month + 1)
        ]
    days = [str(day).zfill(2) for day in range(1, 32)]
    # create datestring

    c.retrieve(
        name="reanalysis-era5-single-levels",
        request={
            "product_type": ["reanalysis"],
            "data_format": "grib",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            # "date": datestring,
            "year": years,
            "month": months,
            "day": days,
            "time": ["%02d:00" % (hour) for hour in range(24)],
            "area": area,
            "download_format": "unarchived",
            # "grid": "0.08/0.08",
        },
        target=tmp_file,
    )
    shutil.move(tmp_file, path_wind)
    data = xr.open_dataset(path_wind)
    data.to_netcdf(path_wind.replace(".grib", ".nc"))
    return path_wind


def download_era5(config: dict) -> None:
    """Download ERA5 data for a region of interest

    Args:
        config: Dataset dictionary
    """
    latitudes = config["latitudes"]
    longitudes = config["longitudes"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    era5_path = config["era5_path"]
    context_delta = config["context_delta"]
    south, north = latitudes
    west, east = longitudes
    roi_str = f"n{north}_w{west}_s{south}_e{east}"
    roi_str = roi_str.replace(".", "dot")
    path_wind = f"{era5_path}/era5_time{start_date.strftime('y%Ym%md%d')}_{end_date.strftime('y%Ym%md%d')}_{roi_str}.nc"
    if not os.path.exists(era5_path):
        os.makedirs(era5_path)
    if not os.path.exists(path_wind):
        # c = cdsapi.Client()
        configp = configparser.ConfigParser()
        configp.read(config_paths.CREDENTIALS_PATH)
        c = cdsapi.Client(
            url=configp["copernicus"]["url"], key=configp["copernicus"]["key"]
        )
        era5_path = _download_era5_roi_data(
            c, latitudes, longitudes, context_delta, start_date, end_date, era5_path
        )
        print("ERA5 data downloaded successfully")
    else:
        print("ERA5 data already downloaded")


# def download_era5_wind_data(
#     c: cdsapi.Client, trajectory: DrifterTrajectory, path: str, delta: float = 0.1
# ) -> str:
#     roi_traj = trajectory.get_trajectory_region(delta=delta)
#     north, west, south, east = roi_traj
#     roi_str = f"n{north}_w{west}_s{south}_e{east}"
#     toi_traj = trajectory.get_time_period()
#     min_date, max_date = toi_traj
#     data_str_min = min_date.strftime("%Y-%m-%d")
#     data_str_max = max_date.strftime("%Y-%m-%d")
#     if data_str_min == data_str_max:
#         max_date = max_date + timedelta(days=1)
#         data_str_max = max_date.strftime("%Y-%m-%d")
#     toi_str = f"{toi_traj[0].strftime('%Y-%m-%d')}_{toi_traj[1].strftime('%Y-%m-%d')}"
#     datestring = get_datestr(min_date, max_date)
#     file_name = f"{toi_str}_{roi_str}_era5.nc"
#     if os.path.exists(f"{path}/{file_name}"):
#         return f"{path}/{file_name}"
#     c.retrieve(
#         "reanalysis-era5-single-levels",
#         {
#             "product_type": "reanalysis",
#             "format": "netcdf",
#             "variable": [
#                 "10m_u_component_of_wind",
#                 "10m_v_component_of_wind",
#                 "u_component_stokes_drift",
#                 "v_component_stokes_drift",
#                 "model_bathymetry",
#             ],
#             "date": datestring,
#             "time": ["%02d:00" % (hour) for hour in range(24)],
#             "area": [north, west, south, east],
#             "grid": "0.08/0.08",
#         },
#         f"{path}/{file_name}",
#     )
#     return path + file_name


def download_copernicus(config: dict) -> None:
    """Download Copernicus data for a region of interest

    Args:
        config: Dataset dictionary
    """
    latitudes = config["latitudes"]
    longitudes = config["longitudes"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    context_delta = config["context_delta"]
    south, north = latitudes
    west, east = longitudes
    roi_str = f"n{north}_w{west}_s{south}_e{east}"
    roi_str = roi_str.replace(".", "dot")
    time_str = f"{start_date.strftime('y%Ym%md%d')}_{end_date.strftime('y%Ym%md%d')}"
    copernicus_path = config["copernicus_path"]
    credentials = configparser.ConfigParser()
    credentials.read(config_paths.CREDENTIALS_PATH)
    if not os.path.exists(copernicus_path):
        os.makedirs(copernicus_path)
    """
    if not os.path.exists(os.path.join(copernicus_path, f"copernicus_hourly_time{time_str}_{roi_str}.nc")):
        try:
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
                variables=["uo", "vo"],
                minimum_longitude=west - context_delta,
                maximum_longitude=east + context_delta,
                minimum_latitude=south - context_delta,
                maximum_latitude=north + context_delta,
                start_datetime=f"{start_date.strftime('%Y-%m-%d')}T00:00:00",
                end_datetime=f"{end_date.strftime('%Y-%m-%d')}T23:59:59",
                output_filename=f"copernicus_hourly_time{time_str}_{roi_str}.nc",
                output_directory=TMP_PATH,
                username=credentials["copernicusmarine"]["username"],
                password=credentials["copernicusmarine"]["password"],
                force_download=True,
                no_metadata_cache=True
                
            )
            # copy file to copernicus path and remove
            #os.rename(f"/tmp/copernicus_hourly_time{time_str}_{roi_str}.nc", os.path.join(copernicus_path, f"copernicus_hourly_time{time_str}_{roi_str}.nc"))
            shutil.move(os.path.join(TMP_PATH, f"copernicus_hourly_time{time_str}_{roi_str}.nc"), 
                        os.path.join(copernicus_path, f"copernicus_hourly_time{time_str}_{roi_str}.nc"))
            print("Copernicus hourly data downloaded successfully")
        except copernicusmarine.core_functions.exceptions.CoordinatesOutOfDatasetBounds as e:
            print("Coordinates out of dataset bounds")
    else:
        print("Copernicus daily data already downloaded")
    """
    if not os.path.exists(
        os.path.join(copernicus_path, f"copernicus_daily_time{time_str}_{roi_str}.nc")
    ):
        if start_date < datetime(2021, 7, 1) and end_date < datetime(2021, 7, 1):
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
                variables=["uo", "vo"],
                minimum_longitude=west - context_delta,
                maximum_longitude=east + context_delta,
                minimum_latitude=south - context_delta,
                maximum_latitude=north + context_delta,
                maximum_depth=1,
                start_datetime=f"{start_date.strftime('%Y-%m-%d')}T00:00:00",
                end_datetime=f"{end_date.strftime('%Y-%m-%d')}T23:59:59",
                output_filename=f"copernicus_daily_time{time_str}_{roi_str}.nc",
                output_directory=TMP_PATH,
                username=credentials["copernicusmarine"]["username"],
                password=credentials["copernicusmarine"]["password"],
                # force_download=True,
                # no_metadata_cache=True
            )
        elif start_date >= datetime(2021, 7, 1) and end_date >= datetime(2021, 7, 1):
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",
                variables=["uo", "vo"],
                minimum_longitude=west - context_delta,
                maximum_longitude=east + context_delta,
                minimum_latitude=south - context_delta,
                maximum_latitude=north + context_delta,
                maximum_depth=1,
                start_datetime=f"{start_date.strftime('%Y-%m-%d')}T00:00:00",
                end_datetime=f"{end_date.strftime('%Y-%m-%d')}T23:59:59",
                output_filename=f"copernicus_daily_time{time_str}_{roi_str}.nc",
                output_directory=TMP_PATH,
                username=credentials["copernicusmarine"]["username"],
                password=credentials["copernicusmarine"]["password"],
                # force_download=True,
                # no_metadata_cache=True
            )
        elif start_date < datetime(2021, 7, 1) and end_date >= datetime(2021, 7, 1):
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
                variables=["uo", "vo"],
                minimum_longitude=west - context_delta,
                maximum_longitude=east + context_delta,
                minimum_latitude=south - context_delta,
                maximum_latitude=north + context_delta,
                maximum_depth=1,
                start_datetime=f"{start_date.strftime('%Y-%m-%d')}T00:00:00",
                end_datetime="2021-06-30T00:00:00",
                output_filename=f"copernicus_daily_time{time_str}_{roi_str}1.nc",
                output_directory=TMP_PATH,
                username=credentials["copernicusmarine"]["username"],
                password=credentials["copernicusmarine"]["password"],
                # force_download=True,
                # no_metadata_cache=True
            )
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",
                variables=["uo", "vo"],
                minimum_longitude=west - context_delta,
                maximum_longitude=east + context_delta,
                minimum_latitude=south - context_delta,
                maximum_latitude=north + context_delta,
                maximum_depth=1,
                start_datetime="2021-06-30T00:00:00",
                end_datetime=f"{end_date.strftime('%Y-%m-%d')}T23:59:59",
                output_filename=f"copernicus_daily_time{time_str}_{roi_str}2.nc",
                output_directory=TMP_PATH,
                username=credentials["copernicusmarine"]["username"],
                password=credentials["copernicusmarine"]["password"],
                # force_download=True,
                # no_metadata_cache=True
            )
            # merge files
            ds1 = xr.open_dataset(
                os.path.join(TMP_PATH, f"copernicus_daily_time{time_str}_{roi_str}1.nc")
            )
            ds2 = xr.open_dataset(
                os.path.join(TMP_PATH, f"copernicus_daily_time{time_str}_{roi_str}2.nc")
            )
            ds = xr.concat([ds1, ds2], dim="time")
            ds.to_netcdf(
                os.path.join(TMP_PATH, f"copernicus_daily_time{time_str}_{roi_str}.nc")
            )
            os.remove(
                os.path.join(TMP_PATH, f"copernicus_daily_time{time_str}_{roi_str}1.nc")
            )
            os.remove(
                os.path.join(TMP_PATH, f"copernicus_daily_time{time_str}_{roi_str}2.nc")
            )

        # copy file to copernicus path and remove
        shutil.move(
            os.path.join(TMP_PATH, f"copernicus_daily_time{time_str}_{roi_str}.nc"),
            os.path.join(
                copernicus_path, f"copernicus_daily_time{time_str}_{roi_str}.nc"
            ),
        )
        print("Copernicus daily data downloaded successfully")
    else:
        print("Copernicus daily data already downloaded")


################### DEPRECATED FUNCTIONS ###################
# @deprecated
# def _daterange(start_date, end_date):
#     # https://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
#     for n in range(int((end_date - start_date).days) + 1):
#         yield start_date + timedelta(n)

# @deprecated
# def _get_datestr(min_date: datetime, max_date: datetime) -> str:
#     datestr = ""
#     for day in _daterange(min_date, max_date):
#         datestr += f"{day.strftime('%Y-%m-%d')}/"
#     return datestr[:-1]

# @deprecated
# def download_hycom_data_global(
#     north: float,
#     west: float,
#     south: float,
#     east: float,
#     depth: int,
#     year: int,
#     path: str,
# ) -> str:
#     roi_str = f"n{north}_w{west}_s{south}_e{east}"
#     start_date = datetime(year, 1, 1)
#     end_date = datetime(year + 1, 1, 1)

#     current_date = start_date
#     all_days = []

#     while current_date < end_date:
#         all_days.append(current_date)
#         current_date += timedelta(days=1)

#     for day in all_days:
#         start_date = day  # datetime(year=year, month=month, day=1)
#         # if month < 12:
#         #    end_date = datetime(year=year, month=month + 1, day=1)
#         # else:
#         #    end_date = datetime(year=year + 1, month=1, day=1)
#         end_date = day + timedelta(days=1)
#         day_str = day.strftime("%Y-%m-%d")
#         path_water = f"{path}/global_{year}_{day_str}_{roi_str}_hycom_d{depth}.nc4"
#         if os.path.exists(path_water):
#             continue
#         url = create_hycom_url(
#             (start_date, end_date), (north, west, south, east), depth=depth
#         )
#         print(start_date, end_date)
#         print(url)
#         response = requests.get(url)

#         with open(path_water, "wb") as f:
#             f.write(response.content)
#     return path_water

# @deprecated
# def download_hycom_data(
#     trajectory: DrifterTrajectory, path: str, depth: int = 0, delta: float = 0.1
# ) -> str:
#     roi_traj = trajectory.get_trajectory_region(delta=delta)
#     roi_str = f"n{roi_traj[0]}_w{roi_traj[1]}_s{roi_traj[2]}_e{roi_traj[3]}"
#     toi_traj = trajectory.get_time_period()
#     toi_str = f"{toi_traj[0].strftime('%Y-%m-%d')}_{toi_traj[1].strftime('%Y-%m-%d')}"
#     path_water = (
#         f"{path}/{trajectory.drifter_id}_{toi_str}_{roi_str}_hycom_d{depth}.nc4"
#     )
#     if os.path.exists(path_water):
#         return path_water
#     url = create_hycom_url(toi_traj, roi_traj, depth=depth)
#     response = requests.get(url)

#     with open(path_water, "wb") as f:
#         f.write(response.content)
#     return path_water


# @deprecated
# def download_era5_wind_data_global(
#     c: cdsapi.Client,
#     north: float,
#     west: float,
#     south: float,
#     east: float,
#     year: int,
#     path: str,
# ) -> str:
#     roi_str = f"n{north}_w{west}_s{south}_e{east}"
#     start_date = datetime(year=year, month=1, day=1)
#     end_date = datetime(year=year, month=12, day=31)
#     path_wind = f"{path}/global_{year}_{roi_str}_era5.nc"
#     if os.path.exists(path_wind):
#         return path_wind
#     # datestring = get_datestr(start_date, end_date)
#     datestring = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
#     c.retrieve(
#         "reanalysis-era5-single-levels",
#         {
#             "product_type": "reanalysis",
#             "format": "netcdf",
#             "variable": [
#                 "10m_u_component_of_wind",
#                 "10m_v_component_of_wind",
#                 "u_component_stokes_drift",
#                 "v_component_stokes_drift",
#                 "model_bathymetry",
#             ],
#             "date": datestring,
#             "time": ["%02d:00" % (hour) for hour in range(24)],
#             "area": [north, west, south, east],
#             # "grid": "0.08/0.08",
#         },
#         path_wind,
#     )
#     return path_wind
# # %%
