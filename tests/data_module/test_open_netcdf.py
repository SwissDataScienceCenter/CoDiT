"""``utils.open_netcdf`` picks the backend from the file signature, not from which
optional backends happen to be installed."""

import zipfile

import numpy as np
import pytest
import xarray as xr

from codit.data_module import utils


def _dataset():
    return xr.Dataset(
        {
            "latitude": ("obs", np.array([1.0, 2.0, 3.0])),
            "ID": ("obs", np.array(["300234064833010", "300234064833010", "42"])),
        }
    )


@pytest.mark.parametrize("engine", ["scipy", "h5netcdf"])
def test_roundtrip_uses_matching_backend(tmp_path, engine):
    path = tmp_path / f"data_{engine}.nc"
    _dataset().to_netcdf(path, engine=engine)
    ds = utils.open_netcdf(str(path))
    assert ds.encoding["source"] == str(path)
    np.testing.assert_array_equal(ds["latitude"].values, [1.0, 2.0, 3.0])
    assert list(ds["ID"].values.astype(str)) == ["300234064833010", "300234064833010", "42"]
    ds.close()


def test_zip_raises_value_error(tmp_path):
    # context_functions relies on a ValueError to fall back to unzipping ERA5 downloads
    nc = tmp_path / "inner.nc"
    _dataset().to_netcdf(nc, engine="scipy")
    path = tmp_path / "era5.nc"
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(nc, "data_stream-oper_stepType-instant.nc")
    with pytest.raises(ValueError, match="neither"):
        utils.open_netcdf(str(path))
