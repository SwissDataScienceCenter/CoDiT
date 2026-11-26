"""Tests for the ``noise_schedule`` parameter of ``CoDiTModel``."""

import os

import numpy as np
import pytest

from codit.data_module import config_paths
from codit.model_module.diffusion_model import CoDiTModel

PARAMS = {
    "prediction_length": 24,
    "dt": 1.0 / 24,
    "vars": ["hycom_current", "copernicus_current", "era5_wind", "bathymetry"],
    "aligned_context": True,
}


def _alpha_bar_end(model: CoDiTModel) -> float:
    return float(model.schedule_dict["alpha_bar"][-1])


def test_default_is_paper_linear_schedule():
    model = CoDiTModel("t", dict(PARAMS))
    assert model.params["noise_schedule"] == "linear"
    assert model.params["beta_max"] == 0.02
    np.testing.assert_allclose(_alpha_bar_end(model), 0.364, atol=1e-3)


def test_cosine_schedule_reaches_noise():
    model = CoDiTModel("t", PARAMS | {"noise_schedule": "cosine"})
    assert model.params["beta_max"] == 0.2
    assert _alpha_bar_end(model) < 1e-4
    ab = np.asarray(model.schedule_dict["alpha_bar"])
    assert np.all(np.diff(ab) < 0)


def test_configs_with_removed_sampler_params_still_load():
    """Models trained while the Heun sampler existed store these keys."""
    old = {"sampler": "heun", "num_sampling_steps": 18, "rho": 7.0}
    model = CoDiTModel("t", PARAMS | {"noise_schedule": "cosine"} | old)
    assert _alpha_bar_end(model) < 1e-4


def test_explicit_beta_max_is_kept():
    model = CoDiTModel("t", PARAMS | {"noise_schedule": "cosine", "beta_max": 0.1})
    assert model.params["beta_max"] == 0.1
    assert model.schedule_dict["beta"][-1] == pytest.approx(0.1)


def test_unknown_schedule_raises():
    with pytest.raises(ValueError, match="noise_schedule"):
        CoDiTModel("t", PARAMS | {"noise_schedule": "quadratic"})


@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            config_paths.MODEL_PATH, "trained_models", "codit_GRU_open_ocean", "model_config.yml"
        )
    ),
    reason="codit_GRU_open_ocean checkpoint not found",
)
def test_shipped_checkpoint_keeps_linear_schedule():
    model = CoDiTModel.import_trained_model("codit_GRU_open_ocean")
    assert model.params["noise_schedule"] == "linear"
    np.testing.assert_allclose(_alpha_bar_end(model), 0.364, atol=1e-3)


def test_export_import_raw_weights(tmp_path):
    model = CoDiTModel("ema_model", PARAMS | {"noise_schedule": "cosine"})
    model.nn_params = {"w": np.ones(3)}
    model.nn_params_raw = {"w": np.zeros(3)}
    model.std = np.array([1.0, 2.0])
    model.export_trained_model(export_path=str(tmp_path))
    ema = CoDiTModel.import_trained_model("ema_model", import_path=str(tmp_path))
    raw = CoDiTModel.import_trained_model("ema_model", import_path=str(tmp_path), weights="raw")
    np.testing.assert_array_equal(ema.nn_params["w"], np.ones(3))
    np.testing.assert_array_equal(raw.nn_params["w"], np.zeros(3))
    assert raw.params["noise_schedule"] == "cosine"


def test_import_raw_weights_without_ema_raises(tmp_path):
    model = CoDiTModel("plain", dict(PARAMS))
    model.nn_params = {"w": np.ones(3)}
    model.std = np.array([1.0, 2.0])
    model.export_trained_model(export_path=str(tmp_path))
    with pytest.raises(ValueError, match="raw weights"):
        CoDiTModel.import_trained_model("plain", import_path=str(tmp_path), weights="raw")
