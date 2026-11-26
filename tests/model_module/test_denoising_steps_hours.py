"""Tests for the diagnostics-only ``denoising_steps_hours`` option of ``CoDiTModel.predict``."""

import datetime
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from codit.data_module import config_paths

_MODELS_DIR = os.path.join(config_paths.MODEL_PATH, "trained_models")
_CKPT = os.path.join(_MODELS_DIR, "codit_GRU_coastline", "nn_params.npz")
pytestmark = pytest.mark.skipif(
    not os.path.exists(_CKPT), reason="codit_GRU_coastline checkpoint not found"
)

L, S = 6, 3
START_POS = jnp.array([64.0, -15.5])
START_TIME = datetime.datetime(2021, 7, 6)


class ZeroContextDataset:
    def __init__(self, num_vars: int):
        self.num_vars = num_vars

    def get_context(self, times, positions, interpolator):
        return jnp.zeros((positions.shape[0], 16, 16, self.num_vars))


@pytest.fixture(scope="module")
def model_and_data():
    from codit.model_module.diffusion_model import CoDiTModel

    model = CoDiTModel.import_trained_model("codit_GRU_coastline", import_path=_MODELS_DIR)
    return model, ZeroContextDataset(int(np.max(model.var_idx)) + 1)


def _predict(model, data, **kw):
    return model.predict(
        START_POS, START_TIME, L, None, data, num_samples=S,
        key=jax.random.PRNGKey(0), return_denoising_steps=True, **kw,
    )


def test_traced_hours_shape_and_last_hour_matches_default(model_and_data):
    model, data = model_and_data
    pred, steps = _predict(model, data)
    pred_t, traced = _predict(model, data, denoising_steps_hours=[2, L])
    assert jnp.all(pred == pred_t)
    assert traced.shape == (S, 2) + steps.shape[1:]
    np.testing.assert_allclose(traced[:, 1], steps, rtol=0, atol=1e-5)


def test_traced_hour_reproduces_kept_position(model_and_data):
    model, data = model_and_data
    pred, traced = _predict(model, data, denoising_steps_hours=[3])
    np.testing.assert_allclose(traced[:, 0, -1, 3], pred[:, 2], rtol=0, atol=1e-3)


def test_out_of_range_hour_raises(model_and_data):
    model, data = model_and_data
    with pytest.raises(ValueError, match="out of range"):
        _predict(model, data, denoising_steps_hours=[0])
