"""Tests for the diagnostics-only ``initial_y`` override of ``CoDiTModel.predict``."""

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

L, S = 24, 4
START_POS = jnp.array([64.0, -15.5])
START_TIME = datetime.datetime(2021, 7, 6)


class ZeroContextDataset:
    """Stands in for OceanDataset: zero context fields, no NetCDF needed."""

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
        key=jax.random.PRNGKey(0), **kw,
    )


def _default_initial_y(model):
    """Reproduce predict()'s own draw of the starting noise for PRNGKey(0)."""
    key, _ = jax.random.split(jax.random.PRNGKey(0))
    keys = jax.random.split(key, S + 1)
    keys = jax.random.split(keys[0], S + 1)
    return jax.vmap(model.noise_model, in_axes=(0, None))(keys[1:], L)[0]


def test_explicit_default_start_is_bit_identical(model_and_data):
    model, data = model_and_data
    default = _predict(model, data)
    explicit = _predict(model, data, initial_y=_default_initial_y(model))
    assert jnp.all(default == explicit)


def test_different_start_changes_prediction(model_and_data):
    model, data = model_and_data
    y0 = _default_initial_y(model)
    assert not jnp.allclose(_predict(model, data), _predict(model, data, initial_y=0.5 * y0))


def test_wrong_shape_raises(model_and_data):
    model, data = model_and_data
    with pytest.raises(ValueError, match="initial_y must have shape"):
        _predict(model, data, initial_y=jnp.zeros((S, 5, 2)))
