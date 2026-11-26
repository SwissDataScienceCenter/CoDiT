"""Encoding each context grid once in ``predict`` must not change the predictions.

``CoDiTModel.predict`` passes precomputed context encodings to the denoiser
(``history_context_enc`` / ``prediction_context_enc``) instead of the raw grids, so the
context encoder runs once per prediction hour rather than at every denoising step.
These tests check that this gives the same output as the original path, both for a
single network call and end to end through ``predict``.
"""

import datetime
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from codit.codit_module import diffusion_models
from codit.data_module import config_paths

_MODELS_DIR = os.path.join(config_paths.MODEL_PATH, "trained_models")
SHIPPED = ["codit_GRU_coastline", "codit_Transformer_coastline"]
L, S, G = 24, 4, 16
START_POS = jnp.array([64.0, -15.5])
START_TIME = datetime.datetime(2021, 7, 6)
# float32, different XLA kernels for different batch shapes
ATOL = 1e-4


def _have(name):
    return os.path.exists(os.path.join(_MODELS_DIR, name, "nn_params.npz"))


def _load(name):
    from codit.model_module.diffusion_model import CoDiTModel

    if not _have(name):
        pytest.skip(f"{name} checkpoint not found")
    return CoDiTModel.import_trained_model(name, import_path=_MODELS_DIR)


class PositionContextDataset:
    """Stands in for OceanDataset: smooth context fields that depend on position and time,
    so each sample's context differs and the per-hour slot updates matter."""

    def __init__(self, num_vars: int):
        self.num_vars = num_vars

    def get_context(self, times, positions, interpolator):
        g = jnp.linspace(-1.0, 1.0, G)
        phase = jnp.arange(self.num_vars)
        lat = positions[:, 0, None, None, None]
        lon = positions[:, 1, None, None, None]
        t = jnp.asarray(times, dtype=jnp.float32).reshape(-1)[0]
        return jnp.sin(
            40.0 * lat + g[None, :, None, None] + phase
        ) * jnp.cos(40.0 * lon + g[None, None, :, None] + 0.1 * t)


def _network_inputs(num_vars, key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    batch = {
        "history_target_vals": jax.random.normal(k1, (1, 2)),
        "history_context": jax.random.normal(k2, (1, G, G, num_vars)),
        "prediction_context": jax.random.normal(k3, (L, G, G, num_vars)),
    }
    y = jax.random.normal(k4, (L, 2))
    return y, jnp.zeros((1, 2)), batch, jnp.array([7])


def _with_encodings(net, params, batch):
    enc = {
        "history_target_vals": batch["history_target_vals"],
        "history_context_enc": diffusion_models.encode_context(
            net, params, batch["history_context"]
        ),
        "prediction_context_enc": diffusion_models.encode_context(
            net, params, batch["prediction_context"]
        ),
    }
    return enc


def _assert_same_call(net, params, num_vars):
    y, y_hist, batch, k = _network_inputs(num_vars, jax.random.PRNGKey(1))
    raw = net.apply(params, y, y_hist, batch, k)
    pre = net.apply(params, y, y_hist, _with_encodings(net, params, batch), k)
    np.testing.assert_allclose(pre, raw, atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", SHIPPED)
def test_network_call_shipped(name):
    model = _load(name)
    _assert_same_call(model.model, model.nn_params, len(model.var_idx))


@pytest.mark.parametrize("encoder", ["conv", "mlp", "gap"])
@pytest.mark.parametrize("cls", ["Autoregressive_Denoiser", "Autoregressive_TransformerDenoiser"])
def test_network_call_random_init(cls, encoder):
    kwargs = {"output_size": 2, "context": True, "num_layers": 1, "hidden_size": 32,
              "diffusion_emb_dim": 16, "K": 10, "context_encoder_type": encoder}
    if cls == "Autoregressive_TransformerDenoiser":
        kwargs.update({"num_heads": 2, "max_len": L + 1})
    net = getattr(diffusion_models, cls)(**kwargs)
    y, y_hist, batch, k = _network_inputs(5, jax.random.PRNGKey(0))
    params = net.init(jax.random.PRNGKey(2), y, y_hist, batch, k)
    _assert_same_call(net, params, 5)


def test_encode_context_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown context_encoder_type"):
        diffusion_models.make_context_encoder("nope", 32)


def _predict(model, data, encode_once):
    model._encode_context_once = encode_once
    return model.predict(
        START_POS, START_TIME, L, None, data, num_samples=S, key=jax.random.PRNGKey(0)
    )


@pytest.mark.parametrize("aligned", [False, True])
def test_predict_matches_original_path(aligned):
    model = _load("codit_GRU_coastline")
    model.aligned_context = aligned  # both context layouts (legacy one writes slot i)
    data = PositionContextDataset(int(np.max(model.var_idx)) + 1)
    try:
        new = _predict(model, data, encode_once=True)
        old = _predict(model, data, encode_once=False)
    finally:
        model._encode_context_once = True

    assert new.shape == (S, L, 2)
    assert np.all(np.isfinite(new))
    # positions in degrees; 1e-4 deg ~ 10 m after 24 autoregressive hours
    np.testing.assert_allclose(new, old, atol=1e-4, rtol=0)
    # the samples must actually differ, i.e. the comparison is not trivial
    assert float(jnp.std(new[:, -1, 0])) > 1e-3


def test_predict_transformer_matches_original_path():
    model = _load("codit_Transformer_coastline")
    data = PositionContextDataset(int(np.max(model.var_idx)) + 1)
    new = _predict(model, data, encode_once=True)
    old = _predict(model, data, encode_once=False)
    model._encode_context_once = True

    np.testing.assert_allclose(new, old, atol=1e-4, rtol=0)
