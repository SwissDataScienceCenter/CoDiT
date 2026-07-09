"""Tests for CoDiTModel correctness."""

import os

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest

from codit.data_module import config_paths

# ── Skip guards for checkpoint-dependent tests ────────────────────────────────
_MODELS_DIR = os.path.join(config_paths._REPO_ROOT, "data", "trained_models")
_CODIT_CKPT = os.path.join(_MODELS_DIR, "codit_GRU_open_ocean", "nn_params.npz")
_CONFIG_FILE = os.path.join(
    config_paths.CONFIG_PATH, "open_ocean", "test", "open_ocean_patch_1_year_2021.yml"
)
_requires_codit = pytest.mark.skipif(
    not os.path.exists(_CODIT_CKPT) or not os.path.exists(_CONFIG_FILE),
    reason="codit_GRU_open_ocean checkpoint or open_ocean test config not found",
)


def test_codit_model_does_not_override_fit():
    """
    CoDiTModel.fit() must NOT be defined as an override.
    If it is, it bypasses BaseTrajectoryModel.fit() which sets trained_successfully.
    The fix: delete CoDiTModel.fit() so the base class method is inherited.
    """
    from codit.model_module.diffusion_model import CoDiTModel

    # CoDiTModel.__dict__ contains only methods defined directly on CoDiTModel.
    # If fit() is not in __dict__, it is inherited from BaseTrajectoryModel (correct).
    assert "fit" not in CoDiTModel.__dict__, (
        "CoDiTModel must not override fit(). "
        "Remove the fit() method so BaseTrajectoryModel.fit() is inherited. "
        "The base class calls self._fit() and then sets self.trained_successfully = True."
    )


def test_linear_gauss_module_mean_velocity_shape():
    """get_mean_velocity must return a (2,) velocity vector for any context grid size."""
    from codit.model_module.linear_gauss_model import LinearGaussModule

    module = LinearGaussModule(rngs=nnx.Rngs(0))
    # Context: (grid_H, grid_W, 8_features) — uses centre cell, any odd grid size works
    context = jnp.zeros((5, 5, 8))
    mean_vel = module.get_mean_velocity(context)
    assert mean_vel.shape == (2,)


def test_linear_gauss_module_sample_shape(jax_key):
    """__call__ must return a sampled velocity with shape (2,)."""
    from codit.model_module.linear_gauss_model import LinearGaussModule

    module = LinearGaussModule(rngs=nnx.Rngs(0))
    context = jnp.zeros((5, 5, 8))
    sampled = module(context, key=jax_key)
    assert sampled.shape == (2,)


def test_linear_gauss_module_sample_differs_from_mean(jax_key):
    """Sampled velocity must differ from the deterministic mean (noise was added)."""
    from codit.model_module.linear_gauss_model import LinearGaussModule

    module = LinearGaussModule(rngs=nnx.Rngs(0))
    context = jnp.zeros((5, 5, 8))
    mean_vel = module.get_mean_velocity(context)
    sampled = module(context, key=jax_key)
    # log_sigma initialised to 0 → sigma = exp(0) = 1, so noise is large
    assert not jnp.allclose(mean_vel, sampled, atol=1e-6)


def test_dynamic_index_matches_static():
    """lax.dynamic_index_in_dim with a traced int must match Python-int indexing."""
    arr = jnp.arange(72, dtype=jnp.float32) * 1.7 + 0.3
    for i in range(72):
        static = arr[i]
        dynamic = jax.lax.dynamic_index_in_dim(arr, jnp.int32(i), axis=0, keepdims=False)
        assert jnp.allclose(static, dynamic, atol=0.0), (
            f"Mismatch at i={i}: static={static}, dynamic={dynamic}"
        )


@_requires_codit
def test_predict_deterministic_same_key():
    """predict() with the same PRNG key must return bit-for-bit identical output."""
    from codit.model_module.diffusion_model import CoDiTModel
    from codit.data_module import config_funcs
    from codit.data_module.dataset import OceanDataset

    model = CoDiTModel.import_trained_model("codit_GRU_open_ocean")
    config = config_funcs.load_config(
        "open_ocean_patch_1_year_2021",
        os.path.join(config_paths.CONFIG_PATH, "open_ocean", "test"),
    )
    dataset = OceanDataset(config, 72, 16, 10000, "linear")
    start_position, start_time, _times, interpolator, _ = dataset[0]
    key = jax.random.PRNGKey(0)

    model._denoise_func = None  # ensure both calls start from a cold cache
    result_a = model.predict(
        start_position, start_time, 72, interpolator, dataset, num_samples=2, key=key
    )
    result_b = model.predict(
        start_position, start_time, 72, interpolator, dataset, num_samples=2, key=key
    )
    assert jnp.allclose(result_a, result_b, atol=0.0), (
        "predict() returned different results for the same PRNG key — "
        "the optimization must not change numerical output"
    )
