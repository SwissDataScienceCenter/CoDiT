"""Tests for CoDiTModel correctness."""

import jax.numpy as jnp
import flax.nnx as nnx


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
