import jax
import jax.numpy as jnp
import pytest

from codit.codit_module.brownian_noise_models import WalkNoiseModel
from codit.codit_module.noising_functions import diffusion_jump, forward_diffusion


class TestForwardDiffusion:
    def test_y_t_has_same_shape_as_input(
        self, jax_key, shared_schedule, walk_noise_model
    ):
        T, D = 11, 2
        y = jnp.ones((T, D))
        k = jnp.array(5)
        y_t, _ = forward_diffusion(y, k, jax_key, shared_schedule, walk_noise_model)
        assert y_t.shape == (T, D)

    def test_noise_has_shape_T_minus_1_D(
        self, jax_key, shared_schedule, walk_noise_model
    ):
        """For white_noise model, dW shape is (T-1, D)."""
        T, D = 11, 2
        y = jnp.ones((T, D))
        k = jnp.array(5)
        _, noise_t = forward_diffusion(y, k, jax_key, shared_schedule, walk_noise_model)
        assert noise_t.shape == (T - 1, D)

    def test_high_noise_step_perturbs_trajectory(
        self, jax_key, shared_schedule, walk_noise_model
    ):
        """At high k (near K), alpha_bar[k] is small, so noise dominates."""
        T, D = 11, 2
        y = jnp.zeros((T, D))
        k_high = jnp.array(shared_schedule["K"] - 1)
        y_t, _ = forward_diffusion(
            y, k_high, jax_key, shared_schedule, walk_noise_model
        )
        # y_t should NOT be all zeros at high noise step (noise dominates)
        assert not jnp.allclose(y_t, y, atol=0.01)

    def test_low_noise_step_preserves_trajectory(
        self, jax_key, shared_schedule, walk_noise_model
    ):
        """At k=0, alpha_bar[0] ≈ 1, so y_t ≈ y (low noise)."""
        T, D = 11, 2
        y = jnp.ones((T, D)) * 5.0
        k_low = jnp.array(0)
        y_t, _ = forward_diffusion(y, k_low, jax_key, shared_schedule, walk_noise_model)
        # At k=0 with alpha_bar[0] ≈ 1 and no noise: y_t ≈ y
        # Allow some tolerance since white noise is added
        assert float(jnp.mean(jnp.abs(y_t - y))) < 3.0

    def test_same_key_gives_identical_output(
        self, shared_schedule, walk_noise_model
    ):
        """Determinism: same PRNG key must yield bit-identical noising."""
        T, D = 11, 2
        y = jnp.ones((T, D))
        k = jnp.array(5)
        key = jax.random.PRNGKey(123)
        y_a, n_a = forward_diffusion(y, k, key, shared_schedule, walk_noise_model)
        y_b, n_b = forward_diffusion(y, k, key, shared_schedule, walk_noise_model)
        assert jnp.array_equal(y_a, y_b)
        assert jnp.array_equal(n_a, n_b)

    def test_different_keys_give_different_output(
        self, shared_schedule, walk_noise_model
    ):
        """Different PRNG keys must produce different noise realisations."""
        T, D = 11, 2
        y = jnp.zeros((T, D))
        k = jnp.array(5)
        y_a, _ = forward_diffusion(
            y, k, jax.random.PRNGKey(0), shared_schedule, walk_noise_model
        )
        y_b, _ = forward_diffusion(
            y, k, jax.random.PRNGKey(1), shared_schedule, walk_noise_model
        )
        assert not jnp.allclose(y_a, y_b, atol=1e-6)

    def test_variance_grows_with_k(self, shared_schedule, walk_noise_model):
        """For fixed y=0, the noised sample's variance must be non-decreasing in k."""
        T, D = 32, 2
        y = jnp.zeros((T, D))
        n_draws = 64
        keys = jax.random.split(jax.random.PRNGKey(0), n_draws)
        var_by_k = []
        for k_val in (0, 20, 60, shared_schedule["K"] - 1):
            samples = jax.vmap(
                lambda kkey: forward_diffusion(
                    y, jnp.array(k_val), kkey, shared_schedule, walk_noise_model
                )[0]
            )(keys)
            var_by_k.append(float(samples.var()))
        assert var_by_k[0] <= var_by_k[1] <= var_by_k[2] <= var_by_k[3]

    def test_brownian_motion_noise_model_works(self, shared_schedule):
        """forward_diffusion must also work with the 'brownian_motion' noise model."""
        brownian = WalkNoiseModel(
            noise_type="brownian_motion",
            num_dims=2,
            dt=1.0,
            sigma=1.0,
            T=10.0,
        )
        T, D = 11, 2
        y = jnp.ones((T, D))
        k = jnp.array(5)
        y_t, dW = forward_diffusion(
            y, k, jax.random.PRNGKey(0), shared_schedule, brownian
        )
        assert y_t.shape == (T, D)
        assert dW.shape == (T - 1, D)


class TestDiffusionJump:
    def test_returns_expected_shapes(self, jax_key, shared_schedule, walk_noise_model):
        T, D = 11, 2
        y = jnp.ones((T, D))
        y_t, dW = diffusion_jump(
            jax_key, y, jnp.array(3), shared_schedule["alpha_bar"], walk_noise_model
        )
        assert y_t.shape == (T, D)
        assert dW.shape == (T - 1, D)

    @pytest.mark.parametrize("k_val", [0, 25, 75, 99])
    def test_forward_diffusion_matches_direct_jump(
        self, shared_schedule, walk_noise_model, k_val
    ):
        """forward_diffusion is a thin wrapper around diffusion_jump — outputs must match."""
        T, D = 8, 2
        y = jnp.ones((T, D)) * 2.0
        k = jnp.array(k_val)
        key = jax.random.PRNGKey(7)
        y_fd, n_fd = forward_diffusion(y, k, key, shared_schedule, walk_noise_model)
        y_dj, n_dj = diffusion_jump(
            key, y, k, shared_schedule["alpha_bar"], walk_noise_model
        )
        assert jnp.array_equal(y_fd, y_dj)
        assert jnp.array_equal(n_fd, n_dj)
