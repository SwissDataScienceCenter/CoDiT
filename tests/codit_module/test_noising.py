import jax.numpy as jnp

from codit.codit_module.noising_functions import forward_diffusion


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
