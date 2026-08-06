"""Forward-pass shape tests for the four main model architectures.

These are checkpoint-free: models are initialised with tiny hyperparameters and
called on synthetic inputs to assert output shapes and simple invariants (e.g.
batching preserves shape, different PRNG keys change outputs).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


# ── CoDiT diffusion denoisers ─────────────────────────────────────────────────


def _codit_inputs(T_pred=4, hist_len=1, H=4, W=4, C=3):
    """Build a minimal input bundle for the CoDiT denoisers."""
    y = jnp.zeros((T_pred, 2))
    y_history = jnp.zeros((hist_len, 2))
    x = {
        "history_target_vals": jnp.zeros((hist_len, 2)),
        "history_context": jnp.zeros((hist_len, H, W, C)),
        "prediction_context": jnp.zeros((T_pred, H, W, C)),
    }
    return y, y_history, x


class TestCoDiTAutoregressiveDenoiser:
    def test_output_shape_matches_prediction_length(self):
        from codit.codit_module.diffusion_models import Autoregressive_Denoiser

        T_pred = 4
        model = Autoregressive_Denoiser(
            context=True,
            num_layers=1,
            hidden_size=8,
            diffusion_emb_dim=4,
            K=10,
            context_encoder_type="mlp",
        )
        y, y_history, x = _codit_inputs(T_pred=T_pred)
        params = model.init(
            jax.random.PRNGKey(0),
            y=y,
            y_history=y_history,
            x=x,
            k=jnp.ones((1,), dtype=jnp.int32),
            train=True,
        )
        out = model.apply(params, y, y_history, x, jnp.ones((1,), dtype=jnp.int32))
        assert out.shape == (T_pred, 2)

    def test_output_shape_scales_with_prediction_length(self):
        """Doubling prediction length must double the output length dimension."""
        from codit.codit_module.diffusion_models import Autoregressive_Denoiser

        model = Autoregressive_Denoiser(
            context=True,
            num_layers=1,
            hidden_size=8,
            diffusion_emb_dim=4,
            K=10,
            context_encoder_type="mlp",
        )
        y, y_history, x = _codit_inputs(T_pred=6)
        params = model.init(
            jax.random.PRNGKey(0),
            y=y,
            y_history=y_history,
            x=x,
            k=jnp.ones((1,), dtype=jnp.int32),
            train=True,
        )
        # Reuse init; apply with a longer horizon by re-initialising (Flax nn is stateless).
        for T_pred in (3, 6, 9):
            y, y_history, x = _codit_inputs(T_pred=T_pred)
            params2 = model.init(
                jax.random.PRNGKey(0),
                y=y,
                y_history=y_history,
                x=x,
                k=jnp.ones((1,), dtype=jnp.int32),
                train=True,
            )
            out = model.apply(
                params2, y, y_history, x, jnp.ones((1,), dtype=jnp.int32)
            )
            assert out.shape == (T_pred, 2)


class TestCoDiTTransformerDenoiser:
    def test_output_shape_matches_prediction_length(self):
        from codit.codit_module.diffusion_models import (
            Autoregressive_TransformerDenoiser,
        )

        T_pred = 4
        model = Autoregressive_TransformerDenoiser(
            context=True,
            num_layers=1,
            hidden_size=8,
            diffusion_emb_dim=4,
            K=10,
            max_len=32,
            num_heads=2,
            context_encoder_type="mlp",
        )
        y, y_history, x = _codit_inputs(T_pred=T_pred)
        params = model.init(
            jax.random.PRNGKey(0),
            y=y,
            y_history=y_history,
            x=x,
            k=jnp.ones((1,), dtype=jnp.int32),
            train=True,
        )
        out = model.apply(params, y, y_history, x, jnp.ones((1,), dtype=jnp.int32))
        assert out.shape == (T_pred, 2)


# ── ConvNet baseline ──────────────────────────────────────────────────────────


class TestConvNetForward:
    def test_returns_batch_of_2d_velocities(self):
        """ConvNet expects (B, 16, 16, C) — Linear1 is fixed at 512 in-features."""
        from codit.model_module.conv_net import ConvNet

        model = ConvNet(in_features=4, rngs=nnx.Rngs(0))
        x = jnp.zeros((3, 16, 16, 4))
        out = model(x)
        assert out.shape == (3, 2)

    def test_output_changes_with_input(self):
        from codit.model_module.conv_net import ConvNet

        model = ConvNet(in_features=4, rngs=nnx.Rngs(0))
        x1 = jnp.zeros((2, 16, 16, 4))
        x2 = jnp.ones((2, 16, 16, 4))
        out1 = model(x1)
        out2 = model(x2)
        assert not jnp.allclose(out1, out2, atol=1e-6)

    def test_mc_dropout_produces_distinct_samples(self):
        """Dropout is active at inference; calling twice must yield different outputs."""
        from codit.model_module.conv_net import ConvNet

        model = ConvNet(in_features=4, rngs=nnx.Rngs(0))
        x = jnp.ones((1, 16, 16, 4))
        out_a = model(x)
        out_b = model(x)
        # With a live Dropout layer the two forward passes must draw distinct masks
        assert not jnp.allclose(out_a, out_b, atol=1e-6)


# ── LinearGauss baseline ──────────────────────────────────────────────────────


class TestLinearGaussModuleForward:
    def test_get_mean_velocity_returns_2d(self):
        from codit.model_module.linear_gauss_model import LinearGaussModule

        module = LinearGaussModule(rngs=nnx.Rngs(0))
        context = jnp.zeros((5, 5, 8))
        assert module.get_mean_velocity(context).shape == (2,)

    def test_call_returns_2d_sample(self):
        from codit.model_module.linear_gauss_model import LinearGaussModule

        module = LinearGaussModule(rngs=nnx.Rngs(0))
        context = jnp.zeros((5, 5, 8))
        out = module(context, key=jax.random.PRNGKey(0))
        assert out.shape == (2,)

    def test_batched_context_returns_batched_velocity(self):
        """A leading batch dim on context must yield a leading batch dim on mean velocity."""
        from codit.model_module.linear_gauss_model import LinearGaussModule

        module = LinearGaussModule(rngs=nnx.Rngs(0))
        context = jnp.zeros((7, 5, 5, 8))
        mean = module.get_mean_velocity(context)
        assert mean.shape == (7, 2)

    @pytest.mark.parametrize("grid_size", [3, 4, 5, 6, 7])
    def test_get_mean_velocity_shape_across_grid_sizes(self, grid_size):
        """Works for both even and odd grids (uses centre cell or 2x2 average)."""
        from codit.model_module.linear_gauss_model import LinearGaussModule

        module = LinearGaussModule(rngs=nnx.Rngs(0))
        context = jnp.zeros((grid_size, grid_size, 8))
        assert module.get_mean_velocity(context).shape == (2,)


# ── GRU (direct prediction) model ─────────────────────────────────────────────


class TestAutoregressiveGRUForward:
    def _inputs(self, T_hist=1, T_pred=4, H=4, W=4, C=3):
        return {
            "history_target_vals": jnp.zeros((T_hist, 2)),
            "history_context": jnp.zeros((T_hist, H, W, C)),
            "prediction_context": jnp.zeros((T_pred, H, W, C)),
            "history_time": jnp.zeros((T_hist, 1), dtype=jnp.int32),
            "prediction_time": jnp.arange(1, T_pred + 1, dtype=jnp.int32).reshape(
                -1, 1
            ),
        }

    def test_gru_backbone_output_shape(self):
        from codit.model_module.gru_model import Autoregressive

        T_hist, T_pred = 1, 4
        model = Autoregressive(
            context=True,
            time_emb_dim=4,
            num_layers=1,
            hidden_size=8,
            output_size=2,
            T=10,
            context_encoder_type="mlp",
            backbone="GRU",
        )
        y_history = jnp.zeros((T_hist, 2))
        x = self._inputs(T_hist=T_hist, T_pred=T_pred)
        params = model.init(jax.random.PRNGKey(0), y_history, x, train=True)
        out = model.apply(params, y_history, x, train=False)
        assert out.shape == (T_pred, 2)

    def test_transformer_backbone_output_shape(self):
        from codit.model_module.gru_model import Autoregressive

        T_hist, T_pred = 1, 4
        model = Autoregressive(
            context=True,
            time_emb_dim=4,
            num_layers=1,
            hidden_size=8,
            output_size=2,
            T=10,
            max_len=16,
            num_heads=2,
            context_encoder_type="mlp",
            backbone="Transformer",
        )
        y_history = jnp.zeros((T_hist, 2))
        x = self._inputs(T_hist=T_hist, T_pred=T_pred)
        params = model.init(jax.random.PRNGKey(0), y_history, x, train=True)
        out = model.apply(params, y_history, x, train=False)
        assert out.shape == (T_pred, 2)
