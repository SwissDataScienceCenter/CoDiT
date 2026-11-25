"""Tests for the EDM Heun sampler (``heun_denoise``) used with DDPM-trained CoDiT weights."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from codit.codit_module.brownian_noise_models import WalkNoiseModel
from codit.codit_module.scheduler import linear_schedule
from codit.data_module import config_paths
from codit.model_module.diffusion_model import (
    denoise,
    edm_sigmas,
    heun_denoise,
    heun_step_indices,
)

_MODELS_DIR = os.path.join(config_paths.MODEL_PATH, "trained_models")
_CODIT_PATH = os.path.join(_MODELS_DIR, "codit_GRU_open_ocean", "nn_params.npz")
requires_codit = pytest.mark.skipif(
    not os.path.exists(_CODIT_PATH),
    reason="codit_GRU_open_ocean checkpoint not found",
)

T = 24  # prediction length
SCHEDULE = linear_schedule(K=100, beta_min=1e-4, beta_max=0.02)  # shipped schedule
NOISE = WalkNoiseModel(
    noise_type="brownian_motion", num_dims=2, dt=1.0 / T, sigma=1.0, T=1.0
)
BATCH = {"history_target_vals": jnp.zeros((1, 2))}


class GaussianOracle:
    """Exact noise predictor for Gaussian data ``y0 = c * L z``, ``z`` white.

    Mimics ``model.apply(params, y_noised[1:], y_history, batch, k)``. With
    ``y_k = sqrt(ab) y0 + sqrt(1 - ab) L n`` the posterior mean of the white
    increments is ``E[n | y_k] = sqrt(1 - ab) / (ab c^2 + 1 - ab) * L^{-1} y_k``.
    """

    def __init__(self, c: float):
        self.c = c

    def apply(self, params, y_in, y_history, batch, k):
        ab = SCHEDULE["alpha_bar"][k[0]]
        path = jnp.concatenate([jnp.zeros((1, 2)), y_in], axis=0)
        whitened = jnp.diff(path, axis=0) / (NOISE.sigma * jnp.sqrt(NOISE.dt))
        return jnp.sqrt(1 - ab) / (ab * self.c**2 + 1 - ab) * whitened


def _initial_noise(key, n):
    keys = jax.random.split(key, n)
    return jax.vmap(NOISE, in_axes=(0, None))(keys, T)[0]  # (n, T+1, 2)


class TestStepIndices:
    @pytest.mark.parametrize("n", [2, 5, 18, 50, 100, 500])
    def test_decreasing_from_last_to_first_step(self, n):
        idx = heun_step_indices(SCHEDULE, n)
        assert idx[0] == SCHEDULE["K"] - 1
        assert idx[-1] == 0
        assert np.all(np.diff(idx) < 0)
        assert len(idx) <= min(n, SCHEDULE["K"])

    def test_more_steps_visit_more_levels(self):
        # snapping to the trained grid merges some levels, so fewer than requested
        assert len(heun_step_indices(SCHEDULE, 100)) > len(
            heun_step_indices(SCHEDULE, 18)
        )

    def test_edm_sigmas_match_alpha_bar(self):
        s = edm_sigmas(SCHEDULE["alpha_bar"])
        np.testing.assert_allclose(
            1.0 / (1.0 + s**2), SCHEDULE["alpha_bar"], rtol=1e-5
        )


class TestHeunGaussianOracle:
    """With the exact posterior mean, the probability-flow ODE has a closed form.

    In EDM coordinates, ``dx/dsigma = sigma / (c^2 + sigma^2) * x``, so
    ``x(0) = x(sigma_max) * c / sqrt(c^2 + sigma_max^2)`` along every sample.
    """

    @staticmethod
    def _relative_error(c, num_steps):
        y_init = _initial_noise(jax.random.PRNGKey(0), 8)
        idx = heun_step_indices(SCHEDULE, num_steps)
        out, steps = jax.vmap(
            lambda y: heun_denoise(
                None, y, BATCH, GaussianOracle(c), NOISE,
                jax.random.PRNGKey(1), SCHEDULE, idx,
            )
        )(y_init)
        assert steps.shape == (8, len(idx), T + 1, 2)
        np.testing.assert_allclose(steps[:, -1], out)
        ab_max = SCHEDULE["alpha_bar"][idx[0]]
        s_max = edm_sigmas(SCHEDULE["alpha_bar"])[idx[0]]
        expected = y_init / jnp.sqrt(ab_max) * c / jnp.sqrt(c**2 + s_max**2)
        return float(jnp.max(jnp.abs(out - expected)) / jnp.max(jnp.abs(expected)))

    @pytest.mark.parametrize("c", [0.3, 1.0, 2.0])
    def test_close_to_closed_form_ode_solution(self, c):
        assert self._relative_error(c, 18) < 0.01

    @pytest.mark.parametrize("c", [0.3, 1.0])
    def test_second_order_convergence(self, c):
        """Halving the step size must cut the error by about 4x (2nd order)."""
        ratio = self._relative_error(c, 10) / self._relative_error(c, 20)
        assert ratio > 3.0

    def test_first_point_stays_at_origin(self):
        y_init = _initial_noise(jax.random.PRNGKey(0), 1)[0]
        idx = heun_step_indices(SCHEDULE, 18)
        out, _ = heun_denoise(
            None, y_init, BATCH, GaussianOracle(0.5), NOISE,
            jax.random.PRNGKey(1), SCHEDULE, idx,
        )
        assert jnp.all(out[0] == 0)

    def test_same_spread_as_ddpm(self):
        """Heun (18 levels) and ancestral DDPM (100 steps) sample the same distribution.

        ``c = 1`` matches the sampler's starting distribution to the forward
        process at the last trained step, so both samplers should return paths with
        unit whitened-increment variance.
        """
        n = 2000
        key_init, key_w = jax.random.split(jax.random.PRNGKey(3))
        y_init = _initial_noise(key_init, n)
        model = GaussianOracle(1.0)
        idx = heun_step_indices(SCHEDULE, 18)

        heun = jax.vmap(
            lambda y: heun_denoise(
                None, y, BATCH, model, NOISE, jax.random.PRNGKey(1), SCHEDULE, idx
            )[0]
        )(y_init)

        W = jax.vmap(
            lambda key: jax.vmap(NOISE, in_axes=(0, None))(
                jax.random.split(key, SCHEDULE["K"]), T
            )[0]
        )(jax.random.split(key_w, n))
        ddpm = jax.vmap(
            lambda y, w: denoise(
                None, y, BATCH, model, w, NOISE, jax.random.PRNGKey(1), SCHEDULE
            )[0]
        )(y_init, W)

        def endpoint_std(paths):
            return float(jnp.std(paths[:, -1]))

        assert endpoint_std(heun) == pytest.approx(1.0, rel=0.05)
        assert endpoint_std(ddpm) == pytest.approx(1.0, rel=0.05)


class MeanShiftOracle(GaussianOracle):
    """Exact noise predictor for ``y0 = mu * t/T + c * L z`` (mean drift ``mu``)."""

    def __init__(self, c: float, mu: float, schedule: dict):
        super().__init__(c)
        self.mu = mu
        self.schedule = schedule

    def apply(self, params, y_in, y_history, batch, k):
        ab = self.schedule["alpha_bar"][k[0]]
        path = jnp.concatenate([jnp.zeros((1, 2)), y_in], axis=0)
        scale = NOISE.sigma * jnp.sqrt(NOISE.dt)
        residual = jnp.diff(path, axis=0) / scale - jnp.sqrt(ab) * self.mu / T / scale
        return jnp.sqrt(1 - ab) / (ab * self.c**2 + 1 - ab) * residual


def _endpoint_stats_heun(schedule, c, mu, n=2000):
    y_init = _initial_noise(jax.random.PRNGKey(0), n)
    idx = heun_step_indices(schedule, 18)
    model = MeanShiftOracle(c, mu, schedule)
    out = jax.vmap(
        lambda y: heun_denoise(
            None, y, BATCH, model, NOISE, jax.random.PRNGKey(1), schedule, idx
        )[0]
    )(y_init)
    return float(out[:, -1, 0].mean()), float(out[:, -1, 0].std())


class TestHeunScheduleDependence:
    """Heun is unbiased only if the schedule ends at (near) pure noise.

    An ODE sampler transports its starting distribution exactly, so starting
    from pure noise when the forward process ends at alpha_bar = 0.36 (the shipped
    schedule) gives biased samples even with a perfect denoiser.
    """

    def test_unbiased_when_schedule_reaches_noise(self):
        schedule = linear_schedule(K=100, beta_min=1e-4, beta_max=0.2)
        mean, std = _endpoint_stats_heun(schedule, c=0.3, mu=2.0)
        assert mean == pytest.approx(2.0, abs=0.05)
        assert std == pytest.approx(0.3, rel=0.15)

    def test_biased_with_shipped_schedule(self):
        """Documents the known bias: drift too short, spread too wide."""
        mean, std = _endpoint_stats_heun(SCHEDULE, c=0.3, mu=2.0)
        assert mean < 1.8
        assert std > 0.34


@requires_codit
class TestHeunWithCheckpoint:
    @pytest.fixture(scope="class")
    def model(self):
        from codit.model_module.diffusion_model import CoDiTModel

        return CoDiTModel.import_trained_model(
            "codit_GRU_open_ocean", import_path=_MODELS_DIR
        )

    def test_default_sampler_is_ddpm(self, model):
        assert model.sampler == "ddpm"

    def test_heun_warns_for_shipped_schedule(self, model):
        with pytest.warns(UserWarning, match="alpha_bar"):
            model.set_sampler("heun")
        model.set_sampler("ddpm")

    @pytest.mark.filterwarnings("ignore:The diffusion schedule ends")
    def test_set_sampler_validates_and_invalidates_cache(self, model):
        model._denoise_func = object()
        model.set_sampler("heun", 10)
        assert model.sampler == "heun"
        assert model._denoise_func is None
        assert model._heun_indices[0] == model.schedule_dict["K"] - 1
        with pytest.raises(ValueError):
            model.set_sampler("euler")
        model.set_sampler("ddpm")

    def test_heun_runs_on_trained_network(self, model):
        C = len(model.var_idx)
        L = model.params["prediction_length"]
        batch = {
            "history_target_vals": jnp.zeros((1, 2)),
            "history_context": jnp.zeros((1, 16, 16, C)),
            "prediction_context": jnp.zeros((L, 16, 16, C)),
        }
        y_init = model.noise_model(jax.random.PRNGKey(0), L)[0]
        idx = heun_step_indices(model.schedule_dict, 18)
        out, steps = jax.jit(
            lambda y: heun_denoise(
                model.nn_params, y, batch, model.model, model.noise_model,
                jax.random.PRNGKey(1), model.schedule_dict, idx,
            )
        )(y_init)
        assert out.shape == (L + 1, 2)
        assert steps.shape == (len(idx), L + 1, 2)
        assert jnp.all(jnp.isfinite(out))
