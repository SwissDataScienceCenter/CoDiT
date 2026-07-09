import jax.numpy as jnp
import numpy as np
import pytest

from codit.eval_module.metrics import (
    get_energy_score,
    get_haversine_error,
    get_metrics,
    get_skill_score,
)


def _stationary_coords(n=4, t=10, lat=0.0, lon=0.0):
    """All N trajectories stay at the same lat/lon for all T timesteps."""
    return jnp.full((n, t, 2), jnp.array([lat, lon]))


def _moving_coords(n=4, t=10):
    """N trajectories that move 0.01 deg/timestep northward."""
    lats = jnp.linspace(0.0, 0.1, t)
    traj = jnp.stack([lats, jnp.zeros(t)], axis=-1)
    return jnp.tile(traj, (n, 1, 1))


class TestGetHaversineError:
    def test_stationary_identical_gives_zero_error(self):
        coords = _stationary_coords()
        error = get_haversine_error(coords, coords)
        assert jnp.all(
            jnp.abs(error) < 1.0
        ), "Identical stationary trajectories should give ~0 m haversine error"

    def test_returns_array(self):
        coords = _moving_coords()
        shifted = coords + 0.01
        error = get_haversine_error(coords, shifted)
        assert hasattr(error, "shape"), "Expected a JAX array"

    def test_error_increases_with_shift(self):
        coords = _moving_coords()
        small_shift = get_haversine_error(coords, coords + 0.001)
        large_shift = get_haversine_error(coords, coords + 1.0)
        assert float(jnp.mean(large_shift)) > float(jnp.mean(small_shift))


class TestGetEnergyScore:
    def test_stationary_identical_gives_zero_score(self):
        coords = _stationary_coords()
        score = get_energy_score(coords, coords)
        assert jnp.all(
            jnp.abs(score) < 1.0
        ), "Identical stationary trajectories should give ~0 energy score"

    def test_returns_array(self):
        coords = _moving_coords()
        score = get_energy_score(coords, coords)
        assert hasattr(score, "shape")


class TestGetSkillScore:
    def test_returns_non_nan_for_moving_trajectory(self):
        true_coords = _moving_coords()
        pred_coords = _moving_coords()
        score = get_skill_score(true_coords, pred_coords)
        assert not jnp.any(jnp.isnan(score)), "Skill score must not be NaN"

    def test_score_is_non_negative(self):
        true_coords = _moving_coords()
        pred_coords = _moving_coords() + 5.0
        score = get_skill_score(true_coords, pred_coords, threshold=1.0)
        assert jnp.all(score >= 0.0)


class TestGetMetrics:
    def test_returns_dict_with_required_keys(self):
        coords = _moving_coords()
        metrics = get_metrics(coords, coords)
        assert set(metrics.keys()) == {"mean_distance", "energy_score", "skill_score"}

    def test_all_values_are_arrays(self):
        coords = _moving_coords()
        metrics = get_metrics(coords, coords)
        for key, val in metrics.items():
            assert hasattr(val, "shape"), f"metrics['{key}'] should be an array"
