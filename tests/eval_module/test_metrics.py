"""Tests for codit.eval_module.metrics.

API contract
------------
* get_haversine_error(true_coords, pred_coords)
    true_coords : (N, T, 2)  — the T *future* timesteps only
    pred_coords : (N, S, T, 2) or (N, T, 2) — S samples over T future steps

* get_energy_score(true_coords, pred_coords)
    Same shape convention as get_haversine_error.

* get_skill_score(true_coords, pred_coords)
    true_coords : (N, T+1, 2)  — includes the starting position at t=0
    pred_coords : (N, S, T, 2) or (N, T, 2) — S samples over T future steps

* get_metrics(true_coords, pred_coords)
    true_coords : (N, T+1, 2)  — includes the starting position at t=0
    pred_coords : (N, S, T, 2) or (N, T, 2) — S samples over T future steps
    All returned arrays have shape (N, T).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from codit.eval_module.metrics import (
    get_energy_score,
    get_haversine_error,
    get_metrics,
    get_skill_score,
)

# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _stationary_coords(n=4, t=10, lat=0.0, lon=0.0):
    """(N, T, 2) — all trajectories stay at the same lat/lon."""
    return jnp.full((n, t, 2), jnp.array([lat, lon]))


def _moving_coords(n=4, t=10):
    """(N, T, 2) — trajectories move 0.01 deg/step northward."""
    lats = jnp.linspace(0.0, 0.1, t)
    traj = jnp.stack([lats, jnp.zeros(t)], axis=-1)
    return jnp.tile(traj, (n, 1, 1))


def _moving_coords_with_start(n=4, t=10):
    """(N, T+1, 2) — includes the starting position at t=0.

    For use as *true_coords* in get_skill_score / get_metrics.
    """
    lats = jnp.linspace(0.0, 0.1, t + 1)
    traj = jnp.stack([lats, jnp.zeros(t + 1)], axis=-1)
    return jnp.tile(traj, (n, 1, 1))


def _stationary_coords_with_start(n=4, t=10, lat=0.0, lon=0.0):
    """(N, T+1, 2) — stationary, with start position included."""
    return jnp.full((n, t + 1, 2), jnp.array([lat, lon]))


def _moving_coords_4d(n=4, s=5, t=10):
    """(N, S, T, 2) — S identical northward-moving samples."""
    lats = jnp.linspace(0.0, 0.1, t)
    traj = jnp.stack([lats, jnp.zeros(t)], axis=-1)
    return jnp.tile(traj, (n, s, 1, 1))


def _stationary_coords_4d(n=4, s=5, t=10, lat=0.0, lon=0.0):
    """(N, S, T, 2) — stationary samples."""
    return jnp.full((n, s, t, 2), jnp.array([lat, lon]))


# ---------------------------------------------------------------------------
# 3-D tests — legacy single-sample interface (N, T, 2)
# ---------------------------------------------------------------------------


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


def _haversine_np(a, b):
    """Great-circle distance in metres between (..., 2) lat/lon arrays."""
    a, b = np.radians(a), np.radians(b)
    dlat, dlon = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]
    h = (
        np.sin(dlat / 2) ** 2
        + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin(dlon / 2) ** 2
    )
    return 2 * 6371000.0 * np.arcsin(np.sqrt(h))


def _energy_score_reference(true_coords, pred_coords):
    """Brute-force energy score: at each (n, t), use only the finite samples."""
    true_coords, pred_coords = np.asarray(true_coords), np.asarray(pred_coords)
    n_traj, _, n_t, _ = pred_coords.shape
    out = np.full((n_traj, n_t), np.nan)
    for n in range(n_traj):
        for t in range(n_t):
            x = pred_coords[n, :, t]
            x = x[~np.isnan(x).any(axis=-1)]
            if len(x) == 0:
                continue
            err = np.mean([_haversine_np(true_coords[n, t], xi) for xi in x])
            spread = np.mean([_haversine_np(xi, xj) for xi in x for xj in x])
            out[n, t] = err - 0.5 * spread
    return out


def _random_ensemble(n=3, s=8, t=6, seed=0):
    rng = np.random.default_rng(seed)
    true = np.stack(
        [np.full((n, t), 45.0), np.tile(np.linspace(0.0, 0.5, t), (n, 1))], axis=-1
    )
    pred = true[:, None] + rng.normal(0.0, 0.05, (n, s, t, 2))
    return true, pred


class TestEnergyScoreNaN:
    """Samples that leave the context grid become NaN from some timestep on."""

    def test_matches_reference_without_nan(self):
        true, pred = _random_ensemble()
        score = get_energy_score(jnp.array(true), jnp.array(pred))
        np.testing.assert_allclose(
            score, _energy_score_reference(true, pred), rtol=1e-4
        )

    def test_partially_nan_sample_counts_where_finite(self):
        true, pred = _random_ensemble()
        pred[0, 0, 3:] = np.nan  # sample 0 of traj 0 leaves the grid at t=3
        pred[1, 2, 5:, 1] = np.nan  # a single NaN coordinate also invalidates
        score = np.asarray(get_energy_score(jnp.array(true), jnp.array(pred)))
        np.testing.assert_allclose(
            score, _energy_score_reference(true, pred), rtol=1e-4
        )

    def test_before_nan_equals_full_ensemble(self):
        true, pred = _random_ensemble()
        full = np.asarray(get_energy_score(jnp.array(true), jnp.array(pred)))
        pred_nan = pred.copy()
        pred_nan[0, 0, 3:] = np.nan
        part = np.asarray(get_energy_score(jnp.array(true), jnp.array(pred_nan)))
        np.testing.assert_allclose(part[0, :3], full[0, :3], rtol=1e-5)
        np.testing.assert_allclose(part[1:], full[1:], rtol=1e-5)

    def test_after_nan_equals_ensemble_without_sample(self):
        true, pred = _random_ensemble()
        pred_nan = pred.copy()
        pred_nan[0, 0, 3:] = np.nan
        part = np.asarray(get_energy_score(jnp.array(true), jnp.array(pred_nan)))
        dropped = np.asarray(
            get_energy_score(jnp.array(true[:1]), jnp.array(pred[:1, 1:]))
        )
        np.testing.assert_allclose(part[0, 3:], dropped[0, 3:], rtol=1e-5)

    def test_all_samples_nan_gives_nan(self):
        true, pred = _random_ensemble()
        pred[0, :, 4] = np.nan
        score = np.asarray(get_energy_score(jnp.array(true), jnp.array(pred)))
        assert np.isnan(score[0, 4])
        assert np.isfinite(np.delete(score.ravel(), 4)).all()

    def test_nan_truth_gives_nan(self):
        true, pred = _random_ensemble()
        true[2, 1] = np.nan
        score = np.asarray(get_energy_score(jnp.array(true), jnp.array(pred)))
        assert np.isnan(score[2, 1])
        assert np.isfinite(np.delete(score.ravel(), 2 * score.shape[1] + 1)).all()


class TestGetSkillScore:
    """true_coords has T+1 rows; pred_coords has T rows."""

    def test_returns_non_nan_for_moving_trajectory(self):
        true_coords = _moving_coords_with_start()  # (N, T+1, 2)
        pred_coords = _moving_coords()  # (N, T, 2)
        score = get_skill_score(true_coords, pred_coords)
        assert not jnp.any(jnp.isnan(score)), "Skill score must not be NaN"

    def test_score_is_non_negative(self):
        true_coords = _moving_coords_with_start()
        pred_coords = _moving_coords() + 5.0
        score = get_skill_score(true_coords, pred_coords, threshold=1.0)
        assert jnp.all(score >= 0.0)


class TestGetMetrics:
    """true_coords has T+1 rows; pred_coords has T rows."""

    def test_returns_dict_with_required_keys(self):
        true_coords = _moving_coords_with_start()
        pred_coords = _moving_coords()
        metrics = get_metrics(true_coords, pred_coords)
        assert set(metrics.keys()) == {"mean_distance", "energy_score", "skill_score"}

    def test_all_values_are_arrays(self):
        true_coords = _moving_coords_with_start()
        pred_coords = _moving_coords()
        metrics = get_metrics(true_coords, pred_coords)
        for key, val in metrics.items():
            assert hasattr(val, "shape"), f"metrics['{key}'] should be an array"


# ---------------------------------------------------------------------------
# 4-D tests — production signature (N, S, T, 2)
# ---------------------------------------------------------------------------


class TestGetHaversineError4D:
    """Tests for get_haversine_error with 4-D pred_coords (N, S, T, 2)."""

    def test_shape_is_n_t(self):
        n, s, t = 3, 5, 8
        true = _moving_coords(n=n, t=t)
        pred = _moving_coords_4d(n=n, s=s, t=t)
        error = get_haversine_error(true, pred)
        assert error.shape == (n, t), f"Expected ({n}, {t}), got {error.shape}"

    def test_identical_samples_gives_zero_error(self):
        """When all S samples equal the true trajectory, median error is ~0."""
        coords = _stationary_coords(n=3, t=6)
        pred_4d = coords[:, None, :, :]  # (N, 1, T, 2)
        error = get_haversine_error(coords, pred_4d)
        assert jnp.all(jnp.abs(error) < 1.0), (
            "Error should be ~0 m when pred equals true"
        )

    def test_larger_shift_gives_larger_error_4d(self):
        n, s, t = 4, 10, 10
        true = _moving_coords(n=n, t=t)
        pred_small = _moving_coords_4d(n=n, s=s, t=t) + 0.001
        pred_large = _moving_coords_4d(n=n, s=s, t=t) + 1.0
        err_small = get_haversine_error(true, pred_small)
        err_large = get_haversine_error(true, pred_large)
        assert float(jnp.mean(err_large)) > float(jnp.mean(err_small))

    def test_consistent_with_3d_single_sample(self):
        """4-D with S=1 should give the same result as 3-D input."""
        n, t = 4, 10
        true = _moving_coords(n=n, t=t)
        pred_3d = _moving_coords(n=n, t=t) + 0.05
        pred_4d = pred_3d[:, None, :, :]  # (N, 1, T, 2)
        err_3d = get_haversine_error(true, pred_3d)
        err_4d = get_haversine_error(true, pred_4d)
        np.testing.assert_allclose(
            np.array(err_3d),
            np.array(err_4d),
            rtol=1e-5,
            err_msg="3-D and 4-D (S=1) should agree",
        )

    def test_median_aggregation_uses_sample_median(self):
        """Median over S samples should equal applying haversine to the median coords."""
        n, t = 2, 5
        # Create pred with S=3 samples: two at +0.1, one at -0.1 — median should be +0.1
        true = _moving_coords(n=n, t=t)
        base = _moving_coords(n=n, t=t)
        pred_samples = jnp.stack(
            [base + 0.1, base + 0.1, base - 0.1], axis=1
        )  # (N, 3, T, 2)
        error = get_haversine_error(true, pred_samples, aggregation="median")
        # median should be base + 0.1; compute reference directly
        ref_median = np.nanmedian(np.array(pred_samples), axis=1)  # (N, T, 2)
        from jax import vmap as _vmap
        from codit.data_module import data_transform as _dt

        expected = _vmap(_vmap(_dt.haversine_distance))(true, jnp.array(ref_median))
        np.testing.assert_allclose(np.array(error), np.array(expected), rtol=1e-5)


class TestGetEnergyScore4D:
    """Tests for get_energy_score with 4-D pred_coords (N, S, T, 2)."""

    def test_shape_is_n_t(self):
        n, s, t = 3, 5, 8
        true = _moving_coords(n=n, t=t)
        pred = _moving_coords_4d(n=n, s=s, t=t)
        score = get_energy_score(true, pred)
        assert score.shape == (n, t), f"Expected ({n}, {t}), got {score.shape}"

    def test_identical_samples_gives_zero_score(self):
        """When all S samples equal the true trajectory, energy score is ~0."""
        n, s, t = 3, 4, 6
        coords = _stationary_coords(n=n, t=t)
        pred_4d = jnp.tile(coords[:, None, :, :], (1, s, 1, 1))
        score = get_energy_score(coords, pred_4d)
        assert jnp.all(jnp.abs(score) < 1.0), (
            "Energy score should be ~0 when pred equals true"
        )

    def test_non_negative_for_dispersed_predictions(self):
        """Energy score must be >= 0 by definition."""
        n, s, t = 4, 6, 8
        rng = np.random.default_rng(42)
        true = _moving_coords(n=n, t=t)
        pred = jnp.array(
            np.tile(np.array(true)[:, None], (1, s, 1, 1))
            + rng.normal(scale=0.5, size=(n, s, t, 2))
        )
        score = get_energy_score(true, pred)
        assert jnp.all(score >= -1.0), (
            "Energy score should be >= 0 (allowing small numerics)"
        )

    def test_consistent_with_3d_single_sample(self):
        """4-D with S=1 should give the same result as 3-D input."""
        n, t = 3, 8
        true = _moving_coords(n=n, t=t)
        pred_3d = _moving_coords(n=n, t=t) + 0.05
        pred_4d = pred_3d[:, None, :, :]
        score_3d = get_energy_score(true, pred_3d)
        score_4d = get_energy_score(true, pred_4d)
        np.testing.assert_allclose(
            np.array(score_3d),
            np.array(score_4d),
            rtol=1e-5,
            err_msg="3-D and 4-D (S=1) energy scores should agree",
        )

    def test_larger_spread_gives_larger_score(self):
        """Higher prediction uncertainty (spread) should increase energy score."""
        n, t = 3, 6
        true = _moving_coords(n=n, t=t)
        rng = np.random.default_rng(0)
        base = np.array(_moving_coords(n=n, t=t))
        pred_tight = jnp.array(
            np.tile(base[:, None], (1, 10, 1, 1))
            + rng.normal(scale=0.01, size=(n, 10, t, 2))
        )
        pred_spread = jnp.array(
            np.tile(base[:, None], (1, 10, 1, 1))
            + rng.normal(scale=2.0, size=(n, 10, t, 2))
        )
        score_tight = get_energy_score(true, pred_tight)
        score_spread = get_energy_score(true, pred_spread)
        assert float(jnp.mean(score_spread)) > float(jnp.mean(score_tight))


class TestGetSkillScore4D:
    """Tests for get_skill_score with 4-D pred_coords (N, S, T, 2).

    true_coords has T+1 rows (includes start position at t=0).
    pred_coords has T rows (future predictions).
    """

    def test_shape_is_n_t(self):
        n, s, t = 3, 5, 9
        true = _moving_coords_with_start(n=n, t=t)  # (N, T+1, 2)
        pred = _moving_coords_4d(n=n, s=s, t=t)  # (N, S, T, 2)
        score = get_skill_score(true, pred)
        assert score.shape == (n, t), f"Expected ({n}, {t}), got {score.shape}"

    def test_non_nan_for_moving_trajectory_4d(self):
        true = _moving_coords_with_start(n=4, t=10)  # (N, T+1, 2)
        pred = _moving_coords_4d(n=4, s=5, t=10)
        score = get_skill_score(true, pred)
        assert not jnp.any(jnp.isnan(score)), "Skill score must not be NaN"

    def test_score_is_non_negative_4d(self):
        true = _moving_coords_with_start(n=4, t=10)
        pred = _moving_coords_4d(n=4, s=5, t=10) + 5.0
        score = get_skill_score(true, pred, threshold=1.0)
        assert jnp.all(score >= 0.0)

    def test_perfect_prediction_gives_high_skill(self):
        """When the median of samples equals true, skill score should be 1."""
        n, t = 3, 8
        true = _moving_coords_with_start(n=n, t=t)  # (N, T+1, 2)
        # Single sample equals the true future trajectory
        pred = true[:, None, 1:, :]  # (N, 1, T, 2) — skip start pos
        score = get_skill_score(true, pred, threshold=1.0)
        np.testing.assert_allclose(
            np.array(score),
            np.ones((n, t)),
            atol=1e-4,
            err_msg="Perfect prediction should give skill score of 1",
        )

    def test_consistent_with_3d_single_sample(self):
        """4-D with S=1 should give the same result as 3-D input."""
        n, t = 4, 10
        true = _moving_coords_with_start(n=n, t=t)  # (N, T+1, 2)
        pred_3d = _moving_coords(n=n, t=t) + 0.05  # (N, T, 2)
        pred_4d = pred_3d[:, None, :, :]  # (N, 1, T, 2)
        score_3d = get_skill_score(true, pred_3d)
        score_4d = get_skill_score(true, pred_4d)
        np.testing.assert_allclose(
            np.array(score_3d),
            np.array(score_4d),
            rtol=1e-5,
            err_msg="3-D and 4-D (S=1) skill scores should agree",
        )


class TestGetMetrics4D:
    """Tests for the get_metrics wrapper with 4-D pred_coords (N, S, T, 2).

    true_coords has T+1 rows (includes start position at t=0).
    pred_coords has T rows (future predictions).
    All returned arrays have shape (N, T).
    """

    def test_returns_dict_with_required_keys_4d(self):
        n, s, t = 3, 5, 10
        true = _moving_coords_with_start(n=n, t=t)
        pred = _moving_coords_4d(n=n, s=s, t=t)
        result = get_metrics(true, pred)
        assert set(result.keys()) == {"mean_distance", "energy_score", "skill_score"}

    def test_output_shapes_4d(self):
        """All metric values should have shape (N, T)."""
        n, s, t = 4, 6, 9
        true = _moving_coords_with_start(n=n, t=t)
        pred = _moving_coords_4d(n=n, s=s, t=t)
        result = get_metrics(true, pred)
        for key, val in result.items():
            assert val.shape == (
                n,
                t,
            ), f"metrics['{key}'] shape {val.shape} != expected ({n}, {t})"

    def test_all_values_finite_4d(self):
        n, s, t = 4, 5, 8
        true = _moving_coords_with_start(n=n, t=t)
        pred = _moving_coords_4d(n=n, s=s, t=t)
        result = get_metrics(true, pred)
        for key, val in result.items():
            assert not np.any(np.isnan(np.array(val))), f"metrics['{key}'] contains NaN"

    def test_consistent_3d_vs_4d_s1(self):
        """get_metrics with 3-D and 4-D (S=1) inputs should give the same results."""
        n, t = 4, 10
        true = _moving_coords_with_start(n=n, t=t)  # (N, T+1, 2)
        pred_3d = _moving_coords(n=n, t=t) + 0.03  # (N, T, 2)
        pred_4d = pred_3d[:, None, :, :]  # (N, 1, T, 2)
        m3 = get_metrics(true, pred_3d)
        m4 = get_metrics(true, pred_4d)
        for key in m3:
            np.testing.assert_allclose(
                np.array(m3[key]),
                np.array(m4[key]),
                rtol=1e-5,
                err_msg=f"3-D and 4-D (S=1) disagree for metrics['{key}']",
            )

    def test_energy_score_output_shape_n_t(self):
        """energy_score from get_metrics should have shape (N, T)."""
        n, s, t = 3, 8, 12
        true = _moving_coords_with_start(n=n, t=t)
        pred = _moving_coords_4d(n=n, s=s, t=t)
        result = get_metrics(true, pred)
        assert result["energy_score"].shape == (n, t)


# ---------------------------------------------------------------------------
# Reference-value and edge-case tests
# ---------------------------------------------------------------------------


class TestReferenceValues:
    """Sanity-check metric outputs against hand-computed reference values."""

    def test_haversine_error_one_degree_latitude_is_111km(self):
        """One degree of latitude on a sphere of R=6_371_000 m is ~111.195 km."""
        true = jnp.array([[[0.0, 0.0]]])  # (1, 1, 2)
        pred = jnp.array([[[1.0, 0.0]]])  # (1, 1, 2) — shifted 1 deg north
        error = get_haversine_error(true, pred)
        assert error.shape == (1, 1)
        expected_m = np.pi / 180.0 * 6_371_000.0  # ~111_195 m
        np.testing.assert_allclose(float(error[0, 0]), expected_m, rtol=1e-5)

    def test_haversine_error_zero_shift(self):
        """Zero displacement gives exactly zero error (no floating-point drift > 1 um)."""
        true = jnp.array([[[10.0, 20.0], [10.1, 20.1]]])
        pred = true[:, None, :, :]  # (1, 1, T, 2)
        error = get_haversine_error(true, pred)
        assert float(jnp.max(jnp.abs(error))) < 1e-6

    def test_skill_score_perfect_forecast_equals_one(self):
        """When the forecast trajectory matches truth exactly, skill == 1."""
        n, t = 2, 5
        true_with_start = _moving_coords_with_start(n=n, t=t)  # (N, T+1, 2)
        pred = true_with_start[:, None, 1:, :]  # perfect single-sample forecast
        score = get_skill_score(true_with_start, pred, threshold=1.0)
        np.testing.assert_allclose(np.array(score), np.ones((n, t)), atol=1e-4)

    def test_energy_score_reduces_to_haversine_for_single_sample(self):
        """With S=1, energy score = expected error - 0.5 * self-distance = haversine.

        The cross-distance term is d(s, s) = 0 when S=1, so the score equals
        the pointwise haversine distance between the sample and the truth.
        """
        n, t = 2, 4
        true = _moving_coords(n=n, t=t)
        pred_3d = true + 0.01  # small shift
        pred_4d = pred_3d[:, None, :, :]  # (N, 1, T, 2)
        energy = get_energy_score(true, pred_4d)
        # Reference: pointwise haversine between pred and true
        from jax import vmap as _vmap
        from codit.data_module import data_transform as _dt

        ref = _vmap(_vmap(_dt.haversine_distance))(true, pred_3d)
        np.testing.assert_allclose(np.array(energy), np.array(ref), rtol=1e-5)


class TestInvalidAggregation:
    """Both get_haversine_error and get_skill_score reject unknown aggregation."""

    def test_haversine_error_rejects_bad_aggregation(self):
        coords = _moving_coords()
        with pytest.raises(ValueError, match="Unknown aggregation"):
            get_haversine_error(coords, coords, aggregation="mode")

    def test_skill_score_rejects_bad_aggregation(self):
        true = _moving_coords_with_start()
        pred = _moving_coords()
        with pytest.raises(ValueError, match="Unknown aggregation"):
            get_skill_score(true, pred, aggregation="mode")


class TestMeanAggregation:
    """Mean aggregation path is a supported alternative to median."""

    def test_haversine_error_mean_agrees_with_reference(self):
        n, s, t = 2, 4, 5
        rng = np.random.default_rng(0)
        true = _moving_coords(n=n, t=t)
        pred = jnp.array(
            np.tile(np.array(true)[:, None], (1, s, 1, 1))
            + rng.normal(scale=0.05, size=(n, s, t, 2))
        )
        error = get_haversine_error(true, pred, aggregation="mean")
        # Reference: compute mean over S, then haversine to true
        from jax import vmap as _vmap
        from codit.data_module import data_transform as _dt

        aggr = np.nanmean(np.array(pred), axis=1)  # (N, T, 2)
        ref = _vmap(_vmap(_dt.haversine_distance))(true, jnp.array(aggr))
        np.testing.assert_allclose(np.array(error), np.array(ref), rtol=1e-5)


class TestSkillScoreThreshold:
    """Threshold parameter linearly scales the (1 - c/threshold) core."""

    def test_higher_threshold_gives_higher_or_equal_skill(self):
        """Doubling the threshold cannot decrease the (non-clipped) skill score."""
        n, t = 3, 6
        true = _moving_coords_with_start(n=n, t=t)
        pred = _moving_coords(n=n, t=t) + 0.02  # moderate forecast error
        low = get_skill_score(true, pred, threshold=1.0)
        high = get_skill_score(true, pred, threshold=2.0)
        # Both are clipped at 0 from below, so high >= low pointwise
        assert bool(jnp.all(high >= low - 1e-6))

    def test_skill_score_clipped_at_zero(self):
        """Very bad forecast (huge shift) is clipped at 0, not negative."""
        n, t = 2, 5
        true = _moving_coords_with_start(n=n, t=t)
        pred = _moving_coords(n=n, t=t) + 50.0  # far away
        score = get_skill_score(true, pred, threshold=1.0)
        assert bool(jnp.all(score >= 0.0))
        assert bool(jnp.all(score < 1e-6))  # essentially zero


class TestNanHandling:
    """Energy score uses nansum-style aggregation over samples."""

    def test_energy_score_ignores_nan_samples(self):
        """A single NaN sample should not blow up the score; result stays finite."""
        n, s, t = 2, 4, 3
        base = np.array(_moving_coords(n=n, t=t))
        pred = np.tile(base[:, None], (1, s, 1, 1)) + 0.01
        pred[:, 0, :, :] = np.nan  # invalidate one sample per trajectory
        score = get_energy_score(jnp.array(base), jnp.array(pred))
        assert score.shape == (n, t)
        assert not np.any(np.isnan(np.array(score))), (
            "NaN samples should be filtered, not propagate through the score"
        )
