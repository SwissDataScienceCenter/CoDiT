import jax.numpy as jnp
import pytest

from codit.codit_module.scheduler import cosine_schedule, linear_schedule


class TestLinearSchedule:
    def test_returns_required_keys(self):
        sched = linear_schedule(K=10)
        assert set(sched.keys()) == {"beta", "alpha", "alpha_bar", "K"}

    def test_K_stored_correctly(self):
        sched = linear_schedule(K=10)
        assert sched["K"] == 10

    def test_arrays_have_length_K(self):
        K = 10
        sched = linear_schedule(K=K)
        assert len(sched["beta"]) == K
        assert len(sched["alpha"]) == K
        assert len(sched["alpha_bar"]) == K

    def test_alpha_bar_monotonically_non_increasing(self):
        sched = linear_schedule(K=50)
        diffs = jnp.diff(sched["alpha_bar"])
        assert jnp.all(diffs <= 0), "alpha_bar must be non-increasing"

    def test_alpha_bar_starts_near_one(self):
        sched = linear_schedule(K=200, beta_min=1e-4, beta_max=0.2)
        assert float(sched["alpha_bar"][0]) == pytest.approx(1.0, abs=0.02)

    def test_alpha_bar_ends_near_zero(self):
        sched = linear_schedule(K=1000, beta_min=1e-4, beta_max=0.2)
        assert float(sched["alpha_bar"][-1]) < 0.1

    def test_beta_in_valid_range(self):
        sched = linear_schedule(K=50)
        assert jnp.all(sched["beta"] >= 0)
        assert jnp.all(sched["beta"] <= 1)


class TestCosineSchedule:
    def test_returns_required_keys(self):
        sched = cosine_schedule(K=10)
        assert set(sched.keys()) == {"beta", "alpha", "alpha_bar", "K"}

    def test_alpha_bar_monotonically_non_increasing(self):
        sched = cosine_schedule(K=50)
        diffs = jnp.diff(sched["alpha_bar"])
        assert jnp.all(diffs <= 0)

    def test_K_stored_correctly(self):
        sched = cosine_schedule(K=20)
        assert sched["K"] == 20

    def test_arrays_have_length_K(self):
        K = 25
        sched = cosine_schedule(K=K)
        assert len(sched["beta"]) == K
        assert len(sched["alpha"]) == K
        assert len(sched["alpha_bar"]) == K

    def test_beta_bounded_by_beta_min_and_beta_max(self):
        beta_min, beta_max = 1e-4, 0.2
        sched = cosine_schedule(K=50, beta_min=beta_min, beta_max=beta_max)
        # Small numerical tolerance for rounding at endpoints
        assert jnp.all(sched["beta"] >= beta_min - 1e-6)
        assert jnp.all(sched["beta"] <= beta_max + 1e-6)

    def test_alpha_bar_starts_near_one(self):
        sched = cosine_schedule(K=200)
        assert float(sched["alpha_bar"][0]) == pytest.approx(1.0, abs=0.02)

    def test_alpha_bar_ends_smaller_than_start(self):
        sched = cosine_schedule(K=1000, beta_min=1e-4, beta_max=0.2)
        assert float(sched["alpha_bar"][-1]) < float(sched["alpha_bar"][0])


@pytest.mark.parametrize("schedule_fn", [linear_schedule, cosine_schedule])
class TestSchedulerInvariants:
    """Invariants that must hold for any schedule generator."""

    def test_alpha_equals_one_minus_beta(self, schedule_fn):
        sched = schedule_fn(K=30)
        assert jnp.allclose(sched["alpha"], 1 - sched["beta"], atol=1e-6)

    def test_alpha_bar_is_cumprod_of_alpha(self, schedule_fn):
        sched = schedule_fn(K=30)
        expected = jnp.cumprod(sched["alpha"])
        assert jnp.allclose(sched["alpha_bar"], expected, atol=1e-6)

    def test_K_equals_one_produces_length_one(self, schedule_fn):
        sched = schedule_fn(K=1)
        assert sched["beta"].shape == (1,)
        assert sched["alpha"].shape == (1,)
        assert sched["alpha_bar"].shape == (1,)
