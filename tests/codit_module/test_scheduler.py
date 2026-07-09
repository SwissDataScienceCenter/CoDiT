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
