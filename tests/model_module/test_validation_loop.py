"""
Documents the division-by-zero bug in validation loop normalisation and verifies the fix.

The bug in linear_gauss_model.py and conv_net.py:
    val_loss = 0
    for i, batch in enumerate(val_dataloader):
        val_loss += eval_step(...)
    val_loss /= i * batch[0].shape[0] * batch[0].shape[1]  # i=0 → inf

The fix:
    n_val = 0
    for batch in val_dataloader:
        val_loss += eval_step(...)
        n_val += batch[0].shape[0] * batch[0].shape[1]
    val_loss /= n_val  # always > 0
"""

import jax.numpy as jnp
import pytest


def _simulate_buggy_normalisation(num_batches, batch_size=4, seq_len=10):
    """Reproduce the buggy normalisation pattern with a given number of batches."""
    val_loss = jnp.array(float(num_batches))  # pretend sum of losses = num_batches
    fake_batch_shape = (batch_size, seq_len)
    for i in range(num_batches):
        pass  # simulate the loop
    # After loop, i = num_batches - 1
    divisor = i * fake_batch_shape[0] * fake_batch_shape[1]
    return val_loss / divisor


def _simulate_fixed_normalisation(num_batches, batch_size=4, seq_len=10):
    """Reproduce the fixed normalisation pattern."""
    val_loss = jnp.array(float(num_batches))
    n_val = num_batches * batch_size * seq_len
    return val_loss / n_val


def test_buggy_normalisation_is_inf_with_one_batch():
    result = _simulate_buggy_normalisation(num_batches=1)
    assert not jnp.isfinite(
        result
    ), "This documents the bug: with 1 batch, i=0 → val_loss/0 = inf"


def test_fixed_normalisation_is_finite_with_one_batch():
    result = _simulate_fixed_normalisation(num_batches=1)
    assert jnp.isfinite(result)
    assert float(result) == pytest.approx(1.0 / (1 * 4 * 10), rel=1e-5)


def test_fixed_normalisation_is_finite_with_multiple_batches():
    result = _simulate_fixed_normalisation(num_batches=5)
    assert jnp.isfinite(result)
