"""``CoDiTModel.predict_batch`` must reproduce ``predict`` trajectory by trajectory.

Uses a synthetic context field (no NetCDF needed) with the real
``OceanDataset.get_context`` and one ``Interpolator3D`` per trajectory, as
``OceanDataset`` items provide them.
"""

import datetime
import os
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interpax import Interpolator3D

from codit.data_module import config_paths, utils
from codit.data_module.dataset import OceanDataset

_MODELS_DIR = os.path.join(config_paths.MODEL_PATH, "trained_models")
_CKPT = os.path.join(_MODELS_DIR, "codit_GRU_open_ocean", "nn_params.npz")
pytestmark = pytest.mark.skipif(
    not os.path.exists(_CKPT), reason="codit_GRU_open_ocean checkpoint not found"
)

L, S = 6, 3  # short horizon, few samples: fast on CPU
LAT = jnp.arange(-40.0, -30.0, 0.25)
LON = jnp.arange(90.0, 100.0, 0.25)
STARTS = [
    (jnp.array([-35.0, 95.0]), datetime.datetime(2021, 3, 1)),
    (jnp.array([-34.2, 93.1]), datetime.datetime(2021, 3, 1, 5)),
    (jnp.array([-36.7, 96.4]), datetime.datetime(2021, 3, 2, 11)),
]
KEYS = jax.random.split(jax.random.PRNGKey(42), len(STARTS))


def _interp(start_time: datetime.datetime, amp: float) -> Interpolator3D:
    """Smooth 8-channel field on the trajectory's own (L+1)-hour time axis."""
    t0 = float(utils.convert_date_to_hours(np.array([np.datetime64(start_time)]))[0])
    t = jnp.arange(t0, t0 + L + 1, 1.0)
    T, LA, LO = jnp.meshgrid(t, LAT, LON, indexing="ij")
    f = amp * jnp.stack(
        [jnp.sin(0.3 * LA + 0.1 * c) * jnp.cos(0.2 * LO - 0.05 * T + c) for c in range(8)],
        axis=-1,
    )
    return Interpolator3D(t, LAT, LON, f, method="linear")


@pytest.fixture(scope="module")
def setup():
    from codit.model_module.diffusion_model import CoDiTModel

    model = CoDiTModel.import_trained_model("codit_GRU_open_ocean", import_path=_MODELS_DIR)
    ds = types.SimpleNamespace(num_grid_points=16, delta_metres=10000)
    ds.get_context = types.MethodType(OceanDataset.get_context, ds)
    # a different field per trajectory, so using the wrong row's
    # interpolator would change the result
    interps = [_interp(t, amp) for (_, t), amp in zip(STARTS, [1.0, -3.0, 5.0])]
    return model, ds, interps


def _loop(model, ds, interps, initial_y=None):
    return jnp.stack([
        model.predict(p, t, L, it, ds, num_samples=S, key=k,
                      initial_y=None if initial_y is None else initial_y[b])
        for b, ((p, t), it, k) in enumerate(zip(STARTS, interps, KEYS))
    ])


def _batch(model, ds, interps, **kw):
    return model.predict_batch(
        jnp.stack([p for p, _ in STARTS]), [t for _, t in STARTS], L, interps, ds,
        num_samples=S, keys=KEYS, **kw,
    )


@pytest.mark.parametrize("sampler", ["ddpm", "heun"])
def test_batch_matches_per_trajectory_predict(setup, sampler):
    model, ds, interps = setup
    model.set_sampler(sampler, 6)
    expected = _loop(model, ds, interps)
    got = _batch(model, ds, interps)
    assert got.shape == (len(STARTS), S, L, 2)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_each_row_uses_its_own_interpolator(setup):
    model, ds, interps = setup
    model.set_sampler("ddpm")
    got = _batch(model, ds, interps)
    swapped = _batch(model, ds, interps[::-1])
    assert not jnp.allclose(got[0], swapped[0])


def test_shared_interpolator_and_initial_y(setup):
    model, ds, interps = setup
    model.set_sampler("ddpm")
    shared = [interps[0]] * len(STARTS)
    y0 = 0.3 * jax.random.normal(jax.random.PRNGKey(7), (len(STARTS), S, L + 1, 2))
    expected = _loop(model, ds, shared, initial_y=y0)
    got = _batch(model, ds, interps[0], initial_y=y0)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_on_step_reports_every_hour(setup):
    model, ds, interps = setup
    model.set_sampler("ddpm")
    seen = []
    _batch(model, ds, interps, on_step=seen.append)
    assert seen == list(range(1, L + 1))


def test_mismatched_inputs_raise(setup):
    model, ds, interps = setup
    with pytest.raises(ValueError, match="must match"):
        model.predict_batch(jnp.zeros((2, 2)), [STARTS[0][1]] * 3, L, interps, ds, keys=KEYS)
    with pytest.raises(ValueError, match="interpolators"):
        _batch(model, ds, interps[:2])
