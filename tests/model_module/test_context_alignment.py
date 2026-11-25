"""Which context step conditions which predicted position (``aligned_context``).

Position j is the drifter position at step j. Its step j-1 -> j should be conditioned on
the context up to c_{j-1}, the tile at the start of the step. The legacy layout (paper
checkpoints, ``aligned_context=False``) conditions it only up to c_{j-2}.
"""

import datetime

import jax
import numpy as np
import pytest
from interpax import Interpolator3D
from jax import numpy as jnp

from codit.codit_module import diffusion_models as dm
from codit.data_module import utils
from codit.data_module.dataset import OceanDataset

L, G = 6, 16


def _first_affected_position(aligned: bool, C: int = 2) -> dict:
    """{t: first position j whose noise prediction changes when c_t changes} for the
    training path (``transform_batch`` + denoiser); None if c_t is never used."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    traj = jnp.stack([30 + 0.05 * jnp.arange(L + 1.0), -140 + 0.05 * jnp.arange(L + 1.0)],
                     axis=-1)[None]
    ctx = jax.random.normal(k1, (1, L + 1, G, G, C))

    def batch(c):
        b = dm.transform_batch((traj, c), std=jnp.array([1e4, 1e4]), aligned_context=aligned)
        return {k: v[0] for k, v in b.items()}

    model = dm.Autoregressive_Denoiser(context=True, num_layers=2, hidden_size=16,
                                       diffusion_emb_dim=8, K=10)
    y, y_hist, k = jax.random.normal(k2, (L, 2)), jnp.zeros((1, 2)), jnp.array([5])
    params = model.init(k3, y, y_hist, batch(ctx), k)
    base = model.apply(params, y, y_hist, batch(ctx), k)        # row p = position p + 1
    out = {}
    for t in range(L + 1):
        d = np.abs(np.asarray(model.apply(params, y, y_hist, batch(ctx.at[:, t].add(5.0)), k)
                              - base)).max(-1)
        hit = np.nonzero(d > 1e-6)[0]
        out[t] = int(hit[0]) + 1 if len(hit) else None
    return out


def test_training_layout_aligned():
    res = _first_affected_position(aligned=True)
    # c_0 conditions position 1, c_t position t + 1; c_L is past the last step
    assert res == {0: 1, **{t: t + 1 for t in range(1, L)}, L: None}


def test_training_layout_legacy_is_one_step_late():
    res = _first_affected_position(aligned=False)
    # c_0 twice, c_t only from position t + 2; c_{L-1} and c_L unused
    assert res == {0: 1, **{t: t + 2 for t in range(1, L - 1)}, L - 1: None, L: None}


# --------------------------------------------------------------------------- prediction
START = datetime.datetime(2024, 6, 1, 12)
DT_HOURS = 1.0  # predict() steps hourly


def _model(aligned: bool):
    from codit.model_module.diffusion_model import CoDiTModel

    model = CoDiTModel("unit_alignment", {
        "prediction_length": L, "dt": 1.0 / L, "K": 3, "num_layers": 1, "hidden_dims": 8,
        "diffusion_emb": 8, "vars": ["copernicus_current"],
        "aligned_context": aligned})
    y_hist = jnp.zeros((1, 2))
    C = len(model.var_idx)
    model.nn_params = model.model.init(
        jax.random.PRNGKey(0), y=jnp.zeros((L, 2)), y_history=y_hist,
        x={"history_target_vals": y_hist, "history_context": jnp.zeros((1, G, G, C)),
           "prediction_context": jnp.zeros((L, G, G, C))},
        k=jnp.ones((1,), dtype=jnp.int32), train=True)
    model.std = jnp.full(2, 2e4)
    return model


def _interpolator(change_from_step: int | None):
    """8-channel field, channel c = lat + c * lon; from step `change_from_step` on (knots at
    the prediction times, so linear interpolation is exact there) every channel is +5."""
    t0 = float(utils.convert_date_to_hours(np.datetime64(START)))
    t = jnp.asarray(t0 + DT_HOURS * np.arange(-1, L + 2))
    lat = jnp.linspace(20.0, 40.0, 41)
    lon = jnp.linspace(-150.0, -130.0, 41)
    LAT, LON = jnp.meshgrid(lat, lon, indexing="ij")
    field = jnp.stack([LAT + c * LON for c in range(8)], axis=-1)
    steps = np.arange(-1, L + 2)
    shift = np.zeros(len(steps)) if change_from_step is None else \
        5.0 * (steps >= change_from_step)
    vals = field[None] + jnp.asarray(shift)[:, None, None, None]
    return Interpolator3D(t, lat, lon, vals, method="linear")


def _context_source():
    """OceanDataset with only what get_context needs (no data files)."""
    ds = OceanDataset.__new__(OceanDataset)
    ds.num_grid_points, ds.delta_metres = G, 50000
    return ds


def _predict(model, interp):
    x0 = jnp.asarray([30.0, -140.0])
    model._denoise_func = None  # the cached jit closes over the interpolator's dataset
    return np.asarray(model.predict(x0, START, L, interp, _context_source(),
                                    num_samples=2, key=jax.random.PRNGKey(3)))


@pytest.mark.parametrize("aligned", [True, False], ids=["aligned", "legacy"])
def test_prediction_uses_the_training_layout(aligned):
    """Changing the forcing from step 1 on: with the aligned layout position 2 already
    responds (its step 1 -> 2 sees c_1), with the legacy one only position 3 does."""
    model = _model(aligned)
    base = _predict(model, _interpolator(None))
    changed = _predict(model, _interpolator(1))
    differs = np.abs(changed - base).max(axis=(0, 2)) > 1e-9       # per position 1..L
    first = int(np.argmax(differs)) + 1
    assert not differs[0]                                           # position 1: c_0 only
    assert first == (2 if aligned else 3)


def _params_without_flag():
    params = dict(_model(True).params)
    params.pop("aligned_context")
    return params


def test_new_models_are_aligned_by_default():
    from codit.model_module.diffusion_model import CoDiTModel

    assert CoDiTModel("new", _params_without_flag()).aligned_context is True


@pytest.mark.parametrize("aligned", [True, False])
def test_flag_round_trip(tmp_path, aligned):
    from codit.model_module.diffusion_model import CoDiTModel

    _model(aligned).export_trained_model("m", str(tmp_path))
    assert CoDiTModel.import_trained_model("m", str(tmp_path)).aligned_context is aligned


def test_models_saved_before_the_flag_load_as_legacy(tmp_path):
    """Shipped paper checkpoints have no aligned_context in model_config.yml."""
    import yaml

    from codit.model_module.diffusion_model import CoDiTModel

    _model(False).export_trained_model("old", str(tmp_path))
    cfg = tmp_path / "old" / "model_config.yml"
    params = yaml.safe_load(cfg.read_text())
    params.pop("aligned_context")
    cfg.write_text(yaml.dump(params))
    assert CoDiTModel.import_trained_model("old", str(tmp_path)).aligned_context is False


def test_shipped_paper_checkpoints_load_as_legacy():
    import os

    from codit.data_module import config_paths
    from codit.model_module.diffusion_model import CoDiTModel

    path = os.path.join(config_paths.MODEL_PATH, "trained_models")
    if not os.path.isdir(os.path.join(path, "codit_GRU_open_ocean")):
        pytest.skip("shipped checkpoints not present")
    assert CoDiTModel.import_trained_model("codit_GRU_open_ocean", path).aligned_context is False


@pytest.mark.parametrize("saved", [None, True, False], ids=["pre-flag", "aligned", "legacy"])
def test_training_checkpoint_layout_must_match(tmp_path, saved):
    """load_trained_model reads ckpts/<name>/final/params.npz, which has no model config."""
    model = _model(True)
    model.ckpt_dir = str(tmp_path)
    (tmp_path / "final").mkdir()
    extra = {} if saved is None else {"aligned_context": saved}
    jnp.savez(tmp_path / "final" / "params.npz", params=np.zeros(1), std=np.ones(2), **extra)
    if saved is True:
        model.load_trained_model()
    else:
        with pytest.raises(ValueError, match="aligned_context"):
            model.load_trained_model()
