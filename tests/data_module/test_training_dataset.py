"""OceanTrainingDataset keeps its arrays in host memory (NumPy), not on the device."""

import jax.numpy as jnp
import numpy as np

from codit.data_module.dataset import OceanTrainingDataset, merge_training_sets

N, L, H, C = 10, 5, 4, 8


def _make(n=N, seed=0):
    rng = np.random.default_rng(seed)
    traj = rng.normal(size=(n, L + 1, 2)).astype(np.float32)
    ctx = rng.normal(size=(n, L + 1, H, H, C)).astype(np.float32)
    return traj, ctx


def test_arrays_are_numpy_even_from_jax_input():
    traj, ctx = _make()
    ds = OceanTrainingDataset(jnp.asarray(traj), jnp.asarray(ctx))
    assert isinstance(ds.trajectories, np.ndarray)
    assert isinstance(ds.context, np.ndarray)
    assert isinstance(ds.indices, np.ndarray)
    assert ds.context.dtype == np.float32


def test_getitem_selects_context_vars_with_jax_index():
    traj, ctx = _make()
    ds = OceanTrainingDataset(traj, ctx)
    ds.set_context_var_idx(jnp.asarray([0, 3, 5]))
    t, c = ds[2]
    np.testing.assert_array_equal(t, traj[2])
    np.testing.assert_array_equal(c, ctx[2][..., [0, 3, 5]])


def test_shuffle_split_partition_all_samples():
    np.random.seed(0)
    traj, ctx = _make()
    ds = OceanTrainingDataset(traj, ctx)
    ds.shuffle()
    tr, va = ds.split(0.8)
    assert len(tr) == 8 and len(va) == 2
    assert sorted(np.concatenate([tr.indices, va.indices]).tolist()) == list(range(N))
    # split shares the underlying arrays instead of copying them
    assert tr.context is ds.context and va.context is ds.context


def test_remove_nans_drops_only_nan_samples_and_skips_copy_without_nans():
    traj, ctx = _make()
    ds = OceanTrainingDataset(traj, ctx)
    before = ds.context
    ds.remove_nans()
    assert ds.context is before  # nothing to drop -> no copy
    ctx[[1, 4], 0, 0, 0, 0] = np.nan
    ds = OceanTrainingDataset(traj, ctx)
    ds.remove_nans()
    assert len(ds) == N - 2
    assert not np.isnan(ds.context).any()


def test_merge_concatenates_on_host():
    a, b = OceanTrainingDataset(*_make(4, 0)), OceanTrainingDataset(*_make(6, 1))
    merged = merge_training_sets([a, b])
    assert isinstance(merged.context, np.ndarray)
    assert len(merged) == 10
    np.testing.assert_array_equal(merged.context[4:], b.context)


def test_save_and_load_round_trip(tmp_path):
    traj, ctx = _make()
    OceanTrainingDataset(traj, ctx).save(str(tmp_path))
    ds = OceanTrainingDataset.from_disk(str(tmp_path), L)
    assert isinstance(ds.context, np.ndarray)
    np.testing.assert_array_equal(ds.context, ctx)
    np.testing.assert_array_equal(ds.trajectories, traj)
