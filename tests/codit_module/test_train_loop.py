"""Convergence checks of ``diffusion_models.train``: step budget, loss scale,
deterministic validation, early stopping and best-parameter selection."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from torch.utils.data import DataLoader

from codit.codit_module import diffusion_models
from codit.codit_module.brownian_noise_models import WalkNoiseModel
from codit.codit_module.diffusion_models import (
    _cycle,
    _dataset_arrays,
    _device_batches,
    ema_decay_at,
    ema_update,
    train,
    validation_k,
)
from codit.codit_module.scheduler import cosine_schedule
from codit.data_module.dataset import OceanTrainingDataset

L, H, C, K = 6, 16, 3, 10  # encoders assume a 16x16 context grid
SCHEDULE = cosine_schedule(K)
NOISE = WalkNoiseModel(noise_type="brownian_motion", num_dims=2, dt=1.0 / L, sigma=1.0, T=1.0)
STD = jnp.array([5000.0, 5000.0])


def _dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    start = np.c_[rng.uniform(-10, 10, n), rng.uniform(-150, -140, n)]
    steps = rng.normal(0, 0.02, (n, L, 2)).cumsum(1)
    traj = np.concatenate([start[:, None], start[:, None] + steps], 1).astype(np.float32)
    ctx = rng.normal(size=(n, L + 1, H, H, C)).astype(np.float32)
    ds = OceanTrainingDataset(traj, ctx)
    ds.set_context_var_idx(np.arange(C))
    return ds


class CountingLoader:
    """DataLoader wrapper that counts the batches drawn from it."""

    def __init__(self, loader):
        self.loader, self.batch_size, self.drawn = loader, loader.batch_size, 0

    def __iter__(self):
        for b in self.loader:
            self.drawn += 1
            yield b

    def __len__(self):
        return len(self.loader)


@pytest.fixture(scope="module")
def setup():
    model = diffusion_models.Autoregressive_Denoiser(
        output_size=2, context=True, num_layers=1, hidden_size=8,
        diffusion_emb_dim=8, K=K, context_encoder_type="gap",
    )
    batch = diffusion_models.transform_batch(next(iter(DataLoader(_dataset(4), batch_size=4))), STD)
    one = jax.tree.map(lambda a: a[0], batch)
    params = model.init(
        jax.random.PRNGKey(0), y=one["prediction_target_vals"][1:],
        y_history=one["history_target_vals"], x=one, k=jnp.ones((1,), dtype=jnp.int32),
    )
    return model, params


def _run(setup, lr, num_epochs, patience=10, n_train=12, batch=4, steps=5, key=0, ema=0.999):
    model, params = setup
    opt = optax.sgd(lr)
    train_loader = CountingLoader(DataLoader(_dataset(n_train), batch_size=batch, shuffle=True, drop_last=True))
    val_loader = DataLoader(_dataset(8, seed=1), batch_size=4, shuffle=False, drop_last=True)
    out = train(
        train_loader, opt, model, NOISE, params, opt.init(params), jax.random.PRNGKey(key),
        SCHEDULE, num_epochs=num_epochs, num_batches_per_epoch=steps,
        val_dataloader=val_loader, early_stopping_patience=patience, std=STD,
        ema_decay=ema,
    )
    return out, train_loader


def test_validation_k_covers_all_levels_evenly():
    ks = jnp.concatenate([validation_k(i, 4, K) for i in range(5)])[:, 0]
    assert ks.tolist() == [i % K for i in range(20)]


def test_cycle_restarts_and_rejects_empty_loader():
    it = _cycle([1, 2])
    assert [next(it) for _ in range(5)] == [1, 2, 1, 2, 1]
    with pytest.raises(ValueError, match="empty"):
        next(_cycle([]))


def test_epoch_is_exactly_num_batches_per_epoch_steps(setup):
    # 12 samples / batch 4 = 3 batches per pass, but 5 steps per epoch are required
    (_, _, losses, _, _), loader = _run(setup, lr=1e-3, num_epochs=2, steps=5)
    assert len(losses) == 2
    assert loader.drawn == 10


def test_val_loss_is_deterministic_and_patience_is_exact(setup):
    # lr = 0: parameters never change, so the val loss must be identical every epoch
    (_, _, losses, val_losses, info), _ = _run(setup, lr=0.0, num_epochs=20, patience=3)
    assert len(set(val_losses)) == 1
    # with frozen weights the EMA equals the weights
    np.testing.assert_allclose(info["val_loss_ema"], info["val_loss_raw"], rtol=1e-5)
    # epoch 0 is the best; stops after exactly `patience` epochs without improvement
    assert len(val_losses) == 1 + 3
    # loss is a per-sample mean MSE (order 1 for epsilon prediction), not divided by batch size
    assert 0.05 < val_losses[0] < 20 and 0.05 < losses[0] < 20


@pytest.mark.parametrize("ema", [None, 0.999])
def test_returns_parameters_of_best_epoch(setup, ema):
    # sensible lr for epoch 0 (5 steps), then a huge one: later epochs get worse or NaN
    lr = optax.piecewise_constant_schedule(1e-2, {5: 1e6})
    (best, _, _, val_losses, _), _ = _run(setup, lr=lr, num_epochs=4, patience=10, ema=ema)
    assert np.isfinite(val_losses[0])
    best_epoch = int(np.nanargmin(val_losses))
    assert best_epoch < len(val_losses) - 1, "test needs a run that gets worse"
    # re-validate the returned parameters with a zero learning rate and the same key
    model, _ = setup
    opt = optax.sgd(0.0)
    val_loader = DataLoader(_dataset(8, seed=1), batch_size=4, shuffle=False, drop_last=True)
    train_loader = DataLoader(_dataset(12), batch_size=4, shuffle=True, drop_last=True)
    _, _, _, recheck, _ = train(
        train_loader, opt, model, NOISE, best, opt.init(best), jax.random.PRNGKey(0),
        SCHEDULE, num_epochs=1, num_batches_per_epoch=1, val_dataloader=val_loader, std=STD,
        ema_decay=None,
    )
    assert recheck[0] == pytest.approx(val_losses[best_epoch], rel=1e-5)


def test_ema_decay_warm_up():
    assert float(ema_decay_at(0, 0.999)) == pytest.approx(0.1)
    assert float(ema_decay_at(10, 0.999)) == pytest.approx(11 / 20)
    assert float(ema_decay_at(100_000, 0.999)) == pytest.approx(0.999)


def test_ema_update():
    ema = {"w": jnp.array([0.0, 1.0])}
    new = ema_update(ema, {"w": jnp.array([1.0, 1.0])}, 0.9)
    np.testing.assert_allclose(new["w"], [0.1, 1.0], rtol=1e-6)


def test_without_ema_no_ema_losses(setup):
    (_, _, _, val_losses, info), _ = _run(setup, lr=1e-3, num_epochs=2, ema=None)
    assert info["val_loss_ema"] == []
    assert val_losses == info["val_loss_raw"]


def test_ema_differs_from_raw_when_training(setup):
    (best, _, _, _, info), _ = _run(setup, lr=1e-2, num_epochs=2, ema=0.999)
    assert info["val_loss_ema"] != info["val_loss_raw"]
    leaves_best = jax.tree.leaves(best)
    leaves_raw = jax.tree.leaves(info["best_raw_params"])
    assert any(not np.allclose(a, b) for a, b in zip(leaves_best, leaves_raw))


def test_device_batches_shuffle_and_drop_last():
    it = _device_batches(10, 4, jax.random.PRNGKey(0))
    first_pass = np.concatenate([np.asarray(next(it)) for _ in range(2)])
    assert len(first_pass) == 8 and len(set(first_pass.tolist())) == 8  # 2 left over
    second_pass = np.concatenate([np.asarray(next(it)) for _ in range(2)])
    assert not np.array_equal(first_pass, second_pass)  # reshuffled
    ordered = [np.asarray(b).tolist() for b in _device_batches(10, 4, None)]
    assert ordered == [[0, 1, 2, 3], [4, 5, 6, 7]]  # one pass, in order


def test_dataset_arrays_follow_indices_and_channels():
    ds = _dataset(6)
    ds.indices = np.array([5, 0, 3])
    ds.set_context_var_idx(np.array([2, 0]))
    traj, ctx = _dataset_arrays(DataLoader(ds, batch_size=2))
    np.testing.assert_array_equal(traj, ds.trajectories[[5, 0, 3]])
    np.testing.assert_array_equal(ctx, ds.context[[5, 0, 3]][..., [2, 0]])


def test_device_data_matches_dataloader(setup):
    """Same validation batches, order and keys: identical val losses; training runs."""
    model, params = setup
    opt = optax.sgd(0.0)
    val_ds = _dataset(8, seed=1)
    out = {}
    for device_data in (False, True):
        train_loader = DataLoader(_dataset(12), batch_size=4, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, drop_last=True)
        out[device_data] = train(
            train_loader, opt, model, NOISE, params, opt.init(params),
            jax.random.PRNGKey(0), SCHEDULE, num_epochs=2, num_batches_per_epoch=5,
            val_dataloader=val_loader, std=STD, ema_decay=None, device_data=device_data,
        )
    np.testing.assert_allclose(out[True][3], out[False][3], rtol=1e-5)
    assert len(out[True][2]) == 2 and np.all(np.isfinite(out[True][2]))


def test_device_data_trains(setup):
    model, params = setup
    opt = optax.adam(1e-2)
    train_loader = DataLoader(_dataset(12), batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(_dataset(8, seed=1), batch_size=4, shuffle=False, drop_last=True)
    best, _, losses, val_losses, _ = train(
        train_loader, opt, model, NOISE, params, opt.init(params), jax.random.PRNGKey(0),
        SCHEDULE, num_epochs=3, num_batches_per_epoch=5, val_dataloader=val_loader,
        std=STD, device_data=True,
    )
    assert len(losses) == 3 and np.all(np.isfinite(val_losses))
    assert any(not np.allclose(a, b) for a, b in zip(jax.tree.leaves(best), jax.tree.leaves(params)))
