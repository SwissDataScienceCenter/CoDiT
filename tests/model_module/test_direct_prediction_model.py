import datetime

import jax.numpy as jnp
from jax import random

from codit.codit_module.direct_models import AutoregressiveDirectPredictor
from codit.model_module.direct_prediction_model import (
    DirectPredictionModel,
    make_teacher_forced_batch,
)


def make_example(sequence_length: int = 4, num_vars: int = 8) -> tuple[jnp.ndarray, dict]:
    history_target_vals = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [0.2, -0.05],
            [0.3, 0.0],
        ],
        dtype=jnp.float32,
    )[:sequence_length]
    history_context = jnp.ones((sequence_length, 16, 16, num_vars), dtype=jnp.float32)
    prediction_context = 2.0 * jnp.ones(
        (sequence_length, 16, 16, num_vars), dtype=jnp.float32
    )
    batch = {
        "history_context": history_context,
        "prediction_context": prediction_context,
    }
    return history_target_vals, batch


def test_direct_backbone_forward_shape() -> None:
    y_history, batch = make_example()
    model = AutoregressiveDirectPredictor(
        output_size=2,
        context=True,
        num_layers=2,
        hidden_size=32,
        context_encoder_type="gap",
    )
    params = model.init(random.PRNGKey(0), y_history=y_history, x=batch, train=False)
    preds = model.apply(params, y_history=y_history, x=batch, train=False)
    assert preds.shape == y_history.shape


def test_make_teacher_forced_batch_shifts_targets() -> None:
    trajectory = jnp.array(
        [
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [3.0, 5.0],
                [8.0, 13.0],
            ]
        ],
        dtype=jnp.float32,
    )
    context = jnp.zeros((1, 4, 16, 16, 8), dtype=jnp.float32)
    batch = make_teacher_forced_batch(trajectory, context)
    assert batch["history_target_vals"].shape == (1, 3, 2)
    assert batch["prediction_target_vals"].shape == (1, 3, 2)
    assert jnp.allclose(batch["history_target_vals"][0, 0], jnp.array([0.0, 0.0]))
    assert jnp.allclose(
        batch["prediction_target_vals"][0, -1], jnp.array([8.0, 13.0])
    )


class DummyDataset:
    def get_context(self, times, coordinates, interpolator):
        return jnp.zeros((len(times), 16, 16, 8), dtype=jnp.float32)


class DummyPredictor:
    def apply(self, params, *, y_history, x, train=False):
        next_coord = y_history[-1] + jnp.array([0.05, 0.10], dtype=jnp.float32)
        return jnp.concatenate([y_history[:-1], next_coord[None]], axis=0)


def test_predict_rolls_out_requested_horizon() -> None:
    model = DirectPredictionModel(
        name="unit_test_direct_model",
        params={
            "prediction_length": 3,
            "model_type": "GRU",
            "context_encoder_type": "gap",
        },
    )
    model.model = DummyPredictor()
    model.nn_params = {}
    model.std = jnp.array([1.0, 1.0], dtype=jnp.float32)

    pred = model.predict(
        start_position=jnp.array([45.0, -20.0], dtype=jnp.float32),
        start_time=datetime.datetime(2020, 1, 1, 0, 0, 0),
        prediction_length=3,
        interpolator=None,
        test_dataset=DummyDataset(),
        num_samples=1,
    )
    assert pred.shape == (1, 3, 2)
    assert jnp.isfinite(pred).all()