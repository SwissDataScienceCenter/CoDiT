"""
Tests for the shared eval loop in eval_module/eval_runner.py.

These tests use mock objects so they do not require real data or disk access.
The batch format matches the actual eval script conventions:
  test_dataset[i] -> (start_position, start_time, times, interpolator, coordinates)
and model.predict() is called with positional args:
  model.predict(start_position, start_time, length, interpolator, test_dataset, num_samples=N)
"""

import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------

PRED_LENGTH = 9  # number of prediction steps (coordinates[1:] has this many rows)
N_SAMPLES = 4


class _MockModel:
    """Minimal model whose predict() returns plausible coordinate arrays."""

    def predict(
        self,
        start_position,
        start_time,
        length,
        interpolator,
        test_dataset,
        num_samples=1,
    ):
        # Returns (num_samples, length, 2) — same lat/lon as start_position
        return jnp.tile(start_position[None, None, :], (num_samples, length, 1))


class _MockDataset:
    """
    Mimics OceanDataset: len() + __getitem__ returning the 5-tuple.
    coordinates has shape (PRED_LENGTH + 1, 2) — first row is start_position,
    remaining PRED_LENGTH rows are the trajectory steps used for distance.
    """

    def __init__(self, size=3, raise_on=None):
        self._size = size
        self._raise_on = raise_on or set()

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        if idx in self._raise_on:
            raise ValueError(f"Simulated error at index {idx}")
        start_position = jnp.array([10.0, 20.0])
        start_time = None
        times = None
        interpolator = None
        # (PRED_LENGTH + 1, 2): row 0 = start, rows 1..PRED_LENGTH = trajectory
        coordinates = jnp.zeros((PRED_LENGTH + 1, 2))
        return start_position, start_time, times, interpolator, coordinates


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunEvalBasic:
    def test_returns_three_arrays(self):
        from codit.eval_module.eval_runner import run_eval

        model = _MockModel()
        dataset = _MockDataset(size=2)
        distances, pred_coords, true_coords = run_eval(
            model, dataset, indices=range(2), length=PRED_LENGTH, num_samples=N_SAMPLES
        )
        assert hasattr(distances, "shape"), "distances must be an array"
        assert hasattr(pred_coords, "shape"), "pred_coords must be an array"
        assert hasattr(true_coords, "shape"), "true_coords must be an array"

    def test_distances_shape(self):
        from codit.eval_module.eval_runner import run_eval

        model = _MockModel()
        dataset = _MockDataset(size=3)
        distances, pred_coords, true_coords = run_eval(
            model, dataset, indices=range(3), length=PRED_LENGTH, num_samples=N_SAMPLES
        )
        # distances: (N_items, num_samples, PRED_LENGTH) — per-sample per-timestep
        assert distances.shape[0] == 3, "one distance entry per valid sample"

    def test_pred_coords_shape(self):
        from codit.eval_module.eval_runner import run_eval

        model = _MockModel()
        dataset = _MockDataset(size=2)
        distances, pred_coords, true_coords = run_eval(
            model, dataset, indices=range(2), length=PRED_LENGTH, num_samples=N_SAMPLES
        )
        # pred_coords: (N_items, num_samples, PRED_LENGTH, 2)
        assert pred_coords.shape[0] == 2
        assert pred_coords.shape[-1] == 2

    def test_true_coords_shape(self):
        from codit.eval_module.eval_runner import run_eval

        model = _MockModel()
        dataset = _MockDataset(size=2)
        distances, pred_coords, true_coords = run_eval(
            model, dataset, indices=range(2), length=PRED_LENGTH, num_samples=N_SAMPLES
        )
        # true_coords: (N_items, PRED_LENGTH + 1, 2) — includes start_position row
        assert true_coords.shape[0] == 2
        assert true_coords.shape[-1] == 2


class TestRunEvalErrorHandling:
    def test_skips_indices_that_raise_value_error(self):
        from codit.eval_module.eval_runner import run_eval

        model = _MockModel()
        # indices 0 and 2 raise; only index 1 succeeds
        dataset = _MockDataset(size=3, raise_on={0, 2})
        distances, pred_coords, true_coords = run_eval(
            model, dataset, indices=range(3), length=PRED_LENGTH, num_samples=N_SAMPLES
        )
        assert distances.shape[0] == 1, "Only one valid sample should be returned"

    def test_all_errors_raises(self):
        from codit.eval_module.eval_runner import run_eval

        model = _MockModel()
        dataset = _MockDataset(size=2, raise_on={0, 1})
        with pytest.raises(ValueError, match="No valid samples"):
            run_eval(
                model,
                dataset,
                indices=range(2),
                length=PRED_LENGTH,
                num_samples=N_SAMPLES,
            )


class TestRunEvalSubsetIndices:
    def test_subset_indices_respected(self):
        from codit.eval_module.eval_runner import run_eval

        model = _MockModel()
        dataset = _MockDataset(size=10)
        indices = np.linspace(0, 9, 4, dtype=int)
        distances, pred_coords, true_coords = run_eval(
            model, dataset, indices=indices, length=PRED_LENGTH, num_samples=N_SAMPLES
        )
        assert distances.shape[0] == 4

    def test_single_index(self):
        from codit.eval_module.eval_runner import run_eval

        model = _MockModel()
        dataset = _MockDataset(size=5)
        distances, pred_coords, true_coords = run_eval(
            model, dataset, indices=[2], length=PRED_LENGTH, num_samples=N_SAMPLES
        )
        assert distances.shape[0] == 1


class TestRunEvalDefaultNumSamples:
    def test_default_num_samples_is_one(self):
        from codit.eval_module.eval_runner import run_eval

        class _TrackingModel:
            called_with_num_samples = None

            def predict(
                self,
                start_position,
                start_time,
                length,
                interpolator,
                test_dataset,
                num_samples=1,
            ):
                _TrackingModel.called_with_num_samples = num_samples
                return jnp.tile(start_position[None, None, :], (num_samples, length, 1))

        model = _TrackingModel()
        dataset = _MockDataset(size=1)
        run_eval(model, dataset, indices=[0], length=PRED_LENGTH)
        assert _TrackingModel.called_with_num_samples == 1
