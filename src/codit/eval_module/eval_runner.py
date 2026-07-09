"""
Shared evaluation loop for CoDiT eval scripts.

Both ``codit_eval.py`` and ``lingauss_eval.py`` iterate over a test dataset,
call ``model.predict()``, and accumulate per-sample haversine distances together
with predicted and true coordinates.  This module extracts that inner loop so
the two scripts stay in sync.
"""

from __future__ import annotations

from typing import Any

from jax import numpy as jnp
from jax import vmap

from codit.data_module import data_transform


def run_eval(
    model: Any,
    dataset: Any,
    indices: Any,
    length: int,
    num_samples: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the evaluation loop over a dataset and return accumulated results.

    For each index in *indices* the function calls ``dataset[idx]`` to obtain
    the 5-tuple ``(start_position, start_time, times, interpolator, coordinates)``
    that ``OceanDataset.__getitem__`` returns, then calls ``model.predict()``
    with the corresponding positional arguments.

    Haversine distances are computed per sample and per timestep between the
    predicted coordinates and ``coordinates[1:]`` (the ground-truth trajectory
    excluding the starting position).

    Args:
        model: Any model that implements
            ``predict(start_position, start_time, length, interpolator,
            test_dataset, num_samples=N) -> jnp.ndarray`` of shape
            ``(num_samples, length, 2)``.
        dataset: An indexable object that returns
            ``(start_position, start_time, times, interpolator, coordinates)``
            on ``__getitem__``.  ``coordinates`` must have shape
            ``(length + 1, 2)``.
        indices: Iterable of integer indices to evaluate (e.g. ``range(len(dataset))``
            or a sub-sampled array from ``np.linspace``).
        length: Number of prediction steps (passed directly to ``model.predict``).
        num_samples: Number of trajectory samples to draw from the model.
            Defaults to 1.

    Returns:
        A 3-tuple ``(distances, pred_coords, true_coords)`` where

        * **distances** — ``jnp.ndarray`` of shape ``(N, num_samples, length)``,
          per-sample per-timestep haversine distance in metres.
        * **pred_coords** — ``jnp.ndarray`` of shape ``(N, num_samples, length, 2)``,
          predicted lat/lon coordinates.
        * **true_coords** — ``jnp.ndarray`` of shape ``(N, length + 1, 2)``,
          ground-truth coordinates (including the starting position at row 0).

    Raises:
        ValueError: If *all* indices raised ``ValueError`` during evaluation
            (i.e. no valid samples could be produced).
    """
    distances_list: list[jnp.ndarray] = []
    pred_coords_list: list[jnp.ndarray] = []
    true_coords_list: list[jnp.ndarray] = []

    for idx in indices:
        try:
            start_position, start_time, times, interpolator, coordinates = dataset[idx]
            pred_coord = model.predict(
                start_position,
                start_time,
                length,
                interpolator,
                dataset,
                num_samples=num_samples,
            )
            # distance shape: (num_samples, length)
            distance = vmap(vmap(data_transform.haversine_distance), in_axes=(None, 0))(
                coordinates[1:, :], pred_coord[:, :]
            )
            distances_list.append(distance[None])
            pred_coords_list.append(pred_coord[None])
            true_coords_list.append(coordinates[:, :][None])
        except ValueError:
            continue

    if not distances_list:
        raise ValueError(
            "No valid samples were produced during evaluation. "
            "All indices raised ValueError."
        )

    distances = jnp.concatenate(distances_list, axis=0)
    pred_coords = jnp.concatenate(pred_coords_list, axis=0)
    true_coords = jnp.concatenate(true_coords_list, axis=0)
    return distances, pred_coords, true_coords
