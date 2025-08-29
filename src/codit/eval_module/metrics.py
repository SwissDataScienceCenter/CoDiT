import numpy as np
from jax import numpy as jnp
from jax import vmap

from codit.data_module import data_transform


def get_energy_score(true_coords: jnp.ndarray, pred_coords: jnp.ndarray) -> jnp.ndarray:
    """Calculate the energy score between true and predicted coordinates.

    Args:
        true_coords: True coordinates of the trajectory. (N, T, 2)
        pred_coords: Predicted coordinates of the trajectory. (N, T, 2)

    Returns:
        Energy score between true and predicted coordinates.
    """
    num_samples = jnp.sum(jnp.prod(~jnp.isnan(pred_coords), axis=-1), axis=1)
    expected_error = (
        jnp.nansum(
            vmap(vmap(vmap(data_transform.haversine_distance), in_axes=(None, 0)))(
                true_coords, pred_coords
            ),
            axis=1,
        )
        / num_samples
    )
    num_traj = true_coords.shape[0]
    expected_cross_distance = []
    for itraj in range(num_traj):
        expected_distance_pred = vmap(
            vmap(vmap(data_transform.haversine_distance), in_axes=(0, None)),
            in_axes=(None, 0),
        )(pred_coords[itraj], pred_coords[itraj])
        expected_cross_distance.append(
            jnp.nansum(expected_distance_pred, axis=(0, 1)) / num_samples[itraj] ** 2
        )
    expected_cross_distance = jnp.array(expected_cross_distance)
    return expected_error - 0.5 * expected_cross_distance

def get_haversine_error(
    true_coords: jnp.ndarray, pred_coords: jnp.ndarray, aggregation: str = "median"
) -> jnp.ndarray:
    """Calculate the haversine error between true and predicted coordinates.

    Args:
        true_coords: True coordinates of the trajectory. (N, T, 2)
        pred_coords: Predicted coordinates of the trajectory. (N, T, 2)
        aggregation: Aggregation method to use. (median or mean)

    Returns:
        Haversine error between true and predicted coordinates.
    """
    if aggregation == "median":
        aggr_coords = np.nanmedian(pred_coords, axis=1)
    elif aggregation == "mean":
        aggr_coords = np.nanmean(pred_coords, axis=1)
    return vmap(vmap(data_transform.haversine_distance))(true_coords, aggr_coords)


def get_skill_score(
    true_coords: jnp.ndarray,
    pred_coords: jnp.ndarray,
    threshold: float = 1,
    aggregation: str = "median",
) -> jnp.ndarray:
    """Calculate the skill score between true and predicted coordinates.

    Args:
        true_coords: True coordinates of the trajectory. (N, T, 2)
        pred_coords: Predicted coordinates of the trajectory. (N, T, 2)
        threshold: Threshold for skill score. Defaults to 1.
        aggregation: Aggregation method to use. (median or mean)

    Returns:
        Skill score between true and predicted coordinates.
    """
    if aggregation == "median":
        aggr_coords = np.nanmedian(pred_coords, axis=1)
    elif aggregation == "mean":
        aggr_coords = np.nanmean(pred_coords, axis=1)
    partial_lengths = vmap(vmap(data_transform.haversine_distance, in_axes=(-2, -2)))(
        true_coords[..., :-1, :], true_coords[..., 1:, :]
    )
    lengths = jnp.cumsum(partial_lengths, axis=-1)
    cum_lengths = np.cumsum(lengths, axis=-1)
    distances = vmap(vmap(data_transform.haversine_distance, in_axes=(-2, -2)))(
        aggr_coords, true_coords[..., 1:, :]
    )
    cum_distances = jnp.cumsum(distances, axis=-1)
    c = cum_distances / cum_lengths
    skill_score = jnp.maximum(1.0 - c / threshold, 0)
    return skill_score


def get_metrics(
    true_coords: jnp.ndarray, pred_coords: jnp.ndarray, aggregation: str = "median"
) -> dict:
    """Calculate evaluation metrics for the trajectory.

    Args:
        true_coords: True coordinates of the trajectory. (N, T, 2)
        pred_coords: Predicted coordinates of the trajectory. (N, T, 2)
        aggregation: Aggregation method to use. (median or mean). Defaults to "median".

    Returns:
        Dictionary of evaluation metrics.
    """
    mean_distance = get_haversine_error(
        true_coords[..., 1:, :], pred_coords, aggregation=aggregation
    )
    energy_score = get_energy_score(true_coords[..., 1:, :], pred_coords)
    skill_score = get_skill_score(true_coords, pred_coords, aggregation=aggregation)
    return {
        "mean_distance": mean_distance,
        "energy_score": energy_score,
        "skill_score": skill_score,
    }