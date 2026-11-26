import numpy as np
from jax import numpy as jnp
from jax import vmap

from codit.data_module import data_transform


def _ensure_4d(pred_coords: jnp.ndarray) -> jnp.ndarray:
    """Promote a 3-D ``(N, T, 2)`` array to 4-D ``(N, 1, T, 2)``.

    Production callers always pass ``(N, S, T, 2)``.  This helper makes the
    single-sample case a special case of the general one so the rest of the
    code only has to handle the 4-D layout.
    """
    if pred_coords.ndim == 3:
        return pred_coords[:, None, :, :]  # (N, 1, T, 2)
    return pred_coords  # already (N, S, T, 2)


def _dist_to_samples(traj: jnp.ndarray, samples: jnp.ndarray) -> jnp.ndarray:
    """Per-timestep distance of one ``(T, 2)`` trajectory to ``(S, T, 2)`` samples.

    Returns shape ``(S, T)``.
    """
    return vmap(vmap(data_transform.haversine_distance), in_axes=(None, 0))(
        traj, samples
    )


def get_energy_score(true_coords: jnp.ndarray, pred_coords: jnp.ndarray) -> jnp.ndarray:
    """Calculate the energy score between true and predicted coordinates.

    Args:
        true_coords: True coordinates of the trajectory. Shape ``(N, T, 2)``.
        pred_coords: Predicted coordinates of the trajectory.
            Shape ``(N, S, T, 2)`` or ``(N, T, 2)`` for a single sample.

    Returns:
        Energy score per trajectory and timestep, shape ``(N, T)``. Samples
        are used at every timestep where they are finite (NaN samples, e.g.
        rollouts that left the context grid, are skipped per timestep). The
        score is NaN where the truth is NaN or no sample is finite.
    """
    pred_coords = _ensure_4d(pred_coords)  # (N, S, T, 2)
    # A sample is valid at a timestep if both its coordinates are finite.
    # Samples become NaN once a rollout leaves the context grid, so validity
    # is decided per timestep: a sample counts where it is valid and nowhere
    # else, in both the numerators and the denominators.
    valid = jnp.all(~jnp.isnan(pred_coords), axis=-1)  # (N, S, T)
    num_valid = jnp.sum(valid, axis=1)  # (N, T)

    # E|X - y|: mean distance of the valid samples to the truth, (N, T)
    dists = vmap(_dist_to_samples)(true_coords, pred_coords)  # (N, S, T)
    expected_error = jnp.sum(jnp.where(valid, dists, 0.0), axis=1) / num_valid

    # E|X - X'|: mean pairwise distance over pairs of valid samples, (N, T).
    # V-statistic (divides by num_valid**2, self-pairs included), as before.
    num_traj = true_coords.shape[0]
    expected_cross_distance = []
    for itraj in range(num_traj):
        pred_samples = pred_coords[itraj]  # (S, T, 2)
        valid_traj = valid[itraj]  # (S, T)
        # pairwise distances between all S samples, per timestep: (S, S, T)
        pred_distances = vmap(_dist_to_samples, in_axes=(0, None))(
            pred_samples, pred_samples
        )
        pair_valid = valid_traj[:, None, :] & valid_traj[None, :, :]  # (S, S, T)
        cross_dist = jnp.sum(
            jnp.where(pair_valid, pred_distances, 0.0), axis=(0, 1)
        ) / (num_valid[itraj] ** 2)
        expected_cross_distance.append(cross_dist)
    expected_cross_distance = jnp.array(expected_cross_distance)  # (N, T)
    # NaN where the truth is NaN or no sample is valid (0 / 0).
    return expected_error - 0.5 * expected_cross_distance


def get_haversine_error(
    true_coords: jnp.ndarray, pred_coords: jnp.ndarray, aggregation: str = "median"
) -> jnp.ndarray:
    """Calculate the haversine error between true and predicted coordinates.

    Args:
        true_coords: True coordinates of the trajectory. Shape ``(N, T, 2)``.
        pred_coords: Predicted coordinates of the trajectory.
            Shape ``(N, S, T, 2)`` or ``(N, T, 2)`` for a single sample.
        aggregation: Aggregation method over samples. ``"median"`` or ``"mean"``.

    Returns:
        Haversine error per trajectory and timestep, shape ``(N, T)``.
    """
    pred_coords = _ensure_4d(pred_coords)  # (N, S, T, 2)
    if aggregation == "median":
        aggr_coords = np.nanmedian(pred_coords, axis=1)  # (N, T, 2)
    elif aggregation == "mean":
        aggr_coords = np.nanmean(pred_coords, axis=1)  # (N, T, 2)
    else:
        raise ValueError(
            f"Unknown aggregation: {aggregation!r}. Use 'median' or 'mean'."
        )
    return vmap(vmap(data_transform.haversine_distance))(true_coords, aggr_coords)


def get_skill_score(
    true_coords: jnp.ndarray,
    pred_coords: jnp.ndarray,
    threshold: float = 1,
    aggregation: str = "median",
) -> jnp.ndarray:
    """Calculate the skill score between true and predicted coordinates.

    Args:
        true_coords: True coordinates of the trajectory. Shape ``(N, T, 2)``.
        pred_coords: Predicted coordinates of the trajectory.
            Shape ``(N, S, T, 2)`` or ``(N, T, 2)`` for a single sample.
        threshold: Threshold for skill score. Defaults to 1.
        aggregation: Aggregation method over samples. ``"median"`` or ``"mean"``.

    Returns:
        Skill score per trajectory and timestep (excluding t=0), shape ``(N, T-1)``.
    """
    pred_coords = _ensure_4d(pred_coords)  # (N, S, T, 2)
    if aggregation == "median":
        aggr_coords = np.nanmedian(pred_coords, axis=1)  # (N, T, 2)
    elif aggregation == "mean":
        aggr_coords = np.nanmean(pred_coords, axis=1)  # (N, T, 2)
    else:
        raise ValueError(
            f"Unknown aggregation: {aggregation!r}. Use 'median' or 'mean'."
        )
    # Compute distances between consecutive true points
    partial_lengths = vmap(vmap(data_transform.haversine_distance))(
        true_coords[..., :-1, :], true_coords[..., 1:, :]
    )
    lengths = jnp.cumsum(partial_lengths, axis=-1)
    cum_lengths = np.cumsum(lengths, axis=-1)
    # Compute per-timestep distances from aggregated prediction to true points: (N, T-1)
    distances = vmap(vmap(data_transform.haversine_distance))(
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
        true_coords: True coordinates of the trajectory. Shape ``(N, T, 2)``.
        pred_coords: Predicted coordinates of the trajectory.
            Shape ``(N, S, T, 2)`` or ``(N, T, 2)`` for a single sample.
        aggregation: Aggregation method over samples. ``"median"`` or ``"mean"``.
            Defaults to ``"median"``.

    Returns:
        Dictionary with keys ``"mean_distance"``, ``"energy_score"``,
        ``"skill_score"``.  All values have shape ``(N, T-1)`` (the starting
        position at ``t=0`` is excluded).
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
