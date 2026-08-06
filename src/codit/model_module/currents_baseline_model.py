import datetime
from typing import Sequence

from flax import linen as nn
from flax.core import FrozenDict
from interpax import Interpolator3D
from jax import lax
from jax import numpy as jnp

from codit.data_module import data_transform
from codit.data_module.dataset import OceanDataset
from codit.model_module.base_model import BaseTrajectoryModel

# Mapping from source name to the velocity channel indices used in the interpolator output.
# HYCOM stores (u, v) at positions 0-1; GLORYS/Copernicus stores (u, v) at positions 2-3.
_VELOCITY_INDICES: dict[str, Sequence[int]] = {
    "hycom": [1, 0],
    "glorys": [3, 2],
}


class CurrentsDriftDynamics(nn.Module):
    """Flax module that advances a particle by one time step using ocean current data.

    Attributes:
        velocity_indices: Channel indices to extract (u, v) from the interpolator output.
            Use [1, 0] for HYCOM and [3, 2] for GLORYS/Copernicus.
    """

    velocity_indices: Sequence[int]

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        interpolator: Interpolator3D,
        dt: float,
    ) -> jnp.ndarray:
        water_velocity = interpolator(t, x[0], x[1])[..., self.velocity_indices]
        speed, bearing = data_transform.velocity_to_distance_bearing(water_velocity)
        dcoord = data_transform.inverse_haversine_grad(x, speed * dt, bearing)
        return dcoord


def predict_trajectory(
    params: dict | FrozenDict,
    model: CurrentsDriftDynamics,
    x0: jnp.ndarray,
    t: jnp.ndarray,
    inputs: Interpolator3D,
) -> jnp.ndarray:
    dt = (t[1] - t[0]) * 3600

    def dynamics(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return model.apply(params, x, t, inputs, dt)  # type: ignore

    def scan_fun(carry: tuple, t: jnp.ndarray) -> tuple:
        x, v = carry
        dcoord = dynamics(x, t)
        x = x + dcoord
        return (x, dcoord), (x, dcoord)

    _, (predicted_trajectory, _) = lax.scan(scan_fun, (x0, jnp.zeros_like(x0)), t[:-1])
    return predicted_trajectory


class CurrentsBaselineModel(BaseTrajectoryModel):
    """Physics-based baseline: advect a particle using ocean current data.

    Args:
        name: Model name.
        params: Model parameter dictionary (passed to BaseTrajectoryModel).
        source: Which current dataset to use.  Must be ``"hycom"`` or ``"glorys"``.

    The ``type`` attribute is set to ``f"{source}_model"`` so that, e.g.,
    ``HycomModel().type == "hycom_model"``.
    """

    def __init__(self, name: str, params: dict, source: str) -> None:
        if source not in _VELOCITY_INDICES:
            raise ValueError(
                f"Unknown source {source!r}. Expected one of {list(_VELOCITY_INDICES)}."
            )
        super().__init__(name, params)
        self.source = source
        self.type = f"{source}_model"
        self.model = CurrentsDriftDynamics(velocity_indices=_VELOCITY_INDICES[source])

    def _fit(
        self,
        train_dataloader,
        val_dataloader,
        train_params: dict | None = None,
    ) -> "CurrentsBaselineModel":
        return self

    def _create_data(self, datasets: dict, trajectory_length: int) -> tuple:
        data = []
        data_time = []
        for dataset in datasets.values():
            for k in dataset["trajectories"].keys():
                x = dataset["trajectories"][k]
                if x.shape[0] == (trajectory_length + 1):
                    batch_time = x[:, 0]
                    batch = x[:, 1:]
                    data.append(batch[None])
                    data_time.append(batch_time[None])
        data = jnp.concatenate(data, axis=0)
        data_time = jnp.concatenate(data_time, axis=0)
        return data, data_time

    def predict(
        self,
        start_position: jnp.ndarray,
        start_time: datetime.datetime,
        prediction_length: int,
        interpolator: Interpolator3D,
        test_dataset: OceanDataset,
        **kwargs: dict,
    ) -> jnp.ndarray:
        times = self._get_prediction_steps(start_time, prediction_length)
        x_pred = predict_trajectory({}, self.model, start_position, times, interpolator)
        pred_coords = x_pred[None]
        return pred_coords

    def export_trained_model(
        self, model_name: str | None = None, export_path: str | None = None
    ) -> None:
        """Exports the trained model to a specified path.

        Args:
            model_name: The name of the model to export. If None, uses the model's name.
            export_path: The directory where the model will be exported. If None, uses a
                default path.

        Raises:
            ValueError: If the model parameters are not set.
        """
        pass

    @classmethod
    def import_trained_model(
        cls, model_name: str, import_path: str | None
    ) -> "CurrentsBaselineModel":
        """Imports a trained model from the specified path.

        Args:
            model_name: The name of the model to import.
            import_path: The path from which to import the model. If None, uses a default
                path.

        Returns:
            CurrentsBaselineModel: An instance of the model with the loaded parameters.

        Raises:
            FileNotFoundError: If the configuration file or model file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML configuration file.
            pickle.UnpicklingError: If there is an error unpickling the model file.
        """
        raise NotImplementedError(
            "import_trained_model must be called on a concrete subclass "
            "(HycomModel or GlorysModel)."
        )
