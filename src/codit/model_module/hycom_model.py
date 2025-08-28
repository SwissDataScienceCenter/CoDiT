import datetime

from flax import linen as nn
from flax.core import FrozenDict
from interpax import Interpolator3D
from jax import lax
from jax import numpy as jnp

from codit.data_module import data_transform
from codit.data_module.dataset import OceanDataset
from codit.model_module.base_model import BaseTrajectoryModel


class HycomDriftDynamics(nn.Module):
    
    @nn.compact
    def __call__(
        self, x: jnp.ndarray, t: jnp.ndarray, interpolator: Interpolator3D, dt: float
    ) -> jnp.ndarray:
        water_velocity = interpolator(t, x[0], x[1])[..., [1, 0]]
        speed, bearing = data_transform.velocity_to_distance_bearing(water_velocity)
        dcoord = data_transform.inverse_haversine_grad(x, speed * dt, bearing)
        return dcoord


def predict_trajectory(
    params: dict | FrozenDict,
    model: HycomDriftDynamics,
    x0: jnp.ndarray,
    t: jnp.ndarray,
    inputs: Interpolator3D,
) -> jnp.ndarray:
    dt = (t[1] - t[0]) * 3600

    def dynamics(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return model.apply(params, x, t, inputs, dt) # type: ignore

    def scan_fun(carry: tuple, t: jnp.ndarray) -> tuple:
        x, v = carry
        dcoord = dynamics(x, t)
        x = x + dcoord
        return (x, dcoord), (x, dcoord)

    _, (predicted_trajectory, _) = lax.scan(scan_fun, (x0, jnp.zeros_like(x0)), t[:-1])
    return predicted_trajectory


class HycomModel(BaseTrajectoryModel):

    def __init__(self, name: str, params: dict):
        super().__init__(name, params)
        self.type = "hycom"
        self.model = HycomDriftDynamics()

    def _fit(self, train_configs: list) -> "HycomModel":
        self.model = HycomDriftDynamics()
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
        **kwargs: dict
    ) -> jnp.ndarray:

        times = self._get_prediction_steps(start_time, prediction_length)
        x_pred = predict_trajectory({}, self.model, start_position, times, interpolator)

        pred_coords = x_pred[None]
        return pred_coords