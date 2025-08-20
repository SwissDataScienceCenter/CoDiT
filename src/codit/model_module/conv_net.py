import os

import jax
import optax
import orbax.checkpoint as ocp
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from interpax import Interpolator3D
from jax import numpy as jnp
from jax import vmap
from torch.utils import data
from tqdm import tqdm

from codit import config_paths
from codit.data_module.dataset import OceanDataset, OceanTrainingDataset
from codit.data_module import data_transform
from codit.model_module.base_model import BaseTrajectoryModel


def merge_state(dst: nnx.State, src: nnx.State) -> None:
    for k, v in src.items():
        if isinstance(v, nnx.State):
            merge_state(dst[k], v)
        else:
            dst[k] = v


class ConvNet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        """
        Initializes the convolutional neural network layers.

        Args:
            rngs (nnx.Rngs): Random number generators for initializing layer parameters.

        Attributes:
            Conv1 (nnx.Conv): First convolutional layer with 8 input features, 16 output features, 3x3 kernel, and stride 2.
            Conv2 (nnx.Conv): Second convolutional layer with 16 input features, 32 output features, 3x3 kernel, and stride 2.
            Linear1 (nnx.Linear): Fully connected layer with 512 input features and 128 output features.
            Dropout (nnx.Dropout): Dropout layer with a dropout rate of 0.2.
            Linear2 (nnx.Linear): Output fully connected layer with 128 input features and 2 output features.
        """
        self.Conv1 = nnx.Conv(
            in_features=8, out_features=16, kernel_size=(3, 3), strides=2, rngs=rngs
        )
        self.Conv2 = nnx.Conv(
            in_features=16, out_features=32, kernel_size=(3, 3), strides=2, rngs=rngs
        )
        self.Linear1 = nnx.Linear(in_features=512, out_features=64, rngs=rngs)
        self.Dropout = nnx.Dropout(rate=0.2, rngs=rngs)
        self.Linear2 = nnx.Linear(in_features=64, out_features=2, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        x = self.Conv1(x)
        x = nnx.relu(x)
        # x = self.Dropout(x)
        x = self.Conv2(x)
        x = nnx.relu(x)
        # x = self.Dropout(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.Linear1(x)
        x = nnx.relu(x)
        x = self.Dropout(x)
        x = self.Linear2(x)
        return x


def loss(model: ConvNet, context: jnp.ndarray, trajectory: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the mean squared error loss between the predicted and true increments of a trajectory.

    Args:
        model: A callable that takes the context and returns predicted increments.
        context: The input context for the model, typically representing the current state or features.
        trajectory: A sequence (e.g., array) of positions over time.

    Returns:
        jnp.ndarray: The mean squared error between the model's predicted increments and the true increments computed from the trajectory.
    """
    true_increments = vmap(data_transform.velocity_vector_from_pos_time, in_axes=(0, 0, None))(
        trajectory[:-1], trajectory[1:], 3600
    )
    increment_prediction = model(context)
    return jnp.mean(jnp.pow(increment_prediction - true_increments, 2))


def load_checkpoint(model: nnx.Module, ckpt_dir: str) -> nnx.Module:
    """
    Loads the model state from a checkpoint directory.

    Args:
        model: The model to load the state into.
        ckpt_dir (str): The directory where the checkpoint is stored.

    Returns:
        nnx.Module: The model with the loaded state.
    """
    with ocp.CheckpointManager(
        ckpt_dir,
        item_names=("state", "opt_state"),
    ) as ckpt_mgr:
        graphdef, abstract_state = nnx.split(model)
        rngs_key = jax.tree.map(jax.random.key_data, abstract_state.filter(nnx.RngKey))
        merge_state(abstract_state, rngs_key)

        restored = ckpt_mgr.restore(
            1,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_state),
            ),
        )
    state_restored = jax.device_put(restored["state"], jax.devices()[0])
    return nnx.merge(graphdef, state_restored)


class ConvNetModel(BaseTrajectoryModel):

    def __init__(self, name: str, params: dict):
        """
        Initializes the ConvNet model module.

        Args:
            name (str): The name of the model instance.
            params (dict): A dictionary of parameters for the model.

        Attributes:
            type (str): The type of the model, set to "conv_net".
            params (dict): The parameters for the model.
            model (ConvNet): The ConvNet model instance.
            ckpt_dir (str): Directory path for model checkpoints.
            trained_sucessfully (bool): Indicates whether the model was successfully restored from a checkpoint.

        Raises:
            Exception: If restoring the model state from checkpoint fails, sets `trained_sucessfully` to False.
        """
        super().__init__(name, params)
        self.type = "conv_net"
        self.params = params
        self.model = ConvNet(rngs=nnx.Rngs(0))
        self.ckpt_dir = os.path.join(config_paths.MODEL_PATH, "ckpts", self.name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        try:
            self.model = load_checkpoint(self.model, self.ckpt_dir)
            self.trained_sucessfully = True
        except Exception as e:
            print(f"Error restoring model state: {e}")
            self.trained_sucessfully = False

    def _fit(
        self,
        train_dataset: OceanTrainingDataset,
        val_dataset: OceanTrainingDataset,
        train_params: dict = {},
    ) -> "ConvNetModel":

        ckpt_dir = ocp.test_utils.erase_and_create_empty(self.ckpt_dir)
        train_dataloader = data.DataLoader(
            train_dataset, batch_size=train_params.get("batch_size", 32), shuffle=True
        )
        val_dataloader = data.DataLoader(
            val_dataset, batch_size=train_params.get("batch_size", 128), shuffle=False
        )
        optimizer = nnx.Optimizer(self.model, optax.adam(train_params.get("lr", 1e-3)))

        @nnx.jit
        def train_step(
            model: nnx.Module,
            optimizer: nnx.Optimizer,
            trajectory: jnp.ndarray,
            context: jnp.ndarray,
        ) -> jnp.ndarray:
            def loss_fn(
                model: nnx.Module, trajectory: jnp.ndarray, context: jnp.ndarray
            ) -> jnp.ndarray:
                true_increments = vmap(
                    vmap(data_transform.velocity_vector_from_pos_time, in_axes=(0, 0, None)),
                    in_axes=(0, 0, None),
                )(trajectory[:, :-1], trajectory[:, 1:], 3600)
                increment_prediction = vmap(model)(context[:, :-1])
                return jnp.mean(jnp.pow(increment_prediction - true_increments, 2))

            loss, grad = nnx.value_and_grad(loss_fn)(model, trajectory, context)
            optimizer = optimizer.update(grad)
            return loss

        @nnx.jit
        def eval_step(
            model: nnx.Module, trajectory: jnp.ndarray, context: jnp.ndarray
        ) -> jnp.ndarray:
            def loss_fn(model: nnx.Module) -> jnp.ndarray:
                true_increments = vmap(
                    vmap(data_transform.velocity_vector_from_pos_time, in_axes=(0, 0, None)),
                    in_axes=(0, 0, None),
                )(trajectory[:, :-1], trajectory[:, 1:], 3600)
                increment_prediction = vmap(model)(context[:, :-1])
                return jnp.mean(jnp.pow(increment_prediction - true_increments, 2))

            loss = loss_fn(model)
            return loss

        num_batches_per_epoch = train_params.get("num_batches_per_epoch", 100)
        num_epochs = train_params.get("num_epochs", 100)
        early_stop = EarlyStopping(
            min_delta=train_params.get("min_delta", 0.001),
            patience=train_params.get("patience", 5),
        )
        for epoch in range(num_epochs):
            for i, batch in tqdm(enumerate(train_dataloader)):
                trajectory, context = jnp.asarray(batch[0]), jnp.asarray(batch[1])
                train_step(self.model, optimizer, trajectory, context)
                if i >= num_batches_per_epoch:
                    break

            with ocp.CheckpointManager(
                ckpt_dir,
                item_names=("state", "opt_state"),
            ) as ckpt_mgr:
                _, state = nnx.split(self.model)
                rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
                merge_state(state, rngs_key)

                _, opt_state = nnx.split(optimizer.opt_state)
                ckpt_mgr.save(
                    1,
                    args=ocp.args.Composite(
                        state=ocp.args.StandardSave(state),
                        opt_state=ocp.args.StandardSave(opt_state),
                    ),
                )
            val_loss = 0
            for i, batch in tqdm(enumerate(val_dataloader)):
                trajectory, context = jnp.asarray(batch[0]), jnp.asarray(batch[1])
                val_loss += eval_step(self.model, trajectory, context)
            val_loss /= i * batch[0].shape[0] * batch[0].shape[1]
            early_stop.update(val_loss)

            if early_stop.should_stop:
                break
            print(f"Epoch {epoch} - Val Loss: {val_loss}")
        return self

    def load_trained_model(self) -> None:
        self.model = load_checkpoint(self.model, self.ckpt_dir)

    def predict(
        self,
        start_position: jnp.ndarray,
        times: jnp.ndarray,
        interpolator: Interpolator3D,
        test_dataset: OceanDataset,
        num_samples: int = 1,
    ) -> jnp.ndarray:
        num_preds = num_samples
        cur_coord = jnp.array([start_position])
        cur_coord = jnp.repeat(cur_coord, num_preds, axis=0)

        def pred_trajectory(
            model: nnx.Module,
            start_position: jnp.ndarray,
            times: jnp.ndarray,
            interpolator: Interpolator3D,
            i: int,
        ) -> tuple[nnx.Module, jnp.ndarray]:
            @nnx.scan
            def scan_fn(
                carry: tuple[nnx.Module, jnp.ndarray], t: jnp.ndarray
            ) -> tuple[tuple[nnx.Module, jnp.ndarray], jnp.ndarray]:
                model, cur_coord = carry[0], carry[1]
                context = test_dataset.get_context(
                    jnp.array([t]), cur_coord, interpolator
                )
                pred_vel = model(context)
                speed, bearing = data_transform.velocity_to_distance_bearing(pred_vel[0])
                new_coord = data_transform.inverse_haversine(
                    cur_coord[0], 3600 * speed, bearing
                )
                return (
                    model,
                    jnp.array([new_coord]),
                ), new_coord

            init = (
                model,
                jnp.array([start_position]),
            )

            _, pred_coords = scan_fn(init, times)
            return model, pred_coords

        model, pred_coords = nnx.scan(
            lambda model, i: pred_trajectory(
                model, start_position, times[:-1], interpolator, i
            )
        )(self.model, jnp.arange(num_preds))

        return pred_coords
