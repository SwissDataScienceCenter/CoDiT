import datetime
import os
import pickle

# from anyio import key
import jax
import optax
import orbax.checkpoint as ocp
import yaml
from jax import random
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from interpax import Interpolator3D
from jax import numpy as jnp
from jax import vmap
from torch.utils import data
from tqdm import tqdm

from codit.data_module import config_paths, data_transform
from codit.data_module.dataset import OceanDataset, OceanTrainingDataset
from codit.model_module.base_model import BaseTrajectoryModel


def merge_state(dst: nnx.State, src: nnx.State) -> None:
    for k, v in src.items():
        if isinstance(v, nnx.State):
            merge_state(dst[k], v)
        else:
            dst[k] = v


class LinearGaussModule(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        """
        Initializes the convolutional neural network layers.

        Args:
            rngs (nnx.Rngs): Random number generators for initializing layer parameters.

        Attributes:
            alpha (nnx.Param): Learnable parameter for wind influence.
            log_sigma (nnx.Param): Learnable parameter for log standard deviation of Gaussian noise.
        """
        self.alpha = nnx.Param(jnp.zeros((1,)))
        self.log_sigma = nnx.Param(jnp.zeros((1,)))
        # self.Conv1 = nnx.Conv(
        #     in_features=in_features, out_features=16, kernel_size=(3, 3), strides=2, rngs=rngs
        # )
        # self.Conv2 = nnx.Conv(
        #     in_features=16, out_features=32, kernel_size=(3, 3), strides=2, rngs=rngs
        # )
        # self.Linear1 = nnx.Linear(in_features=512, out_features=64, rngs=rngs)
        # self.Dropout = nnx.Dropout(rate=0.2, rngs=rngs)
        # self.Linear2 = nnx.Linear(in_features=64, out_features=2, rngs=rngs)

    def get_mean_velocity(self, context: jax.Array) -> jnp.ndarray:
        # get the center of the context features for axis=-2 and -3
        grid_size = context.shape[-2]
        half_gridsize = grid_size // 2
        if grid_size % 2 == 0:
            center_context = context[
                ...,
                half_gridsize - 1 : half_gridsize + 1,
                half_gridsize - 1 : half_gridsize + 1,
                :,
            ].mean(axis=(-2, -3))
        else:
            center_context = context[..., half_gridsize, half_gridsize, :]

        water_velocity = center_context[..., [3, 2]]
        wind_velocity = center_context[..., [5, 4]]
        mean_velocity = water_velocity + self.alpha * wind_velocity
        return mean_velocity

    def __call__(self, context: jax.Array, key: jax.Array) -> jnp.ndarray:
        mean_velocity = self.get_mean_velocity(context)
        sigma = jnp.exp(self.log_sigma)
        rand_vec = random.normal(key=key, shape=mean_velocity.shape)
        sampled_velocity = mean_velocity + sigma * rand_vec
        return sampled_velocity


def gauss_loss(
    true_increments: jnp.ndarray,
    pred_mean_velocity: jnp.ndarray,
    log_sigma: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes the mean squared error loss between the predicted and true increments of a trajectory.

    Args:
        model: A callable that takes the context and returns predicted increments.
        context: The input context for the model, typically representing the current state or features.
        trajectory: A sequence (e.g., array) of positions over time.

    Returns:
        jnp.ndarray: The mean squared error between the model's predicted increments and the true increments computed from the trajectory.
    """
    # true_increments = vmap(data_transform.velocity_vector_from_pos_time, in_axes=(0, 0, None))(
    #     trajectory[:-1], trajectory[1:], 3600
    # )
    # mean_velocity = model.get_mean_velocity(context)
    gauss_loss = true_increments.shape[-1] * log_sigma + 0.5 * jnp.sum(
        jnp.pow((true_increments - pred_mean_velocity) * jnp.exp(-log_sigma), 2),
        axis=-1,
    )
    return jnp.mean(gauss_loss)
    # return jnp.mean(jnp.pow(mean_velocity - true_increments, 2))


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
        rngs_key = jax.tree.map(jax.random.key_data, abstract_state.filter(nnx.RngKey))  # type: ignore
        merge_state(abstract_state, rngs_key)  # type: ignore

        restored = ckpt_mgr.restore(
            1,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_state),  # type: ignore
            ),
        )
    state_restored = jax.device_put(restored["state"], jax.devices()[0])
    return nnx.merge(graphdef, state_restored)


class LinearGaussModel(BaseTrajectoryModel):
    """Linear Gaussian velocity baseline (WDM in the paper).

    Predicts the next-step displacement as ``alpha * v_context``, where
    ``v_context`` is the ocean/wind velocity at the drifter's current position
    read from the conditioning context, and ``alpha`` is a scalar coefficient
    learned from the training trajectories. Prediction uncertainty is a
    Gaussian with an isotropic learned standard deviation
    ``sigma = exp(log_sigma)``. Trajectories are integrated by rolling the
    predicted velocity forward one hour at a time via the inverse haversine
    formula.

    The model has exactly two trainable parameters (``alpha`` and
    ``log_sigma``) and serves as the simplest sanity-check baseline in the
    paper.

    Args:
        name: Unique model identifier used as the checkpoint directory name.
        params: Reserved for future extensions; currently unused.
        load_ckpt: If ``True``, attempt to restore ``alpha`` and ``log_sigma``
            from an existing checkpoint on construction.

    Example::

        model = LinearGaussModel(
            name="linear_gauss_open_ocean",
            params={"prediction_length": 72},
        )
    """

    def __init__(self, name: str, params: dict, load_ckpt: bool = False):
        """Initialise the Linear Gaussian model.

        Args:
            name: Unique model identifier used for checkpoint directory naming.
            params: Model parameters (currently unused; reserved for future extensions).
            load_ckpt: If ``True``, attempt to restore parameters from an existing
                checkpoint at ``{config_paths.MODEL_PATH}/ckpts/{name}/`` on construction.

        Attributes:
            type (str): Fixed to ``"linear_gauss"``.
            model (LinearGaussModule): The Flax NNX module containing ``alpha`` and ``log_sigma``.
            ckpt_dir (str): Path to the checkpoint directory.
            trained_successfully (bool): ``True`` if parameters were loaded or fitted successfully.
        """
        super().__init__(name, params)
        self.type = "linear_gauss"
        self.params = params
        self.model = LinearGaussModule(rngs=nnx.Rngs(0))
        self.ckpt_dir = os.path.join(config_paths.MODEL_PATH, "ckpts", self.name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        if load_ckpt:
            try:
                self.model = load_checkpoint(self.model, self.ckpt_dir)
                self.trained_successfully = True
            except Exception as e:
                print(f"Error restoring model state: {e}")
                self.trained_successfully = False
        else:
            self.trained_successfully = False

    def _fit(
        self,
        train_dataset: OceanTrainingDataset,
        val_dataset: OceanTrainingDataset,
        train_params: dict | None = None,
    ) -> "LinearGaussModel":
        """Fit ``alpha`` and ``log_sigma`` by minimising negative log-likelihood.

        At each step the model predicts ``alpha * current_velocity`` and computes a
        Gaussian log-likelihood with variance ``exp(log_sigma)^2``. Uses ``optax.adam``
        and PyTorch's ``DataLoader`` for batching. Early stopping is applied if
        ``patience`` is set.

        Args:
            train_dataset: Training split; context variables must be set before calling.
            val_dataset: Validation split used for early-stopping evaluation.
            train_params: Training hyperparameters. Supported keys:

                - ``num_epochs`` (int): maximum epochs (default ``100``)
                - ``num_batches_per_epoch`` (int): gradient steps per epoch (default ``100``)
                - ``batch_size`` (int): samples per gradient step (default ``32``)
                - ``lr`` (float): Adam learning rate (default ``1e-3``)
                - ``min_delta`` (float): minimum improvement for early stopping (default ``0.001``)
                - ``patience`` (int): epochs without val improvement before stopping (default ``5``)

        Returns:
            self, with ``model.alpha`` and ``model.log_sigma`` fitted.
        """
        if train_params is None:
            train_params = {}

        ckpt_dir = ocp.test_utils.erase_and_create_empty(self.ckpt_dir)
        train_dataset.set_context_var_idx(self.var_idx)
        train_dataloader = data.DataLoader(
            train_dataset, batch_size=train_params.get("batch_size", 32), shuffle=True
        )
        val_dataset.set_context_var_idx(self.var_idx)
        val_dataloader = data.DataLoader(
            val_dataset, batch_size=train_params.get("batch_size", 32), shuffle=False
        )
        optimizer = nnx.Optimizer(self.model, optax.adam(train_params.get("lr", 1e-3)))

        state_axes = nnx.StateAxes({"Dropout": 0, ...: None})

        @nnx.split_rngs(splits=train_params.get("batch_size", 32), only="dropout")
        @nnx.vmap(
            in_axes=(
                state_axes,
                0,
            ),  # model is not vmapped, x is vmapped, train is not vmapped
            out_axes=0,  # Output is vmapped
        )
        def batched_apply(model, x: jnp.ndarray):
            """Applies the model to a batch of inputs."""
            # NNX will handle the state and RNGs of the model instance passed to this function.
            return model.get_mean_velocity(x)

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
                    vmap(
                        data_transform.velocity_vector_from_pos_time,
                        in_axes=(0, 0, None),
                    ),
                    in_axes=(0, 0, None),
                )(trajectory[:, :-1], trajectory[:, 1:], 3600)
                increment_prediction = batched_apply(model, context[:, :-1])  # type: ignore
                return gauss_loss(
                    true_increments, increment_prediction, model.log_sigma
                )

            loss, grad = nnx.value_and_grad(loss_fn)(model, trajectory, context)
            optimizer = optimizer.update(grad)  # type: ignore
            return loss

        @nnx.jit
        def eval_step(
            model: nnx.Module, trajectory: jnp.ndarray, context: jnp.ndarray
        ) -> jnp.ndarray:
            def loss_fn(model: nnx.Module) -> jnp.ndarray:
                true_increments = vmap(
                    vmap(
                        data_transform.velocity_vector_from_pos_time,
                        in_axes=(0, 0, None),
                    ),
                    in_axes=(0, 0, None),
                )(trajectory[:, :-1], trajectory[:, 1:], 3600)
                increment_prediction = batched_apply(model, context[:, :-1])  # type: ignore
                return gauss_loss(
                    true_increments, increment_prediction, model.log_sigma
                )

            loss = loss_fn(model)
            return loss

        num_batches_per_epoch = train_params.get("num_batches_per_epoch", 100)
        num_epochs = train_params.get("num_epochs", 100)
        early_stop = EarlyStopping(
            min_delta=train_params.get("min_delta", 0.001),
            patience=train_params.get("patience", 5),
        )
        self.model.train()
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
                rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))  # type: ignore
                merge_state(state, rngs_key)  # type: ignore

                _, opt_state = nnx.split(optimizer.opt_state)
                ckpt_mgr.save(
                    1,
                    args=ocp.args.Composite(
                        state=ocp.args.StandardSave(state),  # type: ignore
                        opt_state=ocp.args.StandardSave(opt_state),  # type: ignore
                    ),
                )
            val_loss = 0.0
            n_val = 0
            for batch in tqdm(val_dataloader):
                trajectory, context = jnp.asarray(batch[0]), jnp.asarray(batch[1])
                val_loss += eval_step(self.model, trajectory, context)
                n_val += batch[0].shape[0] * batch[0].shape[1]
            val_loss /= n_val
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
        start_time: datetime.datetime,
        prediction_length: int,
        interpolator: Interpolator3D,
        test_dataset: OceanDataset,
        key: jax.Array = random.PRNGKey(0),
        num_samples: int = 1,
    ) -> jnp.ndarray:
        """Sample trajectory predictions from the fitted Linear Gaussian model.

        At each step, the predicted displacement is ``alpha * current_velocity + noise``
        where ``noise ~ N(0, exp(log_sigma)^2 * I_2)``.

        Args:
            start_position: Lat/lon of the drifter at ``start_time``, shape ``(2,)`` in decimal degrees.
            start_time: Absolute UTC datetime for the first prediction step.
            prediction_length: Number of hourly time steps to predict.
            interpolator: Pre-built ``interpax.Interpolator3D`` covering the domain.
            test_dataset: ``OceanDataset`` providing context feature extraction.
            key: JAX PRNG key (uses ``random.PRNGKey(0)`` if not provided).
            num_samples: Number of independent trajectory samples to draw.

        Returns:
            Predicted trajectories as ``jnp.ndarray`` of shape ``(num_samples, prediction_length, 2)``
            in decimal degrees (lat, lon).
        """
        times = self._get_prediction_steps(start_time, prediction_length)
        num_preds = num_samples
        cur_coord = jnp.array([start_position])
        cur_coord = jnp.repeat(cur_coord, num_preds, axis=0)

        def pred_trajectory(
            carry: tuple[nnx.Module, jax.Array],
            start_position: jnp.ndarray,
            times: jnp.ndarray,
            interpolator: Interpolator3D,
            i: int,
        ) -> tuple[nnx.Module, jnp.ndarray]:

            model, key = carry
            key, subkey = random.split(key)

            @nnx.scan
            def scan_fn(
                carry: tuple[nnx.Module, jnp.ndarray, jax.Array], t: jnp.ndarray
            ) -> tuple[tuple[nnx.Module, jnp.ndarray, jax.Array], jnp.ndarray]:
                model, cur_coord, key = carry[0], carry[1], carry[2]
                key, subkey = random.split(key)
                context = test_dataset.get_context(
                    jnp.array([t]), cur_coord, interpolator
                )[..., self.var_idx]
                pred_vel = model(context, key=subkey)  # type: ignore
                speed, bearing = data_transform.velocity_to_distance_bearing(
                    pred_vel[0]
                )
                new_coord = data_transform.inverse_haversine(
                    cur_coord[0], 3600 * speed, bearing  # type: ignore
                )
                return (
                    model,
                    jnp.array([new_coord]),
                    key,
                ), new_coord

            init = (model, jnp.array([start_position], dtype=float), subkey)

            _, pred_coords = scan_fn(init, times)
            return (model, key), pred_coords

        init = (self.model, key)
        _, pred_coords = nnx.scan(
            lambda carry, i: pred_trajectory(
                carry, start_position, times[:-1], interpolator, i
            )
        )(init, jnp.arange(num_preds))

        return pred_coords

    def export_trained_model(
        self, model_name: str | None = None, export_path: str | None = None
    ) -> None:
        """Save the fitted model parameters to disk as a pickle file.

        Writes ``model_config.yml`` (YAML of ``self.params``) and
        ``nnx_model.pkl`` (pickle of the Flax NNX state) into
        ``{export_path}/{model_name}/``.

        Args:
            model_name: Checkpoint folder name. Defaults to ``self.name``.
            export_path: Parent directory. Defaults to
                ``{config_paths.MODEL_PATH}/trained_models``.

        Raises:
            Exception: If a directory named ``model_name`` already exists at ``export_path``.
        """

        if model_name is None:
            model_name = self.name
        if export_path is None:
            export_path = os.path.join(config_paths.MODEL_PATH, "trained_models")

        model_path = os.path.join(export_path, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        else:
            raise Exception(
                f"Model {model_name} already exists at {model_path}. Please delete it or change the model name."
            )

        # nn_params_path = os.path.join(model_path, "nn_params.npz")
        # jnp.savez(nn_params_path, params=self.nn_params, allow_pickle=True)
        with open(os.path.join(model_path, "model_config.yml"), "w") as f:
            yaml.dump(self.params, f, default_flow_style=False)

        nnx_model_path = os.path.join(model_path, "nnx_model.pkl")
        graph, state = nnx.split(self.model)
        with open(nnx_model_path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def import_trained_model(
        cls, model_name: str, import_path: str | None = None
    ) -> "LinearGaussModel":
        """Restore a fitted LinearGaussModel from a pickle checkpoint.

        Reads ``model_config.yml`` and ``nnx_model.pkl`` from
        ``{import_path}/{model_name}/`` and merges the restored NNX state back
        into a freshly constructed model.

        Args:
            model_name: Name of the checkpoint directory.
            import_path: Parent directory to search. Defaults to
                ``{config_paths.MODEL_PATH}/trained_models``.

        Returns:
            A ``LinearGaussModel`` with ``alpha`` and ``log_sigma`` restored from the checkpoint.

        Raises:
            FileNotFoundError: If ``nnx_model.pkl`` does not exist in the checkpoint directory.
        """
        if import_path is None:
            import_path = os.path.join(config_paths.MODEL_PATH, "trained_models")

        model_path = os.path.join(import_path, model_name)
        with open(os.path.join(model_path, "model_config.yml"), "r") as f:
            model_config = yaml.safe_load(f)

        nnx_model_path = os.path.join(model_path, "nnx_model.pkl")
        if not os.path.exists(nnx_model_path):
            raise FileNotFoundError(
                f"NNX parameters file {nnx_model_path} does not exist."
            )

        with open(nnx_model_path, "rb") as f:
            state = pickle.load(f)

        model = cls(model_name, model_config, load_ckpt=False)
        graph, _ = nnx.split(model.model)
        model.model = nnx.merge(graph, state)
        model.trained_successfully = True
        return model
