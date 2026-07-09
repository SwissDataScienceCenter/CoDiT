import datetime
import os
import pickle
import shutil
import warnings

# Suppress the SerializationWarning
from functools import partial

import jax
import optax
import orbax.checkpoint
import orbax.checkpoint as ocp
import xarray as xr
import yaml
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import orbax_utils
from interpax import Interpolator3D
from jax import jit, lax, random, vmap

# only use cpu
# jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

from codit.codit_module import (
    diffusion_models,
    noising_functions,
    scheduler,
)
from codit.codit_module.brownian_noise_models import (
    WalkNoiseModel,
)

# Suppress the SerializationWarning
from codit.data_module import config_paths, data_transform
from codit.data_module.dataset import OceanDataset, OceanTrainingDataset
from codit.model_module.base_model import BaseTrajectoryModel

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

# config.update("jax_enable_x64", True)


def backward_denoising_step(
    carry: tuple,
    i: int,
    W: jnp.ndarray,
    batch: dict,
    schedule_dict: dict,
    noise_model: WalkNoiseModel,
    model: nn.Module,
    params: FrozenDict | dict,
) -> tuple[tuple, jnp.ndarray]:
    """
    A single step of the backward denoising algorithm.

    Args:
        carry: The state of the algorithm.
        i: The current time step.
        beta: The beta schedule.
        alpha: The alpha schedule.
        alpha_bar: The alpha bar schedule.
        noise_model: The noise model.
        model: The denoising model.
        params: The parameters of the model.
        K: The number of time steps.
        init_cond: The initial condition.
        T: The time horizon.

    Returns:
        carry: The updated state of the algorithm.
        output: The output of the algorithm.
    """
    key, y = carry
    # key, subkey = random.split(key)
    beta_tmp = schedule_dict["beta"][i]
    alpha_tmp = schedule_dict["alpha"][i]
    alpha_bar_tmp = schedule_dict["alpha_bar"][i]
    k = jnp.array([i])
    y_proj = y
    y_history_reversed = batch["history_target_vals"][..., ::-1, :]
    y_history_input = noise_model.center_data(y_history_reversed)
    # y_history_delta = y_history_input[...,:1,:]
    key, subkey = random.split(key)
    y_noised_history, _ = noising_functions.forward_diffusion(
        y_history_input, k, subkey, schedule_dict, noise_model
    )
    # y_noised_history = y_noised_history + y_history_delta
    # key, subkey = random.split(key)
    y_proj_history = y_noised_history
    y_proj_history = y_proj_history[..., ::-1, :]
    noise_pred = model.apply(params, y_proj[1:], y_proj_history, batch, k)
    assert isinstance(noise_pred, jnp.ndarray), "Model output is not a jnp.ndarray"
    W_pred = noise_model.transform(noise_pred)
    mu_pred = y / jnp.sqrt(1 - beta_tmp)
    mu_pred -= beta_tmp / (jnp.sqrt(1 - alpha_bar_tmp) * jnp.sqrt(alpha_tmp)) * W_pred
    sigma_pred = jnp.sqrt(beta_tmp)
    # key, subkey = random.split(key)
    # W, dW = noise_model(subkey, y.shape[-2]-1, cov=cov)
    k_is_not_zero = i != 0
    # For the last step the mean is returned
    y = mu_pred + k_is_not_zero * sigma_pred * W[i]
    y = y.at[:1].set(0)
    # y += y_delta
    return (key, y), y


def denoise(
    params: FrozenDict | dict,
    y_init: jnp.ndarray,
    batch: dict,
    model: nn.Module,
    W: jnp.ndarray,
    noise_model: WalkNoiseModel,
    key: jax.Array,
    schedule_dict: dict,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Denoise a time series.

    Args:
        params: The parameters of the model.
        model: The denoising model.
        noise_model: The noise model.
        key: The random key.
        T: The time horizon.
        y_init: The initial data.
        beta: The beta schedule.
        alpha: The alpha schedule.
        alpha_bar: The alpha bar schedule.
        init_cond: The initial condition.

    Returns:
        y: The denoised time series.
        noise_pred: The predicted noise
    """
    init = (key, y_init)
    K = schedule_dict["K"]
    iterations = jnp.arange(0, K)[::-1]

    def step_func(carry: tuple, i: int) -> tuple[tuple, jnp.ndarray]:
        return backward_denoising_step(
            carry, i, W, batch, schedule_dict, noise_model, model, params
        )

    # step_func = lambda carry, i: backward_denoising_step(
    #     carry, i, W, batch, schedule_dict, noise_model, model, params
    # )
    denoise_results = lax.scan(step_func, init, iterations)  # type: ignore
    y = denoise_results[0][1]
    denoising_steps = denoise_results[1]
    return y, denoising_steps


def store_checkpoint(
    ckpt_file: str,
    model: nn.Module,
    params: dict,
    loss_list: list,
    val_loss_list: list,
    std: jnp.ndarray,
) -> None:
    if os.path.exists(ckpt_file):
        shutil.rmtree(ckpt_file)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = {
        "model": params,
        "loss_list": loss_list,
        "val_loss_list": val_loss_list,
        "std": std,
    }
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(ckpt_file, ckpt, save_args=save_args)


def restore_checkpoint(ckpt_file: str) -> tuple[nn.Module, list, list, jnp.ndarray]:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(ckpt_file)
    params = ckpt["model"]
    loss_list = ckpt["loss_list"]
    val_loss_list = ckpt["val_loss_list"]
    std = ckpt["std"]
    return params, loss_list, val_loss_list, std


class CoDiTModel(BaseTrajectoryModel):

    def __init__(self, name: str, params: dict) -> None:

        self.type = "diffusion"

        default_params = {
            "model_type": "GRU",
            "num_layers": 2,
            "hidden_dims": 64,
            "diffusion_emb": 64,
            "K": 100,
            "sigma": 1.0,
            "dt": 0.1,
            "beta_min": 1e-4,
            "beta_max": 0.02,  # before 0.02
            "noise": "brownian_motion",
            "interp_method": "linear",
            "max_len": 100,
            "num_heads": 8,
            "context_encoder_type": "conv",  # conv, mlp, gap
        }
        params = default_params | params
        super().__init__(name, params)
        model_type = params.get("model_type", default_params["model_type"])
        context_encoder_type = params.get(
            "context_encoder_type", default_params["context_encoder_type"]
        )

        layers = params.get("num_layers", default_params["num_layers"])
        hidden_dims = params.get("hidden_dims", default_params["hidden_dims"])
        config_paths.MODEL_PATH
        self.ckpt_dir = f"{config_paths.MODEL_PATH}/ckpts/{self.name}/"

        # layers = params.get('num_layers', default_params['num_layers'])

        diffusion_emb = params.get("diffusion_emb", default_params["diffusion_emb"])
        K = params.get("K", default_params["K"])
        noise = params.get("noise", default_params["noise"])
        dt = params.get("dt", default_params["dt"])
        prediction_length = params["prediction_length"]
        if model_type == "GRU":
            self.model = diffusion_models.Autoregressive_Denoiser(
                output_size=2,
                context=True,
                num_layers=layers,
                hidden_size=hidden_dims,
                diffusion_emb_dim=diffusion_emb,
                K=K,
                context_encoder_type=context_encoder_type,
            )
        elif model_type == "Transformer":
            max_len = params.get("max_len", default_params["max_len"])
            num_heads = params.get("num_heads", default_params["num_heads"])
            self.model = diffusion_models.Autoregressive_TransformerDenoiser(
                output_size=2,
                context=True,
                num_layers=layers,
                hidden_size=hidden_dims,
                diffusion_emb_dim=diffusion_emb,
                K=K,
                max_len=max_len,
                num_heads=num_heads,
                context_encoder_type=context_encoder_type,
            )
        self.schedule_dict = scheduler.linear_schedule(
            K,
            beta_min=params.get("beta_min", default_params["beta_min"]),
            beta_max=params.get("beta_max", default_params["beta_max"]),
        )

        self.noise_model = WalkNoiseModel(
            noise_type=noise,  # type: ignore
            num_dims=2,  # type: ignore
            dt=dt,  # type: ignore
            sigma=params.get("sigma", default_params["sigma"]),  # type: ignore
            T=prediction_length * dt,  # type: ignore
        )
        self.nn_params = None

    def center_and_scale_data(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        if len(trajectory.shape) == 2:
            return (trajectory - trajectory[0]) / self.std
        else:
            return (trajectory - trajectory[:, 0]) / self.std

    def scale(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        if len(trajectory.shape) == 2:
            start = trajectory[0]
        else:
            start = trajectory[:, 0]
        return (trajectory - start) / self.std + start

    def _fit(
        self,
        train_dataset: OceanTrainingDataset,
        val_dataset: OceanTrainingDataset,
        train_params: dict | None = None,
    ) -> "CoDiTModel":
        if train_params is None:
            train_params = {}
        # return self
        default_train_params = {
            "seed": 0,
            "batch_size": 128,
            "num_batches": 100,
            "num_epochs": 1000,
            "num_batches_per_epoch": 100,
            "force_retrain": False,
            "learning_rate": 1e-4,
            "optimizer": "adam",
            "scheduler": None,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 1e-4,
        }

        # todo: This is not very nice, should happen in the dataset or more downstream
        def transform_traj_to_metre(traj: jnp.ndarray) -> jnp.ndarray:
            velocity_vectors = vmap(
                data_transform.velocity_vector_from_pos_time, in_axes=(None, 0, None)
            )(traj[0], traj, 3600)
            traj_in_metres = velocity_vectors * 3600
            return traj_in_metres

        # assert isinstance(train_dataset.trajectories, jnp.ndarray), (
        #     "Trajectories should be a jnp.ndarray, "
        #     "but got: {}".format(type(train_dataset.trajectories))
        # )
        traj_in_metres = vmap(transform_traj_to_metre)(
            jnp.asarray(train_dataset.trajectories)
        )
        self.std = jnp.std(traj_in_metres[:, -1], axis=0)
        train_dataset.set_context_var_idx(self.var_idx)
        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=train_params.get(
                "batch_size", default_train_params["batch_size"]
            ),
            shuffle=True,
            drop_last=True,
        )
        val_dataset.set_context_var_idx(self.var_idx)
        val_dataloader = data.DataLoader(
            val_dataset, batch_size=128, shuffle=False, drop_last=True
        )
        num_epochs = train_params.get("num_epochs", default_train_params["num_epochs"])
        # num_batches_per_epoch = train_params.get('num_batches_per_epoch', default_train_params['num_batches_per_epoch'])
        key = random.PRNGKey(train_params.get("seed", default_train_params["seed"]))
        log_path = os.path.join(self.ckpt_dir, "..", "logs", self.name)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        writer = SummaryWriter(log_dir=log_path)
        data_sample = train_dataset[0]
        # assert isinstance(data_sample[0], jnp.ndarray), (
        #     "Data sample should be a jnp.ndarray, "
        #     "but got: {}".format(type(data_sample[0]))
        # )
        trajectory = self.center_and_scale_data(jnp.asarray(data_sample[0]))
        y = trajectory[1:]
        y_hist = trajectory[:1]
        input = {
            "history_target_vals": y_hist,
            "history_context": data_sample[1][:1],
            "prediction_context": data_sample[1][1:],
        }
        params = self.model.init(
            key,
            y=y,
            y_history=y_hist,
            x=input,
            k=jnp.ones((1,), dtype=jnp.int32),
            train=True,
        )
        # output = self.model.apply(params, y=y, y_history=y_hist, x=input, k=jnp.ones((1,), dtype=jnp.int32),
        #                     train=True, mutable=['batch_stats'])

        scheduler = train_params.get("scheduler", default_train_params["scheduler"])
        if scheduler is None:
            lr = train_params.get(
                "learning_rate", default_train_params["learning_rate"]
            )
        else:
            lr = scheduler

        optimizer_type = train_params.get(
            "optimizer", default_train_params["optimizer"]
        )

        if optimizer_type == "adam":
            optimizer = optax.adam(learning_rate=lr)
        elif optimizer_type == "adamw":
            optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-4)
        elif optimizer_type == "sgd":
            optimizer = optax.sgd(learning_rate=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        opt_state = optimizer.init(params)

        orbax_path = os.path.join(self.ckpt_dir, "orbax")
        if not os.path.isdir(orbax_path):
            os.makedirs(orbax_path)
        key, subkey = random.split(key)
        loss_list: list = []
        val_loss_list: list = []

        options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1)
        mngr = ocp.CheckpointManager(orbax_path, options=options)
        # latest_ckpt = mngr.latest_step()
        # if latest_ckpt is not None:
        # self.load_trained_model()
        if not os.path.isdir(
            os.path.join(os.path.join(self.ckpt_dir, "final"), "d")
        ) or train_params.get("force_retrain", False):
            params, opt_state, loss_list, val_loss_list = diffusion_models.train(
                train_dataloader,
                optimizer,
                self.model,
                self.noise_model,
                params,
                opt_state,
                key,
                self.schedule_dict,
                num_epochs=num_epochs,
                loss_list=loss_list,
                checkpoint_mngr=mngr,
                val_dataloader=val_dataloader,
                val_loss_list=val_loss_list,
                writer=writer,
                std=self.std,
                early_stopping_patience=train_params.get("early_stopping_patience", 10),
                early_stopping_min_delta=train_params.get(
                    "early_stopping_min_delta", 1e-4
                ),
            )
            writer.close()
            print("storing", self.ckpt_dir)
            store_checkpoint(
                os.path.join(self.ckpt_dir, "final"),
                self.model,
                params,
                loss_list,
                val_loss_list,
                self.std,
            )
            jnp.savez(
                os.path.join(self.ckpt_dir, "final", "params.npz"),
                params=params,
                loss_list=loss_list,
                val_loss_list=val_loss_list,
                std=self.std,
            )
            self.nn_params = params
        return self

    def load_trained_model(self) -> None:
        # params, loss_list, val_loss_list, std = restore_checkpoint(
        trained_model = jnp.load(
            os.path.join(self.ckpt_dir, "final", "params.npz"), allow_pickle=True
        )
        self.nn_params = trained_model["params"]
        self.std = trained_model["std"]

    def predict(
        self,
        start_position: jax.Array,
        start_time: datetime.datetime,
        prediction_length: int,
        interpolator: Interpolator3D,
        test_dataset: OceanDataset,
        num_samples: int = 1,
        key: jax.Array = random.PRNGKey(0),
        return_denoising_steps: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:

        times = self._get_prediction_steps(start_time, prediction_length)
        prediction_length = len(times) - 1
        start_context = test_dataset.get_context(
            times[:1], start_position[None], interpolator
        )[..., self.var_idx]
        coords_denoised = [start_position]

        key, subkey = random.split(key)
        K = self.schedule_dict["K"]
        batch = {
            "history_target_vals": jnp.zeros((1, 2)),
            "history_context": start_context,
            "prediction_context": start_context[None],
        }
        keys = random.split(key, num_samples + 1)
        key, subkeys = keys[0], keys[1:]

        def get_noise_model(
            key: jax.Array, prediction_length: int
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            keys = random.split(key, K + 1)
            key, subkeys = keys[0], keys[1:]
            return vmap(self.noise_model, in_axes=(0, None))(subkeys, prediction_length)

        W, dW = vmap(get_noise_model, in_axes=(0, None))(subkeys, prediction_length)
        coords_denoised = [jnp.tile(start_position[None], (num_samples, 1, 1))]
        keys = random.split(key, num_samples + 1)
        key, subkeys = keys[0], keys[1:]
        initial_y, _ = vmap(self.noise_model, in_axes=(0, None))(
            subkeys, prediction_length
        )

        @partial(vmap, in_axes=(0, 0, 0, 0, None, None))
        def get_denoised_trajectory(
            key: jax.Array,
            W: jnp.ndarray,
            initial_y: jnp.ndarray,
            batch: dict,
            numeric_time: jnp.ndarray,
            i: int,
        ) -> (
            tuple[jnp.ndarray, jnp.ndarray]
            | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ):
            denoised, denoised_steps = denoise(
                self.nn_params,  # type: ignore
                initial_y,
                batch,
                self.model,
                W,
                self.noise_model,
                key,
                self.schedule_dict,
            )
            denoised_scaled = denoised * self.std
            distance_in_metres = jnp.sqrt(
                denoised_scaled[:, 0] ** 2 + denoised_scaled[:, 1] ** 2
            )
            bearing_angle = jnp.arctan2(denoised_scaled[:, 1], denoised_scaled[:, 0])
            bearing_angle = jnp.degrees(bearing_angle)
            dcoord = data_transform.inverse_haversine_grad(
                start_position, distance_in_metres[i], bearing_angle[i]
            )
            new_coord = start_position[None] + dcoord
            new_context = test_dataset.get_context(
                numeric_time, new_coord, interpolator
            )[..., self.var_idx]
            if return_denoising_steps:
                return new_coord, new_context, denoised_steps
            else:
                return new_coord, new_context

        keys = random.split(key, num_samples + 1)
        key, subkeys = keys[0], keys[1:]
        coords_denoised = [jnp.tile(start_position[None], (num_samples, 1, 1))]
        batch = {
            "history_target_vals": jnp.zeros((num_samples, 1, 2)),
            "history_context": jnp.tile(start_context[None], (num_samples, 1, 1, 1, 1)),
            "prediction_context": jnp.tile(
                start_context[None], (num_samples, prediction_length, 1, 1, 1)
            ),
        }

        denoise_func = jit(get_denoised_trajectory)
        for i in range(1, prediction_length + 1):
            if return_denoising_steps:
                new_coord, new_context, denoise_steps_scaled = denoise_func(
                    subkeys, W, initial_y, batch, times[i : i + 1], i
                )
            else:
                new_coord, new_context = denoise_func(
                    subkeys, W, initial_y, batch, times[i : i + 1], i
                )
            coords_denoised.append(new_coord)
            batch["prediction_context"] = (
                batch["prediction_context"].at[:, i : i + 1].set(new_context)
            )  # jnp.concatenate([batch['prediction_context'], new_context], axis=1)
            batch["history_target_vals"] = jnp.zeros((num_samples, 1, 2))
            batch["history_context"] = jnp.tile(
                start_context[None], (num_samples, 1, 1, 1, 1)
            )
        pred_coords = jnp.concatenate(coords_denoised, axis=1)
        if return_denoising_steps:
            denoised_steps_scaled = denoise_steps_scaled * self.std
            distance_in_metres = jnp.sqrt(
                denoised_steps_scaled[..., 0] ** 2 + denoised_steps_scaled[..., 1] ** 2
            )
            bearing_angle = jnp.arctan2(
                denoised_steps_scaled[..., 1], denoised_steps_scaled[..., 0]
            )
            bearing_angle = jnp.degrees(bearing_angle)
            denoised_coords = vmap(
                vmap(
                    vmap(data_transform.inverse_haversine, in_axes=(None, 0, 0)),
                    in_axes=(None, 0, 0),
                ),
                in_axes=(None, 0, 0),
            )(start_position, distance_in_metres, bearing_angle)
            return pred_coords[:, 1:], denoised_coords
        else:
            return pred_coords[:, 1:]

    def export_trained_model(
        self, model_name: str | None = None, export_path: str | None = None
    ) -> None:
        if model_name is None:
            model_name = self.name
        if export_path is None:
            export_path = os.path.join(config_paths.MODEL_PATH, "trained_models")

        model_path = os.path.join(export_path, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        else:
            raise Exception(
                f"Model {model_name} already exists in {model_path}. Please delete it or change the model name."
            )

        with open(os.path.join(model_path, "model_config.yml"), "w") as f:
            yaml.dump(self.params, f, default_flow_style=False)

        # save nn_params and std
        nn_params_path = os.path.join(model_path, "nn_params.npz")

        if self.nn_params is None:
            raise ValueError(
                "Model parameters (nn_params) are None. Train or load the model before exporting."
            )
        # jnp.savez(nn_params_path, params=self.nn_params, std=self.std, allow_pickle=True)
        with open(nn_params_path, "wb") as f:
            pickle.dump({"params": self.nn_params, "std": self.std}, f)

    @classmethod
    def import_trained_model(
        cls, model_name: str, import_path: str | None = None
    ) -> "CoDiTModel":
        """
        Imports a trained model from the specified path.

        Args:
            model_name (str): The name of the model to import.
            model_path (str): The path where the model is stored.

        Returns:
            DiffusionModel: An instance of the DiffusionModel class with the loaded parameters.
        """
        if import_path is None:
            import_path = os.path.join(config_paths.MODEL_PATH, "trained_models")
        model_path = os.path.join(import_path, model_name)
        config_file = os.path.join(model_path, "model_config.yml")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")

        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)

        nn_params_path = os.path.join(model_path, "nn_params.npz")
        if not os.path.exists(nn_params_path):
            raise FileNotFoundError(
                f"NN parameters file {nn_params_path} does not exist."
            )

        # nn_params = jnp.load(nn_params_path, allow_pickle=True)
        with open(nn_params_path, "rb") as f:
            nn_params = pickle.load(f)

        model = cls(model_name, model_config)
        assert isinstance(nn_params, dict), "Loaded nn_params should be a dictionary."
        model.nn_params = nn_params["params"]
        model.std = nn_params["std"]

        return model
