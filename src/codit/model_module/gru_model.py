import datetime
import os
import pickle
import warnings
from functools import partial
from typing import Optional

import jax
import optax
import orbax.checkpoint as ocp
import xarray as xr
import yaml
from flax import linen as nn
from flax.training.early_stopping import EarlyStopping
from interpax import Interpolator3D
from jax import jit, random, vmap
from jax import numpy as jnp
from orbax.checkpoint import CheckpointManager
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from codit.codit_module.diffusion_models import (
    Autoregressive_RNN,
    ConvContextEncoder,
    GAPContextEncoder,
    HistoryModule,
    MLP,
    MLPContextEncoder,
    ResDecoder,
    TemporalEncoding,
    Transformer,
)
from codit.data_module import config_paths, data_transform
from codit.data_module.data_transform import (
    transform_traj_to_metre,
    velocity_vector_from_pos_time,
)
from codit.data_module.dataset import OceanDataset, OceanTrainingDataset
from codit.model_module.base_model import BaseTrajectoryModel

warnings.filterwarnings("ignore", category=xr.SerializationWarning)


@partial(jax.jit, static_argnums=(1,))
@jax.value_and_grad
def mse_loss(
    params: dict,
    model: nn.Module,
    batch: dict,
    key: random.PRNGKey,
) -> jnp.ndarray:
    y_history = batch["history_target_vals"]
    increment_prediction = model_predict(params, y_history, batch, model, key)
    trajectory = batch["prediction_target_vals"]
    true_increments = vmap(
        vmap(velocity_vector_from_pos_time, in_axes=(0, 0, None)),
        in_axes=(0, 0, None),
    )(trajectory[:, :-1], trajectory[:, 1:], 3600)
    return jnp.mean(jnp.pow(increment_prediction - true_increments, 2))


class Autoregressive(nn.Module):
    context: bool = False
    time_emb_dim: int = 64
    num_layers: int = 1
    hidden_size: int = 128
    max_len: int = 100
    num_heads: int = 8
    output_size: int = 2
    T: int = 100
    context_encoder_type: str = "conv"  # conv, mlp, gap
    backbone: str = "GRU"  # GRU, Transformer

    @nn.compact
    def __call__(
        self,
        y_history: jnp.ndarray,
        x: dict,
        train: bool = True,
    ) -> jnp.ndarray:
        history_module = HistoryModule(
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            num_freqs=self.time_emb_dim,
            num_layers=2,
            K=self.T,
        )
        encoder = MLP(
            num_layers=2, hidden_size=self.hidden_size, output_size=self.hidden_size
        )
        decoder = ResDecoder(
            hidden_size=self.hidden_size, output_size=self.hidden_size, num_layers=2
        )

        t_history, t_pred = x["history_time"], x["prediction_time"]
        t = jnp.concatenate([t_history, t_pred], axis=-2)
        time_enc = TemporalEncoding(K=self.T, num_freqs=self.time_emb_dim)(
            t.squeeze(-1)
        )

        history_context = x["history_context"]
        if self.context_encoder_type == "conv":
            context_enc = ConvContextEncoder(kernel_size=3, hidden_base=8, num_layers=4)
        elif self.context_encoder_type == "mlp":
            context_enc = MLPContextEncoder(output_dim=self.hidden_size)
        elif self.context_encoder_type == "gap":
            context_enc = GAPContextEncoder(output_dim=self.hidden_size)
        else:
            raise ValueError(
                f"Unknown context encoder type: {self.context_encoder_type}"
            )

        history_context_enc = context_enc(history_context)
        history_enc = history_module(history_context_enc, x["history_target_vals"], t)

        y_encoder = MLP(
            num_layers=2, hidden_size=self.hidden_size, output_size=y_history.shape[-1]
        )
        y_pred = y_encoder(t_pred)
        y = jnp.concatenate([y_history, y_pred], axis=-2)

        if self.context:
            prediction_context = x["prediction_context"]
            prediction_context_enc = context_enc(prediction_context)
            context = jnp.concatenate(
                [history_context_enc, prediction_context_enc], axis=-2
            )
            context = jnp.concatenate(
                [jnp.zeros((1, context.shape[-1])), context[..., :-1, :]], axis=-2
            )
            context = jnp.concatenate(
                [
                    context,
                    time_enc,
                    jnp.tile(history_enc, (y.shape[-2], 1)),
                ],
                axis=-1,
            )
        else:
            context = jnp.concatenate(
                [
                    time_enc,
                    jnp.tile(history_enc, (y.shape[-2], 1)),
                ],
                axis=-1,
            )

        if self.backbone == "GRU":
            inputs = encoder(y)
            outputs = Autoregressive_RNN(
                num_layers=self.num_layers, num_hidden=self.hidden_size
            )(inputs, context)[..., y_history.shape[-2] :, :]
        elif self.backbone == "Transformer":
            inputs = jnp.concatenate([y, context], axis=-1)
            inputs = encoder(inputs)
            outputs = Transformer(
                d_model=self.hidden_size,
                num_layers=self.num_layers,
                max_len=self.max_len,
                num_heads=self.num_heads,
            )(inputs, train)[..., y_history.shape[-2] :, :]
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

        outputs = decoder(
            inputs[..., y_history.shape[-2] :, :],
            outputs,
            context[..., y_history.shape[-2] :, :],
            train,
        )
        outputs = nn.Dense(self.output_size)(outputs)
        return outputs


def train_step(
    carry: tuple,
    batch: dict,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
) -> tuple:
    params, opt_state, key, loss = carry
    value, grads = mse_loss(params, model, batch, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    loss += value
    return (params, opt_state, key, loss), value


def train(
    train_dataloader: DataLoader,
    optimizer: optax.GradientTransformation,
    model: nn.Module,
    params: dict,
    opt_state: optax.OptState,
    key: random.PRNGKey,
    num_epochs: int = 100,
    num_batches_per_epoch: int = 100,
    loss_list: Optional[list] = None,
    checkpoint_mngr: Optional[CheckpointManager] = None,
    val_dataloader: Optional[DataLoader] = None,
    val_loss_list: Optional[list] = None,
    early_stopping_kwargs: Optional[dict] = None,
    writer: Optional[SummaryWriter] = None,
    std: float = 1.0,
) -> tuple:
    if loss_list is None:
        loss_list = []
    if val_loss_list is None:
        val_loss_list = []
    if early_stopping_kwargs is None:
        early_stopping_kwargs = {"patience": 30, "min_delta": 1e-4}

    @jit
    def step_func(carry: tuple, batch: dict) -> tuple:
        return train_step(carry, batch, model, optimizer)

    @jit
    def eval_step(carry: tuple, batch: dict) -> tuple:
        eval_params, eval_opt_state, eval_key, eval_loss = carry
        value, _ = mse_loss(eval_params, model, batch, eval_key)
        eval_loss += value
        return (eval_params, eval_opt_state, eval_key, eval_loss), value

    early_stop = EarlyStopping(**early_stopping_kwargs)
    pbar = tqdm(total=num_epochs, desc="Training", dynamic_ncols=True)

    for epoch_idx in range(num_epochs):
        carry = (params, opt_state, key, 0)
        processed_train_steps = 0
        for batch in train_dataloader:
            if processed_train_steps >= num_batches_per_epoch:
                break
            jax_batch = transform_batch(batch, std)
            carry, _ = step_func(carry, jax_batch)
            processed_train_steps += 1

        params, opt_state, key, loss = carry
        train_denom = max(processed_train_steps, 1)
        loss_list.append(float(loss) / train_denom)

        if val_dataloader is not None:
            carry = (params, opt_state, key, 0)
            processed_val_steps = 0
            for batch in val_dataloader:
                jax_batch = transform_batch(batch, std)
                carry, _ = eval_step(carry, jax_batch)
                processed_val_steps += 1
            params, opt_state, key, val_loss = carry
            if processed_val_steps > 0:
                val_loss_list.append(float(val_loss) / processed_val_steps)
                early_stop = early_stop.update(val_loss_list[-1])
            else:
                early_stop = early_stop.update(loss_list[-1])
        else:
            early_stop = early_stop.update(loss_list[-1])

        if checkpoint_mngr is not None:
            ckpt = {
                "params": params,
                "opt_state": opt_state,
                "key": key,
                "epoch": epoch_idx,
            }
            checkpoint_mngr.save(epoch_idx, args=ocp.args.StandardSave(ckpt))

        if early_stop.should_stop:
            print(f"Early stopping at epoch {epoch_idx}")
            break

        if writer is not None:
            writer.add_scalar("Loss/train", loss_list[-1], epoch_idx)
        if writer is not None and val_loss_list:
            writer.add_scalar("Loss/val", val_loss_list[-1], epoch_idx)

        if val_loss_list:
            pbar.set_postfix(val_loss=val_loss_list[-1], train_loss=loss_list[-1])
        else:
            pbar.set_postfix(train_loss=loss_list[-1])
        pbar.update(1)

    return params, opt_state, loss_list, val_loss_list


def transform_batch(torch_batch: tuple, std: float) -> dict:
    trajectory = jnp.asarray(torch_batch[0])
    traj_in_metres = jax.vmap(transform_traj_to_metre)(trajectory)
    trajectory = traj_in_metres / std

    context = jnp.asarray(torch_batch[1])

    batch_size = trajectory.shape[0]
    seq_len = trajectory.shape[1]
    time = jnp.tile(
        jnp.arange(seq_len, dtype=jnp.float32)[None, :, None],
        (batch_size, 1, 1),
    )

    prediction_target_vals = trajectory[:, :]
    history_target_vals = trajectory[:, :1]
    history_context = context[:, :1]
    prediction_context = context[:, :-1]
    history_time = time[:, :1]
    prediction_time = time[:, 1:]

    jax_batch = {
        "history_target_vals": history_target_vals,
        "history_time": history_time,
        "prediction_time": prediction_time,
        "prediction_target_vals": prediction_target_vals,
        "history_context": history_context,
        "prediction_context": prediction_context,
    }
    return jax_batch


def model_predict(
    params: dict,
    y_history: jnp.ndarray,
    batch: dict,
    model: nn.Module,
    key: random.PRNGKey,
) -> jnp.ndarray:
    del key
    increment_prediction = jax.vmap(model.apply, in_axes=(None, 0, 0))(
        params,
        y_history,
        batch,
    )
    return increment_prediction


class AutoregressiveModel(BaseTrajectoryModel):
    def __init__(self, name: str, params: dict):
        super().__init__(name, params)

        default_params = {
            "model_type": "GRU",
            "num_layers": 2,
            "hidden_dims": 64,
            "time_emb": 64,
            "num_heads": 8,
            "max_len": 100,
            "context_encoder_type": "conv",  # conv, mlp, gap
        }

        model_type = params.get("model_type", default_params["model_type"])
        context_encoder_type = params.get(
            "context_encoder_type", default_params["context_encoder_type"]
        )
        layers = params.get("num_layers", default_params["num_layers"])
        hidden_dims = params.get("hidden_dims", default_params["hidden_dims"])

        time_emb = params.get("time_emb", default_params["time_emb"])
        self.ckpt_dir = os.path.join(config_paths.MODEL_PATH, "ckpts", self.name)

        prediction_length = params["prediction_length"]
        t_horizon = prediction_length

        max_len = params.get("max_len", default_params["max_len"])
        num_heads = params.get("num_heads", default_params["num_heads"])
        self.model = Autoregressive(
            context=True,
            hidden_size=hidden_dims,
            num_layers=layers,
            num_heads=num_heads,
            max_len=max_len,
            time_emb_dim=time_emb,
            T=t_horizon,
            context_encoder_type=context_encoder_type,
            backbone=model_type,
            output_size=2,
        )
        self.nn_params = None

    def center_and_scale_data(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        if len(trajectory.shape) == 2:
            return (trajectory - trajectory[0]) / self.std
        return (trajectory - trajectory[:, 0]) / self.std

    def scale(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        if len(trajectory.shape) == 2:
            start = trajectory[0]
        else:
            start = trajectory[:, 0]
        return (trajectory - start) / self.std + start

    def fit(
        self,
        train_dataloader: OceanTrainingDataset,
        val_dataloader: OceanTrainingDataset,
        train_params: Optional[dict] = None,
    ) -> "AutoregressiveModel":
        if train_params is None:
            train_params = {}
        return self._fit(train_dataloader, val_dataloader, train_params)

    def _fit(
        self,
        train_dataset: OceanTrainingDataset,
        val_dataset: OceanTrainingDataset,
        train_params: Optional[dict] = None,
    ) -> "AutoregressiveModel":
        if train_params is None:
            train_params = {}
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
        }

        train_dataset.set_context_var_idx(self.var_idx)
        val_dataset.set_context_var_idx(self.var_idx)

        traj_in_metres = vmap(transform_traj_to_metre)(train_dataset.trajectories)
        self.std = jnp.std(traj_in_metres[:, -1], axis=0)

        batch_size = train_params.get("batch_size", default_train_params["batch_size"])
        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_dataloader = data.DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=False,
            drop_last=True,
        )

        key = random.PRNGKey(train_params.get("seed", default_train_params["seed"]))
        log_path = os.path.join(self.ckpt_dir, "..", "logs", self.name)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        writer = SummaryWriter(log_dir=log_path)

        data_sample = train_dataset[0]
        trajectory = data_sample[0]
        y_hist = trajectory[:1]
        input_batch = {
            "history_target_vals": y_hist,
            "history_time": jnp.arange(1, dtype=jnp.float32)[:, None],
            "history_context": data_sample[1][:1],
            "prediction_context": data_sample[1][:-1],
            "prediction_time": jnp.arange(1, len(trajectory), dtype=jnp.float32)[
                :, None
            ],
        }
        params = self.model.init(
            key,
            y_history=y_hist,
            x=input_batch,
            train=True,
        )

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

        loss_list: list = []
        val_loss_list: list = []

        options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1)
        mngr = ocp.CheckpointManager(orbax_path, options=options)

        final_dir = os.path.join(self.ckpt_dir, "final")
        params_path = os.path.join(final_dir, "params.npz")
        force_retrain = train_params.get("force_retrain", False)
        if force_retrain or not os.path.isfile(params_path):
            params, opt_state, loss_list, val_loss_list = train(
                train_dataloader,
                optimizer,
                self.model,
                params,
                opt_state,
                key,
                num_epochs=train_params.get(
                    "num_epochs", default_train_params["num_epochs"]
                ),
                num_batches_per_epoch=train_params.get(
                    "num_batches_per_epoch",
                    default_train_params["num_batches_per_epoch"],
                ),
                loss_list=loss_list,
                checkpoint_mngr=mngr,
                val_dataloader=val_dataloader,
                val_loss_list=val_loss_list,
                writer=writer,
                std=self.std,
            )
            print("storing", self.ckpt_dir)
            if not os.path.isdir(final_dir):
                os.makedirs(final_dir)
            jnp.savez(
                os.path.join(final_dir, "params.npz"),
                params=params,
                loss_list=loss_list,
                val_loss_list=val_loss_list,
                std=self.std,
            )
            self.nn_params = params
        else:
            self.load_trained_model()

        writer.close()
        return self

    def load_trained_model(self) -> None:
        trained_model = jnp.load(
            os.path.join(self.ckpt_dir, "final", "params.npz"), allow_pickle=True
        )
        self.nn_params = trained_model["params"].tolist()
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
    ) -> jnp.ndarray:
        if self.nn_params is None:
            raise ValueError(
                "Model parameters are not loaded. Train or load the model before predict."
            )

        times = self._get_prediction_steps(start_time, prediction_length)
        prediction_length = len(times) - 1
        start_context = test_dataset.get_context(
            times[:1], start_position[None], interpolator
        )[..., self.var_idx]

        model_time = jnp.tile(
            jnp.arange(times.shape[0], dtype=jnp.float32)[None, :, None],
            (num_samples, 1, 1),
        )
        keys = random.split(key, num_samples + 1)
        _, subkeys = keys[0], keys[1:]

        @partial(vmap, in_axes=(0, 0, 0, None, None))
        def get_prediction_trajectory(
            single_key: random.PRNGKey,
            current_position: jnp.ndarray,
            incremental_prediction: jnp.ndarray,
            numeric_time: jnp.ndarray,
            i: int,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            del single_key
            incremental_scaled = incremental_prediction * self.std
            distance_in_metres = jnp.sqrt(
                incremental_scaled[:, 0] ** 2 + incremental_scaled[:, 1] ** 2
            )
            bearing_angle = jnp.degrees(
                jnp.arctan2(incremental_scaled[:, 1], incremental_scaled[:, 0])
            )
            dcoord = data_transform.inverse_haversine_grad(
                current_position, distance_in_metres[i], bearing_angle[i]
            )
            new_coord = current_position + dcoord
            new_context = test_dataset.get_context(
                numeric_time, new_coord[None], interpolator
            )[..., self.var_idx]
            return new_coord, new_context

        coords_predicted = [jnp.tile(start_position[None, None], (num_samples, 1, 1))]
        current_positions = coords_predicted[0][:, 0]
        current_context = jnp.tile(start_context[None], (num_samples, 1, 1, 1, 1))
        prediction_context_full = jnp.tile(
            start_context[None], (num_samples, prediction_length, 1, 1, 1)
        )
        batch = {
            "history_target_vals": coords_predicted[0],
            "history_time": model_time[:, :1],
            "prediction_time": model_time[:, 1:],
            "history_context": current_context,
            "prediction_context": prediction_context_full,
        }

        predict_func = jit(get_prediction_trajectory)
        for step_idx in range(prediction_length):
            batch["history_target_vals"] = current_positions[:, None, :]
            batch["history_time"] = model_time[:, step_idx : step_idx + 1]
            batch["prediction_time"] = model_time[:, step_idx + 1 :]
            batch["history_context"] = current_context
            prediction_context_full = prediction_context_full.at[
                :, step_idx : step_idx + 1
            ].set(current_context)
            batch["prediction_context"] = prediction_context_full[:, step_idx:, ...]

            incremental_prediction = jax.vmap(self.model.apply, in_axes=(None, 0, 0))(
                self.nn_params,
                batch["history_target_vals"],
                batch,
            )
            new_coord, new_context = predict_func(
                subkeys,
                current_positions,
                incremental_prediction,
                times[step_idx + 1 : step_idx + 2],
                0,
            )
            coords_predicted.append(new_coord[:, None])
            current_positions = new_coord
            current_context = new_context

        pred_coords = jnp.concatenate(coords_predicted, axis=1)
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

        nn_params_path = os.path.join(model_path, "nn_params.pkl")

        if self.nn_params is None:
            raise ValueError(
                "Model parameters (nn_params) are None. Train or load the model before exporting."
            )

        with open(nn_params_path, "wb") as f:
            pickle.dump({"params": self.nn_params, "std": self.std}, f)

    @classmethod
    def import_trained_model(
        cls, model_name: str, import_path: str | None = None
    ) -> "AutoregressiveModel":
        if import_path is None:
            import_path = os.path.join(config_paths.MODEL_PATH, "trained_models")
        model_path = os.path.join(import_path, model_name)
        config_file = os.path.join(model_path, "model_config.yml")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")

        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)

        nn_params_path = os.path.join(model_path, "nn_params.pkl")
        if not os.path.exists(nn_params_path):
            raise FileNotFoundError(
                f"NN parameters file {nn_params_path} does not exist."
            )

        with open(nn_params_path, "rb") as f:
            nn_params = pickle.load(f)

        model = cls(model_name, model_config)
        assert isinstance(nn_params, dict), "Loaded nn_params should be a dictionary."
        model.nn_params = nn_params["params"]
        model.std = nn_params["std"]

        return model
