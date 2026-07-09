import datetime
import os
import pickle
from typing import Any

import jax
import optax
import yaml
from flax.core import FrozenDict
from flax.training.early_stopping import EarlyStopping
from interpax import Interpolator3D
from jax import numpy as jnp
from jax import random, value_and_grad, vmap
from torch.utils import data
import logging

from codit.codit_module.direct_models import (
    AutoregressiveDirectPredictor,
    AutoregressiveTransformerDirectPredictor,
)
from codit.data_module import config_paths, data_transform
from codit.data_module.data_transform import transform_traj_to_metre
from codit.data_module.dataset import OceanDataset, OceanTrainingDataset
from codit.model_module.base_model import BaseTrajectoryModel


def make_teacher_forced_batch(
    trajectory: jnp.ndarray, context: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    return {
        "history_target_vals": trajectory[:, :-1],
        "prediction_target_vals": trajectory[:, 1:],
        "history_context": context[:, :-1],
        "prediction_context": context[:, 1:],
    }


def batched_predict(
    model: Any,
    params: FrozenDict | dict,
    batch: dict[str, jnp.ndarray],
    train: bool,
) -> jnp.ndarray:
    return vmap(
        lambda y_history, history_context, prediction_context: model.apply(
            params,
            y_history=y_history,
            x={
                "history_context": history_context,
                "prediction_context": prediction_context,
            },
            train=train,
        )
    )(
        batch["history_target_vals"],
        batch["history_context"],
        batch["prediction_context"],
    )


def direct_mse_loss(
    model: Any,
    params: FrozenDict | dict,
    batch: dict[str, jnp.ndarray],
    train: bool,
) -> jnp.ndarray:
    preds = batched_predict(model, params, batch, train=train)
    return jnp.mean((preds - batch["prediction_target_vals"]) ** 2)


class DirectPredictionModel(BaseTrajectoryModel):
    def __init__(self, name: str, params: dict) -> None:
        default_params = {
            "model_type": "GRU",
            "num_layers": 2,
            "hidden_dims": 64,
            "context_encoder_type": "conv",
            "max_len": 100,
            "num_heads": 8,
        }
        params = default_params | params
        super().__init__(name, params)
        self.type = "direct_prediction"
        self.ckpt_dir = os.path.join(
            config_paths.MODEL_PATH, "trained_models", self.name
        )

        if params["model_type"] == "GRU":
            self.model = AutoregressiveDirectPredictor(
                output_size=2,
                context=True,
                num_layers=params["num_layers"],
                hidden_size=params["hidden_dims"],
                context_encoder_type=params["context_encoder_type"],
            )
        elif params["model_type"] == "Transformer":
            self.model = AutoregressiveTransformerDirectPredictor(
                output_size=2,
                context=True,
                num_layers=params["num_layers"],
                hidden_size=params["hidden_dims"],
                num_heads=params["num_heads"],
                max_len=params["max_len"],
                context_encoder_type=params["context_encoder_type"],
            )
        else:
            raise ValueError(f"Unknown model type: {params['model_type']}")

        self.nn_params = None
        self.std = None

    def center_and_scale_data(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        if self.std is None:
            raise ValueError("Model std is not set.")
        if len(trajectory.shape) == 2:
            return (trajectory - trajectory[0]) / self.std
        return (trajectory - trajectory[:, :1]) / self.std

    def _fit(
        self,
        train_dataset: OceanTrainingDataset,
        val_dataset: OceanTrainingDataset,
        train_params: dict | None = None,
    ) -> "DirectPredictionModel":
        if train_params is None:
            train_params = {}
        default_train_params = {
            "seed": 0,
            "batch_size": 128,
            "num_epochs": 100,
            "num_batches_per_epoch": 100,
            "learning_rate": 1e-3,
            "optimizer": "adam",
            "patience": 10,
            "min_delta": 1e-4,
        }

        traj_in_metres = vmap(transform_traj_to_metre)(
            jnp.asarray(train_dataset.trajectories)
        )
        self.std = jnp.std(traj_in_metres[:, -1], axis=0)

        train_dataset.set_context_var_idx(self.var_idx)
        val_dataset.set_context_var_idx(self.var_idx)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=train_params.get(
                "batch_size", default_train_params["batch_size"]
            ),
            shuffle=True,
            drop_last=True,
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=train_params.get(
                "batch_size", default_train_params["batch_size"]
            ),
            shuffle=False,
            drop_last=True,
        )

        sample_traj, sample_context = train_dataset[0]
        sample_traj = self.center_and_scale_data(
            transform_traj_to_metre(jnp.asarray(sample_traj))
        )
        sample_batch = {
            "history_context": jnp.asarray(sample_context[:-1]),
            "prediction_context": jnp.asarray(sample_context[1:]),
        }
        params = self.model.init(
            random.PRNGKey(train_params.get("seed", default_train_params["seed"])),
            y_history=sample_traj[:-1],
            x=sample_batch,
            train=True,
        )

        optimizer_name = train_params.get(
            "optimizer", default_train_params["optimizer"]
        )
        learning_rate = train_params.get(
            "learning_rate", default_train_params["learning_rate"]
        )
        if optimizer_name == "adam":
            optimizer = optax.adam(learning_rate)
        elif optimizer_name == "adamw":
            optimizer = optax.adamw(learning_rate)
        elif optimizer_name == "sgd":
            optimizer = optax.sgd(learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_name}")
        opt_state = optimizer.init(params)

        def prepare_batch(torch_batch: tuple) -> dict[str, jnp.ndarray]:
            trajectory = jnp.asarray(torch_batch[0])
            context = jnp.asarray(torch_batch[1])
            normalized = vmap(transform_traj_to_metre)(trajectory)
            normalized = self.center_and_scale_data(normalized)
            return make_teacher_forced_batch(normalized, context)

        @jax.jit
        def train_step(
            params: FrozenDict | dict,
            opt_state: optax.OptState,
            batch: dict[str, jnp.ndarray],
        ) -> tuple[FrozenDict | dict, optax.OptState, jnp.ndarray]:
            loss, grads = value_and_grad(
                lambda p: direct_mse_loss(self.model, p, batch, True)
            )(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        @jax.jit
        def eval_step(
            params: FrozenDict | dict,
            batch: dict[str, jnp.ndarray],
        ) -> jnp.ndarray:
            return direct_mse_loss(self.model, params, batch, False)

        num_epochs = train_params.get("num_epochs", default_train_params["num_epochs"])
        num_batches_per_epoch = train_params.get(
            "num_batches_per_epoch",
            default_train_params["num_batches_per_epoch"],
        )

        best_params = params
        best_metric = jnp.inf
        early_stop = EarlyStopping(
            min_delta=float(
                train_params.get("min_delta", default_train_params["min_delta"])
            ),
            patience=int(
                train_params.get("patience", default_train_params["patience"])
            ),
        )
        for epoch_idx in range(num_epochs):
            logging.info(f"Epoch {epoch_idx + 1}/{num_epochs}")
            print(f"Epoch {epoch_idx + 1}/{num_epochs}")
            train_losses = []
            for batch_idx, torch_batch in enumerate(train_loader):
                batch = prepare_batch(torch_batch)
                params, opt_state, train_loss = train_step(params, opt_state, batch)
                train_losses.append(float(train_loss))
                if batch_idx + 1 >= num_batches_per_epoch:
                    break

            val_losses = []
            for torch_batch in val_loader:
                batch = prepare_batch(torch_batch)
                val_losses.append(eval_step(params, batch))

            if val_losses:
                metric = float(jnp.mean(jnp.asarray(val_losses)))
            else:
                metric = float(jnp.mean(jnp.asarray(train_losses)))

            if metric < best_metric:
                logging.info(
                    f"New best metric: {best_metric:.4f} Improvement: {best_metric - metric:.4f}"
                )
                print(
                    f"New best metric: {best_metric:.4f} Improvement: {best_metric - metric:.4f}"
                )
                best_metric = metric
                best_params = params

            early_stop = early_stop.update(metric)
            if early_stop.should_stop:
                break

        self.nn_params = best_params
        return self

    def export_trained_model(
        self,
        model_name: str | None = None,
        export_path: str | None = None,
    ) -> None:
        if model_name is None:
            model_name = self.name
        if export_path is None:
            export_path = os.path.join(config_paths.MODEL_PATH, "trained_models")
        if self.nn_params is None or self.std is None:
            raise ValueError("Model parameters are not set.")

        model_path = os.path.join(export_path, model_name)
        os.makedirs(model_path, exist_ok=False)
        with open(os.path.join(model_path, "model_config.yml"), "w") as file_obj:
            yaml.dump(self.params, file_obj, default_flow_style=False)
        with open(os.path.join(model_path, "nn_params.npz"), "wb") as file_obj:
            pickle.dump({"params": self.nn_params, "std": self.std}, file_obj)

    @classmethod
    def import_trained_model(
        cls,
        model_name: str,
        import_path: str | None = None,
    ) -> "DirectPredictionModel":
        if import_path is None:
            import_path = os.path.join(config_paths.MODEL_PATH, "trained_models")

        model_path = os.path.join(import_path, model_name)
        with open(os.path.join(model_path, "model_config.yml"), "r") as file_obj:
            params = yaml.safe_load(file_obj)
        with open(os.path.join(model_path, "nn_params.npz"), "rb") as file_obj:
            state = pickle.load(file_obj)

        model = cls(model_name, params)
        model.nn_params = state["params"]
        model.std = state["std"]
        return model

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
        del key
        if num_samples != 1:
            raise ValueError(
                "DirectPredictionModel.predict currently supports num_samples=1 only."
            )
        if self.nn_params is None or self.std is None:
            raise ValueError("Load or train the model before calling predict.")

        times = self._get_prediction_steps(start_time, prediction_length)
        current_coord = start_position
        coords = []
        history_target_vals = jnp.zeros((1, 2), dtype=jnp.float32)
        history_context = test_dataset.get_context(
            times[:1], start_position[None], interpolator
        )[..., self.var_idx]
        # OceanDataset.get_context returns (time, n_coords, H, W, C) at inference.
        # The direct model expects (time, H, W, C), so we drop n_coords (always 1 here).
        if history_context.ndim == 5:
            history_context = history_context[:, 0]

        for step_idx in range(1, len(times)):
            prediction_context_step = test_dataset.get_context(
                times[step_idx : step_idx + 1],
                current_coord[None],
                interpolator,
            )[..., self.var_idx]
            if prediction_context_step.ndim == 5:
                prediction_context_step = prediction_context_step[:, 0]

            # The direct model expects prediction_context length to match y_history length.
            # During autoregressive rollout we only observe one new context frame per step,
            # so we broadcast that frame across the current history length for the model call.
            prediction_context = jnp.tile(
                prediction_context_step,
                (history_target_vals.shape[0], 1, 1, 1),
            )

            preds = self.model.apply(
                self.nn_params,
                y_history=history_target_vals,
                x={
                    "history_context": history_context,
                    "prediction_context": prediction_context,
                },
                train=False,
            )
            next_normalized = preds[-1]
            next_displacement = next_normalized * self.std
            distance_in_metres = jnp.sqrt(
                next_displacement[0] ** 2 + next_displacement[1] ** 2
            )
            bearing_angle = jnp.degrees(
                jnp.arctan2(next_displacement[1], next_displacement[0])
            )
            dcoord = data_transform.inverse_haversine_grad(
                start_position,
                distance_in_metres,
                bearing_angle,
            )
            current_coord = start_position + dcoord
            coords.append(current_coord[None])
            history_target_vals = jnp.concatenate(
                [history_target_vals, next_normalized[None]],
                axis=0,
            )
            history_context = jnp.concatenate(
                [history_context, prediction_context_step],
                axis=0,
            )

        return jnp.concatenate(coords, axis=0)[None]
