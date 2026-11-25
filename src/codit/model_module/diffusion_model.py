import datetime
import os
import pickle
import shutil
import warnings
from typing import Callable, Optional

# Suppress the SerializationWarning
from functools import partial

import jax
import numpy as np
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
from codit.data_module.data_transform import transform_traj_to_metre
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


# Heun sampling is only trusted when the forward process ends close to pure noise;
# see the warning in CoDiTModel.set_sampler.
HEUN_MAX_TERMINAL_ALPHA_BAR = 1e-3


def edm_sigmas(alpha_bar: jnp.ndarray) -> jnp.ndarray:
    """Noise level of each DDPM step in EDM (variance-exploding) form.

    A DDPM state ``y_k = sqrt(ab_k) y_0 + sqrt(1 - ab_k) B`` corresponds to the EDM
    state ``x = y_k / sqrt(ab_k) = y_0 + sigma_k B`` with
    ``sigma_k = sqrt((1 - ab_k) / ab_k)``.
    """
    return jnp.sqrt((1.0 - alpha_bar) / alpha_bar)


def heun_step_indices(
    schedule_dict: dict, num_steps: int, rho: float = 7.0
) -> np.ndarray:
    """Pick the DDPM steps the Heun sampler visits.

    Uses the EDM time discretisation (Karras et al. 2022, Eq. 5) between the
    smallest and largest trained noise level, snapped (in log-sigma) to the
    nearest trained step, so the network is only ever queried at step indices
    it was trained on.

    Args:
        schedule_dict: DDPM schedule (needs ``alpha_bar``).
        num_steps: Number of noise levels to visit (EDM's N). Clipped to ``[2, K]``.
        rho: EDM spacing exponent; larger values put more steps at low noise.

    Returns:
        Strictly decreasing integer step indices, starting at ``K - 1`` and ending
        at ``0``. Can be shorter than ``num_steps`` if snapping merges levels.
    """
    sigmas = np.asarray(edm_sigmas(schedule_dict["alpha_bar"]), dtype=np.float64)
    K = len(sigmas)
    num_steps = int(np.clip(num_steps, 2, K))
    s_min, s_max = sigmas[0], sigmas[-1]
    ramp = np.linspace(0.0, 1.0, num_steps)
    targets = (
        s_max ** (1 / rho) + ramp * (s_min ** (1 / rho) - s_max ** (1 / rho))
    ) ** rho
    idx = np.abs(np.log(sigmas)[:, None] - np.log(targets)[None, :]).argmin(axis=0)
    return np.unique(idx)[::-1].astype(np.int32)


def _ode_slope(
    params: FrozenDict | dict,
    model: nn.Module,
    noise_model: WalkNoiseModel,
    schedule_dict: dict,
    batch: dict,
    x: jnp.ndarray,
    k_idx: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """dx/dsigma of the probability-flow ODE at EDM state ``x`` and DDPM step ``k_idx``.

    For ``x = y_0 + sigma * L n`` (``L`` = ``noise_model.transform``) the ODE is
    ``dx/dsigma = (x - D(x; sigma)) / sigma = L E[n | x]``, and ``E[n | x]`` is
    exactly what the DDPM network predicts. The history handling mirrors
    ``backward_denoising_step``.
    """
    k = jnp.array([k_idx])
    y_vp = x * jnp.sqrt(schedule_dict["alpha_bar"][k_idx])
    y_history_input = noise_model.center_data(
        batch["history_target_vals"][..., ::-1, :]
    )
    y_noised_history, _ = noising_functions.forward_diffusion(
        y_history_input, k, key, schedule_dict, noise_model
    )
    noise_pred = model.apply(params, y_vp[1:], y_noised_history[..., ::-1, :], batch, k)
    assert isinstance(noise_pred, jnp.ndarray), "Model output is not a jnp.ndarray"
    return noise_model.transform(noise_pred)


def heun_denoise(
    params: FrozenDict | dict,
    y_init: jnp.ndarray,
    batch: dict,
    model: nn.Module,
    noise_model: WalkNoiseModel,
    key: jax.Array,
    schedule_dict: dict,
    step_indices: np.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic 2nd-order Heun sampler (EDM Algorithm 1) for a DDPM-trained model.

    Solves the probability-flow ODE from the largest trained noise level down to
    zero, visiting only ``step_indices``. Uses ``2 * len(step_indices) - 1``
    network evaluations (Heun steps between levels, a final Euler step to
    sigma = 0), against ``K`` for the ancestral ``denoise``.

    The output is a deterministic function of ``y_init``: no noise is added
    during sampling (``key`` only feeds the history noising, which is zero for
    the single-point history used in CoDiT).

    Args:
        params: Network parameters.
        y_init: Initial noise in DDPM space at step ``step_indices[0]``, same
            distribution as for ``denoise`` (a noise-model path).
        batch: Conditioning batch (history and context).
        model: The denoising network.
        noise_model: The noise model.
        key: PRNG key.
        schedule_dict: DDPM schedule.
        step_indices: Decreasing DDPM step indices, from ``heun_step_indices``.

    Returns:
        y: The denoised trajectory (centered, scaled), shape like ``y_init``.
        steps: State after each level in DDPM space, shape
            ``(len(step_indices), *y_init.shape)``; the last entry equals ``y``.
    """
    alpha_bar = schedule_dict["alpha_bar"]
    sigmas = edm_sigmas(alpha_bar)
    idx = jnp.asarray(step_indices)

    def slope(x: jnp.ndarray, k_idx: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        return _ode_slope(
            params, model, noise_model, schedule_dict, batch, x, k_idx, key
        )

    def heun_step(carry: tuple, ks: tuple) -> tuple[tuple, jnp.ndarray]:
        key, x = carry
        k_cur, k_next = ks
        key, key_cur, key_next = random.split(key, 3)
        h = sigmas[k_next] - sigmas[k_cur]
        d_cur = slope(x, k_cur, key_cur)
        x_euler = x + h * d_cur
        d_next = slope(x_euler, k_next, key_next)
        x = x + 0.5 * h * (d_cur + d_next)
        x = x.at[:1].set(0)
        return (key, x), x * jnp.sqrt(alpha_bar[k_next])

    x = y_init / jnp.sqrt(alpha_bar[idx[0]])
    (key, x), steps = lax.scan(heun_step, (key, x), (idx[:-1], idx[1:]))
    # final Euler step from the smallest trained level to sigma = 0
    key, subkey = random.split(key)
    x = x - sigmas[idx[-1]] * slope(x, idx[-1], subkey)
    x = x.at[:1].set(0)
    steps = jnp.concatenate([steps, x[None]], axis=0)
    return x, steps


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
    """Conditional denoising diffusion model for ocean drifter trajectory prediction.

    CoDiT frames trajectory generation as a reverse diffusion process conditioned on
    spatiotemporal ocean context (currents, wind, bathymetry). At inference time it
    draws ``num_samples`` trajectory samples from the learned conditional distribution
    by running K reverse-diffusion steps.

    Supported denoiser architectures (``params["model_type"]``):

    - ``"GRU"``: autoregressive GRU denoiser (default, faster to train)
    - ``"Transformer"``: autoregressive Transformer denoiser (more expressive)

    Args:
        name: Unique model identifier used as the checkpoint directory name.
        params: Model hyperparameters. Unset keys fall back to internal defaults:

            - ``prediction_length`` (int): number of trajectory steps to predict (**required**)
            - ``dt`` (float): time step in fractional days (default ``0.1``; set to ``1/prediction_length``)
            - ``model_type`` (str): ``"GRU"`` or ``"Transformer"`` (default ``"GRU"``)
            - ``context_encoder_type`` (str): ``"conv"``, ``"mlp"``, or ``"gap"`` (default ``"conv"``)
            - ``K`` (int): number of diffusion steps (default ``100``)
            - ``num_layers`` (int): GRU/Transformer depth (default ``2``)
            - ``hidden_dims`` (int): hidden state dimensionality (default ``64``)
            - ``diffusion_emb`` (int): diffusion-timestep embedding size (default ``64``)
            - ``vars`` (list[str]): context variable names used to select ``var_idx``
            - ``noise`` (str): noise model type, ``"brownian_motion"`` (default)
            - ``beta_min`` / ``beta_max`` (float): diffusion schedule bounds (defaults ``1e-4`` / ``0.02``)
            - ``aligned_context`` (bool): ``False`` for the legacy context layout of older
              checkpoints (default ``True``)
            - ``sampler`` (str): ``"ddpm"`` (ancestral, K steps; default, used for the
              paper) or ``"heun"`` (deterministic EDM Heun sampler on a subset of the
              trained steps). Can be changed after loading with ``set_sampler``.
            - ``num_sampling_steps`` (int): noise levels visited by the Heun sampler
              (default ``18``, i.e. 35 network evaluations)
            - ``rho`` (float): EDM step-spacing exponent for the Heun sampler (default ``7``)

    Example::

        model = CoDiTModel(
            name="codit_GRU_open_ocean",
            params={
                "prediction_length": 72,
                "dt": 1 / 72,
                "model_type": "GRU",
                "context_encoder_type": "conv",
                "K": 100,
                "vars": ["hycom_current", "copernicus_current", "era5_wind", "bathymetry"],
            },
        )
    """

    def __init__(self, name: str, params: dict) -> None:
        """Initialise a CoDiT diffusion model.

        See the class docstring for the full list of supported ``params`` keys.
        Unset keys fall back to the internal defaults; ``prediction_length`` and
        ``vars`` are the only keys the caller is expected to always provide.

        Args:
            name: Unique model identifier; also the checkpoint subdirectory name
                under ``config_paths.MODEL_PATH/ckpts/``.
            params: Hyperparameter dictionary. See class docstring for details.

        Attributes set:
            type (str): Always ``"diffusion"``.
            model (nn.Module): Flax denoiser (GRU or Transformer).
            noise_model (WalkNoiseModel): Brownian noise process.
            schedule_dict (dict): Precomputed linear beta schedule for K steps.
            ckpt_dir (str): Absolute path to the model's checkpoint directory.
            nn_params: Set to ``None`` at construction; populated by ``fit`` or
                ``import_trained_model``.
        """
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
            "aligned_context": True,
            "sampler": "ddpm",
            "num_sampling_steps": 18,
            "rho": 7.0,
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
        self._denoise_func: Optional[Callable] = None  # cached jit-compiled step function
        # same, batched over trajectories; keyed by "interpolator shared across batch"
        self._denoise_func_batch: dict[bool, Callable] | None = None
        self.aligned_context = bool(params["aligned_context"])
        self.set_sampler(
            params["sampler"], params["num_sampling_steps"], params["rho"]
        )

    def set_sampler(
        self, sampler: str, num_sampling_steps: int = 18, rho: float = 7.0
    ) -> None:
        """Choose the reverse-diffusion sampler used by ``predict``.

        Args:
            sampler: ``"ddpm"`` (ancestral, all K steps) or ``"heun"``
                (deterministic EDM Heun ODE sampler, ``2 * num_sampling_steps - 1``
                network evaluations). Both work with the same trained weights.
            num_sampling_steps: Noise levels visited by the Heun sampler.
            rho: EDM step-spacing exponent for the Heun sampler.
        """
        if sampler not in ("ddpm", "heun"):
            raise ValueError(f"Unknown sampler {sampler!r}: use 'ddpm' or 'heun'")
        alpha_bar_end = float(self.schedule_dict["alpha_bar"][-1])
        if sampler == "heun" and alpha_bar_end > HEUN_MAX_TERMINAL_ALPHA_BAR:
            warnings.warn(
                f"The diffusion schedule ends at alpha_bar={alpha_bar_end:.3g}, far from "
                "pure noise, but sampling starts from pure noise. The deterministic Heun "
                "sampler cannot correct that mismatch (ancestral DDPM partly can): "
                "expect too wide ensembles and too short mean drift. Use 'ddpm' for "
                "this model, or retrain with a schedule that reaches noise (e.g. "
                "beta_max=0.2 for K=100). See docs/EDM_IMPROVEMENTS.md.",
                stacklevel=2,
            )
        self.sampler = sampler
        self.params["sampler"] = sampler
        self.params["num_sampling_steps"] = int(num_sampling_steps)
        self.params["rho"] = float(rho)
        self._heun_indices = heun_step_indices(
            self.schedule_dict, num_sampling_steps, rho
        )
        self._denoise_func = self._denoise_func_batch = None  # invalidate cached jit (sampler changed)

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
        """Train CoDiT via denoising score matching.

        Iterates ``num_epochs`` epochs of ``num_batches_per_epoch`` gradient steps.
        Each step samples a batch, applies the forward diffusion process, predicts
        the noise with the denoiser, and minimises the MSE between predicted and
        true noise. Early stopping is applied on the validation loss.

        Args:
            train_dataset: Training split with trajectories and context arrays loaded.
            val_dataset: Validation split used for early stopping.
            train_params: Training hyperparameters. Supported keys:

                - ``num_epochs`` (int): maximum training epochs (default ``1000``)
                - ``num_batches_per_epoch`` (int): gradient steps per epoch (default ``100``)
                - ``batch_size`` (int): samples per gradient step (default ``128``)
                - ``learning_rate`` (float): initial learning rate (default ``1e-4``)
                - ``optimizer`` (str): ``"adam"``, ``"adamw"``, or ``"sgd"`` (default ``"adam"``)
                - ``scheduler``: an ``optax`` schedule instance or ``None`` for constant LR
                - ``early_stopping_patience`` (int): val-epochs without improvement before stopping (default ``10``)
                - ``early_stopping_min_delta`` (float): minimum improvement threshold (default ``1e-4``)
                - ``force_retrain`` (bool): retrain even if a checkpoint already exists (default ``False``)

        Returns:
            self, with ``nn_params`` populated and ``trained_successfully`` set to ``True``.
        """
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
            "val_batch_size": 128,
        }

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
            val_dataset,
            batch_size=train_params.get(
                "val_batch_size", default_train_params["val_batch_size"]
            ),
            shuffle=False,
            drop_last=True,
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
                aligned_context=self.aligned_context,
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
                aligned_context=self.aligned_context,
            )
            self.nn_params = params
            self._denoise_func = self._denoise_func_batch = None  # invalidate cached jit (nn_params changed)
        return self

    def load_trained_model(self) -> None:
        # params, loss_list, val_loss_list, std = restore_checkpoint(
        trained_model = jnp.load(
            os.path.join(self.ckpt_dir, "final", "params.npz"), allow_pickle=True
        )
        # older checkpoints use the legacy context layout
        ckpt_aligned = bool(trained_model["aligned_context"]) \
            if "aligned_context" in trained_model else False
        if ckpt_aligned != self.aligned_context:
            raise ValueError(
                f"checkpoint {self.ckpt_dir} was trained with aligned_context={ckpt_aligned}, "
                f"the model has aligned_context={self.aligned_context}: set it to match, or retrain")
        self.nn_params = trained_model["params"]
        self._denoise_func = self._denoise_func_batch = None  # invalidate cached jit (nn_params changed)
        self.std = trained_model["std"]

    def _context_slot(self, i: int) -> int:
        """Slot in ``prediction_context`` of the context sampled at predicted position ``i``."""
        return i - 1 if self.aligned_context else i

    def _prepare_rollout(
        self,
        start_position: jax.Array,
        start_time: datetime.datetime,
        prediction_length: int,
        interpolator: Interpolator3D,
        test_dataset: OceanDataset,
        num_samples: int,
        key: jax.Array,
        initial_y: jnp.ndarray | None,
    ) -> tuple:
        """Eager per-trajectory setup shared by ``predict`` and ``predict_batch``.

        Returns ``(times, start_context, W, subkeys, initial_y, batch)``: the
        prediction time grid, the context at the start, the Brownian paths used
        by the DDPM sampler, the per-sample PRNG keys of the reverse process,
        the reverse-process starting state and the initial conditioning batch.
        The PRNG splits are kept exactly as in the original ``predict``.
        """
        times = self._get_prediction_steps(start_time, prediction_length)
        prediction_length = len(times) - 1
        start_context = test_dataset.get_context(
            times[:1], start_position[None], interpolator
        )[..., self.var_idx]

        key, _ = random.split(key)
        K = self.schedule_dict["K"]
        keys = random.split(key, num_samples + 1)
        key, subkeys = keys[0], keys[1:]

        def get_noise_model(
            key: jax.Array, prediction_length: int
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            keys = random.split(key, K + 1)
            key, subkeys = keys[0], keys[1:]
            return vmap(self.noise_model, in_axes=(0, None))(subkeys, prediction_length)

        W, _ = vmap(get_noise_model, in_axes=(0, None))(subkeys, prediction_length)
        keys = random.split(key, num_samples + 1)
        key, subkeys = keys[0], keys[1:]
        sampled_y, _ = vmap(self.noise_model, in_axes=(0, None))(
            subkeys, prediction_length
        )
        if initial_y is None:
            initial_y = sampled_y
        elif initial_y.shape != sampled_y.shape:
            raise ValueError(
                f"initial_y must have shape {sampled_y.shape}, got {initial_y.shape}"
            )

        keys = random.split(key, num_samples + 1)
        key, subkeys = keys[0], keys[1:]
        batch = {
            "history_target_vals": jnp.zeros((num_samples, 1, 2)),
            "history_context": jnp.tile(start_context[None], (num_samples, 1, 1, 1, 1)),
            "prediction_context": jnp.tile(
                start_context[None], (num_samples, prediction_length, 1, 1, 1)
            ),
        }
        return times, start_context, W, subkeys, initial_y, batch

    def _make_denoise_step(
        self, test_dataset: OceanDataset, return_denoising_steps: bool
    ) -> Callable:
        """Build the per-hour step: full reverse diffusion, then the new position
        and its context. Vectorised over samples; not yet jit-compiled."""

        @partial(vmap, in_axes=(0, 0, 0, 0, None, None, None, None))
        def _get_denoised_trajectory(
            key: jax.Array,
            W: jnp.ndarray,
            initial_y: jnp.ndarray,
            batch: dict,
            numeric_time: jnp.ndarray,
            i: jax.Array,
            start_pos: jnp.ndarray,
            interp: Interpolator3D,
        ) -> tuple:
            if self.sampler == "heun":
                denoised, denoised_steps = heun_denoise(
                    self.nn_params,  # type: ignore
                    initial_y,
                    batch,
                    self.model,
                    self.noise_model,
                    key,
                    self.schedule_dict,
                    self._heun_indices,
                )
            else:
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
                start_pos,
                jax.lax.dynamic_index_in_dim(distance_in_metres, i, axis=0, keepdims=False),
                jax.lax.dynamic_index_in_dim(bearing_angle, i, axis=0, keepdims=False),
            )
            new_coord = start_pos[None] + dcoord
            new_context = test_dataset.get_context(
                numeric_time, new_coord, interp
            )[..., self.var_idx]
            # NOTE: return_denoising_steps is a Python bool resolved at trace
            # time. This function is never cached (self._denoise_func) when
            # return_denoising_steps=True, so the two return arities are safe.
            if return_denoising_steps:
                return new_coord, new_context, denoised_steps
            else:
                return new_coord, new_context

        return _get_denoised_trajectory

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
        initial_y: jnp.ndarray | None = None,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Sample trajectory predictions from the trained diffusion model.

        Runs the reverse diffusion chain independently ``num_samples`` times, each
        starting from a different Brownian-motion initialisation. The sampler is set
        by ``set_sampler``: ``"ddpm"`` runs all K ancestral steps, ``"heun"`` runs the
        deterministic EDM Heun sampler on a subset of them.

        Args:
            start_position: Lat/lon of the drifter at ``start_time``, shape ``(2,)`` in decimal degrees.
            start_time: Absolute UTC datetime for the first prediction step.
            prediction_length: Number of hourly time steps to predict.
            interpolator: Pre-built ``interpax.Interpolator3D`` covering the spatial and
                temporal domain of the prediction window.
            test_dataset: ``OceanDataset`` providing context feature extraction via ``get_context()``.
            num_samples: Number of independent trajectory samples to draw.
            key: JAX PRNG key used for reproducible noise sampling.
            return_denoising_steps: If ``True``, also return the intermediate denoising
                trajectory of shape ``(num_samples, S, prediction_length, 2)``, with
                ``S = K`` for ``"ddpm"`` and ``S = len(step_indices)`` for ``"heun"``.
            initial_y: Diagnostics only. Overrides the starting state of the reverse
                process, shape ``(num_samples, prediction_length + 1, 2)``, in the
                model's centred and scaled units. Default ``None`` draws it from
                the noise model. Used by
                ``scripts/dev/debug/diagnose_prior_mismatch.py``.

        Returns:
            Predicted trajectories as ``jnp.ndarray`` of shape ``(num_samples, prediction_length, 2)``
            in decimal degrees (lat, lon). If ``return_denoising_steps=True``, returns a tuple
            ``(trajectories, denoising_steps)``.
        """
        times, start_context, W, subkeys, initial_y, batch = self._prepare_rollout(
            start_position, start_time, prediction_length, interpolator,
            test_dataset, num_samples, key, initial_y,
        )
        prediction_length = len(times) - 1

        if self._denoise_func is None or return_denoising_steps:
            denoise_func = jit(
                self._make_denoise_step(test_dataset, return_denoising_steps)
            )
            if not return_denoising_steps:
                self._denoise_func = denoise_func
        else:
            denoise_func = self._denoise_func

        coords_denoised = [jnp.tile(start_position[None], (num_samples, 1, 1))]

        for i in range(1, prediction_length + 1):
            if return_denoising_steps:
                new_coord, new_context, denoise_steps_scaled = denoise_func(
                    subkeys, W, initial_y, batch, times[i : i + 1], jnp.int32(i),
                    start_position, interpolator,
                )
            else:
                new_coord, new_context = denoise_func(
                    subkeys, W, initial_y, batch, times[i : i + 1], jnp.int32(i),
                    start_position, interpolator,
                )
            coords_denoised.append(new_coord)
            j = self._context_slot(i)
            batch["prediction_context"] = (
                batch["prediction_context"].at[:, j : j + 1].set(new_context)
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

    def predict_batch(
        self,
        start_positions: jax.Array,
        start_times: list[datetime.datetime],
        prediction_length: int,
        interpolator: Interpolator3D | list[Interpolator3D],
        test_dataset: OceanDataset,
        num_samples: int = 1,
        keys: jax.Array | None = None,
        initial_y: jnp.ndarray | None = None,
        on_step: Callable[[int], None] | None = None,
    ) -> jnp.ndarray:
        """Run ``predict`` for several trajectories at once.

        Trajectory ``b`` gets exactly the computation of
        ``predict(start_positions[b], start_times[b], ..., key=keys[b],
        initial_y=initial_y[b])``: the same PRNG splits, noise paths and reverse
        process. The trajectories are only evaluated side by side (one more
        ``vmap``), which keeps the GPU busy instead of running the latency-bound
        reverse process one trajectory at a time. Results agree with ``predict``
        up to floating-point rounding.

        Memory grows with ``B``: the conditioning batch holds
        ``B * num_samples * prediction_length`` context grids, and per-trajectory
        interpolators are stacked on the device (an ``OceanDataset`` item's
        interpolator stores its data plus an 8x derivative array, ~0.7 GB for a
        72 h open_ocean patch).

        Args:
            start_positions: Start lat/lon per trajectory, shape ``(B, 2)``.
            start_times: ``B`` start datetimes.
            prediction_length: Number of hourly time steps to predict.
            interpolator: One ``Interpolator3D`` shared by all trajectories, or a
                list with one per trajectory (as returned by ``OceanDataset``
                items). Per-trajectory interpolators must have identical shapes,
                i.e. come from the same patch and prediction length.
            test_dataset: ``OceanDataset`` providing ``get_context()``.
            num_samples: Samples per trajectory.
            keys: One PRNG key per trajectory, shape ``(B, 2)`` (or typed keys of
                shape ``(B,)``). Default: ``random.split(PRNGKey(0), B)``.
            initial_y: Diagnostics only. Starting states, shape
                ``(B, num_samples, prediction_length + 1, 2)``; see ``predict``.
            on_step: Optional callback, called with the hour ``i`` after each of
                the ``prediction_length`` autoregressive steps (e.g. to advance a
                progress bar).

        Returns:
            Predicted trajectories, shape ``(B, num_samples, prediction_length, 2)``,
            in decimal degrees (lat, lon).
        """
        n = len(start_times)
        if keys is None:
            keys = random.split(random.PRNGKey(0), n)
        if start_positions.shape[0] != n or keys.shape[0] != n:
            raise ValueError(
                f"got {start_positions.shape[0]} start positions, {n} start times "
                f"and {keys.shape[0]} keys; they must match"
            )
        if initial_y is not None and initial_y.shape[0] != n:
            raise ValueError(f"initial_y has {initial_y.shape[0]} rows, expected {n}")
        shared = not isinstance(interpolator, (list, tuple))
        interps = [interpolator] * n if shared else list(interpolator)
        if len(interps) != n:
            raise ValueError(f"got {len(interps)} interpolators for {n} trajectories")

        prepared = [
            self._prepare_rollout(
                start_positions[b], start_times[b], prediction_length, interps[b],
                test_dataset, num_samples, keys[b],
                None if initial_y is None else initial_y[b],
            )
            for b in range(n)
        ]
        times, _, W, subkeys, init_y, batch = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *prepared
        )
        prediction_length = times.shape[1] - 1

        if not shared:
            interpolator = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *interps)
        if self._denoise_func_batch is None:
            self._denoise_func_batch = {}
        if shared not in self._denoise_func_batch:
            self._denoise_func_batch[shared] = jit(
                vmap(
                    self._make_denoise_step(test_dataset, False),
                    in_axes=(0, 0, 0, 0, 0, None, 0, None if shared else 0),
                )
            )
        denoise_func = self._denoise_func_batch[shared]

        coords_denoised = [
            jnp.tile(start_positions[:, None, None], (1, num_samples, 1, 1))
        ]
        for i in range(1, prediction_length + 1):
            new_coord, new_context = denoise_func(
                subkeys, W, init_y, batch, times[:, i : i + 1], jnp.int32(i),
                start_positions, interpolator,
            )
            coords_denoised.append(new_coord)
            j = self._context_slot(i)
            batch["prediction_context"] = (
                batch["prediction_context"].at[:, :, j : j + 1].set(new_context)
            )
            if on_step is not None:
                new_coord.block_until_ready()  # report finished work, not dispatch
                on_step(i)
        return jnp.concatenate(coords_denoised, axis=2)[:, :, 1:]

    def export_trained_model(
        self, model_name: str | None = None, export_path: str | None = None
    ) -> None:
        """Save the model's neural-network parameters to a checkpoint directory.

        Args:
            model_name: Checkpoint folder name. Defaults to ``self.name``.
            export_path: Parent directory for the checkpoint. Defaults to
                ``config_paths.MODEL_PATH``.
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
        """Restore a trained model from a checkpoint directory.

        Args:
            model_name: Name of the checkpoint directory to load.
            import_path: Parent directory to search. Defaults to
                ``config_paths.MODEL_PATH``.

        Returns:
            A ``CoDiTModel`` instance with parameters restored from the checkpoint.

        Raises:
            FileNotFoundError: If ``model_config.yml`` does not exist in the checkpoint
                directory.
            FileNotFoundError: If ``nn_params.npz`` does not exist in the checkpoint
                directory.
        """
        if import_path is None:
            import_path = os.path.join(config_paths.MODEL_PATH, "trained_models")
        model_path = os.path.join(import_path, model_name)
        config_file = os.path.join(model_path, "model_config.yml")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")

        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)
        # older checkpoints use the legacy context layout
        model_config.setdefault("aligned_context", False)

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
