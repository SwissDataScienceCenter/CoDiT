import jax
import jax.numpy as jnp
from jax import random

from codit.codit_module.brownian_noise_models import WalkNoiseModel


def diffusion_jump(
    key: jax.Array,
    y: jnp.ndarray,
    k: int | jnp.ndarray,
    alpha_bar: jnp.ndarray,
    noise_model: WalkNoiseModel,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform a jump in the diffusion process.

    Args:
        key: The random key.
        y: The current state of the diffusion process.
        k: Denoising step.
        alpha_bar: The alpha bar schedule.
        T: The time horizon.
        noise_model: The noise model.

    Returns:
        y_t: The new state of the diffusion process.
        noise_t: The noise at the new state.
    """
    key, subkey = random.split(key)
    B, noise_x = noise_model(key=subkey, time_steps=y.shape[-2] - 1)
    y = jnp.sqrt(alpha_bar[k]) * y + jnp.sqrt(1 - alpha_bar[k]) * B
    return y, noise_x


def forward_diffusion(
    y: jnp.ndarray,
    k: jnp.ndarray,
    key: jax.Array,
    schedule_dict: dict,
    noise_model: WalkNoiseModel,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform a forward diffusion step.

    Args:
        y: The current state of the diffusion process.
        k: Denoising step.
        key: The random key.
        schedule_dict: Diffusion schedule.
        T: The time horizon.
        noise_model: The noise model.

    Returns:
        y_t: The new state of the diffusion process.
        noise_t: The noise at the new state.
    """
    y_t, noise_t = diffusion_jump(key, y, k, schedule_dict["alpha_bar"], noise_model)
    return y_t, noise_t
