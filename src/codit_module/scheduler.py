from jax import numpy as jnp


def linear_schedule(K: int, beta_min: float = 1e-4, beta_max: float = 0.2) -> dict:
    beta = jnp.linspace(beta_min, beta_max, num=K)
    alpha = 1 - beta
    alpha_bar = jnp.cumprod(alpha)
    schedule_dict = {"beta": beta, "alpha": alpha, "alpha_bar": alpha_bar, "K": K}
    return schedule_dict


def cosine_schedule(K: int, beta_min: float = 1e-4, beta_max: float = 0.2) -> dict:
    x = jnp.linspace(0, 1, num=K)[::-1]
    cos = (1 + jnp.cos(jnp.pi * x)) / 2
    beta = beta_min + (beta_max - beta_min) * cos
    alpha = 1 - beta
    alpha_bar = jnp.cumprod(alpha)
    schedule_dict = {"beta": beta, "alpha": alpha, "alpha_bar": alpha_bar, "K": K}
    return schedule_dict
