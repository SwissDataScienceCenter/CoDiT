from flax import linen as nn
from flax import struct
from jax import lax, random
from jax import numpy as jnp
import jax


def sample_white_noise(
    key: jax.Array, num_dims: int, time_steps: int, dt: float, sigma: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample a Brownian motion.

    Args:
        key: The random key.
        num_dims: The number of dimensions.
        T: The time horizon.
        dt: The time step.
        sigma: The diffusion coefficient of the Brownian motion.

    Returns:
        W: The Brownian path.
        dW: The increments of the Brownian motion.
    """
    key, subkey = random.split(key)
    dW = sigma * random.normal(subkey, (time_steps, num_dims))
    W = jnp.concatenate([jnp.zeros((1, num_dims)), dW], axis=0)
    return W, dW


def make_brownian_path(dW: jnp.ndarray, dt: float, sigma: float) -> jnp.ndarray:
    """
    Generate a Brownian path from a sequence of increments dW.

    Args:
        dW: The increments of the Brownian motion.
        dt: The time step.
        sigma: The diffusion coefficient of the Brownian motion.

    Returns:
        W: The Brownian path.
    """
    dims = dW.shape[-1]
    W = sigma * jnp.cumsum(dW, axis=0) * jnp.sqrt(dt)
    W = jnp.concatenate([jnp.zeros((1, dims)), W], axis=0)
    return W


def make_ou_path(
    dW: jnp.ndarray, dt: float, sigma: float, theta: float = 1
) -> jnp.ndarray:
    def ou_step(X: jnp.ndarray, dW: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        X = X - theta * X * dt + sigma * jnp.sqrt(dt) * dW
        return X, X

    X0 = jnp.zeros(dW.shape[-1])
    _, X = lax.scan(ou_step, X0, dW)
    X = jnp.concatenate([X0[None], X], axis=0)
    return X


def make_brownian_bridge(dW: jnp.ndarray, dt: float, sigma: float) -> jnp.ndarray:
    """
    Generate a Brownian bridge from a sequence of increments dW.

    Args:
        dW: The increments of the Brownian motion.
        T: The time horizon.
        dt: The time step.
        sigma: The diffusion coefficient of the Brownian motion.

    Returns:
        B: The Brownian bridge.
    """

    W = make_brownian_path(dW, dt, sigma)
    T = len(W) * dt
    t_range = jnp.linspace(0, T, int(T / dt))
    B = W - t_range[:, None] * W[-1] / T
    return B


def make_ou_bridge(
    dW: jnp.ndarray, dt: float, sigma: float, theta: float = 1
) -> jnp.ndarray:
    T = (len(dW) + 1) * dt

    def ou_step(carry: tuple, dW: jnp.ndarray) -> tuple[tuple, jnp.ndarray]:
        X, t = carry
        drift = -theta * X - 2.0 * X * jnp.exp(2 * theta * t) / (
            jnp.exp(2 * theta * T) - jnp.exp(2 * theta * t)
        )
        X = X + drift * dt + sigma * jnp.sqrt(dt) * dW
        t += dt
        return (X, t), X

    X0 = jnp.zeros(dW.shape[-1])
    init_carry = (X0, dt)
    carry, X = lax.scan(ou_step, init_carry, dW[:-1])
    X = jnp.concatenate([X0[None], X, X0[None]], axis=0)
    return X


def sample_brownian_motion(
    key: jax.Array, num_dims: int, time_steps: int, dt: float, sigma: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample a Brownian motion.

    Args:
        key: The random key.
        num_dims: The number of dimensions.
        T: The time horizon.
        dt: The time step.
        sigma: The diffusion coefficient of the Brownian motion.

    Returns:
        W: The Brownian path.
        dW: The increments of the Brownian motion.
    """
    key, subkey = random.split(key)
    dW = random.normal(subkey, (time_steps, num_dims))
    W = make_brownian_path(dW, dt, sigma)
    return W, dW


def sample_ou_motion(
    key: jax.Array,
    num_dims: int,
    time_steps: int,
    dt: float,
    sigma: float,
    theta: float = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample an Ornstein-Uhlenbeck process.

    Args:
        key: The random key.
        num_dims: The number of dimensions.
        T: The time horizon.
        dt: The time step.
        sigma: The diffusion coefficient of the Ornstein-Uhlenbeck process.
        theta: The mean reversion rate.

    Returns:
        X: The Ornstein-Uhlenbeck path.
        dX: The increments of the Ornstein-Uhlenbeck process.
    """
    key, subkey = random.split(key)
    dW = random.normal(subkey, (time_steps, num_dims))
    X = make_ou_path(dW, dt, sigma, theta)
    return X, dW


def sample_brownian_bridge(
    key: jax.Array, num_dims: int, time_steps: int, dt: float, sigma: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample a Brownian bridge.

    Args:
        key: The random key.
        num_dims: The number of dimensions.
        T: The time horizon.
        dt: The time step.
        sigma: The diffusion coefficient of the Brownian motion.

    Returns:
        B: The Brownian bridge.
        dW: The increments of the Brownian motion.
    """
    key, subkey = random.split(key)
    dW = random.normal(subkey, (time_steps - 1, num_dims))
    B = make_brownian_bridge(dW, dt, sigma)
    return B, dW


def sample_ou_bridge(
    key: jax.Array,
    num_dims: int,
    T: float,
    dt: float,
    sigma: float,
    theta: float = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample an Ornstein-Uhlenbeck bridge.

    Args:
        key: The random key.
        num_dims: The number of dimensions.
        T: The time horizon.
        dt: The time step.
        sigma: The diffusion coefficient of the Ornstein-Uhlenbeck process.
        theta: The mean reversion rate.

    Returns:
        X: The Ornstein-Uhlenbeck bridge.
        dX: The increments of the Ornstein-Uhlenbeck process.
    """
    time_steps = int(T / dt)
    key, subkey = random.split(key)
    dW = random.normal(subkey, (time_steps - 1, num_dims))
    X = make_ou_bridge(dW, dt, sigma, theta)
    return X, dW


@struct.dataclass
class WalkNoiseModel:
    noise_type: str
    num_dims: int
    dt: float
    sigma: float
    T: float

    def __call__(
        self, key: jax.Array, time_steps: int
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate a Brownian motion or bridge.

        Args:
            key: The random key.
            T: The time horizon.

        Returns:
            W: The Brownian path.
            dW: The increments of the Brownian motion.
        """
        if self.noise_type == "white_noise":
            W, dW = sample_white_noise(
                key, self.num_dims, time_steps, self.dt, self.sigma
            )
        elif self.noise_type == "brownian_motion":
            W, dW = sample_brownian_motion(
                key, self.num_dims, time_steps=time_steps, dt=self.dt, sigma=self.sigma
            )
        elif self.noise_type == "ou_motion":
            W, dW = sample_ou_motion(
                key, self.num_dims, time_steps=time_steps, dt=self.dt, sigma=self.sigma
            )
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        # W = self.auto_encoder.decode(W)
        # W = jnp.dot(W, cov)
        return W, dW

    def transform(self, dW: jnp.ndarray) -> jnp.ndarray:
        """
        Transform a sequence of increments into a Brownian path or bridge.

        Args:
            dW: The increments of the Brownian motion.

        Returns:
            W: The Brownian path.
        """
        if self.noise_type == "white_noise":
            W = dW
        elif self.noise_type == "brownian_motion":
            W = make_brownian_path(dW=dW, dt=self.dt, sigma=self.sigma)
        elif self.noise_type == "ou_motion":
            W = make_ou_path(dW=dW, dt=self.dt, sigma=self.sigma)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        # W = self.auto_encoder.decode(W)
        # W = jnp.dot(W, cov)
        return W

    def center_data(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Center the data, ie. subtract the initial condition for Brownian motion. For Brownian bridge, subtract the initial condition and remove linear trend.

        Args:
            data: The data.

        Returns:
            centered_data: The centered data.
        """
        # if self.noise_type == 'brownian_motion' or self.noise_type == 'ou_motion':
        # center_point = jnp.dot(jnp.dot(data[...,:1,:], cov_inv), cov)
        # return data - center_point
        if self.noise_type == "white_noise":
            return data
        else:
            return data - data[..., :1, :]
        # else:
        #    raise ValueError(f"Unknown noise type: {noise_type}")

    def uncenter_data(
        self, centered_data: jnp.ndarray, init_condition: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Uncenter the data, ie. add the initial condition for Brownian motion. For Brownian bridge, add the initial condition and add linear trend.

        Args:
            centered_data: The centered data.
            uncentered_data: The uncentered data.

        Returns:
            uncentered_data: The uncentered data.
        """
        # if self.noise_type == 'brownian_motion' or self.noise_type == 'ou_motion':
        # uncenter_point = jnp.dot(jnp.dot(init_condition, cov_inv), cov)
        # return centered_data + uncenter_point
        if self.noise_type == "white_noise":
            return centered_data
        else:
            return centered_data + init_condition
        # else:
        #    raise ValueError(f"Unknown noise type: {noise_type}")


class BrownianNoiseModel(nn.Module):
    noise_type: str
    num_dims: int
    dt: float
    sigma: float
    Cov: jnp.ndarray | None= None

    @nn.compact
    def __call__(self, key: jax.Array, T: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate a Brownian motion or bridge.

        Args:
            key: The random key.
            T: The time horizon.

        Returns:
            W: The Brownian path.
            dW: The increments of the Brownian motion.
        """
        if self.noise_type == "white_noise":
            return sample_white_noise(key, self.num_dims, T, self.dt, self.sigma)
        elif self.noise_type == "brownian_motion":
            return sample_brownian_motion(
                key, self.num_dims, time_steps=T, dt=self.dt, sigma=self.sigma
            )
        elif self.noise_type == "ou_motion":
            return sample_ou_motion(
                key, self.num_dims, time_steps=T, dt=self.dt, sigma=self.sigma
            )
        elif self.noise_type == "brownian_bridge":
            return sample_brownian_bridge(
                key, self.num_dims, time_steps=T, dt=self.dt, sigma=self.sigma
            )
        elif self.noise_type == "ou_bridge":
            return sample_ou_bridge(
                key, self.num_dims, T=T, dt=self.dt, sigma=self.sigma
            )
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def transform(self, dW: jnp.ndarray) -> jnp.ndarray:
        """
        Transform a sequence of increments into a Brownian path or bridge.

        Args:
            dW: The increments of the Brownian motion.

        Returns:
            W: The Brownian path.
        """
        if self.noise_type == "white_noise":
            return dW
        elif self.noise_type == "brownian_motion":
            return make_brownian_path(dW=dW, dt=self.dt, sigma=self.sigma)
        elif self.noise_type == "ou_motion":
            return make_ou_path(dW=dW, dt=self.dt, sigma=self.sigma)
        elif self.noise_type == "brownian_bridge":
            return make_brownian_bridge(dW=dW, dt=self.dt, sigma=self.sigma)
        elif self.noise_type == "ou_bridge":
            return make_ou_bridge(dW=dW, dt=self.dt, sigma=self.sigma)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def center_data(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Center the data, ie. subtract the initial condition for Brownian motion. For Brownian bridge, subtract the initial condition and remove linear trend.

        Args:
            data: The data.

        Returns:
            centered_data: The centered data.
        """
        if self.noise_type == "white_noise":
            return data
        elif self.noise_type == "brownian_motion" or self.noise_type == "ou_motion":
            return data - data[..., :1, :]
        elif self.noise_type == "brownian_bridge" or self.noise_type == "ou_bridge":
            return (
                data
                - data[..., :1, :]
                - (data[..., -1:, :] - data[..., :1, :])
                * jnp.linspace(0, 1, data.shape[-2])[..., None]
            )
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def uncenter_data(
        self, centered_data: jnp.ndarray, uncentered_data: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Uncenter the data, ie. add the initial condition for Brownian motion. For Brownian bridge, add the initial condition and add linear trend.

        Args:
            centered_data: The centered data.
            uncentered_data: The uncentered data.

        Returns:
            uncentered_data: The uncentered data.
        """
        if self.noise_type == "white_noise":
            return centered_data
        elif self.noise_type == "brownian_motion" or self.noise_type == "ou_motion":
            return centered_data + uncentered_data[..., :1, :]
        elif self.noise_type == "brownian_bridge" or self.noise_type == "ou_bridge":
            return (
                centered_data
                + uncentered_data[..., :1, :]
                + (uncentered_data[..., -1:, :] - uncentered_data[..., :1, :])
                * jnp.linspace(0, 1, uncentered_data.shape[-2])[..., None]
            )
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def get_condition(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Get the initial condition for Brownian motion. For Brownian bridge, concatenate the initial and final conditions.

        Args:
            data: The data.

        Returns:
            condition: The boundary conditions.
        """
        if self.noise_type == "white_noise":
            return jnp.zeros(data.shape[-1])
        elif self.noise_type == "brownian_motion" or self.noise_type == "ou_motion":
            return data[..., 0, :]
        elif self.noise_type == "brownian_bridge" or self.noise_type == "ou_bridge":
            return jnp.concatenate([data[..., 0, :], data[..., -1, :]], axis=-1)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
