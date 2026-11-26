from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
from flax.core import FrozenDict
from jax import jit, random
from orbax.checkpoint import CheckpointManager
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# import jax_dataloader as jdl
from tqdm import tqdm

from codit.codit_module import noising_functions
from codit.codit_module.brownian_noise_models import WalkNoiseModel
from codit.data_module.data_transform import transform_traj_to_metre

##### Model stuff #####


class TemporalEncoding(nn.Module):
    """
    Temporal encoding iwth cosine functions.

    Args:
        K: The number of diffusion steps.
        num_freqs: The number of frequencies.
    """

    K: int
    num_freqs: int

    @nn.compact
    def __call__(self, k: jnp.ndarray) -> jnp.ndarray:
        freqs = jnp.arange(1, self.num_freqs + 1)
        args = 2 * jnp.pi * k[:, None] / self.K * freqs
        emb = jnp.concatenate([jnp.sin(0.5 * args), jnp.cos(0.5 * args)], axis=-1)
        return emb


class MLP(nn.Module):
    """
    A simple multi-layer perceptron model.

    Args:
        num_layers: The number of layers.
        hidden_size: The hidden size.
        output_size: The output size.
    """

    num_layers: int = 1
    hidden_size: int = 64
    output_size: int = 2

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_size)(inputs)
        for _ in range(self.num_layers - 1):
            x_tmp = nn.leaky_relu(x)
            x_tmp = nn.Dense(self.hidden_size)(x_tmp)
            x = x_tmp + x
        x = nn.leaky_relu(x)
        outputs = nn.Dense(self.output_size)(x)
        return outputs


class HistoryModule(nn.Module):

    hidden_size: int
    output_size: int
    num_layers: int
    K: int
    num_freqs: int = 64

    @nn.compact
    def __call__(
        self,
        history_context: jnp.ndarray,
        history_target_vals: jnp.ndarray,
        k: jnp.ndarray,
    ) -> jnp.ndarray:
        # temp_enc = TemporalEncoding(K=self.K, num_freqs=self.num_freqs)(k)
        x = jnp.concat([history_target_vals, history_context], axis=-1)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.leaky_relu(x)
        for i in range(self.num_layers):
            x = nn.RNN(nn.GRUCell(self.hidden_size))(x)  # type: ignore
        x = MLP(
            num_layers=1, hidden_size=self.hidden_size, output_size=self.output_size
        )(x[-1].flatten())
        return x


class DiffusionEmbedding(nn.Module):
    dim: int
    proj_dim: int
    K: int = 500

    def setup(self) -> None:
        self.embedding = self._build_embedding(self.dim, self.K)
        self.projection1 = nn.Dense(self.proj_dim)
        self.projection2 = nn.Dense(self.proj_dim)

    def __call__(self, diffusion_step: jnp.ndarray) -> jnp.ndarray:
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = nn.silu(x)
        x = self.projection2(x)
        x = nn.silu(x)
        return x

    @staticmethod
    def _build_embedding(dim: int, max_steps: int) -> jnp.ndarray:
        steps = jnp.arange(max_steps).reshape(-1, 1)  # [T,1]
        dims = jnp.arange(dim).reshape(1, -1)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = jnp.concatenate([jnp.sin(table), jnp.cos(table)], axis=1)
        return table


class ResBlock(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray, train: bool) -> jnp.ndarray:
        y = jnp.concatenate([x, emb], axis=-1)
        y = nn.Dense(self.hidden_size)(y)
        y = nn.leaky_relu(y)
        y = nn.Dense(x.shape[-1])(y)
        return x + y


class ResDecoder(nn.Module):
    hidden_size: int = 8
    num_layers: int = 1
    output_size: int = 2

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, h: jnp.ndarray, diff_enc: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        emb = jnp.concatenate([h, diff_enc], axis=-1)
        for i in range(self.num_layers):
            x = ResBlock(self.hidden_size)(x, emb, train)
        return x


class EncoderBlock(nn.Module):
    hidden_size: int
    kernel_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            self.hidden_size,
            kernel_size=(
                self.kernel_size,
                self.kernel_size,
            ),
            padding="SAME",
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.4)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class ConvContextEncoder(nn.Module):
    kernel_size: int
    hidden_base: int
    num_layers: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for ilayer in range(self.num_layers):
            x = EncoderBlock(
                hidden_size=self.hidden_base * (ilayer + 1),
                kernel_size=self.kernel_size,
            )(x)
        return x.squeeze(axis=-2).squeeze(axis=-2)


class MLPContextEncoder(nn.Module):
    output_dim: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Flatten input (24, 16, 16, 8) → (24, 2048)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.output_dim)(x)  # (24, 32)
        return x


class GAPContextEncoder(nn.Module):
    output_dim: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Apply global average pooling to reduce spatial dimensions to 1x1
        x_av = nn.avg_pool(x, window_shape=(16, 16), strides=(16, 16))  # (24, 1, 1, 8)
        x_max = nn.max_pool(x, window_shape=(16, 16), strides=(16, 16))
        x = jnp.concatenate(
            [
                x_av,
                x_max,
            ],
            axis=-1,
        )
        x = x.reshape(x.shape[0], -1)  # (24, 8)
        x = nn.Dense(self.output_dim)(x)  # (24, 32)
        return x


def get_positional_encoding_table(max_len: int, d_model: int) -> jnp.ndarray:
    """
    Get the positional encoding table.

    Args:
        max_len: The maximum length of the sequence.
        d_model: The dimension of the model.

    Returns:
        The positional encoding table.
    """
    position = jnp.arange(max_len, dtype=jnp.float32).reshape(-1, 1)
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
    pe = jnp.zeros((max_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool, mask: jnp.ndarray) -> jnp.ndarray:
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
        )(x, x, x, mask=mask)
        x = nn.LayerNorm()(x + attn_output)
        ff_output = nn.Dense(int(4 * self.d_model))(x)
        ff_output = nn.relu(ff_output)
        ff_output = nn.Dense(self.d_model)(ff_output)
        return nn.LayerNorm()(x + ff_output)


class Transformer(nn.Module):
    d_model: int
    num_heads: int
    num_layers: int
    max_len: int

    def __init__(
        self, d_model: int, num_heads: int, num_layers: int, max_len: int
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.pos_enc_table = get_positional_encoding_table(self.max_len, self.d_model)
        self.transformer_blocks = [
            TransformerBlock(self.d_model, self.num_heads)
            for _ in range(self.num_layers)
        ]

    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = x + self.pos_enc_table[..., : x.shape[-2], :]
        mask = jnp.tril(jnp.ones((x.shape[-2], x.shape[-2])))
        mask = jnp.expand_dims(mask, axis=0)
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x, train, mask)
        return x


class Autoregressive_TransformerDenoiser(nn.Module):
    """
    A bidirectional RNN denoiser model.

    Args:
        condition_encoder: The encoder for the initial condition.
        encoder: The encoder for the data.
        decoder: The decoder for the data.
        num_freqs: The number of frequencies for the temporal encoding.
        num_layers: The number of layers for the RNN.
        hidden_size: The hidden size of the RNN.
        output_size: The output size of the RNN.
        K: The number of time steps.
    """

    context: bool = False
    residual_hidden: int = 64
    diffusion_emb_dim: int = 64
    num_layers: int = 1
    hidden_size: int = 128
    output_size: int = 2
    K: int = 1000
    num_heads: int = 8
    max_len: int = 100
    context_encoder_type: str = "conv"  # conv, mlp, gap

    @nn.compact
    def __call__(
        self,
        y: jnp.ndarray,
        y_history: jnp.ndarray,
        x: jnp.ndarray,
        k: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        history_module = HistoryModule(
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            num_freqs=self.diffusion_emb_dim,
            num_layers=2,
            K=self.K,
        )
        encoder = MLP(
            num_layers=2, hidden_size=self.hidden_size, output_size=self.hidden_size
        )
        decoder = ResDecoder(
            hidden_size=self.hidden_size, output_size=self.hidden_size, num_layers=2
        )
        diff_enc = TemporalEncoding(K=self.K, num_freqs=self.diffusion_emb_dim)(k)
        # diff_enc = DiffusionEmbedding(self.diffusion_emb_dim, self.residual_hidden, self.K)(k)
        history_context = x["history_context"]
        # history_width = history_context.shape[-2]
        if self.context_encoder_type == "conv":
            context_enc = ConvContextEncoder(kernel_size=3, hidden_base=8, num_layers=4)
        elif self.context_encoder_type == "mlp":
            context_enc = MLPContextEncoder(output_dim=self.hidden_size)
        elif self.context_encoder_type == "gap":
            context_enc = GAPContextEncoder(output_dim=self.hidden_size)
        history_context_enc = context_enc(history_context)
        history_enc = history_module(history_context_enc, x["history_target_vals"], k)
        y = jnp.concatenate([y_history, y], axis=-2)

        if self.context:
            prediction_context = x["prediction_context"]
            prediction_context_enc = context_enc(prediction_context)
            context = jnp.concatenate(
                [history_context_enc, prediction_context_enc], axis=-2
            )
            # shift context to the right
            context = jnp.concatenate(
                [jnp.zeros((1, context.shape[-1])), context[..., :-1, :]], axis=-2
            )
            context = jnp.concatenate(
                [
                    context,
                    jnp.tile(diff_enc, (y.shape[-2], 1)),
                    jnp.tile(history_enc, (y.shape[-2], 1)),
                ],
                axis=-1,
            )
        else:
            context = jnp.concatenate(
                [
                    jnp.tile(diff_enc, (y.shape[-2], 1)),
                    jnp.tile(history_enc, (y.shape[-2], 1)),
                ],
                axis=-1,
            )
        inputs = jnp.concatenate([y, context], axis=-1)
        inputs = encoder(inputs)
        outputs = Transformer(
            self.hidden_size, self.num_heads, self.num_layers, self.max_len
        )(inputs, train)[..., y_history.shape[-2] :, :]
        outputs = decoder(
            inputs[..., y_history.shape[-2] :, :],
            outputs,
            context[..., y_history.shape[-2] :, :],
            train,
        )
        # %%
        outputs = nn.Dense(self.output_size)(outputs)
        return outputs


class Autoregressive_RNN(nn.Module):
    num_layers: int
    num_hidden: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        context: jnp.ndarray,
        initial_carry: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        for i in range(0, self.num_layers):
            x = jnp.concatenate([x, context], axis=-1)
            x = nn.RNN(nn.GRUCell(self.num_hidden))(x, initial_carry=initial_carry)  # type: ignore
        return x


class Autoregressive_Denoiser(nn.Module):
    """
    A bidirectional RNN denoiser model.

    Args:
        condition_encoder: The encoder for the initial condition.
        encoder: The encoder for the data.
        decoder: The decoder for the data.
        num_freqs: The number of frequencies for the temporal encoding.
        num_layers: The number of layers for the RNN.
        hidden_size: The hidden size of the RNN.
        output_size: The output size of the RNN.
        K: The number of time steps.
    """

    context: bool = False
    residual_hidden: int = 64
    diffusion_emb_dim: int = 64
    num_layers: int = 1
    hidden_size: int = 128
    output_size: int = 2
    K: int = 1000
    context_encoder_type: str = "conv"

    @nn.compact
    def __call__(
        self,
        y: jnp.ndarray,
        y_history: jnp.ndarray,
        x: jnp.ndarray,
        k: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        history_module = HistoryModule(
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            num_freqs=self.diffusion_emb_dim,
            num_layers=2,
            K=self.K,
        )
        encoder = MLP(
            num_layers=2, hidden_size=self.hidden_size, output_size=self.hidden_size
        )
        decoder = ResDecoder(
            hidden_size=self.hidden_size, output_size=self.hidden_size, num_layers=2
        )
        diff_enc = TemporalEncoding(K=self.K, num_freqs=self.diffusion_emb_dim)(k)
        # diff_enc = DiffusionEmbedding(self.diffusion_emb_dim, self.residual_hidden, self.K)(k)
        history_context = x["history_context"]
        # history_width = history_context.shape[-2]
        if self.context_encoder_type == "conv":
            context_enc = ConvContextEncoder(kernel_size=3, hidden_base=8, num_layers=4)
        elif self.context_encoder_type == "mlp":
            context_enc = MLPContextEncoder(output_dim=self.hidden_size)
        elif self.context_encoder_type == "gap":
            context_enc = GAPContextEncoder(output_dim=self.hidden_size)
        history_context_enc = context_enc(history_context)
        history_enc = history_module(history_context_enc, x["history_target_vals"], k)
        y = jnp.concatenate([y_history, y], axis=-2)
        inputs = encoder(y)
        if self.context:
            prediction_context = x["prediction_context"]
            prediction_context_enc = context_enc(prediction_context)
            context = jnp.concatenate(
                [history_context_enc, prediction_context_enc], axis=-2
            )
            # shift context to the right
            context = jnp.concatenate(
                [jnp.zeros((1, context.shape[-1])), context[..., :-1, :]], axis=-2
            )
            context = jnp.concatenate(
                [
                    context,
                    jnp.tile(diff_enc, (inputs.shape[-2], 1)),
                    jnp.tile(history_enc, (inputs.shape[-2], 1)),
                ],
                axis=-1,
            )
        else:
            context = jnp.concatenate(
                [
                    jnp.tile(diff_enc, (inputs.shape[-2], 1)),
                    jnp.tile(history_enc, (inputs.shape[-2], 1)),
                ],
                axis=-1,
            )

        # init_cond_f = history_enc[: self.hidden_size]
        x = Autoregressive_RNN(num_layers=self.num_layers, num_hidden=self.hidden_size)(
            inputs, context
        )[..., y_history.shape[-2] :, :]
        outputs = decoder(
            inputs[..., y_history.shape[-2] :, :],
            x,
            context[..., y_history.shape[-2] :, :],
            train,
        )
        outputs = nn.Dense(self.output_size)(outputs)
        return outputs


##### train


def transform_batch(
    torch_batch: tuple, std: float | jnp.ndarray, aligned_context: bool = True
) -> dict:
    trajectory = jnp.asarray(torch_batch[0])
    traj_in_metres = jax.vmap(transform_traj_to_metre)(trajectory)
    trajectory = traj_in_metres / std
    context = jnp.asarray(torch_batch[1])
    prediction_target_vals = trajectory[:, :]
    history_target_vals = trajectory[:, :1]
    history_context = context[:, :1]
    prediction_context = context[:, 1:] if aligned_context else context[:, :-1]
    jax_batch = {
        "history_target_vals": history_target_vals,
        "prediction_target_vals": prediction_target_vals,
        "history_context": history_context,
        "prediction_context": prediction_context,
    }
    return jax_batch


def train_step(
    carry: tuple,
    batch: dict,
    model: nn.Module,
    noise_model: WalkNoiseModel,
    optimizer: optax.GradientTransformation,
    schedule_dict: dict,
    K: int,
    batch_size: int,
) -> tuple:
    params, opt_state, key, loss = carry
    key, subkey = random.split(key)
    k = random.randint(subkey, (batch_size, 1), 0, K)
    keys = random.split(key, batch_size + 1)
    key, subkey = keys[0], keys[1:]
    value, grads = denoising_loss(
        params, model, batch, k, subkey, schedule_dict, noise_model
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    loss += value
    return (params, opt_state, key, loss), value


def validation_k(batch_index: int, batch_size: int, K: int) -> jnp.ndarray:
    """Diffusion steps for a validation batch, cycling through all K levels.

    Sample ``j`` of batch ``i`` gets step ``(i * batch_size + j) % K``, so the val set
    covers every noise level evenly and the same sample always sees the same level.
    """
    return ((batch_index * batch_size + jnp.arange(batch_size)) % K)[:, None]


def ema_decay_at(step: int | jnp.ndarray, max_decay: float) -> jnp.ndarray:
    """EMA decay with warm-up: ``min(max_decay, (1 + step) / (10 + step))``.

    Early on the average follows the weights closely, so it is not dominated by the
    random initialisation; later it averages over ~``1 / (1 - max_decay)`` steps.
    """
    return jnp.minimum(max_decay, (1.0 + step) / (10.0 + step))


@jit
def ema_update(ema_params, params, decay):
    """One EMA step: ``ema <- decay * ema + (1 - decay) * params``."""
    return optax.incremental_update(params, ema_params, step_size=1.0 - decay)


def _dataset_arrays(dataloader: DataLoader) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Trajectories and selected context channels of a DataLoader's dataset, on device.

    Takes the samples in the dataset's index order (``OceanTrainingDataset.indices``)
    and only the context channels the model uses (``context_var_idx``), so the copy on
    the device is as small as possible.
    """
    ds = dataloader.dataset
    idx = np.asarray(ds.indices)
    traj = np.asarray(ds.trajectories)[idx]
    ctx = np.take(np.asarray(ds.context)[idx], np.asarray(ds.context_var_idx), axis=-1)
    return jax.device_put(traj), jax.device_put(ctx)


def _device_batches(n: int, batch_size: int, key: jax.Array | None):
    """Yield index arrays for batches over ``n`` samples, like a DataLoader with
    ``drop_last=True``: a fresh permutation per pass if ``key`` is given (shuffle),
    otherwise one pass in order."""
    num_batches = n // batch_size
    if num_batches == 0:
        raise ValueError(f"{n} samples are fewer than one batch of {batch_size}.")
    while True:
        if key is None:
            order = jnp.arange(n)
        else:
            key, subkey = random.split(key)
            order = random.permutation(subkey, n)
        for b in range(num_batches):
            yield order[b * batch_size : (b + 1) * batch_size]
        if key is None:
            return


def _cycle(dataloader: DataLoader):
    """Iterate over a DataLoader indefinitely (reshuffling on every pass)."""
    while True:
        yielded = False
        for batch in dataloader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError(
                "The training DataLoader is empty: the dataset is smaller than one batch."
            )


def train(
    train_dataloader: DataLoader,
    optimizer: optax.GradientTransformation,
    model: nn.Module,
    noise_model: WalkNoiseModel,
    params: dict | FrozenDict,
    opt_state: optax.OptState,
    key: jax.Array,
    schedule_dict: dict,
    num_epochs: int = 100,
    num_batches_per_epoch: int = 100,
    loss_list: Optional[list] = None,
    checkpoint_mngr: Optional[CheckpointManager] = None,
    val_dataloader: Optional[DataLoader] = None,
    val_loss_list: Optional[list] = None,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    writer: SummaryWriter | None = None,
    std: float | jnp.ndarray = 1.0,
    aligned_context: bool = True,
    ema_decay: float | None = 0.999,
    device_data: bool = False,
) -> tuple:
    """Train the denoiser, with early stopping on the val loss.

    An epoch is exactly ``num_batches_per_epoch`` gradient steps; the training
    DataLoader is cycled (and reshuffled) as often as needed, so the budget does
    not depend on the dataset size. Losses are the mean denoising MSE per sample.

    The val loss is deterministic: a fixed key and noise levels spread evenly over
    all K steps (``validation_k``), so epochs can be compared directly. Training
    stops after ``early_stopping_patience`` epochs without an improvement of more
    than ``early_stopping_min_delta`` (absolute, in MSE), and the parameters of the
    best epoch are returned.

    With ``ema_decay`` set, an exponential moving average of the weights is kept
    (with warm-up, see ``ema_decay_at``). Both the raw and the EMA weights are
    validated every epoch; early stopping and best-epoch selection use the EMA
    weights, which are returned as the model. The raw weights of their own best
    epoch are returned as well, for comparison.

    Args:
        aligned_context: ``False`` for the legacy context layout of older checkpoints.
        ema_decay: Maximum EMA decay (``0.999`` averages over ~1000 steps), or
            ``None`` to train without EMA.
        device_data: Copy the training and validation samples (only the selected
            context channels) to the device once and assemble batches there, instead
            of through the DataLoaders, which then only provide the datasets, batch
            sizes and shuffle settings. Much faster when the DataLoader is the
            bottleneck; needs the data to fit in device memory.

    Returns:
        ``(best_params, opt_state, loss_list, val_loss_list, info)``. ``val_loss_list``
        holds the selection metric (EMA val loss with EMA, raw otherwise);
        ``opt_state`` is the optimizer state at the last epoch. ``info`` has
        ``best_epoch``, ``val_loss_raw`` and ``val_loss_ema`` (per epoch; the latter
        empty without EMA), and ``best_raw_params`` / ``best_raw_epoch``.
    """
    if loss_list is None:
        loss_list = []
    if val_loss_list is None:
        val_loss_list = []
    assert isinstance(train_dataloader.batch_size, int), "Batch size must be an integer."
    K = schedule_dict["K"]
    key, val_key = random.split(key)
    if device_data:
        key, data_key = random.split(key)
        train_arrays = _dataset_arrays(train_dataloader)
        val_arrays = _dataset_arrays(val_dataloader) if val_dataloader is not None else None

        @jit
        def gather(traj, ctx, idx):
            return transform_batch((traj[idx], ctx[idx]), std, aligned_context)

        def train_batches():
            n = train_arrays[0].shape[0]
            for idx in _device_batches(n, train_dataloader.batch_size, data_key):
                yield gather(*train_arrays, idx)

        def val_batches():
            assert val_dataloader is not None and val_arrays is not None
            n = val_arrays[0].shape[0]
            for idx in _device_batches(n, val_dataloader.batch_size, None):
                yield gather(*val_arrays, idx)

    else:

        def train_batches():
            for batch in _cycle(train_dataloader):
                yield transform_batch(batch, std, aligned_context)

        def val_batches():
            assert val_dataloader is not None
            for batch in val_dataloader:
                yield transform_batch(batch, std, aligned_context)

    @jit
    def step_func(carry: tuple, batch: dict) -> tuple:
        return train_step(
            carry,
            batch,
            model,
            noise_model,
            optimizer,
            schedule_dict,
            K,
            train_dataloader.batch_size,
        )

    @jit
    def val_loss_func(params, batch: dict, k: jnp.ndarray, keys: jax.Array):
        return denoising_loss_value(
            params, model, batch, k, keys, schedule_dict, noise_model
        )

    def validate(params) -> float:
        assert val_dataloader is not None
        total, count = 0.0, 0
        for i, jax_batch in enumerate(val_batches()):
            n = jax_batch["prediction_target_vals"].shape[0]
            k = validation_k(i, n, K)
            keys = random.split(random.fold_in(val_key, i), n)
            total += float(val_loss_func(params, jax_batch, k, keys)) * n
            count += n
        if count == 0:
            raise ValueError("The validation DataLoader is empty.")
        return total / count

    use_ema = ema_decay is not None
    ema_params = params
    val_raw: list = []
    val_ema: list = []
    best_metric, best_params, best_epoch, epochs_since_best = np.inf, params, -1, 0
    best_raw_metric, best_raw_params, best_raw_epoch = np.inf, params, -1
    step = 0
    train_iter = train_batches()
    pbar = tqdm(total=num_epochs, desc="Training", dynamic_ncols=True)
    for epoch in range(num_epochs):
        carry = (params, opt_state, key, 0.0)
        for _ in range(num_batches_per_epoch):
            carry, _ = step_func(carry, next(train_iter))
            if use_ema:
                ema_params = ema_update(ema_params, carry[0], ema_decay_at(step, ema_decay))
            step += 1
        params, opt_state, key, loss = carry
        # each step adds the batch-mean MSE
        loss_list.append(float(loss) / num_batches_per_epoch)
        if val_dataloader is not None:
            val_raw.append(validate(params))
            raw_metric = val_raw[-1]
            if use_ema:
                val_ema.append(validate(ema_params))
            val_loss_list.append(val_ema[-1] if use_ema else val_raw[-1])
            metric = val_loss_list[-1]
        else:
            # without validation data, select on the training loss (raw weights)
            raw_metric = metric = loss_list[-1]
        selected = ema_params if use_ema else params

        if raw_metric < best_raw_metric:
            best_raw_metric, best_raw_params, best_raw_epoch = raw_metric, params, epoch
        if best_metric - metric > early_stopping_min_delta:
            best_metric, best_params, best_epoch = metric, selected, epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if checkpoint_mngr is not None:
            ckpt = {
                "params": params,
                "ema_params": ema_params,
                "opt_state": opt_state,
                "key": key,
                "epoch": epoch,
            }
            checkpoint_mngr.save(epoch, args=ocp.args.StandardSave(ckpt))  # type: ignore
        if writer is not None:
            writer.add_scalar("Loss/train", loss_list[-1], epoch)
            if val_dataloader is not None:
                writer.add_scalar("Loss/val", val_loss_list[-1], epoch)
                writer.add_scalar("Loss/val_raw", val_raw[-1], epoch)
                if use_ema:
                    writer.add_scalar("Loss/val_ema", val_ema[-1], epoch)
        postfix = {"train_loss": loss_list[-1], "best_epoch": best_epoch}
        if val_dataloader is not None:
            postfix["val_raw"] = val_raw[-1]
            if use_ema:
                postfix["val_ema"] = val_ema[-1]
        pbar.set_postfix(postfix)
        pbar.update(1)
        if epochs_since_best >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    pbar.close()
    what = ("val" if val_dataloader is not None else "train") + (" (EMA)" if use_ema else "")
    print(
        f"Best epoch {best_epoch} ({(best_epoch + 1) * num_batches_per_epoch} steps), "
        f"{what} loss {best_metric:.5g}"
    )
    if use_ema and val_dataloader is not None:
        print(f"Best raw weights: epoch {best_raw_epoch}, val loss {best_raw_metric:.5g}")
    info = {
        "best_epoch": best_epoch,
        "val_loss_raw": val_raw,
        "val_loss_ema": val_ema,
        "best_raw_params": best_raw_params,
        "best_raw_epoch": best_raw_epoch,
    }
    return best_params, opt_state, loss_list, val_loss_list, info


def denoising_loss_value(
    params: FrozenDict,
    model: nn.Module,
    batch: dict,
    k: jnp.ndarray,
    key: jax.Array,
    schedule_dict: dict,
    noise_model: WalkNoiseModel,
) -> jnp.ndarray:
    """
    The denoising loss function.

    Args:
        params: The parameters of the model.
        model: The denoising model.
        y: The data.
        k: The diffusion steps.
        key: The random key.
        alpha_bar: The diffusion coefficient.
        T: Time series length.
        noise_model: The noise model.

    Returns:
        The MSE noise loss.
    """
    # init_condition = batch['history']['target_vals'][...,-1:,:]
    y = batch["prediction_target_vals"]
    y_input = noise_model.center_data(y)
    # y_delta = y_input[...,:1,:]
    y_noised, noise_true = jax.vmap(
        noising_functions.forward_diffusion, in_axes=(0, 0, 0, None, None)
    )(y_input, k, key, schedule_dict, noise_model)
    # y_noised = y_noised + y_delta

    # ae = auto_encoder.bind(ae_params)
    y_history_reversed = batch["history_target_vals"][...][..., ::-1, :]
    y_history_input = noise_model.center_data(y_history_reversed)
    # y_history_delta = y_history_input[...,:1,:]
    y_noised_history, _ = jax.vmap(
        noising_functions.forward_diffusion, in_axes=(0, 0, 0, None, None)
    )(y_history_input, k, key, schedule_dict, noise_model)
    # y_proj_history = jnp.dot(y_noised_history, cov_inv)
    y_noised_history = y_noised_history[..., ::-1, :]
    batch["history_target_vals"] = batch["history_target_vals"]
    noise_pred = jax.vmap(model.apply, in_axes=(None, 0, 0, 0, 0))(
        params, y_noised[:, 1:], y_noised_history, batch, k
    )
    return jnp.mean((noise_true - noise_pred) ** 2)


# Loss and gradients w.r.t. ``params`` (model and noise_model are static).
denoising_loss = partial(jax.jit, static_argnums=(1, 6))(
    jax.value_and_grad(denoising_loss_value)
)
