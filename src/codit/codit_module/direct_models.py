import jax.numpy as jnp
from flax import linen as nn

from codit.codit_module.diffusion_models import (
    Autoregressive_RNN,
    ConvContextEncoder,
    GAPContextEncoder,
    HistoryModule,
    MLP,
    MLPContextEncoder,
    ResDecoder,
    Transformer,
)


def get_context_encoder(context_encoder_type: str, hidden_size: int) -> nn.Module:
    if context_encoder_type == "conv":
        return ConvContextEncoder(kernel_size=3, hidden_base=8, num_layers=4)
    if context_encoder_type == "mlp":
        return MLPContextEncoder(output_dim=hidden_size)
    if context_encoder_type == "gap":
        return GAPContextEncoder(output_dim=hidden_size)
    raise ValueError(f"Unknown context encoder type: {context_encoder_type}")


class AutoregressiveDirectPredictor(nn.Module):
    context: bool = True
    num_layers: int = 2
    hidden_size: int = 64
    output_size: int = 2
    context_encoder_type: str = "conv"

    @nn.compact
    def __call__(self, y_history: jnp.ndarray, x: dict, train: bool = True) -> jnp.ndarray:
        context_enc = get_context_encoder(self.context_encoder_type, self.hidden_size)
        encoder = MLP(
            num_layers=2,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
        )
        decoder = ResDecoder(
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            num_layers=2,
        )

        history_context_enc = context_enc(x["history_context"])
        history_summary = HistoryModule(
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            num_layers=2,
            K=1,
        )(
            history_context_enc,
            y_history,
            jnp.zeros((1,), dtype=jnp.int32),
        )
        history_summary = jnp.tile(history_summary[None], (y_history.shape[0], 1))

        inputs = encoder(y_history)
        if self.context:
            prediction_context_enc = context_enc(x["prediction_context"])
            context = jnp.concatenate([prediction_context_enc, history_summary], axis=-1)
        else:
            context = history_summary

        hidden = Autoregressive_RNN(
            num_layers=self.num_layers,
            num_hidden=self.hidden_size,
        )(inputs, context)
        outputs = decoder(inputs, hidden, context, train)
        return nn.Dense(self.output_size)(outputs)


class AutoregressiveTransformerDirectPredictor(nn.Module):
    context: bool = True
    num_layers: int = 2
    hidden_size: int = 64
    output_size: int = 2
    num_heads: int = 8
    max_len: int = 100
    context_encoder_type: str = "conv"

    @nn.compact
    def __call__(self, y_history: jnp.ndarray, x: dict, train: bool = True) -> jnp.ndarray:
        context_enc = get_context_encoder(self.context_encoder_type, self.hidden_size)
        encoder = MLP(
            num_layers=2,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
        )
        decoder = ResDecoder(
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            num_layers=2,
        )

        history_context_enc = context_enc(x["history_context"])
        history_summary = HistoryModule(
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            num_layers=2,
            K=1,
        )(
            history_context_enc,
            y_history,
            jnp.zeros((1,), dtype=jnp.int32),
        )
        history_summary = jnp.tile(history_summary[None], (y_history.shape[0], 1))

        if self.context:
            prediction_context_enc = context_enc(x["prediction_context"])
            context = jnp.concatenate([prediction_context_enc, history_summary], axis=-1)
        else:
            context = history_summary

        inputs = jnp.concatenate([y_history, context], axis=-1)
        inputs = encoder(inputs)
        hidden = Transformer(
            self.hidden_size,
            self.num_heads,
            self.num_layers,
            self.max_len,
        )(inputs, train)
        outputs = decoder(inputs, hidden, context, train)
        return nn.Dense(self.output_size)(outputs)