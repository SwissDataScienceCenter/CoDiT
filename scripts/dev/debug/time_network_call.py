"""Time one CoDiT network call at eval shapes, split into context encoder and GRU.

During sampling (``CoDiTModel.predict``) every denoising step calls ``model.apply``,
which re-encodes all context grids (1 history + prediction_length slots) with the
conv encoder although the context does not change across the K denoising steps.
This script measures how much of one call the encoder takes, to decide whether
encoding the context once per prediction hour (instead of once per denoising step)
is worth implementing.

Timed, each jitted and vmapped over ``--num_samples`` like ``predict``:

  full     ``model.apply`` as called in ``denoise`` (one network call)
  encoder  the ``ConvContextEncoder`` alone on the same context grids
  gru      the ``Autoregressive_RNN`` alone on inputs of the same shape

Inputs are synthetic (timing depends only on shapes), so no dataset is needed.
Run in the GPU session:

    python scripts/dev/debug/time_network_call.py
    python scripts/dev/debug/time_network_call.py --model codit_GRU_open_ocean_..._cosine
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, vmap

from codit.codit_module.diffusion_models import Autoregressive_RNN, ConvContextEncoder
from codit.model_module.diffusion_model import CoDiTModel


def timed(fn, repeats):
    jax.block_until_ready(fn())  # compile + warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t)
    return np.median(times), np.std(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="codit_GRU_open_ocean")
    p.add_argument("--num_samples", type=int, default=100, help="as codit_eval.py")
    p.add_argument("--prediction_length", type=int, default=72)
    p.add_argument("--grid", type=int, default=16, help="context grid points per side")
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()

    model = CoDiTModel.import_trained_model(args.model)
    net, params = model.model, model.nn_params
    if not (getattr(net, "context", False) and net.context_encoder_type == "conv"):
        raise SystemExit("expects an Autoregressive_Denoiser with context and conv encoder")
    S, T, G = args.num_samples, args.prediction_length, args.grid
    C = len(model.var_idx)
    K = int(model.schedule_dict["K"])
    print(f"backend {jax.default_backend()}, devices {jax.devices()}")
    print(f"model {args.model}: K={K}, S={S}, T={T}, context {G}x{G}x{C}")

    keys = random.split(random.PRNGKey(0), 6)
    y = random.normal(keys[0], (S, T, 2))
    y_hist = random.normal(keys[1], (S, 1, 2))
    batch = {
        "history_target_vals": jnp.zeros((S, 1, 2)),
        "history_context": random.normal(keys[2], (S, 1, G, G, C)),
        "prediction_context": random.normal(keys[3], (S, T, G, G, C)),
    }
    k = jnp.array([K // 2])

    # full call, as in denoise(): model.apply(params, y[1:], y_history, batch, k)
    full = jit(vmap(lambda y_, yh, b: net.apply(params, y_, yh, b, k)))

    # encoder alone, same grids (history + prediction context)
    enc_mod = ConvContextEncoder(kernel_size=3, hidden_base=8, num_layers=4)
    enc_params = {"params": params["params"]["ConvContextEncoder_0"]}
    enc = jit(vmap(lambda b: (
        enc_mod.apply(enc_params, b["history_context"]),
        enc_mod.apply(enc_params, b["prediction_context"]),
    )))

    # GRU alone; input/context widths read off the first GRU cell's kernels
    rnn_params = {"params": params["params"]["Autoregressive_RNN_0"]}
    cell0 = rnn_params["params"]["GRUCell_0"]
    hidden = cell0["hn"]["kernel"].shape[-1]
    in_dim = cell0["ir"]["kernel"].shape[0]
    ctx_dim = in_dim - hidden  # layer 0 input = encoder(y) (hidden) ++ context
    num_layers = sum(name.startswith("GRUCell_") for name in rnn_params["params"])
    rnn_mod = Autoregressive_RNN(num_layers=num_layers, num_hidden=hidden)
    rnn_x = random.normal(keys[4], (S, T + 1, hidden))
    rnn_ctx = random.normal(keys[5], (S, T + 1, ctx_dim))
    gru = jit(vmap(lambda x_, c_: rnn_mod.apply(rnn_params, x_, c_)))

    t_full, s_full = timed(lambda: full(y, y_hist, batch), args.repeats)
    t_enc, s_enc = timed(lambda: enc(batch), args.repeats)
    t_gru, s_gru = timed(lambda: gru(rnn_x, rnn_ctx), args.repeats)

    ms = lambda t: f"{1e3 * t:8.2f} ms"
    print(f"\none network call (median of {args.repeats}, vmapped over {S} samples):")
    print(f"  full       {ms(t_full)} +- {1e3 * s_full:.2f}")
    print(f"  encoder    {ms(t_enc)} +- {1e3 * s_enc:.2f}   ({t_enc / t_full:5.1%} of full)")
    print(f"  GRU        {ms(t_gru)} +- {1e3 * s_gru:.2f}   ({t_gru / t_full:5.1%} of full)"
          f"  [{num_layers} layer(s), hidden {hidden}, context {ctx_dim}]")
    rest = t_full - t_enc - t_gru
    print(f"  rest       {ms(rest)}             ({rest / t_full:5.1%}; MLPs, decoder, overhead)")

    # Projection for one trajectory: T prediction hours x K denoising steps.
    # Standalone kernels do not fuse like inside `full`, so the split is approximate.
    calls = T * K
    now = calls * t_full
    hoisted = calls * max(t_full - t_enc, 0.0) + T * t_enc
    print(f"\nper trajectory (DDPM, {T} x {K} = {calls} calls):")
    print(f"  now                    ~{now:6.1f} s   (eval logs: ~34 s/traj for the shipped model)")
    print(f"  encoder once per hour  ~{hoisted:6.1f} s   (~{now / hoisted:.1f}x faster)")


if __name__ == "__main__":
    main()
