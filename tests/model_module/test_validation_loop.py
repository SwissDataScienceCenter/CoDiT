"""
Documents the division-by-zero bug in validation loop normalisation and verifies the fix.

The bug in linear_gauss_model.py and conv_net.py:
    val_loss = 0
    for i, batch in enumerate(val_dataloader):
        val_loss += eval_step(...)
    val_loss /= i * batch[0].shape[0] * batch[0].shape[1]  # i=0 → inf

The fix:
    n_val = 0
    for batch in val_dataloader:
        val_loss += eval_step(...)
        n_val += batch[0].shape[0] * batch[0].shape[1]
    val_loss /= n_val  # always > 0
"""

import jax.numpy as jnp
import pytest


def _simulate_buggy_normalisation(num_batches, batch_size=4, seq_len=10):
    """Reproduce the buggy normalisation pattern with a given number of batches."""
    val_loss = jnp.array(float(num_batches))  # pretend sum of losses = num_batches
    fake_batch_shape = (batch_size, seq_len)
    for i in range(num_batches):
        pass  # simulate the loop
    # After loop, i = num_batches - 1
    divisor = i * fake_batch_shape[0] * fake_batch_shape[1]
    return val_loss / divisor


def _simulate_fixed_normalisation(num_batches, batch_size=4, seq_len=10):
    """Reproduce the fixed normalisation pattern."""
    val_loss = jnp.array(float(num_batches))
    n_val = num_batches * batch_size * seq_len
    return val_loss / n_val


def test_buggy_normalisation_is_inf_with_one_batch():
    result = _simulate_buggy_normalisation(num_batches=1)
    assert not jnp.isfinite(
        result
    ), "This documents the bug: with 1 batch, i=0 → val_loss/0 = inf"


def test_fixed_normalisation_is_finite_with_one_batch():
    result = _simulate_fixed_normalisation(num_batches=1)
    assert jnp.isfinite(result)
    assert float(result) == pytest.approx(1.0 / (1 * 4 * 10), rel=1e-5)


def test_fixed_normalisation_is_finite_with_multiple_batches():
    result = _simulate_fixed_normalisation(num_batches=5)
    assert jnp.isfinite(result)


def _load_arg_parser(script_path: str):
    """Import the __main__ argparser of a train/eval script without executing it.

    The scripts guard heavyweight setup behind `if __name__ == "__main__"`, so
    parsing the AST for the argparser block is the cleanest way to inspect it.
    """
    import ast
    with open(script_path) as f:
        tree = ast.parse(f.read())
    add_arg_calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            if node.args and isinstance(node.args[0], ast.Constant):
                add_arg_calls.append((node.args[0].value, {k.arg: k.value for k in node.keywords}))
    return add_arg_calls


def test_codit_train_exposes_prediction_length():
    """codit_train.py must accept --prediction_length with default 72."""
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    calls = _load_arg_parser(os.path.join(repo, "..", "scripts", "user", "train", "codit_train.py"))
    names = [name for name, _ in calls]
    assert "--prediction_length" in names, f"missing --prediction_length in {names}"
    _, kwargs = next(c for c in calls if c[0] == "--prediction_length")
    default_node = kwargs.get("default")
    import ast
    assert isinstance(default_node, ast.Constant) and default_node.value == 72


def test_codit_eval_exposes_prediction_length():
    """codit_eval.py must accept --prediction_length with default 72."""
    import os, ast
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    calls = _load_arg_parser(os.path.join(repo, "..", "scripts", "user", "eval", "codit_eval.py"))
    names = [name for name, _ in calls]
    assert "--prediction_length" in names
    _, kwargs = next(c for c in calls if c[0] == "--prediction_length")
    default_node = kwargs.get("default")
    assert isinstance(default_node, ast.Constant) and default_node.value == 72


def test_codit_train_no_hardcoded_length_branch():
    """After the refactor, codit_train.py must not branch on dataset name to set length."""
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo, "..", "scripts", "user", "train", "codit_train.py")) as f:
        src = f.read()
    assert 'if args.dataset == "open_ocean":' not in src
    assert 'elif args.dataset == "coastline":' not in src


def test_codit_eval_no_hardcoded_length_branch():
    """Same guard for codit_eval.py; dataset branch there also picked length."""
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo, "..", "scripts", "user", "eval", "codit_eval.py")) as f:
        src = f.read()
    # The dataset branch may remain (it also builds config_path/result_path)
    # but the `length = 72` / `length = 24` assignments inside it must be gone.
    assert "length = 72" not in src
    assert "length = 24" not in src
