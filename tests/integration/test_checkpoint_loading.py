"""
Integration tests that load the shipped model checkpoints.
Require data/trained_models/ to be present (default repo layout).
Skipped automatically if checkpoints are missing.
"""

import os

import jax
import jax.numpy as jnp
import pytest

from codit.data_module import config_paths

# Capture the models directory at module-load time so the path is not affected
# by any CODIT_DATA_PATH monkey-patching done by other tests.
_MODELS_DIR = os.path.join(config_paths.MODEL_PATH, "trained_models")
_LGAUSS_PATH = os.path.join(_MODELS_DIR, "linear_gauss_open_ocean", "nnx_model.pkl")
_CODIT_PATH = os.path.join(_MODELS_DIR, "codit_GRU_open_ocean", "nn_params.npz")

requires_lgauss = pytest.mark.skipif(
    not os.path.exists(_LGAUSS_PATH),
    reason="linear_gauss_open_ocean checkpoint not found — run download_data.sh first",
)
requires_codit = pytest.mark.skipif(
    not os.path.exists(_CODIT_PATH),
    reason="codit_GRU_open_ocean checkpoint not found — run download_data.sh first",
)


@requires_lgauss
def test_linear_gauss_checkpoint_loads():
    """LinearGaussModel.import_trained_model must load without error."""
    from codit.model_module.linear_gauss_model import LinearGaussModel

    model = LinearGaussModel.import_trained_model(
        "linear_gauss_open_ocean", import_path=_MODELS_DIR
    )
    assert model.trained_successfully is True
    assert model.type == "linear_gauss"


@requires_lgauss
def test_linear_gauss_alpha_is_trained():
    """After loading, alpha must be non-zero (model was actually trained)."""
    from codit.model_module.linear_gauss_model import LinearGaussModel

    model = LinearGaussModel.import_trained_model(
        "linear_gauss_open_ocean", import_path=_MODELS_DIR
    )
    alpha = float(model.model.alpha.value[0])
    assert abs(alpha) > 1e-4, f"alpha={alpha} looks untrained (expected non-zero)"


@requires_lgauss
def test_linear_gauss_forward_pass_with_dummy_context():
    """Loaded model must produce a (2,) velocity for a dummy context grid."""
    from codit.model_module.linear_gauss_model import LinearGaussModel

    model = LinearGaussModel.import_trained_model(
        "linear_gauss_open_ocean", import_path=_MODELS_DIR
    )
    # Context: (grid_H, grid_W, 8_features) — spatial grid with 8 ocean/wind variables
    dummy_context = jnp.zeros((5, 5, 8))
    key = jax.random.PRNGKey(0)
    vel = model.model(dummy_context, key=key)
    assert vel.shape == (2,), f"Expected (2,), got {vel.shape}"


@requires_lgauss
def test_linear_gauss_coastline_checkpoint_loads():
    """Coastline checkpoint must also load cleanly."""
    from codit.model_module.linear_gauss_model import LinearGaussModel

    model = LinearGaussModel.import_trained_model(
        "linear_gauss_coastline", import_path=_MODELS_DIR
    )
    assert model.trained_successfully is True


@requires_codit
def test_codit_gru_checkpoint_loads():
    """CoDiTModel.import_trained_model must load without error."""
    from codit.model_module.diffusion_model import CoDiTModel

    model = CoDiTModel.import_trained_model(
        "codit_GRU_open_ocean", import_path=_MODELS_DIR
    )
    assert model.nn_params is not None
    assert model.std is not None


@requires_codit
def test_codit_gru_std_shape():
    """Loaded CoDiT model std must have shape (2,) — one scale per spatial dimension."""
    from codit.model_module.diffusion_model import CoDiTModel

    model = CoDiTModel.import_trained_model(
        "codit_GRU_open_ocean", import_path=_MODELS_DIR
    )
    assert model.std.shape == (2,), f"Expected std.shape=(2,), got {model.std.shape}"


@requires_codit
def test_codit_gru_schedule_matches_config():
    """Loaded schedule K must match the value in model_config.yml (100)."""
    from codit.model_module.diffusion_model import CoDiTModel

    model = CoDiTModel.import_trained_model(
        "codit_GRU_open_ocean", import_path=_MODELS_DIR
    )
    assert model.schedule_dict["K"] == model.params["K"]


@requires_codit
def test_codit_gru_noise_model_matches_config():
    """Loaded noise model type must match the config."""
    from codit.model_module.diffusion_model import CoDiTModel

    model = CoDiTModel.import_trained_model(
        "codit_GRU_open_ocean", import_path=_MODELS_DIR
    )
    assert model.noise_model.noise_type == model.params["noise"]
