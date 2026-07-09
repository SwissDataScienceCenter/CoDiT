import jax
import pytest
from codit.codit_module.brownian_noise_models import WalkNoiseModel
from codit.codit_module.scheduler import linear_schedule


@pytest.fixture
def shared_schedule():
    return linear_schedule(K=100)


@pytest.fixture
def jax_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def walk_noise_model():
    return WalkNoiseModel(
        noise_type="white_noise",
        num_dims=2,
        dt=1.0,
        sigma=1.0,
        T=10.0,
    )
