"""Smoke tests verifying eval script bugs are fixed."""


def test_lingauss_eval_saves_distances_not_distance():
    """lingauss_eval.py must save 'distances' (concatenated), not 'distance' (last sample)."""
    with open("my_scripts/model_fitting/lingauss_eval.py") as f:
        source = f.read()

    # The jnp.savez call must use distance=distances
    assert "distance=distances," in source


def test_lingauss_eval_assigns_import_result():
    """import_trained_model is a @classmethod returning a new instance.
    The result must be assigned, not discarded."""
    with open("my_scripts/model_fitting/lingauss_eval.py") as f:
        source = f.read()

    assert "model = LinearGaussModel.import_trained_model(" in source


def test_codit_eval_has_model_name_override():
    """codit_eval.py must accept --model_name to override the constructed name."""
    with open("my_scripts/model_fitting/codit_eval.py") as f:
        source = f.read()

    assert "--model_name" in source
