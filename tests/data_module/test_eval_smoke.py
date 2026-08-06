"""Smoke tests verifying eval script bugs are fixed."""


def test_lingauss_eval_saves_distances_not_distance():
    """lingauss_eval.py must save the concatenated distances array under a matching 'distances' key."""
    with open("scripts/user/eval/lingauss_eval.py") as f:
        source = f.read()

    # The jnp.savez call must use distances=distances (key matches value)
    assert "distances=distances," in source


def test_lingauss_eval_assigns_import_result():
    """import_trained_model is a @classmethod returning a new instance.
    The result must be assigned, not discarded."""
    with open("scripts/user/eval/lingauss_eval.py") as f:
        source = f.read()

    assert "model = LinearGaussModel.import_trained_model(" in source


def test_codit_eval_has_model_name_override():
    """codit_eval.py must accept --model_name to override the constructed name."""
    with open("scripts/user/eval/codit_eval.py") as f:
        source = f.read()

    assert "--model_name" in source
