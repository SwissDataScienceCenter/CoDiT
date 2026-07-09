def test_hycom_model_is_importable():
    from codit.model_module.hycom_model import HycomModel

    assert HycomModel is not None


def test_glorys_model_is_importable():
    from codit.model_module.glorys_model import GlorysModel

    assert GlorysModel is not None


def test_hycom_model_has_required_methods():
    from codit.model_module.hycom_model import HycomModel

    assert hasattr(HycomModel, "predict")
    assert hasattr(HycomModel, "_fit")


def test_glorys_model_has_required_methods():
    from codit.model_module.glorys_model import GlorysModel

    assert hasattr(GlorysModel, "predict")
    assert hasattr(GlorysModel, "_fit")
