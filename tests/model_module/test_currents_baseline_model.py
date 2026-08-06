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


def test_hycom_parcels_model_is_importable():
    from codit.model_module.hycom_parcels_model import HycomParcelsModel

    assert HycomParcelsModel is not None


def test_glorys_parcels_model_is_importable():
    from codit.model_module.glorys_parcels_model import GlorysParcelsModel

    assert GlorysParcelsModel is not None


def test_currents_parcels_model_is_importable():
    from codit.model_module.currents_parcels_model import CurrentsParcelsModel

    assert CurrentsParcelsModel is not None


def test_hycom_parcels_model_has_required_methods():
    from codit.model_module.hycom_parcels_model import HycomParcelsModel

    assert hasattr(HycomParcelsModel, "predict")
    assert hasattr(HycomParcelsModel, "_fit")


def test_glorys_parcels_model_has_required_methods():
    from codit.model_module.glorys_parcels_model import GlorysParcelsModel

    assert hasattr(GlorysParcelsModel, "predict")
    assert hasattr(GlorysParcelsModel, "_fit")


def test_hycom_parcels_model_is_subclass_of_currents_parcels():
    from codit.model_module.currents_parcels_model import CurrentsParcelsModel
    from codit.model_module.hycom_parcels_model import HycomParcelsModel

    assert issubclass(HycomParcelsModel, CurrentsParcelsModel)


def test_glorys_parcels_model_is_subclass_of_currents_parcels():
    from codit.model_module.currents_parcels_model import CurrentsParcelsModel
    from codit.model_module.glorys_parcels_model import GlorysParcelsModel

    assert issubclass(GlorysParcelsModel, CurrentsParcelsModel)


def test_hycom_parcels_model_source_and_type():
    from codit.model_module.hycom_parcels_model import HycomParcelsModel

    m = HycomParcelsModel("test", {})
    assert m.source == "hycom"
    assert m.type == "hycom_parcels_model"


def test_glorys_parcels_model_source_and_type():
    from codit.model_module.glorys_parcels_model import GlorysParcelsModel

    m = GlorysParcelsModel("test", {})
    assert m.source == "glorys"
    assert m.type == "glorys_parcels_model"
