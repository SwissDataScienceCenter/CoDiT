from codit.model_module.currents_parcels_model import CurrentsParcelsModel


class GlorysParcelsModel(CurrentsParcelsModel):
    def __init__(self, name: str, params: dict) -> None:
        super().__init__(name, params, source="glorys")

    @classmethod
    def import_trained_model(
        cls, model_name: str, import_path: str | None
    ) -> "GlorysParcelsModel":
        return GlorysParcelsModel(model_name, {})
