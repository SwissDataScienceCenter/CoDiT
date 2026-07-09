from codit.model_module.currents_parcels_model import CurrentsParcelsModel


class HycomParcelsModel(CurrentsParcelsModel):
    def __init__(self, name: str, params: dict) -> None:
        super().__init__(name, params, source="hycom")

    @classmethod
    def import_trained_model(
        cls, model_name: str, import_path: str | None
    ) -> "HycomParcelsModel":
        return HycomParcelsModel(model_name, {})
