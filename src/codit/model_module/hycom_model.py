from codit.model_module.currents_baseline_model import CurrentsBaselineModel


class HycomModel(CurrentsBaselineModel):
    def __init__(self, name: str, params: dict) -> None:
        super().__init__(name, params, source="hycom")

    @classmethod
    def import_trained_model(
        cls, model_name: str, import_path: str | None
    ) -> "HycomModel":
        return HycomModel(model_name, {})
