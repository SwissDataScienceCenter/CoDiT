from codit.model_module.currents_baseline_model import CurrentsBaselineModel


class GlorysModel(CurrentsBaselineModel):
    def __init__(self, name: str, params: dict) -> None:
        super().__init__(name, params, source="glorys")

    @classmethod
    def import_trained_model(
        cls, model_name: str, import_path: str | None
    ) -> "GlorysModel":
        return GlorysModel(model_name, {})
