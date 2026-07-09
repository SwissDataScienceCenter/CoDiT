"""Tests for CoDiTModel correctness."""


def test_codit_model_does_not_override_fit():
    """
    CoDiTModel.fit() must NOT be defined as an override.
    If it is, it bypasses BaseTrajectoryModel.fit() which sets trained_successfully.
    The fix: delete CoDiTModel.fit() so the base class method is inherited.
    """
    from codit.model_module.diffusion_model import CoDiTModel

    # CoDiTModel.__dict__ contains only methods defined directly on CoDiTModel.
    # If fit() is not in __dict__, it is inherited from BaseTrajectoryModel (correct).
    assert "fit" not in CoDiTModel.__dict__, (
        "CoDiTModel must not override fit(). "
        "Remove the fit() method so BaseTrajectoryModel.fit() is inherited. "
        "The base class calls self._fit() and then sets self.trained_successfully = True."
    )
