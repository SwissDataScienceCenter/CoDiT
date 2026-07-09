"""Tests for config_paths and config_funcs."""
import importlib
import os
from datetime import datetime
from unittest.mock import patch

import yaml


class TestConfigPaths:
    """config_paths must resolve from CODIT_DATA_PATH env var."""

    def test_resolves_from_env_var(self):
        with patch.dict(os.environ, {"CODIT_DATA_PATH": "/tmp/test_codit"}):
            from codit.data_module import config_paths

            importlib.reload(config_paths)
            assert config_paths.DATA_PATH == "/tmp/test_codit"
            assert config_paths.CONFIG_PATH == "/tmp/test_codit/configs"
            assert config_paths.RESULT_PATH == "/tmp/test_codit/results"
            assert config_paths.FIG_PATH == "/tmp/test_codit/figs"

    def test_defaults_to_repo_data_dir(self):
        env = os.environ.copy()
        env.pop("CODIT_DATA_PATH", None)
        with patch.dict(os.environ, env, clear=True):
            from codit.data_module import config_paths

            importlib.reload(config_paths)
            assert config_paths.DATA_PATH.endswith("/data")
            assert os.path.isabs(config_paths.DATA_PATH)

    def test_credentials_from_env_var(self):
        with patch.dict(
            os.environ, {"CODIT_CREDENTIALS_PATH": "/tmp/my_creds.ini"}
        ):
            from codit.data_module import config_paths

            importlib.reload(config_paths)
            assert config_paths.CREDENTIALS_PATH == "/tmp/my_creds.ini"

    def test_model_path_resolves_trained_models(self):
        """MODEL_PATH/trained_models/ must point to data/trained_models/."""
        with patch.dict(os.environ, {"CODIT_DATA_PATH": "/tmp/test_codit"}):
            from codit.data_module import config_paths

            importlib.reload(config_paths)
            trained = os.path.join(config_paths.MODEL_PATH, "trained_models")
            assert trained == "/tmp/test_codit/trained_models"


class TestLoadConfig:
    """load_config must reconstruct dataset_path from current DATA_PATH."""

    def test_reconstructs_dataset_path(self, tmp_path):
        config_dir = tmp_path / "configs" / "coastline" / "val"
        config_dir.mkdir(parents=True)
        config_data = {
            "name": "test_patch",
            "latitudes": [1.0, 2.0],
            "longitudes": [7.0, 8.0],
            "start_date": datetime(2020, 3, 7, 16, 0),
            "end_date": datetime(2020, 3, 12, 16, 0),
            "dataset_path": "/old/stale/path/that/should/be/overridden",
            "context_delta": 1.5,
            "type": "region_based",
        }
        with open(config_dir / "test_patch.yml", "w") as f:
            yaml.dump(config_data, f)

        with patch.dict(os.environ, {"CODIT_DATA_PATH": str(tmp_path)}):
            from codit.data_module import config_funcs, config_paths

            importlib.reload(config_paths)
            result = config_funcs.load_config("test_patch", str(config_dir))

        expected = os.path.join(
            str(tmp_path), "datasets", "coastline", "val", "test_patch"
        )
        assert result["dataset_path"] == expected

    def test_complete_config_populates_derived_paths(self, tmp_path):
        config_dir = tmp_path / "configs" / "open_ocean" / "test"
        config_dir.mkdir(parents=True)
        config_data = {
            "name": "test_patch_2",
            "latitudes": [10.0, 20.0],
            "longitudes": [-60.0, -50.0],
            "start_date": datetime(2021, 1, 1),
            "end_date": datetime(2021, 1, 5),
            "dataset_path": "/stale",
            "context_delta": 1.5,
            "type": "region_based",
        }
        with open(config_dir / "test_patch_2.yml", "w") as f:
            yaml.dump(config_data, f)

        with patch.dict(os.environ, {"CODIT_DATA_PATH": str(tmp_path)}):
            from codit.data_module import config_funcs, config_paths

            importlib.reload(config_paths)
            result = config_funcs.load_config("test_patch_2", str(config_dir))

        # complete_config should populate hycom_file, era5_file, etc.
        assert "hycom_file" in result
        assert "era5_file" in result
        assert "trajectories_file" in result
        assert result["hycom_file"].startswith(result["dataset_path"])
