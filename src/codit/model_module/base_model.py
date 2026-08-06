import datetime
import os
import pickle as pkl
from abc import ABC, abstractmethod

import numpy as np
import yaml
from interpax import Interpolator3D
from jax import numpy as jnp

from codit.data_module import utils
from codit.data_module.dataset import OceanDataset, OceanTrainingDataset


class BaseTrajectoryModel(ABC):
    """
    Abstract base class for trajectory models.

    This class provides a template for machine learning models in the trajectory prediction framework,
    for predicting trajectories based on relevant input features. It defines the basic structure
    including methods for fitting, saving, loading, and predicting. Subclasses must
    implement the `_fit` and `predict` methods.

    Attributes:
        name (str): Name of the model.
        type (str): Type of the model (default: "base_model").
        params (dict): Model parameters.
            All models can have the field 'vars', that is a list of strings. Possible fields are 'hycom_current',
            'copernicus_current', 'era5_wind', and 'bathymetry', which selects the corresponding input variables.
            If nothing is indicated, all variables are taken.
        training_sets (list): Names of the training sets used.
        trained_successfully (bool): Indicates if the model has been trained successfully.

    Methods:
        set_model_name(name: str) -> AdoptBaseModel:
            Sets the model's name.

        fit(train_configs: list) -> AdoptBaseModel:
            Fits the model using the provided training configurations.

        _fit(train_configs: list) -> AdoptBaseModel:
            Abstract method to be implemented by subclasses for model-specific fitting.

        create_config() -> dict:
            Creates a configuration dictionary for the model.

        save(model_path: str):
            Saves the trained model and its configuration to disk.

        load(model_name: str, model_path: str) -> tuple[AdoptBaseModel, dict]:
            Loads a model and its configuration from disk.

        predict(test_config: dict, trajectory_length: int) -> dict:
            Abstract method to be implemented by subclasses for making predictions.
    """

    def __init__(self, name: str, params: dict):
        """
        Initializes the base model with a name and parameters.

        Args:
            name (str): The name of the model.
            params (dict): A dictionary of parameters for the model.

        Attributes:
            name (str): The name of the model.
            type (str): The type of the model, set to "base_model".
            params (dict): The parameters for the model.
            training_set: The training dataset, initialized as None.
            trained_successfully (bool): Indicates if the model has been trained successfully, initialized as False.
        """
        self.name = name
        self.type = "base_model"
        self.params = params
        self.training_set = None
        self.trained_successfully = False

        self.vars = self.params.get("vars", None)
        if self.vars is None:
            self.var_idx = jnp.arange(0, 8, dtype=int)
        else:
            var_idx = []
            if "hycom_current" in self.vars:
                var_idx += [0, 1]
            if "copernicus_current" in self.vars:
                var_idx += [2, 3]
            if "era5_wind" in self.vars:
                var_idx += [4, 5]
            if "bathymetry" in self.vars:
                var_idx += [6, 7]
            self.var_idx = jnp.sort(jnp.asarray(var_idx, dtype=int))

    def set_model_name(self, name: str) -> "BaseTrajectoryModel":
        """
        Sets the model's name attribute.

        Args:
            name (str): The name to assign to the model.

        Returns:
            AdoptBaseModel: The instance of the model with the updated name.
        """
        self.name = name
        return self

    def fit(
        self,
        train_dataloader: OceanTrainingDataset,
        val_dataloader: OceanTrainingDataset,
        train_params: dict | None = None,
    ) -> "BaseTrajectoryModel":
        """
        Fits the model using the provided training configurations.

        Args:
            train_configs (list): A list of training configuration dictionaries. Each dictionary should contain at least a "name" key.

        Returns:
            AdoptBaseModel: The instance of the model after fitting.

        Side Effects:
            - Sets the `training_sets` attribute to the list of training set names.
            - Calls the internal `_fit` method with the training configurations.
            - Sets the `trained_successfully` attribute to True.
        """
        if train_params is None:
            train_params = {}
        # self.training_sets = [config["name"] for config in train_configs]
        self._fit(train_dataloader, val_dataloader, train_params)
        self.trained_successfully = True
        return self

    @abstractmethod
    def _fit(
        self,
        train_dataloader: OceanTrainingDataset,
        val_dataloader: OceanTrainingDataset,
        train_params: dict | None = None,
    ) -> "BaseTrajectoryModel":
        """
        Fits the model using the provided training configurations.

        Args:
            train_configs (list): A list of training configuration objects or dictionaries required to train the model.

        Returns:
            AdoptBaseModel: The fitted model instance.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass

    def create_config(self) -> dict:
        """
        Creates a configuration dictionary representing the current state of the model.

        Returns:
            dict: A dictionary containing the model's name, type, parameters, training sets,
                  and training success status.
        """
        return {
            "name": self.name,
            "type": self.type,
            "params": self.params,
            "trained_successfully": self.trained_successfully,
        }

    def save(self, model_path: str) -> None:
        """
        Saves the trained model and its configuration to the specified directory.

        If the model has been trained successfully, this method creates a configuration
        dictionary, sets the save path for the model parameters, and writes the configuration
        to a YAML file. The model itself is serialized and saved as a pickle file. If a model
        file with the same name already exists at the target location, an exception is raised.
        If the model has not been trained, an exception is raised.

        Args:
            model_path (str): The base directory where the model and its configuration will be saved.

        Raises:
            Exception: If the model has not been trained yet.
            Exception: If a model file with the same name already exists.
        """
        if self.trained_successfully:
            model_config = self.create_config()
            model_config["save_path"] = os.path.join(
                model_path, "params", self.name + ".pkl"
            )
            if os.path.exists(model_config["save_path"]):
                raise Exception(
                    "Model already exists, please delete it or change the model name"
                )
            model_config_file = os.path.join(model_path, "configs", self.name + ".yml")
            with open(model_config_file, "w") as f:
                yaml.dump(model_config, f, default_flow_style=False)
            with open(model_config["save_path"], "wb") as f:
                pkl.dump(self, f)
        else:
            raise Exception("Model has not been trained yet")

    @classmethod
    def load(
        cls, model_name: str, model_path: str
    ) -> tuple["BaseTrajectoryModel", dict]:
        """
        Loads a model and its configuration from the specified path.

        Args:
            model_name (str): The name of the model to load. Used to locate the configuration file.
            model_path (str): The base directory where the model and its configuration are stored.

        Returns:
            tuple[AdoptBaseModel, dict]: A tuple containing the loaded model instance and its configuration dictionary.

        Raises:
            FileNotFoundError: If the configuration file or model file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML configuration file.
            pickle.UnpicklingError: If there is an error unpickling the model file.
        """
        model_config_file = os.path.join(model_path, "configs", model_name + ".yml")
        with open(model_config_file, "r") as f:
            model_config = yaml.safe_load(f)
        model_path = model_config["save_path"]
        with open(model_path, "rb") as f:
            model = pkl.load(f)
        return model, model_config

    @staticmethod
    def _get_prediction_steps(
        start_time: datetime.datetime, prediction_length: int
    ) -> jnp.ndarray:
        # Generate an array of prediction steps based on the start time and prediction length
        times = np.arange(
            start_time,
            start_time + datetime.timedelta(hours=prediction_length + 1),
            datetime.timedelta(hours=1),
            dtype=np.datetime64,
        )
        times = jnp.array(utils.convert_date_to_hours(times), dtype=jnp.float32)
        return times

    @abstractmethod
    def predict(
        self,
        start_position: jnp.ndarray,
        start_time: datetime.datetime,
        prediction_length: int,
        interpolator: Interpolator3D,
        test_dataset: OceanDataset,
        num_samples: int = 1,
    ) -> jnp.ndarray:
        """
        Predicts outcomes based on the provided test configuration and trajectory length.

        Args:
            test_config (dict): Configuration parameters for the prediction process.
            trajectory_length (int): The length of the trajectory to predict.

        Returns:
            dict: A dictionary containing the prediction results.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def export_trained_model(
        self, model_name: str | None = None, export_path: str | None = None
    ) -> None:
        """
        Exports the trained model to a specified path.

        Args:
            model_name (str): The name of the model to export. If None, uses the model's name.
            export_path (str): The directory where the model will be exported. If None, uses a default path.

        Raises:
            ValueError: If the model parameters are not set.
        """
        pass

    @classmethod
    @abstractmethod
    def import_trained_model(
        cls, model_name: str, import_path: str | None
    ) -> "BaseTrajectoryModel":
        """
        Imports a trained model from the specified path.

        Args:
            model_name (str): The name of the model to import.
            import_path (str | None): The path from which to import the model. If None, uses a default path.

        Returns:
            BaseTrajectoryModel: An instance of the model with the loaded parameters.

        Raises:
            FileNotFoundError: If the configuration file or model file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML configuration file.
            pickle.UnpicklingError: If there is an error unpickling the model file.
        """
        pass
