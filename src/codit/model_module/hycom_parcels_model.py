import datetime

from jax import numpy as jnp
import numpy as np
import parcels, zarr
import xarray as xr

from codit.data_module import data_transform
from codit.data_module.dataset import OceanDataset
from codit.model_module.base_model import BaseTrajectoryModel

class HycomParcelsModel(BaseTrajectoryModel):
    def __init__(self, name: str, params: dict) -> None:
        # params optional keys:
        #   "kernel": "RK4" | "RK45" | "Euler"  (default "RK4")
        #   "sub_dt": int, sub-hourly timestep in minutes (default 20)
        super().__init__(name, params)
        self.type = "hycom_parcels"
        self.kernel_name = params.get("kernel", "RK4")
        self.sub_dt_minutes = params.get("sub_dt", 60)

    def _fit(self, train_dataloader, val_dataloader, train_params={}) -> "HycomParcelsModel":
        return self  # no-op, physics-only model

    def predict(self, start_position, start_time, prediction_length,
                interpolator, test_dataset, **kwargs) -> jnp.ndarray:
        ds_uv    = self._extract_uv_dataset(test_dataset, start_time, prediction_length + 1)
        fieldset = self._build_fieldset(ds_uv)
        lats, lons = self._run_parcels(fieldset, start_position, start_time, prediction_length + 1)
        coords = np.stack([lats, lons], axis=-1)           # (prediction_length, 2)
        return jnp.array(coords[None], dtype=jnp.float32) # (1, prediction_length, 2)
    
    def export_trained_model(self, model_name: str | None = None, export_path: str | None = None) -> None:
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
    def import_trained_model(cls, model_name: str, import_path: str | None) -> "HycomParcelsModel":
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
        return HycomParcelsModel(model_name, {})
    
    
    def _extract_uv_dataset(self, test_dataset, start_time, prediction_length):
        t_start = start_time - datetime.timedelta(hours=1)   # 1-hour buffer for interpolation
        t_end   = start_time + datetime.timedelta(hours=prediction_length + 1)
        ds = test_dataset.context_xarray.sel(time=slice(t_start, t_end))
        ds_uv = ds[["water_u", "water_v"]].fillna(0.0)  # NaN->0 (land cells = no advection)
        return ds_uv
    
    def _build_fieldset(self, ds):
        fieldset = parcels.FieldSet.from_xarray_dataset(
            ds,
            variables={"U": "water_u", "V": "water_v"},
            dimensions={"lat": "latitude", "lon": "longitude", "time": "time"},
            mesh="spherical",                  # lat/lon degrees, polar correction applied
            allow_time_extrapolation=True,    # explicit error if particle leaves time window
        )
        return fieldset
    
    def _run_parcels(self, fieldset, start_position, start_time, prediction_length):
        

        kernel = getattr(parcels, {"RK4":"AdvectionRK4",
                                   "RK45":"AdvectionRK45",
                                   "Euler":"AdvectionEE"}[self.kernel_name])

        particle_time = float(fieldset.time_origin.reltime(np.datetime64(start_time, 'ns')))  
        
        pset = parcels.ParticleSet(
            fieldset=fieldset,
            pclass=parcels.JITParticle,
            lon=[float(start_position[1])],
            lat=[float(start_position[0])],
            time=[particle_time],
        )

        try:
            store = zarr.storage.MemoryStore()   # zarr v3
        except AttributeError:
            store = zarr.MemoryStore()           # zarr v2 fallback

        output_file = pset.ParticleFile(name=store, outputdt=datetime.timedelta(hours=1))

        pset.execute(
            pyfunc=kernel,
            runtime=datetime.timedelta(hours=prediction_length),
            dt=self.sub_dt_minutes * 60,        # default 1200s (20 min) = 3 RK4 steps/hour
            output_file=output_file,
            verbose_progress=False,
        )

        ds_out = xr.open_zarr(store)
        # obs axis: index 0 = initial position (t=start_time); 1..prediction_length = hourly outputs
        lats = np.array(ds_out["lat"].values[0, 1:])   # (prediction_length,)
        lons = np.array(ds_out["lon"].values[0, 1:])
        return lats, lons
