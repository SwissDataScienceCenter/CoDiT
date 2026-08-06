import datetime

from jax import numpy as jnp
import numpy as np

try:
    import parcels
    import zarr
except ImportError:
    parcels = None  # type: ignore[assignment]
    zarr = None  # type: ignore[assignment]
import xarray as xr

from codit.model_module.base_model import BaseTrajectoryModel

# Mapping from source name to the UV variable names used in the NetCDF dataset.
_UV_VARS: dict[str, tuple[str, str]] = {
    "hycom": ("water_u", "water_v"),
    "glorys": ("uo", "vo"),
}


class CurrentsParcelsModel(BaseTrajectoryModel):
    """Physics-based OceanParcels model: advect a particle using ocean current data.

    Args:
        name: Model name.
        params: Model parameter dictionary (passed to BaseTrajectoryModel).
            Optional keys:
                ``"kernel"``: ``"RK4"`` | ``"RK45"`` | ``"Euler"`` (default ``"RK4"``)
                ``"sub_dt"``: int, sub-hourly timestep in minutes (default 60)
        source: Which current dataset to use.  Must be ``"hycom"`` or ``"glorys"``.

    The ``type`` attribute is set to ``f"{source}_parcels_model"`` so that, e.g.,
    ``HycomParcelsModel().type == "hycom_parcels_model"``.
    """

    def __init__(self, name: str, params: dict, source: str) -> None:
        if source not in _UV_VARS:
            raise ValueError(
                f"Unknown source {source!r}. Expected one of {list(_UV_VARS)}."
            )
        super().__init__(name, params)
        self.source = source
        self.type = f"{source}_parcels_model"
        self.kernel_name = params.get("kernel", "RK4")
        self.sub_dt_minutes = params.get("sub_dt", 60)

    def _fit(
        self, train_dataloader, val_dataloader, train_params=None
    ) -> "CurrentsParcelsModel":
        if train_params is None:
            train_params = {}
        return self  # no-op, physics-only model

    def predict(
        self,
        start_position,
        start_time,
        prediction_length,
        interpolator,
        test_dataset,
        **kwargs,
    ) -> jnp.ndarray:
        if parcels is None:
            raise ImportError(
                "The 'parcels' package is required for CurrentsParcelsModel. "
                "Install it with: conda install -c conda-forge parcels"
            )
        ds_uv = self._extract_uv_dataset(
            test_dataset, start_time, prediction_length + 1
        )
        fieldset = self._build_fieldset(ds_uv)
        lats, lons = self._run_parcels(
            fieldset, start_position, start_time, prediction_length + 1
        )
        coords = np.stack([lats, lons], axis=-1)  # (prediction_length, 2)
        return jnp.array(coords[None], dtype=jnp.float32)  # (1, prediction_length, 2)

    def export_trained_model(
        self, model_name: str | None = None, export_path: str | None = None
    ) -> None:
        """Exports the trained model to a specified path.

        Args:
            model_name: The name of the model to export. If None, uses the model's name.
            export_path: The directory where the model will be exported. If None, uses a
                default path.

        Raises:
            ValueError: If the model parameters are not set.
        """
        pass

    @classmethod
    def import_trained_model(
        cls, model_name: str, import_path: str | None
    ) -> "CurrentsParcelsModel":
        """Imports a trained model from the specified path.

        Args:
            model_name: The name of the model to import.
            import_path: The path from which to import the model. If None, uses a default
                path.

        Returns:
            CurrentsParcelsModel: An instance of the model with the loaded parameters.

        Raises:
            FileNotFoundError: If the configuration file or model file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML configuration file.
            pickle.UnpicklingError: If there is an error unpickling the model file.
        """
        raise NotImplementedError(
            "import_trained_model must be called on a concrete subclass "
            "(HycomParcelsModel or GlorysParcelsModel)."
        )

    def _extract_uv_dataset(self, test_dataset, start_time, prediction_length):
        u_var, v_var = _UV_VARS[self.source]
        t_start = start_time - datetime.timedelta(
            hours=1
        )  # 1-hour buffer for interpolation
        t_end = start_time + datetime.timedelta(hours=prediction_length + 1)
        ds = test_dataset.context_xarray.sel(time=slice(t_start, t_end))
        ds_uv = ds[[u_var, v_var]].fillna(0.0)  # NaN->0 (land cells = no advection)
        return ds_uv

    def _build_fieldset(self, ds):
        u_var, v_var = _UV_VARS[self.source]
        fieldset = parcels.FieldSet.from_xarray_dataset(
            ds,
            variables={"U": u_var, "V": v_var},
            dimensions={"lat": "latitude", "lon": "longitude", "time": "time"},
            mesh="spherical",  # lat/lon degrees, polar correction applied
            allow_time_extrapolation=True,  # explicit error if particle leaves time window
        )
        return fieldset

    def _run_parcels(self, fieldset, start_position, start_time, prediction_length):
        kernel = getattr(
            parcels,
            {"RK4": "AdvectionRK4", "RK45": "AdvectionRK45", "Euler": "AdvectionEE"}[
                self.kernel_name
            ],
        )

        particle_time = float(
            fieldset.time_origin.reltime(np.datetime64(start_time, "ns"))
        )

        pset = parcels.ParticleSet(
            fieldset=fieldset,
            pclass=parcels.JITParticle,
            lon=[float(start_position[1])],
            lat=[float(start_position[0])],
            time=[particle_time],
        )

        try:
            store = zarr.storage.MemoryStore()  # zarr v3
        except AttributeError:
            store = zarr.MemoryStore()  # zarr v2 fallback

        output_file = pset.ParticleFile(
            name=store, outputdt=datetime.timedelta(hours=1)
        )

        pset.execute(
            pyfunc=kernel,
            runtime=datetime.timedelta(hours=prediction_length),
            dt=self.sub_dt_minutes * 60,  # default 3600s (60 min)
            output_file=output_file,
            verbose_progress=False,
        )

        ds_out = xr.open_zarr(store)
        # obs axis: index 0 = initial position (t=start_time); 1..prediction_length = hourly outputs
        lats = np.array(ds_out["lat"].values[0, 1:])  # (prediction_length,)
        lons = np.array(ds_out["lon"].values[0, 1:])
        return lats, lons
