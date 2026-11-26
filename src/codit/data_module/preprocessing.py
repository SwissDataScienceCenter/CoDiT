"""Preprocessing helpers: merge per-patch OceanDatasets into a single training h5.

The helper is the primary entry point; scripts/dev/data/precompute_trainset.py is a
thin CLI wrapper around it.
"""
import gc
import logging
import os

from codit.data_module.dataset import (
    OceanDataset,
    OceanTrainingDataset,
    merge_training_sets,
)

logger = logging.getLogger(__name__)


def preprocess_configs(
    configs: list[dict],
    length: int,
    num_grid_points: int,
    delta_metres: int,
    output_path: str,
    interp_method: str = "linear",
) -> str:
    """Preprocess a list of patch configs into a single merged training h5.

    For each config: build OceanDataset, convert to OceanTrainingDataset, and
    save an intermediate h5 under config["dataset_path"]. Then reload every
    intermediate, strip NaNs, merge, and write the merged h5 into output_path.

    Args:
        configs: Fully-loaded config dicts (already through config_funcs.load_config).
        length: Trajectory snippet length in hours.
        num_grid_points: Grid points per dimension of the context tensor.
        delta_metres: Half-width of the context grid in metres.
        output_path: Directory to write the merged h5 into (created if missing).
        interp_method: Interpolation method for the context (only "linear" supported).

    Returns:
        Absolute path of the merged h5 file: <output_path>/training_data_<length>h.h5.
    """
    os.makedirs(output_path, exist_ok=True)

    for cfg in configs:
        intermediate = os.path.join(cfg["dataset_path"], f"training_data_{length}h.h5")
        if os.path.exists(intermediate):
            logger.info("skipping %s (already cached at %s)", cfg["name"], intermediate)
            continue
        os.makedirs(cfg["dataset_path"], exist_ok=True)
        logger.info("preprocessing %s", cfg["name"])
        raw = training = None
        try:
            raw = OceanDataset(cfg, length, num_grid_points, delta_metres, interp_method)
            if len(raw) == 0:
                logger.warning("no snippets in %s — skipping", cfg["name"])
                continue
            training = OceanTrainingDataset.from_dataset(raw)
            training.save(cfg["dataset_path"])
        except Exception as e:
            # Preserve original per-patch robustness: a single bad patch
            # (missing NetCDF, corrupt file, etc.) must not abort the whole run.
            logger.warning("failed to preprocess %s: %s — skipping", cfg["name"], e)
            continue
        finally:
            # A loaded patch takes tens of GB. Release it before loading the next one,
            # otherwise two patches are alive at once while the next is constructed.
            del raw, training
            gc.collect()

    datasets: list[OceanTrainingDataset] = []
    for cfg in configs:
        intermediate = os.path.join(cfg["dataset_path"], f"training_data_{length}h.h5")
        if not os.path.exists(intermediate):
            continue
        try:
            one = OceanTrainingDataset.from_disk(cfg["dataset_path"], length)
            one.remove_nans()
            datasets.append(one)
        except Exception as e:
            logger.warning(
                "failed to load intermediate for %s: %s — skipping", cfg["name"], e
            )
            continue

    if not datasets:
        raise RuntimeError(
            f"No intermediate training h5 files were produced for any of "
            f"{[c['name'] for c in configs]} — check config paths and inputs."
        )

    merged = merge_training_sets(datasets)
    merged.save(output_path)
    return os.path.abspath(os.path.join(output_path, f"training_data_{length}h.h5"))
