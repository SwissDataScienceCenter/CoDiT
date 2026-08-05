import argparse
import os
import warnings
from pathlib import Path

import numpy as np
from jax import numpy as jnp
from jax import vmap
from tqdm import tqdm

from codit.data_module import config_funcs
from codit.data_module import config_paths
from codit.data_module import data_transform
from codit.data_module.dataset import OceanDataset
from codit.model_module.direct_prediction_model import DirectPredictionModel

warnings.filterwarnings("ignore", message=".*[Ss]erializ.*", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="open_ocean", choices=["open_ocean", "coastline"]
    )
    parser.add_argument("--split", type=str, default="")
    parser.add_argument(
        "--model_type", type=str, default="GRU", choices=["GRU", "Transformer"]
    )
    parser.add_argument(
        "--context_encoder_type",
        type=str,
        default="conv",
        choices=["conv", "mlp", "gap"],
    )
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--evaluate_subset", action="store_true", default=False)
    parser.add_argument(
        "--input_vars",
        nargs="+",
        type=str,
        default=["hycom_current", "copernicus_current", "era5_wind", "bathymetry"],
    )
    args = parser.parse_args()
    np.random.seed(0)

    input_vars = sorted(args.input_vars)
    input_var_str = "_".join(input_vars)
    model_name = (
        f"direct_{args.model_type}_{args.dataset}_"
        f"{args.context_encoder_type}_{args.optimizer}_{input_var_str}"
    )

    if args.dataset == "open_ocean":
        config_path = os.path.join(config_paths.CONFIG_PATH, "open_ocean", args.split)
        result_path = os.path.join(
            config_paths.RESULT_PATH, "open_ocean", args.split, model_name
        )
        length = 72
    elif args.dataset == "coastline":
        config_path = os.path.join(config_paths.CONFIG_PATH, "coastline", args.split)
        result_path = os.path.join(
            config_paths.RESULT_PATH, "coastline", args.split, model_name
        )
        length = 24
    else:
        raise ValueError("Dataset not found")
    os.makedirs(result_path, exist_ok=True)

    val_datasets = []
    for dataset_name in os.listdir(config_path):
        if not dataset_name.endswith(".yml"):
            continue
        config = config_funcs.load_config(Path(dataset_name).stem, config_path)
        if config["start_date"].year >= 2020 and os.path.exists(config["dataset_path"]):
            val_datasets.append(dataset_name)
    print(f"Found {len(val_datasets)} validation datasets for evaluation.")
    model = DirectPredictionModel.import_trained_model(model_name)
    for i, val_dataname in tqdm(enumerate(val_datasets), total=len(val_datasets)):
        data_name = Path(val_dataname).stem
        print(f"Evaluating dataset: {data_name} ({i+1}/{len(val_datasets)})")
        result_file = os.path.join(result_path, f"{data_name}_result.npz")
        if os.path.exists(result_file):
            continue
        try:
            config = config_funcs.load_config(data_name, config_path)
            test_dataset = OceanDataset(config, length, 16, 10000, "linear")
        except Exception as exc:
            print(f"Error loading dataset {val_dataname}: {exc}")
            continue

        if len(test_dataset) == 0:
            print(f"Empty dataset: {val_dataname}")
            continue

        if args.evaluate_subset and len(test_dataset) > 200:
            indices = np.linspace(0, len(test_dataset) - 1, 200, dtype=int)
        else:
            indices = range(len(test_dataset))

        distances = []
        pred_coords = []
        true_coords = []
        for idx in indices:
            print(f"Processing trajectory {idx+1}/{len(test_dataset)} for dataset {data_name}")
            try:
                start_position, start_time, times, interpolator, coordinates = test_dataset[idx]
                pred_coord = model.predict(
                    start_position,
                    start_time,
                    length,
                    interpolator,
                    test_dataset,
                    num_samples=1,
                )
                # distance = vmap(data_transform.haversine_distance)(
                #     coordinates[1:], pred_coord[0]
                # )
                # distances.append(distance[None])
                pred_coords.append(pred_coord)
                true_coords.append(coordinates[1:][None])
            except Exception as exc:
                print(
                    f"Error processing dataset {val_dataname}, idx={idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

        if not pred_coords:
            print(f"No valid trajectories for dataset: {val_dataname}")
            continue

        jnp.savez(
            result_file,
            pred_coords=jnp.concatenate(pred_coords, axis=0),
            true_coords=jnp.concatenate(true_coords, axis=0),
        )