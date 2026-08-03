import argparse
import os
import sys

# mute SerializationWarning
import warnings

from jax import numpy as jnp
from jax import vmap
from tqdm import tqdm
import numpy as np
from codit.data_module import config_paths
from codit.data_module import config_funcs
from codit.data_module.dataset import OceanDataset
from pathlib import Path

from codit.data_module import data_transform
from codit.model_module.gru_model import AutoregressiveModel

warnings.filterwarnings("ignore", message=".*[Ss]erializ.*", category=UserWarning)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="open_ocean", choices=["open_ocean", "coastline"])
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--model_type", type=str, default="GRU", choices=["GRU", "Transformer"])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument(
        "--context_encoder_type", type=str, default="conv", choices=["conv", "mlp", "gap"]
    )
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--evaluate_subset", action="store_true", default=False)
    parser.add_argument(
        "--input_vars",
        nargs="+",
        type=str,
        default=["hycom_current", "copernicus_current", "era5_wind", "bathymetry"],
        help="Input variables for the model",
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num_samples must be > 0")
    np.random.seed(0)

    # Canonicalise ordering so model_name matches regardless of input_vars argument order
    input_vars = sorted(args.input_vars)
    input_var_str = "_".join(input_vars)
    model_name = (
        f"gru_{args.model_type}_{args.dataset}_{args.context_encoder_type}_"
        f"{args.optimizer}_{input_var_str}"
    )
    print(f"Model name: {model_name}")

    if args.dataset == "open_ocean":
        config_path = os.path.join(
            config_paths.CONFIG_PATH,
            "open_ocean",
            args.split,
        )
        result_path = os.path.join(
            config_paths.RESULT_PATH,
            "open_ocean",
            args.split,
            model_name,
        )
        length = 72
        os.makedirs(result_path, exist_ok=True)
    elif args.dataset == "coastline":
        config_path = os.path.join(config_paths.CONFIG_PATH, "coastline", args.split)
        result_path = os.path.join(
            config_paths.RESULT_PATH,
            "coastline",
            args.split,
            model_name,
        )
        length = 24
        os.makedirs(result_path, exist_ok=True)
    else:
        raise ValueError("Dataset not found")

    files = os.listdir(config_path)
    dataset_files = [f for f in files if f.endswith(".yml")]
    val_datasets = []
    for dataset_name in dataset_files:
        config = config_funcs.load_config(Path(dataset_name).stem, config_path)
        if config["start_date"].year >= 2020 and os.path.exists(config["dataset_path"]):
            val_datasets.append(dataset_name)

    model = AutoregressiveModel.import_trained_model(model_name=model_name)

    for val_dataname in tqdm(val_datasets):
        data_name = Path(val_dataname).stem
        result_file = os.path.join(result_path, f"{data_name}_result.npz")
        if os.path.exists(result_file):
            continue

        try:
            config = config_funcs.load_config(data_name, config_path)
            test_dataset = OceanDataset(config, length, 16, 10000, "linear")
        except Exception as e:
            print(f"Error loading dataset {val_dataname}: {e}")
            continue

        if len(test_dataset) == 0:
            print(f"Empty dataset: {val_dataname}")
            continue

        if args.evaluate_subset and len(test_dataset) > 200:
            # select 200 regularly spaced indices from the dataset
            indices = np.linspace(0, len(test_dataset) - 1, 200, dtype=int)
        else:
            indices = range(len(test_dataset))

        distances, pred_coords, true_coords = [], [], []
        for idx in indices:
            try:
                start_position, start_time, times, interpolator, coordinates = test_dataset[idx]
                pred_coord = model.predict(
                    start_position,
                    start_time,
                    length,
                    interpolator,
                    test_dataset,
                    num_samples=args.num_samples,
                )
                # Shape guards before distance computation
                if coordinates.ndim != 2 or coordinates.shape[1] != 2:
                    print(
                        f"Skipping {data_name} idx={idx}: unexpected coordinates shape "
                        f"{coordinates.shape}, expected (T+1, 2)"
                    )
                    continue
                if pred_coord.ndim != 3 or pred_coord.shape[2] != 2:
                    print(
                        f"Skipping {data_name} idx={idx}: unexpected pred_coord shape "
                        f"{pred_coord.shape}, expected (num_samples, T, 2)"
                    )
                    continue
                if coordinates.shape[0] - 1 != pred_coord.shape[1]:
                    print(
                        f"Skipping {data_name} idx={idx}: horizon mismatch — "
                        f"coordinates length {coordinates.shape[0] - 1} vs "
                        f"pred_coord length {pred_coord.shape[1]}"
                    )
                    continue
                distance = vmap(
                    vmap(data_transform.haversine_distance), in_axes=(None, 0)
                )(coordinates[1:, :], pred_coord[:, :])
                distances.append(distance[None])
                pred_coords.append(pred_coord[None])
                true_coords.append(coordinates[None])
            except Exception as e:
                print(
                    f"Error processing dataset {val_dataname}, idx={idx}: "
                    f"{type(e).__name__}: {e}"
                )
                continue

        if not distances:
            print(f"No valid trajectories for dataset: {val_dataname}")
            continue

        distances = jnp.concatenate(distances, axis=0)
        pred_coords = jnp.concatenate(pred_coords, axis=0)
        true_coords = jnp.concatenate(true_coords, axis=0)
        jnp.savez(
            result_file,
            distances=distances,
            pred_coords=pred_coords,
            true_coords=true_coords,
        )
