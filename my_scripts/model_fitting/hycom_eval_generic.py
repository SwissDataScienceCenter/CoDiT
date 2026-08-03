import argparse
import os

# mute SerializationWarning
import warnings

from jax import numpy as jnp
from jax import vmap
from tqdm import tqdm
import numpy as np

from codit.data_module import config_paths
from codit.data_module import config_funcs
from codit.data_module.dataset import OceanDataset
from codit.data_module import data_transform

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="open_ocean")
    parser.add_argument("--current_type", type=str, default="hycom")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--prediction_length", type=int, default=72)
    parser.add_argument("--evaluate_subset", action="store_true", default=False)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    np.random.seed(0)
    config_path = os.path.join(
        config_paths.CONFIG_PATH,
        args.dataset,
        args.split
    )
    length = args.prediction_length
    if length == 72:
        model_train_dataset = "open_ocean"
    elif length == 24:
        model_train_dataset = "coastline"
    else:
        raise ValueError("Unsupported prediction length")
    files = os.listdir(config_path)
    dataset_files = [f for f in files if f.endswith(".yml")]
    if len(dataset_files) == 0:
        raise ValueError("No dataset files found in the specified config path.")
    result_path = os.path.join(config_paths.RESULT_PATH, args.dataset, args.split, args.current_type)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    val_datasets = []
    for dataset_name in dataset_files:
        config = config_funcs.load_config(dataset_name.split(".")[0], config_path)
        #if config["start_date"].year >= 2020 and os.path.exists(config["dataset_path"]):
        val_datasets.append(dataset_name)


    if args.current_type == "hycom":
        from codit.model_module.hycom_model import HycomModel
        model = HycomModel(name=f"hycom_{model_train_dataset}", params={})
    elif args.current_type == "glorys":
        from codit.model_module.glorys_model import GlorysModel
        model = GlorysModel(name=f"glorys_{model_train_dataset}", params={})
    # model.load_trained_model()

    for val_dataname in tqdm(val_datasets):
        data_name = val_dataname.split(".")[0]
        result_file = os.path.join(result_path, f"{data_name}_result.npz")
        if not os.path.exists(result_file):
            try:
                config = config_funcs.load_config(data_name, config_path)
                test_dataset = OceanDataset(config, length, 16, 10000, "linear")
            except Exception as e:
                print(f"Error loading dataset {val_dataname}: {e}")
                continue

            distances, pred_coords, true_coords = [], [], []
            if len(test_dataset) == 0:
                print(f"Empty dataset: {val_dataname}")
                continue
            i = 0
            if args.evaluate_subset and len(test_dataset) > 200:
                # randomly select 200 indices from the dataset
                #indices = np.random.choice(len(test_dataset), 200, replace=False)
                indices = np.linspace(0, len(test_dataset) - 1, 200, dtype=int)
            else:
                indices = range(len(test_dataset))

            for i in indices:
                try:
                    start_position, start_time, times, interpolator, coordinates = test_dataset[i]
                    print(f"Processing {data_name} {i}/{len(test_dataset)}")
                    pred_coord = model.predict(
                        start_position, 
                        start_time, 
                        length, 
                        interpolator, 
                        test_dataset
                    )
                    distance = vmap(
                        vmap(data_transform.haversine_distance), in_axes=(None, 0)
                    )(coordinates[1:, :], pred_coord[:, :])
                    distances.append(distance[None])
                    pred_coords.append(pred_coord[None])
                    true_coords.append(coordinates[:, :][None])
                except ValueError as e:
                    print(f"Error loading dataset {val_dataname}: {e}")
                    continue
            distances = jnp.concatenate(distances, axis=0)
            pred_coords = jnp.concatenate(pred_coords, axis=0)
            true_coords = jnp.concatenate(true_coords, axis=0)
            jnp.savez(
                os.path.join(result_path, f"{data_name}_result.npz"),
                distances=distances,
                pred_coords=pred_coords,
                true_coords=true_coords,
            )
