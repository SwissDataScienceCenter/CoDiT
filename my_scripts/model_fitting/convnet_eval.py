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
from codit.model_module.conv_net import ConvNetModel

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="open_ocean")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--evaluate_subset", action="store_true", default=False)
    args = parser.parse_args()
    np.random.seed(0)
    
    if args.dataset == "open_ocean":
        config_path = os.path.join(
            config_paths.CONFIG_PATH,
            "open_ocean",
            args.split
        )
        result_path = os.path.join(config_paths.RESULT_PATH, "open_ocean", args.split, "conv_net")
        length = 72
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    elif args.dataset == "coastline":
        config_path = os.path.join(config_paths.CONFIG_PATH, "coastline", args.split)
        result_path = os.path.join(config_paths.RESULT_PATH, "coastline", args.split, "conv_net")
        length = 24
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    else:
        raise ValueError("Dataset not found")
    files = os.listdir(config_path)
    dataset_files = [f for f in files if f.endswith(".yml")]
    val_datasets = []
    for dataset_name in dataset_files:
        config = config_funcs.load_config(dataset_name.split(".")[0], config_path)
        if config["start_date"].year >= 2020 and os.path.exists(config["dataset_path"]):
            val_datasets.append(dataset_name)

    model = ConvNetModel(name=f"conv_net_{args.dataset}", params={})
    model.load_trained_model()
    #export_path = os.path.join(config_paths.CONV_NET_PATH, 'trained_models/')
    #model = ConvNetModel.import_trained_model(f'cnn_{args.dataset}', import_path=export_path)

    for val_dataname in tqdm(val_datasets):
        data_name = val_dataname.split(".")[0]
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
            # indices = np.random.choice(len(test_dataset), 200, replace=False)
            indices = np.linspace(0, len(test_dataset) - 1, 200, dtype=int)
        else:
            indices = range(len(test_dataset))
        for i in indices:
            try:
                start_position, start_time, times, interpolator, coordinates = test_dataset[i]
                i += 1
                print(f"Processing {data_name} {i}/{len(test_dataset)}")
                pred_coord = model.predict(
                    start_position, 
                    start_time, 
                    length, 
                    interpolator, 
                    test_dataset, 
                    num_samples=100
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
