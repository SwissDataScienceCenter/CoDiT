import argparse
import os

# mute SerializationWarning
import warnings

from jax import numpy as jnp
from tqdm import tqdm
import numpy as np

from codit.data_module import config_paths
from codit.data_module import config_funcs
from codit.data_module.dataset import OceanDataset
from codit.eval_module.eval_runner import run_eval
from codit.model_module.linear_gauss_model import LinearGaussModel

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
        config_path = os.path.join(config_paths.CONFIG_PATH, "open_ocean", args.split)
        result_path = os.path.join(
            config_paths.RESULT_PATH, "open_ocean", args.split, "linear_gauss"
        )
        length = 72
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    elif args.dataset == "coastline":
        config_path = os.path.join(config_paths.CONFIG_PATH, "coastline", args.split)
        result_path = os.path.join(
            config_paths.RESULT_PATH, "coastline", args.split, "linear_gauss"
        )
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

    model = LinearGaussModel.import_trained_model(f"linear_gauss_{args.dataset}")
    # export_path = os.path.join(config_paths.CONV_NET_PATH, 'trained_models/')
    # model = ConvNetModel.import_trained_model(f'cnn_{args.dataset}', import_path=export_path)

    for val_dataname in tqdm(val_datasets):
        data_name = val_dataname.split(".")[0]
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
            # randomly select 200 indices from the dataset
            # indices = np.random.choice(len(test_dataset), 200, replace=False)
            indices = np.linspace(0, len(test_dataset) - 1, 200, dtype=int)
        else:
            indices = range(len(test_dataset))
        try:
            distances, pred_coords, true_coords = run_eval(
                model,
                test_dataset,
                indices=indices,
                length=length,
                num_samples=args.num_samples,
            )
        except ValueError as e:
            print(f"Error evaluating dataset {val_dataname}: {e}")
            continue
        jnp.savez(
            os.path.join(result_path, f"{data_name}_result.npz"),
            distance=distances,
            pred_coords=pred_coords,
            true_coords=true_coords,
        )
