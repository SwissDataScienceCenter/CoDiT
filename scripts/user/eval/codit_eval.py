import argparse
import gc
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
from codit.model_module.diffusion_model import CoDiTModel

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="open_ocean")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--model_type", type=str, default="GRU")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--context_encoder_type", type=str, default="conv"
    )  # conv, mlp, gap
    parser.add_argument("--optimizer", type=str, default="adam")  # adam, adamw, sgd
    parser.add_argument("--scheduler", type=str, default="constant")  # constant, cosine
    parser.add_argument("--evaluate_subset", action="store_true", default=False)
    parser.add_argument("--K", type=int, default=100)  # number of diffusion steps
    parser.add_argument(
        "--input_vars",
        nargs="+",
        type=str,
        default=["hycom_current", "copernicus_current", "era5_wind", "bathymetry"],
        help="Input variables for the model",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run name for logging and saving results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Override model name (must match directory in data/trained_models/)",
    )
    parser.add_argument(
        "--result_subdir",
        type=str,
        default="",
        help="Optional subdirectory inserted between --split and the model name in the result path. Default '' preserves the original path.",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=72,
        help="Trajectory horizon in hours. Use 72 for open_ocean, 24 for coastline.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddpm",
        choices=["ddpm", "heun"],
        help="Reverse-diffusion sampler: 'ddpm' (ancestral, K steps; paper) or 'heun' "
        "(deterministic EDM Heun sampler, see docs/EDM_IMPROVEMENTS.md).",
    )
    parser.add_argument(
        "--num_sampling_steps",
        type=int,
        default=18,
        help="Noise levels visited by the Heun sampler (2N-1 network evaluations).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="default",
        choices=["default", "raw"],
        help="'default': the exported model (EMA weights if trained with EMA); "
        "'raw': the best non-EMA weights of an EMA-trained model, for comparison.",
    )
    parser.add_argument(
        "--max_per_patch",
        type=int,
        default=0,
        help="Evaluate at most this many trajectories per patch, evenly spaced (the "
        "same ones for every model, so results can be compared pairwise). 0 = all. "
        "Use with --result_subdir to keep partial results apart from full ones.",
    )
    args = parser.parse_args()
    np.random.seed(0)

    if args.model_name:
        model_name = args.model_name
    else:
        input_vars = args.input_vars
        input_var_str = "_".join(input_vars)
        model_name = f"codit_{args.model_type}_{args.dataset}_{args.context_encoder_type}_{args.optimizer}_K{args.K}_{input_var_str}"
        if args.run_name:
            model_name += f"_{args.run_name}"
    print(f"Model name: {model_name}")
    # results of non-default samplers go to their own directory, so cached DDPM
    # results are neither reused nor overwritten
    result_name = model_name
    if args.sampler != "ddpm":
        result_name += f"_{args.sampler}{args.num_sampling_steps}"
    if args.weights != "default":
        result_name += f"_{args.weights}"
    if args.dataset == "open_ocean":
        config_path = os.path.join(config_paths.CONFIG_PATH, "open_ocean", args.split)
        result_path = os.path.join(
            config_paths.RESULT_PATH,
            "open_ocean",
            args.split,
            args.result_subdir,
            result_name,
        )
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    elif args.dataset == "coastline":
        config_path = os.path.join(config_paths.CONFIG_PATH, "coastline", args.split)
        result_path = os.path.join(
            config_paths.RESULT_PATH,
            "coastline",
            args.split,
            args.result_subdir,
            result_name,
        )
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    else:
        raise ValueError("Dataset not found")
    length = args.prediction_length
    files = os.listdir(config_path)
    dataset_files = [f for f in files if f.endswith(".yml")]
    val_datasets = []
    for dataset_name in dataset_files:
        config = config_funcs.load_config(dataset_name.split(".")[0], config_path)
        if config["start_date"].year >= 2020 and os.path.exists(config["dataset_path"]):
            val_datasets.append(dataset_name)

    if args.evaluate_subset and len(val_datasets) > 20:
        patch_indices = np.linspace(0, len(val_datasets) - 1, 20, dtype=int)
        val_datasets = [val_datasets[i] for i in patch_indices]

    # model_name = f"diffusion_model_{args.dataset}_{args.model_type}_{args.context_encoder_type}_{args.optimizer}_{args.scheduler}"
    model = CoDiTModel.import_trained_model(model_name=model_name, weights=args.weights)
    model.set_sampler(args.sampler, args.num_sampling_steps)
    num_predicted_traj = 0
    test_dataset = None
    for val_dataname in tqdm(val_datasets):
        # release the previous patch (tens of GB) before loading the next one
        test_dataset = None
        gc.collect()
        data_name = val_dataname.split(".")[0]
        result_file = os.path.join(result_path, f"{data_name}_result.npz")
        if not os.path.exists(result_file):
            try:
                config = config_funcs.load_config(data_name, config_path)
                test_dataset = OceanDataset(config, length, 16, 10000, "linear")
            except Exception as e:
                tqdm.write(f"Error loading dataset {val_dataname}: {e}")
                continue

            if len(test_dataset) == 0:
                tqdm.write(f"Empty dataset: {val_dataname}")
                continue
            if args.max_per_patch and len(test_dataset) > args.max_per_patch:
                indices = np.linspace(
                    0, len(test_dataset) - 1, args.max_per_patch, dtype=int
                )
            elif args.evaluate_subset and len(test_dataset) > 200:
                # select 200 regularly spaced indices from the dataset
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
                    progress=data_name,
                )
            except ValueError as e:
                tqdm.write(f"Error evaluating dataset {val_dataname}: {e}")
                continue
            jnp.savez(
                result_file,
                distances=distances,
                pred_coords=pred_coords,
                true_coords=true_coords,
            )
        # if args.evaluate_subset and num_predicted_traj >= 200:
        #     break