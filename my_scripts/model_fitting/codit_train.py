# %%
# %load_ext autoreload
# %autoreload 2
import argparse
import os

import jax
import numpy as np

from codit.data_module import config_paths
from codit.data_module.dataset import OceanTrainingDataset
from codit.model_module.diffusion_model import CoDiTModel
import optax

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="open_ocean")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--train_size", type=float, default=0.95)
    parser.add_argument("--model_type", type=str, default="GRU")  # Transformer
    parser.add_argument("--force_retrain", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--context_encoder_type", type=str, default="conv"
    )  # conv, mlp, gap
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam")  # adam, adamw, sgd
    parser.add_argument("--scheduler", type=str, default="constant")  # constant, cosine
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    print("Loading dataset")
    if args.dataset == "open_ocean":
        length = 72
    elif args.dataset == "coastline":
        length = 24
    else:
        raise ValueError("Dataset not found")
    dataset = OceanTrainingDataset.from_disk(
        os.path.join(
            config_paths.DATA_PATH, "datasets", args.dataset, "datasets", "processed"
        ),
        length,
    )

    print("Preprocessing dataset")
    dataset.remove_nans()
    dataset.shuffle()
    train_dataset, val_dataset = dataset.split(args.train_size)
    print("Loading dataset into memory")
    train_dataset.load_into_memory()
    val_dataset.load_into_memory()

    print("Training model")
    input_vars = args.input_vars
    input_var_str = "_".join(input_vars)
    model_name = f"codit_{args.model_type}_{args.dataset}_{args.context_encoder_type}_{args.optimizer}_K{args.K}_{input_var_str}"
    if args.run_name:
        model_name += f"_{args.run_name}"
    print(f"Model name: {model_name}")
    model = CoDiTModel(
        name=model_name,
        params={
            "prediction_length": length,
            "dt": 1.0 / length,
            "model_type": args.model_type,
            "context_encoder_type": args.context_encoder_type,
            "K": args.K,
            "vars": args.input_vars,
        },
    )
    if args.scheduler == "cosine":
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=args.learning_rate * 1e-1,
            peak_value=args.learning_rate,
            warmup_steps=args.num_batches_per_epoch,
            decay_steps=args.num_batches_per_epoch * args.num_epochs // 2,
        )
    else:
        scheduler = None

    try:
        model.import_trained_model(model.name)
    except FileNotFoundError:
        print(
            f"Could not find {model.name} in {config_paths.MODEL_PATH}. Re-training model."
        )
        model.fit(
            train_dataset,
            val_dataset,
            train_params={
                "num_batches_per_epoch": args.num_batches_per_epoch,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "force_retrain": args.force_retrain,
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer,
                "scheduler": scheduler,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
            },
        )
        model.export_trained_model(model.name)


# %%
# class Args:
#     dataset = 'coastline'
#     num_epochs = 10
#     num_batches_per_epoch = 10
#     batch_size = 32
#     train_size = 0.95

# args = Args()

# # %%

# if args.dataset == 'open_ocean':
#     length = 72
#     dataset = AdoptTrainingDataset.from_disk('/mydata/adopt/christian/adopt/data/datasets/open_ocean/datasets', length)
# elif args.dataset == 'coastline':
#     length = 24
#     dataset = AdoptTrainingDataset.from_disk('/mydata/adopt/christian/adopt/data/datasets/coastline_data', length)
# else:
#     raise ValueError('Dataset not found')
# dataset.remove_nans()
# dataset.shuffle()
# train_dataset, val_dataset = dataset.split(0.9)
# #train_dataset.load_into_memory()
# #val_dataset.load_into_memory()
# # %%
# model = DiffusionModel(name=f'diffusion_model_{args.dataset}', params={
#     'prediction_length': length,
# })
# # %%
# model.fit(train_dataset, val_dataset, train_params={
#     'num_batches_per_epoch': args.num_batches_per_epoch,
#     'num_epochs': args.num_epochs,
#     'batch_size': args.batch_size
# })

# # %%


# # %%
