import argparse
import os
import logging

from codit.data_module import config_paths
from codit.data_module.dataset import OceanTrainingDataset
from codit.model_module.direct_prediction_model import DirectPredictionModel

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="open_ocean", choices=["open_ocean", "coastline"]
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--train_size", type=float, default=0.95)
    parser.add_argument(
        "--model_type", type=str, default="GRU", choices=["GRU", "Transformer"]
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument(
        "--context_encoder_type",
        type=str,
        default="conv",
        choices=["conv", "mlp", "gap"],
    )
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument(
        "--input_vars",
        nargs="+",
        type=str,
        default=["hycom_current", "copernicus_current", "era5_wind", "bathymetry"],
    )
    args = parser.parse_args()

    if not (0 < args.train_size < 1):
        parser.error("--train_size must satisfy 0 < train_size < 1")
    if args.num_epochs <= 0:
        parser.error("--num_epochs must be > 0")
    if args.num_batches_per_epoch <= 0:
        parser.error("--num_batches_per_epoch must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch_size must be > 0")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be > 0")
    if args.patience <= 0:
        parser.error("--patience must be > 0")
    if args.min_delta < 0:
        parser.error("--min_delta must be >= 0")

    if args.dataset == "open_ocean":
        length = 72
    elif args.dataset == "coastline":
        length = 24
    else:
        raise ValueError("Dataset not found")

    dataset = OceanTrainingDataset.from_disk(
        os.path.join(
            config_paths.DATA_PATH,
            "datasets",
            args.dataset,
            "datasets",
            "processed",
        ),
        length,
    )
    dataset.remove_nans()
    dataset.shuffle()
    train_dataset, val_dataset = dataset.split(args.train_size)
    train_dataset.load_into_memory()
    val_dataset.load_into_memory()

    input_vars = sorted(args.input_vars)
    input_var_str = "_".join(input_vars)
    model_name = (
        f"direct_{args.model_type}_{args.dataset}_"
        f"{args.context_encoder_type}_{args.optimizer}_{input_var_str}"
    )
    model = DirectPredictionModel(
        name=model_name,
        params={
            "prediction_length": length,
            "model_type": args.model_type,
            "context_encoder_type": args.context_encoder_type,
            "vars": input_vars,
        },
    )
    model.fit(
        train_dataset,
        val_dataset,
        train_params={
            "num_epochs": args.num_epochs,
            "num_batches_per_epoch": args.num_batches_per_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
    )
    model.export_trained_model()