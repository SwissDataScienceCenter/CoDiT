import argparse
import os

import optax

from codit.data_module import config_paths
from codit.data_module.dataset import OceanTrainingDataset
from codit.model_module.gru_model import AutoregressiveModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="open_ocean", choices=["open_ocean", "coastline"]
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--train_size", type=float, default=0.95)
    parser.add_argument("--model_type", type=str, default="GRU", choices=["GRU", "Transformer"])
    parser.add_argument("--force_retrain", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--context_encoder_type", type=str, default="conv", choices=["conv", "mlp", "gap"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument(
        "--input_vars",
        nargs="+",
        type=str,
        default=["hycom_current", "copernicus_current", "era5_wind", "bathymetry"],
        help="Input variables for the model",
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

    # Canonicalise ordering so model_name matches regardless of input_vars argument order
    input_vars = sorted(args.input_vars)
    input_var_str = "_".join(input_vars)
    model_name = (
        f"gru_{args.model_type}_{args.dataset}_{args.context_encoder_type}_"
        f"{args.optimizer}_{input_var_str}"
    )
    print(f"Model name: {model_name}")

    if args.scheduler == "cosine":
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=args.learning_rate * 1e-1,
            peak_value=args.learning_rate,
            warmup_steps=args.num_batches_per_epoch,
            decay_steps=args.num_batches_per_epoch * args.num_epochs // 2,
        )
    elif args.scheduler == "constant":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    should_train = args.force_retrain
    if not args.force_retrain:
        try:
            AutoregressiveModel.import_trained_model(model_name)
            print(
                f"Found existing trained model {model_name} in "
                f"{os.path.join(config_paths.MODEL_PATH, 'trained_models')}"
            )
            should_train = False
        except FileNotFoundError:
            print(
                f"Could not find {model_name} in {config_paths.MODEL_PATH}. "
                "Re-training model."
            )
            should_train = True
    else:
        print("--force_retrain enabled. Re-training model.")

    if should_train:
        print(
            f"Starting training for {model_name}."
        )
        model = AutoregressiveModel(
            name=model_name,
            params={
                "prediction_length": length,
                "dt": 1.0 / length,
                "model_type": args.model_type,
                "context_encoder_type": args.context_encoder_type,
                "vars": input_vars,
            },
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
            },
        )
        model.export_trained_model(model_name)