# %%
import argparse
import os

from codit.data_module import config_paths
from codit.data_module.dataset import OceanTrainingDataset
from codit.model_module.linear_gauss_model import LinearGaussModel

# %%

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="open_ocean")
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_size", type=float, default=0.95)
    args = parser.parse_args()

    if args.dataset == "open_ocean":
        dataset = OceanTrainingDataset.from_disk(
            os.path.join(
                config_paths.DATA_PATH,
                "datasets",
                "open_ocean",
                "datasets",
                "processed",
            ),
            72,
        )
    elif args.dataset == "coastline":
        dataset = OceanTrainingDataset.from_disk(
            os.path.join(
                config_paths.DATA_PATH, "datasets", "coastline", "datasets", "processed"
            ),
            24,
        )
    else:
        raise ValueError("Dataset not found")
    dataset.remove_nans()
    dataset.shuffle()
    train_dataset, val_dataset = dataset.split(0.9)
    train_dataset.load_into_memory()
    val_dataset.load_into_memory()

    model = LinearGaussModel(name=f"linear_gauss_{args.dataset}", params={})
    trained_model = model.fit(
        train_dataset,
        val_dataset,
        train_params={
            "num_batches_per_epoch": args.num_batches_per_epoch,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
        },
    )
    model.export_trained_model()

# %%
