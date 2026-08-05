"""CLI wrapper around preprocess_configs.

Resolves configs from CONFIG_PATH/<dataset>/<split>/*.yml and writes the merged
h5 to DATA_PATH/datasets/<dataset>/<split>/processed/.

Usage:
    python scripts/dev/data/precompute_trainset.py \
        --dataset open_ocean --split train \
        --length 72 --num_grid_points 16 --delta_metres 10000
"""
import argparse
import logging
import os
import warnings

from codit.data_module import config_funcs, config_paths
from codit.data_module.preprocessing import preprocess_configs

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, required=True)
    p.add_argument("--length", type=int, default=72)
    p.add_argument("--num_grid_points", type=int, default=16)
    p.add_argument("--delta_metres", type=int, default=10000)
    p.add_argument("--interp_method", type=str, default="linear")
    args = p.parse_args()

    config_dir = os.path.join(config_paths.CONFIG_PATH, args.dataset, args.split)
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"No config directory at {config_dir}")

    yml_files = sorted(f for f in os.listdir(config_dir) if f.endswith(".yml"))
    if not yml_files:
        raise FileNotFoundError(f"No .yml files in {config_dir}")

    configs = [config_funcs.load_config(f[:-4], config_dir) for f in yml_files]

    output_path = os.path.join(
        config_paths.DATA_PATH, "datasets", args.dataset, args.split, "processed"
    )
    result = preprocess_configs(
        configs=configs,
        length=args.length,
        num_grid_points=args.num_grid_points,
        delta_metres=args.delta_metres,
        output_path=output_path,
        interp_method=args.interp_method,
    )
    print(f"Wrote merged training h5: {result}")


if __name__ == "__main__":
    main()
