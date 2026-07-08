#!/usr/bin/env python3
"""Download CoDiT datasets and bathymetry.

Usage:
    python scripts/download_data.py --dataset open_ocean
    python scripts/download_data.py --dataset coastline
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --bathymetry
"""

import argparse
import os
import subprocess
import sys
import tarfile
import urllib.request

# --------------------------------------------------------------------------
# TODO: Replace placeholder URLs with actual Zenodo URLs once data is uploaded.
# Each tarball, when extracted, must produce the directory structure expected by
# config_funcs.load_config:
#   datasets/{dataset}/{split}/{patch_name}/datasets/{patch_name}/
#   configs/{dataset}/{split}/{patch_name}.yml
# --------------------------------------------------------------------------
DATASET_URLS: dict[str, dict[str, str]] = {
    "open_ocean": {
        "train": "https://zenodo.org/record/XXXXXXX/files/open_ocean_train.tar.gz",
        "val": "https://zenodo.org/record/XXXXXXX/files/open_ocean_val.tar.gz",
        "test": "https://zenodo.org/record/XXXXXXX/files/open_ocean_test.tar.gz",
        "configs": "https://zenodo.org/record/XXXXXXX/files/open_ocean_configs.tar.gz",
    },
    "coastline": {
        "train": "https://zenodo.org/record/XXXXXXX/files/coastline_train.tar.gz",
        "val": "https://zenodo.org/record/XXXXXXX/files/coastline_val.tar.gz",
        "test": "https://zenodo.org/record/XXXXXXX/files/coastline_test.tar.gz",
        "configs": "https://zenodo.org/record/XXXXXXX/files/coastline_configs.tar.gz",
    },
}

BATHYMETRY_URL = "https://polybox.ethz.ch/index.php/s/Gp6zM2JindN6nHA/download"


def download_file(url: str, dest: str) -> None:
    """Download a file with progress reporting."""
    print(f"  Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def extract_tarball(path: str, dest: str) -> None:
    """Extract a .tar.gz archive and remove the tarball."""
    print(f"  Extracting {os.path.basename(path)} -> {dest}")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path=dest)
    os.remove(path)


def download_dataset(dataset: str, data_path: str) -> None:
    """Download and extract all splits + configs for a dataset."""
    print(f"\n=== Downloading {dataset} ===")
    urls = DATASET_URLS[dataset]
    for split, url in urls.items():
        if split == "configs":
            dest_dir = os.path.join(data_path, "configs")
        else:
            dest_dir = os.path.join(data_path, "datasets", dataset, split)
        os.makedirs(dest_dir, exist_ok=True)
        tarball = os.path.join(data_path, f"{dataset}_{split}.tar.gz")
        download_file(url, tarball)
        extract_tarball(tarball, dest_dir)
    print(f"=== {dataset} complete ===")


def download_bathymetry(data_path: str) -> None:
    """Download and extract GEBCO bathymetry data."""
    print("\n=== Downloading bathymetry ===")
    dest_dir = os.path.join(data_path, "bathymetry")
    os.makedirs(dest_dir, exist_ok=True)
    zippath = os.path.join(data_path, "bathymetry.zip")
    download_file(BATHYMETRY_URL, zippath)
    subprocess.run(["unzip", "-o", zippath, "-d", dest_dir], check=True)
    os.remove(zippath)
    print("=== bathymetry complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CoDiT datasets")
    parser.add_argument(
        "--dataset",
        choices=["open_ocean", "coastline", "all"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--bathymetry", action="store_true", help="Download GEBCO bathymetry"
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Target directory (default: CODIT_DATA_PATH env var, or <repo>/data)",
    )
    args = parser.parse_args()

    if not args.dataset and not args.bathymetry:
        parser.print_help()
        sys.exit(1)

    if args.data_path:
        data_path = args.data_path
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.environ.get(
            "CODIT_DATA_PATH", os.path.join(repo_root, "data")
        )

    print(f"Data directory: {data_path}")

    if args.dataset:
        datasets = (
            ["open_ocean", "coastline"] if args.dataset == "all" else [args.dataset]
        )
        for ds in datasets:
            download_dataset(ds, data_path)

    if args.bathymetry:
        download_bathymetry(data_path)

    print("\nDone. You can now run evaluation scripts.")


if __name__ == "__main__":
    main()
