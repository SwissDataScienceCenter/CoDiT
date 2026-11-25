#!/usr/bin/env python3
"""Download CoDiT datasets and bathymetry.

Usage:
    python scripts/user/setup/download_data.py --dataset open_ocean
    python scripts/user/setup/download_data.py --dataset coastline
    python scripts/user/setup/download_data.py --dataset all
    python scripts/user/setup/download_data.py --bathymetry
"""

import argparse
import os
import sys
import tarfile
import urllib.request
import zipfile

# Each tarball, when extracted, produces the directory structure expected by
# config_funcs.load_config:
#   datasets/{dataset}/{split}/{patch_name}/copernicus/
#   datasets/{dataset}/{split}/{patch_name}/era5/
#   datasets/{dataset}/{split}/{patch_name}/hycom/
#   datasets/{dataset}/{split}/{patch_name}/trajectories/
#   configs/{dataset}/{split}/{patch_name}.yml
#
# Zenodo record: https://doi.org/10.5281/zenodo.21644091
ZENODO_RECORD = "21644091"
_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

DATASET_URLS: dict[str, dict[str, str]] = {
    "open_ocean": {
        "train": f"{_BASE}/open_ocean_train.tar.gz",
        "val": f"{_BASE}/open_ocean_val.tar.gz",
        "test": f"{_BASE}/open_ocean_test.tar.gz",
        "configs": f"{_BASE}/open_ocean_configs.tar.gz",
    },
    "coastline": {
        "train": f"{_BASE}/coastline_train.tar.gz",
        "val": f"{_BASE}/coastline_val.tar.gz",
        "test": f"{_BASE}/coastline_test.tar.gz",
        "configs": f"{_BASE}/coastline_configs.tar.gz",
    },
}

BATHYMETRY_URL = "https://polybox.ethz.ch/index.php/s/Gp6zM2JindN6nHA/download"
# Must match the folder name used in codit.data_module.context_functions.load_data.
BATHYMETRY_SUBDIR = "GEBCO_06_Jan_2025_ba69ac5d6a3b"


def download_file(url: str, dest: str) -> None:
    """Download a file with progress reporting."""
    print(f"  Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def extract_tarball(path: str, dest: str) -> None:
    """Extract a .tar.gz archive and remove the tarball."""
    print(f"  Extracting {os.path.basename(path)} -> {dest}")
    dest_abs = os.path.abspath(dest)
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(dest_abs, member.name))
            if not member_path.startswith(dest_abs + os.sep):
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(path=dest_abs)
    os.remove(path)


def download_dataset(dataset: str, data_path: str, dry_run: bool = False) -> None:
    """Download and extract all splits + configs for a dataset."""
    header = "Would download" if dry_run else "Downloading"
    print(f"\n=== {header} {dataset} ===")
    urls = DATASET_URLS[dataset]
    for split, url in urls.items():
        if split == "configs":
            dest_dir = os.path.join(data_path, "configs")
        else:
            dest_dir = os.path.join(data_path, "datasets", dataset, split)
        tarball = os.path.join(data_path, f"{dataset}_{split}.tar.gz")
        if dry_run:
            print(f"  [{split}] {url}")
            print(f"    tarball -> {tarball}")
            print(f"    extract -> {dest_dir}")
            continue
        os.makedirs(dest_dir, exist_ok=True)
        download_file(url, tarball)
        extract_tarball(tarball, dest_dir)
    print(f"=== {dataset} complete ===")


def download_bathymetry(data_path: str, dry_run: bool = False) -> None:
    """Download and extract GEBCO bathymetry data."""
    header = "Would download" if dry_run else "Downloading"
    print(f"\n=== {header} bathymetry ===")
    zippath = os.path.join(data_path, "bathymetry.zip")
    if dry_run:
        print(f"  {BATHYMETRY_URL}")
        print(f"    zip     -> {zippath}")
        print(f"    extract -> {data_path}")
        print("=== bathymetry complete ===")
        return
    os.makedirs(data_path, exist_ok=True)
    download_file(BATHYMETRY_URL, zippath)
    # Zip has a top-level `bathymetry/` folder — extract into data_path, then
    # make sure the final layout is data_path/bathymetry/GEBCO_.../*.nc, which
    # is where context_functions.load_data looks for the tiles.
    print(f"  Extracting {os.path.basename(zippath)} -> {data_path}")
    with zipfile.ZipFile(zippath) as zf:
        zf.extractall(path=data_path)
    os.remove(zippath)
    bathy_dir = os.path.join(data_path, "bathymetry")
    gebco_dir = os.path.join(bathy_dir, BATHYMETRY_SUBDIR)
    os.makedirs(gebco_dir, exist_ok=True)
    for name in os.listdir(bathy_dir):
        src = os.path.join(bathy_dir, name)
        if os.path.isfile(src):
            os.replace(src, os.path.join(gebco_dir, name))
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved URLs and destination paths without downloading",
    )
    args = parser.parse_args()

    if not args.dataset and not args.bathymetry:
        parser.print_help()
        sys.exit(1)

    if args.data_path:
        data_path = args.data_path
    else:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        data_path = os.environ.get("CODIT_DATA_PATH", os.path.join(repo_root, "data"))

    print(f"Data directory: {data_path}")

    if args.dry_run:
        print("(dry-run: no files will be downloaded or extracted)")

    if args.dataset:
        datasets = (
            ["open_ocean", "coastline"] if args.dataset == "all" else [args.dataset]
        )
        for ds in datasets:
            download_dataset(ds, data_path, dry_run=args.dry_run)

    if args.bathymetry:
        download_bathymetry(data_path, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete.")
    else:
        print("\nDone. You can now run evaluation scripts.")


if __name__ == "__main__":
    main()
