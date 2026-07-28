#!/usr/bin/env python3
"""Create CoDiT dataset tar.gz archives for Zenodo upload.

Usage:
    # Create all archives:
    python scripts/make_archives.py --src /path/to/data_final --out /path/to/archives

    # Strip dataset_path from configs in-place (e.g. data/configs):
    python scripts/make_archives.py --strip-inplace data/configs

    # Partial run:
    python scripts/make_archives.py --src ... --out ... --datasets open_ocean --splits train val
"""
import argparse
import io
import os
import subprocess
import tarfile

import yaml

DATASETS = ["open_ocean", "coastline"]
SPLITS = ["train", "val", "test"]


def strip_dataset_path(yml_bytes: bytes) -> bytes:
    """Return YAML bytes with dataset_path field removed."""
    config = yaml.safe_load(yml_bytes)
    config.pop("dataset_path", None)
    return yaml.dump(config, default_flow_style=False, allow_unicode=True).encode()


def strip_inplace(config_dir: str) -> int:
    """Strip dataset_path from every .yml file under config_dir. Returns count."""
    count = 0
    for root, _, files in os.walk(config_dir):
        for fname in files:
            if not fname.endswith(".yml"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "rb") as f:
                raw = f.read()
            with open(fpath, "wb") as f:
                f.write(strip_dataset_path(raw))
            count += 1
    return count


def make_data_archive(dataset: str, split: str, src: str, out_dir: str) -> None:
    """Create {dataset}_{split}.tar.gz.

    Archive paths are relative to the split directory so that
    tar.extractall(dest) produces dest/{patch_name}/... — matching
    what download_data.py expects.
    """
    src_dir = os.path.join(src, "datasets", dataset, split)
    out = os.path.join(out_dir, f"{dataset}_{split}.tar.gz")
    print(f"  Creating {out} ...")
    subprocess.run(["tar", "-czf", out, "-C", src_dir, "."], check=True)
    size_gb = os.path.getsize(out) / 1024**3
    print(f"  -> {size_gb:.2f} GB")


def make_config_archive(dataset: str, src: str, out_dir: str) -> None:
    """Create {dataset}_configs.tar.gz with dataset_path stripped.

    Archive paths are {dataset}/{split}/{name}.yml so that
    tar.extractall(configs_dir) produces configs_dir/{dataset}/{split}/{name}.yml.
    """
    src_dir = os.path.join(src, "configs", dataset)
    out = os.path.join(out_dir, f"{dataset}_configs.tar.gz")
    print(f"  Creating {out} ...")
    with tarfile.open(out, "w:gz") as tar:
        for root, _, files in os.walk(src_dir):
            for fname in sorted(files):
                if not fname.endswith(".yml"):
                    continue
                fpath = os.path.join(root, fname)
                arc_name = os.path.join(
                    dataset, os.path.relpath(fpath, src_dir)
                )
                with open(fpath, "rb") as f:
                    raw = f.read()
                cleaned = strip_dataset_path(raw)
                info = tarfile.TarInfo(name=arc_name)
                info.size = len(cleaned)
                tar.addfile(info, io.BytesIO(cleaned))
    size_mb = os.path.getsize(out) / 1024**2
    print(f"  -> {size_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CoDiT dataset archives")
    parser.add_argument(
        "--src",
        help="Source data directory containing datasets/ and configs/ subdirectories",
    )
    parser.add_argument("--out", help="Output directory for archives")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="Which datasets to archive (default: all)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLITS,
        choices=SPLITS,
        help="Which splits to archive (default: all)",
    )
    parser.add_argument(
        "--strip-inplace",
        metavar="CONFIG_DIR",
        help="Strip dataset_path from all YAML files under CONFIG_DIR in-place, then exit",
    )
    args = parser.parse_args()

    if args.strip_inplace:
        n = strip_inplace(args.strip_inplace)
        print(f"Stripped dataset_path from {n} config files in {args.strip_inplace}")
        return

    if not args.src or not args.out:
        parser.error("--src and --out are required unless using --strip-inplace")

    os.makedirs(args.out, exist_ok=True)
    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        for split in args.splits:
            make_data_archive(dataset, split, args.src, args.out)
        make_config_archive(dataset, args.src, args.out)

    print(
        "\nDone. Upload archives to Zenodo and update DATASET_URLS in "
        "scripts/download_data.py."
    )


if __name__ == "__main__":
    main()
