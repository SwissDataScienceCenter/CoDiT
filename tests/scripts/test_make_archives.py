"""Tests for scripts/dev/data/make_archives.py."""
import sys
import tarfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev" / "data"))
from make_archives import (  # noqa: E402
    make_config_archive,
    make_data_archive,
    strip_dataset_path,
    strip_inplace,
)


def test_strip_removes_dataset_path():
    raw = b"dataset_path: /stale/path\nname: foo\ntype: region_based\n"
    out = yaml.safe_load(strip_dataset_path(raw))
    assert "dataset_path" not in out
    assert out["name"] == "foo"


def test_strip_noop_when_absent():
    raw = b"name: foo\ntype: region_based\n"
    out = yaml.safe_load(strip_dataset_path(raw))
    assert "dataset_path" not in out
    assert out["name"] == "foo"


def test_strip_inplace(tmp_path):
    cfg = tmp_path / "train" / "patch_0.yml"
    cfg.parent.mkdir()
    cfg.write_text(yaml.dump({"name": "patch_0", "dataset_path": "/stale"}))
    count = strip_inplace(str(tmp_path))
    assert count == 1
    assert "dataset_path" not in yaml.safe_load(cfg.read_text())


def test_make_config_archive_strips_path(tmp_path):
    src = tmp_path / "src"
    split_dir = src / "configs" / "test_ds" / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "patch_0.yml").write_text(
        yaml.dump(
            {"name": "patch_0", "dataset_path": "/stale", "type": "region_based"}
        )
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    make_config_archive("test_ds", str(src), str(out_dir))

    with tarfile.open(out_dir / "test_ds_configs.tar.gz") as tar:
        member = tar.getmember("test_ds/train/patch_0.yml")
        loaded = yaml.safe_load(tar.extractfile(member).read())
    assert "dataset_path" not in loaded
    assert loaded["name"] == "patch_0"


def test_make_data_archive_excludes_intermediates(tmp_path):
    """Data archive must not contain hycom/tmp, .grib, .idx, or days_not_loaded.json."""
    src = tmp_path / "src"
    patch = src / "datasets" / "ds" / "train" / "patch_0"
    (patch / "hycom" / "tmp").mkdir(parents=True)
    (patch / "hycom" / "tmp" / "granule.nc4").write_bytes(b"scratch")
    (patch / "hycom" / "days_not_loaded.json").write_text("[]")
    (patch / "hycom" / "hycom_time_final_d0.nc4").write_bytes(b"real")
    (patch / "era5").mkdir()
    (patch / "era5" / "era5_time.grib").write_bytes(b"src")
    (patch / "era5" / "era5_time.grib.abc.idx").write_bytes(b"idx")
    (patch / "era5" / "era5_time.nc").write_bytes(b"nc")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    make_data_archive("ds", "train", str(src), str(out_dir))

    with tarfile.open(out_dir / "ds_train.tar.gz") as tar:
        names = tar.getnames()
    assert not any("hycom/tmp" in n for n in names), names
    assert not any(n.endswith(".grib") for n in names), names
    assert not any(n.endswith(".idx") for n in names), names
    assert not any(n.endswith("days_not_loaded.json") for n in names), names
    # Verify the real files are still present.
    assert any(n.endswith("hycom_time_final_d0.nc4") for n in names), names
    assert any(n.endswith("era5_time.nc") for n in names), names


def test_make_config_archive_internal_paths(tmp_path):
    """Archive paths must be dataset/{split}/{name}.yml for correct extraction."""
    src = tmp_path / "src"
    for split in ("train", "val", "test"):
        d = src / "configs" / "ds" / split
        d.mkdir(parents=True)
        (d / f"patch_{split}.yml").write_text(yaml.dump({"name": f"patch_{split}"}))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    make_config_archive("ds", str(src), str(out_dir))

    with tarfile.open(out_dir / "ds_configs.tar.gz") as tar:
        names = tar.getnames()
    assert "ds/train/patch_train.yml" in names
    assert "ds/val/patch_val.yml" in names
    assert "ds/test/patch_test.yml" in names
