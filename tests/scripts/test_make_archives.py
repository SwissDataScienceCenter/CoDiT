"""Tests for scripts/make_archives.py."""
import sys
import tarfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from make_archives import (  # noqa: E402
    make_config_archive,
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
