# Dataset Reorganisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganise both datasets to the canonical layout (`{split}/{name}/copernicus/...`) and fix `load_config` to match, so the eval pipeline works without symlinks.

**Architecture:** Fix `load_config` first, smoke-test against a temp directory to confirm the fix is correct, then do the full same-filesystem `mv` reorganisation for open_ocean and coastline. Single commit per task.

**Tech Stack:** bash, Python, micromamba `codit` env, pytest.

## Global Constraints

- `CODIT_DATA_PATH=/mydata/adopt/christian/adopt/data_final` for all shell steps
- Run Python via `micromamba run -n codit python ...`
- Run tests via `micromamba run -n codit pytest ...`
- All commits on `dev` branch
- Do not copy data — use `mv` (same filesystem, atomic rename)
- open_ocean has ~91 patches (58 train, 16 val, 17 test); coastline has more — script must handle all

---

## Task 1: Fix `load_config` and smoke-test

**Files:**
- Modify: `src/codit/data_module/config_funcs.py:88-92`
- No new test files — use inline Python to smoke-test

**Interfaces:**
- Produces: `config_funcs.load_config(name, config_path) -> dict` where
  `config["dataset_path"] == DATA_PATH/datasets/{rel}/{name}`

- [ ] **Step 1: Fix `load_config`**

Edit `src/codit/data_module/config_funcs.py`. Replace lines 83–92:

```python
def load_config(data_name: str, config_path: str) -> dict:
    ...
    path = os.path.join(config_path, data_name + ".yml")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Reconstruct dataset_path relative to current DATA_PATH.
    # YAML configs store stale absolute paths from the original environment.
    # data lives at: DATA_PATH/datasets/{dataset}/{split}/{name}/
    rel = os.path.relpath(config_path, config_paths.CONFIG_PATH)
    config["dataset_path"] = os.path.join(
        config_paths.DATA_PATH, "datasets", rel, config["name"]
    )

    config = complete_config(config)
    return config
```

- [ ] **Step 2: Build smoke-test temp directory for open_ocean**

Create a temp dir that mirrors the new layout using symlinks to the real data (no copying):

```bash
TMPDIR=$(mktemp -d)
# open_ocean: new layout is {split}/{name}/copernicus/...
# We symlink one patch from the OLD location datasets/{name}/ into the new {name}/ slot
mkdir -p "$TMPDIR/datasets/open_ocean/test/open_ocean_patch_1_year_2021"
ln -s /mydata/adopt/christian/adopt/data_final/datasets/open_ocean/test/datasets/open_ocean_patch_1_year_2021/copernicus \
      "$TMPDIR/datasets/open_ocean/test/open_ocean_patch_1_year_2021/copernicus"
ln -s /mydata/adopt/christian/adopt/data_final/datasets/open_ocean/test/datasets/open_ocean_patch_1_year_2021/era5 \
      "$TMPDIR/datasets/open_ocean/test/open_ocean_patch_1_year_2021/era5"
ln -s /mydata/adopt/christian/adopt/data_final/datasets/open_ocean/test/datasets/open_ocean_patch_1_year_2021/hycom \
      "$TMPDIR/datasets/open_ocean/test/open_ocean_patch_1_year_2021/hycom"
ln -s /mydata/adopt/christian/adopt/data_final/datasets/open_ocean/test/datasets/open_ocean_patch_1_year_2021/trajectories \
      "$TMPDIR/datasets/open_ocean/test/open_ocean_patch_1_year_2021/trajectories"
# Copy the config dir (small YAML files only)
cp -r /mydata/adopt/christian/adopt/data_final/configs "$TMPDIR/"
echo "TMPDIR=$TMPDIR"
```

- [ ] **Step 3: Build smoke-test temp directory for coastline**

```bash
# coastline: new layout is {split}/{name}/copernicus/...
# We symlink the data from inside the OLD {name}/datasets/{name}/ wrapper
NAME="patch_7_traj_300234067277290_1"
mkdir -p "$TMPDIR/datasets/coastline/test/$NAME"
OLD_BASE="/mydata/adopt/christian/adopt/data_final/datasets/coastline/test/$NAME/datasets/$NAME"
for d in copernicus era5 hycom trajectories; do
  ln -s "$OLD_BASE/$d" "$TMPDIR/datasets/coastline/test/$NAME/$d"
done
```

- [ ] **Step 4: Smoke-test open_ocean with the fixed `load_config`**

```bash
CODIT_DATA_PATH="$TMPDIR" micromamba run -n codit python - <<'EOF'
import os
from codit.data_module import config_paths, config_funcs
from codit.data_module.dataset import OceanDataset

config_path = os.path.join(config_paths.CONFIG_PATH, "open_ocean", "test")
config = config_funcs.load_config("open_ocean_patch_1_year_2021", config_path)
print("dataset_path:", config["dataset_path"])
assert os.path.isdir(config["dataset_path"]), f"Not found: {config['dataset_path']}"
ds = OceanDataset(config, 72, 16, 10000, "linear")
print(f"open_ocean OK — {len(ds)} trajectories loaded")
EOF
```

Expected: `open_ocean OK — 150 trajectories loaded` (exact count may vary slightly).

- [ ] **Step 5: Smoke-test coastline with the fixed `load_config`**

```bash
CODIT_DATA_PATH="$TMPDIR" micromamba run -n codit python - <<'EOF'
import os
from codit.data_module import config_paths, config_funcs
from codit.data_module.dataset import OceanDataset

config_path = os.path.join(config_paths.CONFIG_PATH, "coastline", "test")
config = config_funcs.load_config("patch_7_traj_300234067277290_1", config_path)
print("dataset_path:", config["dataset_path"])
assert os.path.isdir(config["dataset_path"]), f"Not found: {config['dataset_path']}"
ds = OceanDataset(config, 24, 16, 10000, "linear")
print(f"coastline OK — {len(ds)} trajectories loaded")
EOF
```

Expected: `coastline OK — N trajectories loaded` (N > 0).

- [ ] **Step 6: Clean up temp dir**

```bash
rm -rf "$TMPDIR"
```

- [ ] **Step 7: Run existing test suite to confirm no regressions**

```bash
micromamba run -n codit pytest tests/ -q
```

Expected: 107 passed (same as before — no data-dependent tests will run without `CODIT_DATA_PATH` pointing at real data).

- [ ] **Step 8: Commit the `load_config` fix**

```bash
git add src/codit/data_module/config_funcs.py
git commit -m "fix: correct load_config dataset_path reconstruction to DATA_PATH/datasets/{rel}/{name}"
```

---

## Task 2: Reorganise open_ocean

**Files:**
- No source changes — shell operations on `$CODIT_DATA_PATH/datasets/open_ocean/`

- [ ] **Step 1: Dry-run — print what would be moved**

```bash
for split in train val test; do
  base="$CODIT_DATA_PATH/datasets/open_ocean/$split"
  [ -d "$base/datasets" ] || continue
  for patch in "$base/datasets"/*/; do
    echo "mv $patch -> $base/$(basename $patch)"
  done
  echo "rmdir $base/datasets"
done
```

Expected: ~91 `mv` lines (58 train + 16 val + 17 test) and 3 `rmdir` lines. Verify the paths look correct before proceeding.

- [ ] **Step 2: Run the move**

```bash
for split in train val test; do
  base="$CODIT_DATA_PATH/datasets/open_ocean/$split"
  [ -d "$base/datasets" ] || continue
  for patch in "$base/datasets"/*/; do
    mv "$patch" "$base/$(basename "$patch")"
  done
  rmdir "$base/datasets"
done
```

`rmdir` will fail (and abort via `set -e` if used) if any files remain — that is intentional. If it fails, something was not moved and needs investigation before proceeding.

- [ ] **Step 3: Verify one patch loaded correctly**

```bash
CODIT_DATA_PATH=/mydata/adopt/christian/adopt/data_final \
  micromamba run -n codit python - <<'EOF'
import os
from codit.data_module import config_paths, config_funcs
from codit.data_module.dataset import OceanDataset

config_path = os.path.join(config_paths.CONFIG_PATH, "open_ocean", "test")
config = config_funcs.load_config("open_ocean_patch_1_year_2021", config_path)
print("dataset_path:", config["dataset_path"])
assert os.path.isdir(config["dataset_path"])
ds = OceanDataset(config, 72, 16, 10000, "linear")
print(f"OK — {len(ds)} trajectories")
EOF
```

Expected: `OK — N trajectories` (N > 0).

- [ ] **Step 4: Commit**

```bash
git commit --allow-empty -m "chore: reorganise open_ocean datasets to canonical layout {split}/{name}/"
```

(Empty commit records the reorganisation event in git history since the data itself is not tracked.)

---

## Task 3: Reorganise coastline

**Files:**
- No source changes — shell operations on `$CODIT_DATA_PATH/datasets/coastline/`

- [ ] **Step 1: Dry-run — print what would be moved**

```bash
for split in train val test; do
  base="$CODIT_DATA_PATH/datasets/coastline/$split"
  for patch_dir in "$base"/*/; do
    name=$(basename "$patch_dir")
    inner="$patch_dir/datasets/$name"
    [ -d "$inner" ] || continue
    echo "mv $inner/* -> $patch_dir/"
    echo "rmdir $patch_dir/datasets/$name  $patch_dir/datasets"
  done
done
```

Review output before proceeding.

- [ ] **Step 2: Run the move**

```bash
for split in train val test; do
  base="$CODIT_DATA_PATH/datasets/coastline/$split"
  for patch_dir in "$base"/*/; do
    name=$(basename "$patch_dir")
    inner="$patch_dir/datasets/$name"
    [ -d "$inner" ] || continue
    mv "$inner"/* "$patch_dir/"
    rmdir "$patch_dir/datasets/$name"
    rmdir "$patch_dir/datasets"
  done
done
```

- [ ] **Step 3: Verify one patch loaded correctly**

```bash
CODIT_DATA_PATH=/mydata/adopt/christian/adopt/data_final \
  micromamba run -n codit python - <<'EOF'
import os
from codit.data_module import config_paths, config_funcs
from codit.data_module.dataset import OceanDataset

config_path = os.path.join(config_paths.CONFIG_PATH, "coastline", "test")
config = config_funcs.load_config("patch_7_traj_300234067277290_1", config_path)
print("dataset_path:", config["dataset_path"])
assert os.path.isdir(config["dataset_path"])
ds = OceanDataset(config, 24, 16, 10000, "linear")
print(f"OK — {len(ds)} trajectories")
EOF
```

Expected: `OK — N trajectories` (N > 0).

- [ ] **Step 4: Commit**

```bash
git commit --allow-empty -m "chore: reorganise coastline datasets to canonical layout {split}/{name}/"
```

---

## Task 4: Update download script comment and run full test suite

**Files:**
- Modify: `scripts/download_data.py` (comment only)

- [ ] **Step 1: Update the layout comment in `download_data.py`**

Find the comment block near the top of `scripts/download_data.py` that describes the expected extraction layout and update it to:

```python
# Each tarball, when extracted, must produce the directory structure expected by
# config_funcs.load_config:
#   datasets/{dataset}/{split}/{name}/copernicus/
#   datasets/{dataset}/{split}/{name}/era5/
#   datasets/{dataset}/{split}/{name}/hycom/
#   datasets/{dataset}/{split}/{name}/trajectories/
#   configs/{dataset}/{split}/{name}.yml
```

- [ ] **Step 2: Run full test suite**

```bash
micromamba run -n codit pytest tests/ -q
```

Expected: 107 passed.

- [ ] **Step 3: Commit**

```bash
git add scripts/download_data.py
git commit -m "docs: update download_data.py layout comment to match canonical structure"
```
