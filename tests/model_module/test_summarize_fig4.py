"""Smoke test for summarize_fig4.py using synthetic .npz fixtures.

The fixture places one 'patch' file per model in the expected directory
layout, with pred_coords == true_coords (perfect predictions) so PE/ES
are zero, SS is 1.0, and the naive-drift filter is satisfied (trajectories
drift > 100 m from start).
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARIZE_SCRIPT = REPO_ROOT / "scripts" / "dev" / "figures" / "summarize_fig4.py"


def _make_synthetic_patch(model_dir: Path, n_traj: int = 2, n_samples: int = 3, pred_len: int = 5):
    """Write one <name>_result.npz with pred == true (perfect predictions).

    Shapes match what codit_eval / lingauss_eval save:
      true_coords: (N, T, 2)         — N trajectories, T = pred_len + 1 timesteps
      pred_coords: (N, S, T-1, 2)    — N × S samples of predicted (T-1) steps
      distance:    (N, S, T-1)       — filler, not read by summarize

    Coords are (lat, lon) in degrees. Start at (0, 0); drift eastward by
    0.01 deg (~1.1 km) per step so the >100 m naive-drift filter passes.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    # True coords: N trajectories, T timesteps (pred_len + 1 including start)
    T = pred_len + 1
    step_deg = 0.01
    traj = np.stack(
        [np.zeros(T), step_deg * np.arange(T)], axis=-1
    )  # (T, 2)
    true_coords = np.tile(traj, (n_traj, 1, 1)).astype(np.float32)  # (N, T, 2)
    # Perfect predictions: every sample equals the true post-start trajectory
    pred_traj = true_coords[:, 1:, :]  # (N, T-1, 2)
    pred_coords = np.broadcast_to(
        pred_traj[:, None, :, :], (n_traj, n_samples, pred_len, 2)
    ).astype(np.float32)
    distance = np.zeros((n_traj, n_samples, pred_len), dtype=np.float32)
    np.savez(
        model_dir / "synthetic_patch_result.npz",
        true_coords=true_coords,
        pred_coords=pred_coords,
        distance=distance,
    )


@pytest.mark.parametrize("dataset,model_dirs", [
    (
        "open_ocean",
        [
            "codit_GRU_open_ocean",
            "codit_Transformer_open_ocean",
            "linear_gauss",
        ],
    ),
])
def test_summarize_fig4_writes_csv_with_expected_rows(tmp_path, dataset, model_dirs):
    split = "test"
    subdir = "fig4_repro"
    # Layout: <tmp_path>/results/<dataset>/<split>/<subdir>/<model_dir>/synthetic_patch_result.npz
    base = tmp_path / "results" / dataset / split / subdir
    for m in model_dirs:
        _make_synthetic_patch(base / m, pred_len=5)  # small T for speed

    env = os.environ.copy()
    env["CODIT_DATA_PATH"] = str(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SUMMARIZE_SCRIPT),
            "--dataset", dataset,
            "--split", split,
            "--result_subdir", subdir,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"summarize_fig4.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    csv_path = base / "fig4_summary.csv"
    assert csv_path.exists(), f"CSV not written at {csv_path}"

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Expect exactly three rows, one per model
    assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"

    labels = {row["model"] for row in rows}
    assert labels == {"CoDiT (GRU)", "CoDiT (TF)", "WDM"}

    # Each row must have the 9 comparison columns
    expected_cols = {
        "model",
        "PE_new_km", "PE_paper_km", "dPE_km",
        "ES_new_km", "ES_paper_km", "dES_km",
        "SS_new", "SS_paper", "dSS",
    }
    assert expected_cols.issubset(rows[0].keys()), (
        f"missing columns: {expected_cols - set(rows[0].keys())}"
    )

    # With perfect predictions PE and ES must be ~0, SS must be ~1
    for row in rows:
        assert float(row["PE_new_km"]) < 0.05, row
        assert float(row["ES_new_km"]) < 0.05, row
        assert float(row["SS_new"]) > 0.99, row
