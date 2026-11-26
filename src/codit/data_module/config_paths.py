"""Path configuration for CoDiT.

Data paths resolve from the CODIT_DATA_PATH environment variable
(default: <repo_root>/data). Model checkpoints resolve separately from
CODIT_MODEL_PATH (default: <repo_root>/data), so the checkpoints shipped in the
repo under data/trained_models/ stay found when the data lives elsewhere
(e.g. on a mounted store).
"""

import os

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

DATA_PATH = os.environ.get("CODIT_DATA_PATH", os.path.join(_REPO_ROOT, "data"))
CONFIG_PATH = os.path.join(DATA_PATH, "configs")
MODEL_PATH = os.environ.get("CODIT_MODEL_PATH", os.path.join(_REPO_ROOT, "data"))
RESULT_PATH = os.path.join(DATA_PATH, "results")
FIG_PATH = os.path.join(DATA_PATH, "figs")
CREDENTIALS_PATH = os.environ.get(
    "CODIT_CREDENTIALS_PATH", os.path.join(_REPO_ROOT, "config.ini")
)
