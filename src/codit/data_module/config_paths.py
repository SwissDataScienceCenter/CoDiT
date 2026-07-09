"""Path configuration for CoDiT.

All paths resolve from the CODIT_DATA_PATH environment variable.
If unset, defaults to <repo_root>/data.
"""

import os

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

DATA_PATH = os.environ.get("CODIT_DATA_PATH", os.path.join(_REPO_ROOT, "data"))
CONFIG_PATH = os.path.join(DATA_PATH, "configs")
MODEL_PATH = DATA_PATH
RESULT_PATH = os.path.join(DATA_PATH, "results")
FIG_PATH = os.path.join(DATA_PATH, "figs")
CREDENTIALS_PATH = os.environ.get(
    "CODIT_CREDENTIALS_PATH", os.path.join(_REPO_ROOT, "config.ini")
)
