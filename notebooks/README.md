# Notebook release runbook

The two notebooks below are **not** executed in CI. Before every release, run
each once on a checkout with `CODIT_DATA_PATH` set and confirm they finish
without error.

| Notebook | Requires | Wallclock (CPU) |
|----------|----------|-----------------|
| `01_train_quickstart.ipynb` | shipped `open_ocean` merged h5 + one test config | ≈10 min |
| `02_prepare_your_own_region.ipynb` | Copernicus + CDS credentials in `config.ini`; network | ≈30 min |
| `get_started.ipynb` | shipped trained checkpoints; network for a fresh region | ≈15 min |
| `making_predictions.ipynb` | shipped trained checkpoints | ≈5 min |

## Steps

1. `export CODIT_DATA_PATH=/path/to/data`
2. For each notebook:
   `micromamba run -n codit jupyter nbconvert --to notebook --execute <notebook>.ipynb --output /tmp/<notebook>_run.ipynb --ExecutePreprocessor.timeout=1800`
3. Discard the executed output; do not commit it.

If either notebook fails, treat it as a release blocker.
