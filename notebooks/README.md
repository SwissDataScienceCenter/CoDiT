# Notebook release runbook

The notebooks below are **not** executed in CI. Before every release, run
each once on a checkout with `CODIT_DATA_PATH` set and confirm they finish
without error.

| Notebook | Requires | Wallclock (CPU) |
|----------|----------|-----------------|
| `user/01_predict_quickstart.ipynb` | shipped trained checkpoints; Copernicus + CDS credentials in `config.ini`; network | ≈15 min |
| `user/02_train_quickstart.ipynb` | shipped `open_ocean` merged h5 + one test config | ≈10 min |
| `user/03_prepare_your_own_region.ipynb` | Copernicus + CDS credentials in `config.ini`; network | ≈30 min |

## Steps

1. `export CODIT_DATA_PATH=/path/to/data`
2. For each notebook (run from within `notebooks/user/`):
   `micromamba run -n codit jupyter nbconvert --to notebook --execute <notebook>.ipynb --output /tmp/<notebook>_run.ipynb --ExecutePreprocessor.timeout=1800`
3. Discard the executed output; do not commit it.

If either notebook fails, treat it as a release blocker.
