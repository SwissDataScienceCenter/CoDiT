import os
from codit.data_module import config_paths, config_funcs
from codit.data_module.dataset import OceanDataset

config_path = os.path.join(config_paths.CONFIG_PATH, 'open_ocean', 'val')
files = sorted(f for f in os.listdir(config_path) if f.endswith('.yml'))
print('Files:', files[:5])
dataset_name = files[0].split('.')[0]
config = config_funcs.load_config(dataset_name, config_path)
print('Config keys:', list(config.keys()))
print('dataset_path:', config.get('dataset_path'))
print('exists:', os.path.exists(config.get('dataset_path', '')))
ds = OceanDataset(config, 72, 16, 10000, "linear")
print('len(ds):', len(ds))
