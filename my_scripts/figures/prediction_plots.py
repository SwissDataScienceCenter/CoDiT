# %%
from codit.model_module.diffusion_model import CoDiTModel
from codit.model_module.linear_gauss_model import LinearGaussModel
from codit.model_module.conv_net import ConvNetModel
from codit.model_module.hycom_model import HycomModel

import os
from codit.data_module import config_funcs
from codit.data_module.dataset import OceanDataset
from codit.data_module import config_paths
from matplotlib import pyplot as plt
from codit.data_module import inverse_haversine
from cartopy import crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from jax import numpy as jnp
data_set = "open_ocean"  #'coastline'#

import_path = '/mydata/adopt/christian/CoDiT/data/trained_models/'

hycom_model = HycomModel(name='hycom', params={})
linear_model = LinearGaussModel.import_trained_model(f'linear_gauss_{data_set}','/mydata/adopt/christian/CoDiT/data/trained_models/')
cnn_model = ConvNetModel.import_trained_model(f'cnn_{data_set}','/mydata/adopt/christian/CoDiT/data/trained_models/')
gru_model = CoDiTModel.import_trained_model(f'codit_GRU_{data_set}','/mydata/adopt/christian/CoDiT/data/trained_models/')
transformer_model = CoDiTModel.import_trained_model(f'codit_Transformer_{data_set}','/mydata/adopt/christian/CoDiT/data/trained_models/')
# %%

config_path = os.path.join(
    config_paths.CONFIG_PATH,
    data_set,
    'test'
)
if data_set == "open_ocean":
    length = 72
elif data_set == "coastline":
    length = 24
files = os.listdir(config_path)
dataset_files = [f for f in files if f.endswith(".yml")]
val_datasets = []
for dataset_name in dataset_files:
    config = config_funcs.load_config(dataset_name.split(".")[0], config_path)
    if config["start_date"].year >= 2020 and os.path.exists(config["dataset_path"]):
        val_datasets.append(dataset_name)

data_name = val_datasets[2].split(".")[0]
#data_name = 'open_ocean_patch_9_year_2020'
config = config_funcs.load_config(data_name, config_path)
test_dataset = OceanDataset(config, length,)

# %%
idx = 0
start_position, start_time, times, interpolator, coordinates = test_dataset[idx]

# %%
from jax import random
pred_coord = {}
pred_coord['hycom'] = hycom_model.predict(
    start_position,
    start_time,
    length,
    interpolator,
    test_dataset,
)
print('Linear-Gaussian')
pred_coord['linear_gauss'] = linear_model.predict(
    start_position,
    start_time,
    length,
    interpolator,
    test_dataset,
    num_samples=100,
)
print('CNN')
pred_coord['cnn'] = cnn_model.predict(
    start_position,
    start_time,
    length,
    interpolator,
    test_dataset,
    num_samples=100,
)
print('GRU')
pred_coord['gru'] = gru_model.predict(
    start_position,
    start_time,
    length,
    interpolator,
    test_dataset,
    num_samples=100,
)
print('Transformer')
pred_coord['transformer'] = transformer_model.predict(
    start_position,
    start_time,
    length,
    interpolator,
    test_dataset,
    num_samples=100,
    key=random.PRNGKey(101)
)

# %%

colors = {
    'hycom': 'gray',
    'linear_gauss': 'C2',
    'cnn': 'C1',
    'gru': 'C3',
    'transformer': 'C4'
}
model_names = {
    'hycom': 'HYCOM',
    'linear_gauss': 'Linear-Gaussian',
    'cnn': 'CNN',
    'gru': 'CoDiT (GRU)',
    'transformer': 'CoDiT (TF)'
}

fig = plt.figure(figsize=(8, 8))

proj = ccrs.AzimuthalEquidistant(central_longitude=coordinates[0, 1], central_latitude=coordinates[0, 0])
ax = plt.axes(projection=proj)
# add land feature
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.LAND, color="lightgray")
ax.plot(coordinates[:, 1], coordinates[:, 0], 'k', label='True Path', zorder=9, transform=ccrs.PlateCarree())
min_lat = np.inf
min_lon = np.inf
for model, color in colors.items():
    if model == 'hycom':
        ax.plot(pred_coord[model][...,1].T, pred_coord[model][...,0].T, color=color, label=model_names[model], zorder=9, transform=ccrs.PlateCarree())
        ax.plot(pred_coord[model][...,1].T, pred_coord[model][...,0].T, color='w',lw=3, zorder=8, transform=ccrs.PlateCarree())
    else:
        ax.plot(np.nanmean(pred_coord[model][...,1], axis=0), np.nanmean(pred_coord[model][...,0], axis=0), color=color, label=model_names[model], zorder=9, transform=ccrs.PlateCarree())
        ax.plot(np.nanmean(pred_coord[model][...,1], axis=0), np.nanmean(pred_coord[model][...,0], axis=0), color='w',lw=3, zorder=8, transform=ccrs.PlateCarree())
    min_lat = min(min_lat, np.min(pred_coord[model][...,0]))
    min_lon = min(min_lon, np.min(pred_coord[model][...,1]))
plt.legend(frameon=False, loc='upper right')

for model, color in colors.items():
    if model != 'hycom':
        ax.plot(pred_coord[model][...,1].T, pred_coord[model][...,0].T, color=color, alpha=0.2, rasterized=True, transform=ccrs.PlateCarree())

sb1 = jnp.array([min_lat, min_lon]) # ax.get_ylim()[0], ax.get_xlim()[0]
#sb1 = sb1.at[0].add(-.25)
sb2 = inverse_haversine(sb1, 10e3, 90)
ax.set_aspect('equal', adjustable='box')
ax.plot((sb1[1], sb2[1]), (sb1[0], sb2[0]), 'k-', lw=2, zorder=10, transform=ccrs.PlateCarree())
ax.text(sb1[1], sb1[0]+0.01, '10 km', fontsize=12, transform=ccrs.PlateCarree())
#ax.set_xlim(start_position[1]-.5, start_position[1]+.5)
#ax.set_ylim(start_position[0]-.5, start_position[0]+.5)
plt.savefig(os.path.join(config_paths.FIG_PATH, f'prediction_plot_{data_set}_{data_name}_{idx}.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(config_paths.FIG_PATH, f'prediction_plot_{data_set}_{data_name}_{idx}.pdf'), dpi=300, bbox_inches='tight')
plt.show()
# %%
