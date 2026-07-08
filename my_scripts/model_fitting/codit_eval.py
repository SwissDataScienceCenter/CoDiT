import argparse
import os

# mute SerializationWarning
import warnings

from jax import numpy as jnp
from jax import vmap
from tqdm import tqdm
import numpy as np
from codit.data_module import config_paths
from codit.data_module import config_funcs
from codit.data_module.dataset import OceanDataset
from codit.data_module import data_transform
from codit.model_module.diffusion_model import CoDiTModel

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="open_ocean")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--model_type", type=str, default="GRU")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--context_encoder_type", type=str, default="conv"
    )  # conv, mlp, gap
    parser.add_argument("--optimizer", type=str, default="adam")  # adam, adamw, sgd
    parser.add_argument("--scheduler", type=str, default="constant")  # constant, cosine
    parser.add_argument("--evaluate_subset", action="store_true", default=False)
    parser.add_argument("--K", type=int, default=100)  # number of diffusion steps
    parser.add_argument("--input_vars", nargs='+', type=str, 
                    default=["hycom_current", "copernicus_current", "era5_wind", "bathymetry"], 
                    help="Input variables for the model")
    parser.add_argument("--run_name", type=str, default="", help="Optional run name for logging and saving results")
    args = parser.parse_args()
    np.random.seed(0)
    
    input_vars = args.input_vars
    input_var_str = "_".join(input_vars)
    model_name = f"codit_{args.model_type}_{args.dataset}_{args.context_encoder_type}_{args.optimizer}_K{args.K}_{input_var_str}"
    if args.run_name:
        model_name += f"_{args.run_name}"
    print(f"Model name: {model_name}")
    if args.dataset == "open_ocean":
        config_path = os.path.join(
            config_paths.CONFIG_PATH,
            "open_ocean",
            args.split
        )
        result_path = os.path.join(
            config_paths.RESULT_PATH,
            "open_ocean",
            args.split,
            model_name,
        )
        length = 72
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    elif args.dataset == "coastline":
        config_path = os.path.join(config_paths.CONFIG_PATH, "coastline", args.split)
        result_path = os.path.join(
            config_paths.RESULT_PATH,
            "coastline",
            args.split, 
            model_name,
        )
        length = 24
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    else:
        raise ValueError("Dataset not found")
    files = os.listdir(config_path)
    dataset_files = [f for f in files if f.endswith(".yml")]
    val_datasets = []
    for dataset_name in dataset_files:
        config = config_funcs.load_config(dataset_name.split(".")[0], config_path)
        if config["start_date"].year >= 2020 and os.path.exists(config["dataset_path"]):
            val_datasets.append(dataset_name)

    # model_name = f"diffusion_model_{args.dataset}_{args.model_type}_{args.context_encoder_type}_{args.optimizer}_{args.scheduler}"
    model = CoDiTModel.import_trained_model(model_name=model_name)
    num_predicted_traj = 0
    for val_dataname in tqdm(val_datasets):
        data_name = val_dataname.split(".")[0]
        result_file = os.path.join(result_path, f"{data_name}_result.npz")
        if not os.path.exists(result_file):
            try:
                config = config_funcs.load_config(data_name, config_path)
                test_dataset = OceanDataset(config, length, 16, 10000, "linear")
            except Exception as e:
                print(f"Error loading dataset {val_dataname}: {e}")
                continue
            
            distances, pred_coords, true_coords = [], [], []
            if len(test_dataset) == 0:
                print(f"Empty dataset: {val_dataname}")
                continue
            if args.evaluate_subset and len(test_dataset) > 200:
                # select 200 regularly spaced indices from the dataset
                indices = np.linspace(0, len(test_dataset) - 1, 200, dtype=int)
            else:
                indices = range(len(test_dataset))
            for idx in indices:
                try:
                    start_position, start_time, times, interpolator, coordinates = test_dataset[idx]
                    print(f"Processing {data_name} {idx}/{len(test_dataset)}")
                    pred_coord = model.predict(
                        start_position,
                        start_time,
                        length,
                        interpolator,
                        test_dataset,
                        num_samples=100,
                    )
                    distance = vmap(
                        vmap(data_transform.haversine_distance), in_axes=(None, 0)
                    )(coordinates[1:, :], pred_coord[:, :])
                    distances.append(distance[None])
                    pred_coords.append(pred_coord[None])
                    true_coords.append(coordinates[:, :][None])
                except ValueError as e:
                    print(f"Error loading dataset {val_dataname}: {e}")
                    continue
            distances = jnp.concatenate(distances, axis=0)
            pred_coords = jnp.concatenate(pred_coords, axis=0)
            true_coords = jnp.concatenate(true_coords, axis=0)
            jnp.savez(
                result_file,
                distance=distances,
                pred_coords=pred_coords,
                true_coords=true_coords,
            )
        # if args.evaluate_subset and num_predicted_traj >= 200:
        #     break

# # %%
# import os, argparse
# from adopt.model_module.diffusion_model import DiffusionModel
# from adopt.data_module.data_loader import AdoptDataset
# from adopt.data_module import config_funcs
# from adopt import config_paths
# from jax import numpy as jnp
# from tqdm import tqdm
# from jax import vmap
# from adopt.data_processing import jax_transform
# # mute SerializationWarning
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# # %%
# class Args:
#     dataset = 'coastline'
#     num_samples = 100

# args = Args()

# if args.dataset == 'open_ocean':
#     config_path = os.path.join(config_paths.CONFIG_PATH, 'open_ocean',)
#     result_path = os.path.join(config_paths.RESULT_PATH, 'open_ocean', 'conv_net')
#     length = 72
#     if not os.path.exists(result_path):
#         os.makedirs(result_path)
# elif args.dataset == 'coastline':
#     config_path = os.path.join(config_paths.CONFIG_PATH, 'coastline')
#     result_path = os.path.join(config_paths.RESULT_PATH, 'coastline', 'conv_net')
#     length = 24
#     if not os.path.exists(result_path):
#         os.makedirs(result_path)
# else:
#     raise ValueError('Dataset not found')
# files = os.listdir(config_path)
# dataset_files = [f for f in files if f.endswith('.yml')]
# val_datasets = []
# for dataset_name in dataset_files[:100]:
#     config = config_funcs.load_config(dataset_name.split('.')[0], config_path)
#     if config['start_date'].year >= 2020 and os.path.exists(config['dataset_path']):
#         val_datasets.append(dataset_name)

# model = DiffusionModel(name=f'diffusion_model_{args.dataset}', params={
# 'prediction_length': length,
# })
# model.load_trained_model()

# # %%
# val_dataname = val_datasets[5]
# data_name = val_dataname.split('.')[0]
# config = config_funcs.load_config(data_name, config_path)
# test_dataset = AdoptDataset(config, length, 16, 10000, 'linear')
# # %%
# i = 0
# start_position, times, interpolator, coordinates = test_dataset[i]
# # %%
# pred_coord = model.predict(start_position, times, interpolator, test_dataset, num_samples=100)
# # %%
# from jax import random, jit, vmap
# from functools import partial
# from adopt.process_diffusion.brownian_noise_models import BrownianNoiseModel
# from adopt.process_diffusion import denoising_models, noising_functions
# from flax import linen as nn
# from jax import lax
# key = random.PRNGKey(0)
# prediction_length = length
# num_samples = 100

# start_context = test_dataset.get_context(times[:1], start_position[None], interpolator)
# coords_denoised = [start_position]
# key, subkey = random.split(key)
# K = model.schedule_dict['K']
# batch = {'history_target_vals': jnp.zeros((1, 2)), 'history_context': start_context, 'prediction_context': start_context[None]}
# keys = random.split(key, num_samples+1)
# key, subkeys = keys[0], keys[1:]

# def get_noise_model(key, prediction_length):
#     keys = random.split(key, K+1)
#     key, subkeys = keys[0], keys[1:]
#     return vmap(model.noise_model, in_axes=(0, None))(subkeys, prediction_length)

# W, dW = vmap(get_noise_model, in_axes=(0, None))(subkeys, prediction_length)

# traj_samples = []
# denoising_steps = []
# coords_denoised = [jnp.tile(start_position[None], (num_samples, 1, 1))]
# keys = random.split(key, num_samples+1)
# key, subkeys = keys[0], keys[1:]
# initial_y, _ = vmap(model.noise_model, in_axes=(0, None))(subkeys, prediction_length)

# def backward_denoising_step(carry: tuple, i: int, W, batch: dict, schedule_dict: dict, noise_model: BrownianNoiseModel, model: nn.module,
#                             params: dict) -> tuple[tuple, tuple]:
#     """
#     A single step of the backward denoising algorithm.

#     Args:
#         carry: The state of the algorithm.
#         i: The current time step.
#         beta: The beta schedule.
#         alpha: The alpha schedule.
#         alpha_bar: The alpha bar schedule.
#         noise_model: The noise model.
#         model: The denoising model.
#         params: The parameters of the model.
#         K: The number of time steps.
#         init_cond: The initial condition.
#         T: The time horizon.

#     Returns:
#         carry: The updated state of the algorithm.
#         output: The output of the algorithm.
#     """
#     key, y = carry
#     # key, subkey = random.split(key)
#     beta_tmp = schedule_dict['beta'][i]
#     alpha_tmp = schedule_dict['alpha'][i]
#     alpha_bar_tmp = schedule_dict['alpha_bar'][i]
#     k = jnp.array([i])
#     y_proj = y
#     y_history_reversed = batch['history_target_vals'][...,::-1,:]
#     y_history_input = noise_model.center_data(y_history_reversed)
#     #y_history_delta = y_history_input[...,:1,:]
#     key, subkey = random.split(key)
#     y_noised_history, _ = noising_functions.forward_diffusion(y_history_input,
#                                                                 k, subkey,
#                                                                 schedule_dict,
#                                                                 noise_model
#                                                             )
#     #y_noised_history = y_noised_history + y_history_delta
#     #key, subkey = random.split(key)
#     y_proj_history = y_noised_history
#     y_proj_history = y_proj_history[...,::-1,:]
#     noise_pred = model.apply(params, y_proj[1:], y_proj_history, batch, k)
#     y_delta = y[...,:1,:]
#     W_pred = noise_model.transform(noise_pred)
#     mu_pred = y / jnp.sqrt(1-beta_tmp)
#     mu_pred -= beta_tmp / (jnp.sqrt(1 - alpha_bar_tmp) * jnp.sqrt(alpha_tmp)) * W_pred
#     sigma_pred = jnp.sqrt(beta_tmp)
#     #key, subkey = random.split(key)
#     # W, dW = noise_model(subkey, y.shape[-2]-1, cov=cov)
#     k_is_not_zero = i != 0
#     # For the last step the mean is returned
#     y = mu_pred + k_is_not_zero * sigma_pred * W[i]
#     y = y.at[:1].set(0)
#     # y += y_delta
#     return (key, y), y

# def denoise(params: dict, y_init: jnp.ndarray, batch, model: nn.module, W, noise_model: BrownianNoiseModel, key: random.PRNGKey,
#             schedule_dict: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     Denoise a time series.

#     Args:
#         params: The parameters of the model.
#         model: The denoising model.
#         noise_model: The noise model.
#         key: The random key.
#         T: The time horizon.
#         y_init: The initial data.
#         beta: The beta schedule.
#         alpha: The alpha schedule.
#         alpha_bar: The alpha bar schedule.
#         init_cond: The initial condition.

#     Returns:
#         y: The denoised time series.
#         noise_pred: The predicted noise
#     """
#     init = (key, y_init)
#     K = schedule_dict['K']
#     iterations = jnp.arange(0, K)[::-1]
#     step_func = lambda carry, i: backward_denoising_step(carry, i, W, batch, schedule_dict,
#                                                          noise_model, model, params)
#     denoise_results = lax.scan(step_func, init, iterations)
#     y = denoise_results[0][1]
#     denoising_steps = denoise_results[1]
#     return y, denoising_steps


# @partial(vmap, in_axes=(0, 0, 0, 0, None, None))
# def get_denoised_trajectory(key, W, initial_y, batch, numeric_time, i):
#     denoised, denoised_steps = denoise(model.params, initial_y, batch, model.model, W, model.noise_model, key, model.schedule_dict)
#     denoised_scaled = denoised * model.std
#     distance_in_metres = jnp.sqrt(denoised_scaled[:,0]**2 + denoised_scaled[:,1]**2)
#     bearing_angle = jnp.arctan2(denoised_scaled[:,1], denoised_scaled[:,0])
#     bearing_angle = jnp.degrees(bearing_angle)
#     dcoord = jax_transform.inverse_haversine_grad(start_position, distance_in_metres[i], bearing_angle[i])
#     new_coord = start_position[None] + dcoord
#     #new_context = get_interpolation_context(new_coord, numeric_time)
#     new_context = test_dataset.get_context(numeric_time, new_coord, interpolator)
#     return new_coord, new_context

# keys = random.split(key, num_samples+1)
# key, subkeys = keys[0], keys[1:]
# coords_denoised = [jnp.tile(start_position[None], (num_samples, 1, 1))]
# batch = {'history_target_vals': jnp.zeros((num_samples, 1, 2)),
#         'history_context': jnp.tile(start_context[None], (num_samples, 1, 1, 1, 1)),
#         'prediction_context': jnp.tile(start_context[None], (num_samples, length, 1, 1, 1))}

# denoise_func = jit(get_denoised_trajectory)
# for i in range(1, prediction_length+1):
#     new_coord, new_context = denoise_func(subkeys, W, initial_y, batch, times[i:i+1], i)
#     coords_denoised.append(new_coord)
#     batch['prediction_context'] = batch['prediction_context'].at[:,i:i+1].set(new_context) # jnp.concatenate([batch['prediction_context'], new_context], axis=1)
#     batch['history_target_vals'] = jnp.zeros((num_samples, 1, 2))
#     batch['history_context'] = jnp.tile(start_context[None], (num_samples, 1, 1, 1, 1))

# coords_pred = jnp.concatenate(coords_denoised, axis=1)
# # %%
# def get_prediction(model, start_position, times, interpolator, num_samples, dataset, noise_model, prediction_length, key) -> tuple[jnp.ndarray, jnp.ndarray]:
#     start_time = times[0]
#     start_context = dataset.get_context(times[:1], start_position[None], interpolator)
#     coords_denoised = [start_position]
#     key, subkey = random.split(key)
#     initial_y, _ = noise_model(subkey, prediction_length, cov)
#     K = schedule_dict['K']
#     batch = {'history_target_vals': jnp.zeros((1, 2)), 'history_context': start_context, 'prediction_context': start_context[None]}
#     keys = random.split(key, num_samples+1)
#     key, subkeys = keys[0], keys[1:]

#     def get_noise_model(key, prediction_length, cov):
#         keys = random.split(key, K+1)
#         key, subkeys = keys[0], keys[1:]
#         return jax.vmap(noise_model, in_axes=(0, None, None))(subkeys, prediction_length, cov)

#     W, dW = jax.vmap(get_noise_model, in_axes=(0, None, None))(subkeys, prediction_length, cov)

#     traj_samples = []
#     denoising_steps = []
#     coords_denoised = [jnp.tile(start_coord[None], (num_samples, 1, 1))]
#     keys = random.split(key, num_samples+1)
#     key, subkeys = keys[0], keys[1:]
#     initial_y, _ = jax.vmap(noise_model, in_axes=(0, None, None))(subkeys, prediction_length, cov)
#     K = schedule_dict['K']


#     @jit
#     def get_interpolation_context(new_coord, numeric_time):
#         coord_grid = vmap(context_functions.from_grid_to_coords, in_axes=(0,None,None))(new_coord, distance_from_origin, bearing_deg) + new_coord[:,None,None]
#         xq = jnp.asarray(numeric_time)
#         yq = jnp.asarray(coord_grid[:,:,:,0])
#         zq = jnp.asarray(coord_grid[:,:,:,1])
#         new_context = interpolation_function(xq, yq, zq)
#         new_context = jnp.nan_to_num(new_context)
#         return new_context

#     @partial(jax.vmap, in_axes=(0, 0, 0, 0, None, None))
#     def get_denoised_trajectory(key, W, initial_y, batch, numeric_time, i):
#         denoised, denoised_steps = denoise(params, initial_y, batch, model, W,
#                         cov, cov_pinv, noise_model, key, schedule_dict)
#         denoised_scaled = denoised * std
#         distance_in_metres = jnp.sqrt(denoised_scaled[:,0]**2 + denoised_scaled[:,1]**2)
#         bearing_angle = jnp.arctan2(denoised_scaled[:,1], denoised_scaled[:,0])
#         bearing_angle = jnp.degrees(bearing_angle)
#         dcoord = jax_transform.inverse_haversine_grad(start_coord[0], distance_in_metres[i], bearing_angle[i])
#         new_coord = start_coord + dcoord
#         new_context = get_interpolation_context(new_coord, numeric_time)
#         return new_coord, new_context

#     keys = random.split(key, num_samples+1)
#     key, subkeys = keys[0], keys[1:]
#     coords_denoised = [jnp.tile(start_coord[None], (num_samples, 1, 1))]
#     batch = {'history_target_vals': jnp.zeros((num_samples, 1, 2)),
#             'history_context': jnp.tile(start_context[None], (num_samples, 1, 1, 1, 1)),
#             'prediction_context': jnp.tile(start_context[None], (num_samples, length, 1, 1, 1))}

#     denoise_func = jit(get_denoised_trajectory)
#     for i in range(1, prediction_length+1):
#         new_coord, new_context = denoise_func(subkeys, W, initial_y, batch, times, i)
#         coords_denoised.append(new_coord)
#         batch['prediction_context'] = batch['prediction_context'].at[:,i:i+1].set(new_context) # jnp.concatenate([batch['prediction_context'], new_context], axis=1)
#         batch['history_target_vals'] = jnp.zeros((num_samples, 1, 2))
#         batch['history_context'] = jnp.tile(start_context[None], (num_samples, 1, 1, 1, 1))

#     coords_denoised = jnp.concatenate(coords_denoised, axis=1)
#     return coords_denoised, time
# # %%

# if __name__=='__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='open_ocean')
#     parser.add_argument('--num_samples', type=int, default=100)
#     args = parser.parse_args()


#     if args.dataset == 'open_ocean':
#         config_path = os.path.join(config_paths.CONFIG_PATH, 'open_ocean',)
#         result_path = os.path.join(config_paths.RESULT_PATH, 'open_ocean', 'conv_net')
#         length = 72
#         if not os.path.exists(result_path):
#             os.makedirs(result_path)
#     elif args.dataset == 'coastline':
#         config_path = os.path.join(config_paths.CONFIG_PATH, 'coastline')
#         result_path = os.path.join(config_paths.RESULT_PATH, 'coastline', 'conv_net')
#         length = 24
#         if not os.path.exists(result_path):
#             os.makedirs(result_path)
#     else:
#         raise ValueError('Dataset not found')
#     files = os.listdir(config_path)
#     dataset_files = [f for f in files if f.endswith('.yml')]
#     val_datasets = []
#     for dataset_name in dataset_files:
#         config = config_funcs.load_config(dataset_name.split('.')[0], config_path)
#         if config['start_date'].year >= 2020 and os.path.exists(config['dataset_path']):
#             val_datasets.append(dataset_name)

#     model = DiffusionModel(name=f'diffusion_model_{args.dataset}', params={
#     'prediction_length': length,
#     })
#     model.load_trained_model()

#     for val_dataname in tqdm(val_datasets):
#         data_name = val_dataname.split('.')[0]
#         try:
#             config = config_funcs.load_config(data_name, config_path)
#             test_dataset = AdoptDataset(config, length, 16, 10000, 'linear')
#         except Exception as e:
#             print(f"Error loading dataset {val_dataname}: {e}")
#             continue

#         distances, pred_coords, true_coords = [], [], []
#         if len(test_dataset) == 0:
#             print(f"Empty dataset: {val_dataname}")
#             continue
#         i = 0

#         for i in range(len(test_dataset)):
#             try:
#                 start_position, times, interpolator, coordinates = test_dataset[i]
#                 i += 1
#                 print(f"Processing {data_name} {i}/{len(test_dataset)}")
#                 pred_coord = model.predict(start_position, times, interpolator, test_dataset, num_samples=100)
#                 distance = vmap(vmap(jax_transform.haversine_distance), in_axes=(None,0))(coordinates[1:,:], pred_coord[:,:])
#                 distances.append(distance[None])
#                 pred_coords.append(pred_coord[None])
#                 true_coords.append(coordinates[1:,:][None])
#             except ValueError as e:
#                 print(f"Error loading dataset {val_dataname}: {e}")
#                 continue
#         distances = jnp.concatenate(distances, axis=0)
#         pred_coords = jnp.concatenate(pred_coords, axis=0)
#         true_coords = jnp.concatenate(true_coords, axis=0)
#         jnp.savez(os.path.join(result_path, f'{data_name}_result.npz'), distance=distance,
#                   pred_coords=pred_coords, true_coords=true_coords)
# # %%
