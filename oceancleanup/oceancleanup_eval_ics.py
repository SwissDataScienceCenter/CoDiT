import os
import numpy as np
from datetime import timedelta, datetime
from codit.data_module import config_paths, config_funcs
import logging
from codit.data_module.dataset import OceanDataset
import pandas as pd
from argparse import ArgumentParser
from copy import copy
from codit.model_module import CoDiTModel
from codit.model_module.linear_gauss_model import LinearGaussModel
from codit.model_module.hycom_model import HycomModel
from jax import numpy as jnp

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--year", default=2020, type=int)
    #parser.add_argument("--half", default=1, type=int, help="1 for first half, 2 for second half")
    parser.add_argument("--delta_coord", default=2, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    # parser.add_argument("--num_trajectories", default=-1, type=int)
    parser.add_argument("--prediction_length", default=72, type=int)
    args = parser.parse_args()
    
    dataset_name = 'oceancleanup'
    data_split = f'oceancleanup{args.year}'
    np.random.seed(0)
    result_path = os.path.join(
            config_paths.RESULT_PATH,
            dataset_name,
            'codit_analysis',
            data_split,
        )
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    config = config_funcs.load_config(data_split, os.path.join(config_paths.CONFIG_PATH, dataset_name))
    min_lat, max_lat = config['latitudes']
    min_lon, max_lon = config['longitudes']
    name = config['name']

    odf_path = os.path.join(config_paths.DATA_PATH, 'datasets', 'oceancleanup', 'ics_df.csv')
    ics_df = pd.read_csv(odf_path)
    ics_df = ics_df.rename(columns={'Unnamed: 0': 'idx'})
    # filter by date, find those in the year
    # ics_df['date'] = pd.to_datetime(ics_df['time'])
    ics_df_year = ics_df[(ics_df['time'] >= f'{args.year}-01-01') & (ics_df['time'] < f'{args.year+1}-01-01')]
    export_path = os.path.join(config_paths.MODEL_PATH, 'trained_models/')
    segments = ics_df_year.groupby('segment_id')
    export_path = os.path.join(config_paths.MODEL_PATH, 'trained_models/')
    model_name = 'codit_Transformer_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry'
    codit_model = CoDiTModel.import_trained_model(model_name, import_path=export_path)
    hycom_model = HycomModel(name='hycom', params={})
    linear_gauss_model = LinearGaussModel(name="linear_gauss_open_ocean", params={})
    linear_gauss_model.import_trained_model('linear_gauss_open_ocean')
    
    for id, segment in segments:
        segment = segment[1:]
        logging.info(f"Processing segment ID: {id} with {len(segment)} points")
        for idx in range(0, len(segment)-4, 4):
            seg_part = segment.iloc[idx:idx+4]
            start_seg = seg_part.iloc[0]   
            traj_id = start_seg['id']
            traj_config = copy(config)
            traj_config['start_date'] = (datetime.strptime(start_seg['time'], '%Y-%m-%d %H:%M:%S') - timedelta(days=1)).strftime('%Y-%m-%d')
            traj_config['end_date'] = (datetime.strptime(start_seg['time'], '%Y-%m-%d %H:%M:%S') + timedelta(days=4)).strftime('%Y-%m-%d')
            traj_config['name'] = f"{name}_traj_{traj_id}_{idx}"
            traj_config['latitudes'] = [start_seg['lat'] - args.delta_coord, start_seg['lat'] + args.delta_coord]
            traj_config['longitudes'] = [start_seg['lon'] - args.delta_coord, start_seg['lon'] + args.delta_coord]
            logging.info(f"Processing trajectory {traj_config['name']}: {idx+1}/{len(segment) // 4}")
            result_file = os.path.join(result_path, f"{traj_config['name']}_result.npz")
            if os.path.exists(result_file):
                logging.info(f"Result file {result_file} already exists. Skipping...")
                continue
            dataset = OceanDataset(traj_config, length=args.prediction_length)
            start_coord = jnp.array(start_seg[['lat', 'lon']].to_list())
            start_time = datetime.strptime(start_seg['time'], '%Y-%m-%d %H:%M:%S')
            end_time = start_time + timedelta(hours=args.prediction_length+2)
            interpolator = dataset.get_interpolator(min_date=start_time, max_date=end_time)
            hycom_predictions = hycom_model.predict(
                start_coord, 
                start_time, 
                args.prediction_length, 
                interpolator, 
                dataset
            )
            linear_gauss_predictions = linear_gauss_model.predict(
                        start_coord, 
                        start_time, 
                        args.prediction_length, 
                        interpolator, 
                        dataset, 
                        num_samples=100
                    )
            codit_predictions = codit_model.predict(
                start_coord,
                start_time,
                prediction_length=args.prediction_length,
                interpolator=interpolator,
                test_dataset=dataset,
                num_samples=args.num_samples,
            )
            # save
            jnp.savez(
                result_file,
                hycom_predictions=hycom_predictions,
                linear_gauss_predictions=linear_gauss_predictions,
                codit_predictions=codit_predictions,
                true_coords=seg_part[['lat', 'lon']].to_numpy(),
                start_coord=start_coord,
                #start_time=jnp.array(traj_df['date'].strftime('%Y-%m-%dT%H:%M:%S')),
                ics_df_idx=jnp.array(start_seg['idx']),           
            )
            logging.info(f"Saved results to {result_file}")