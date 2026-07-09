import os
import numpy as np
from datetime import timedelta
from codit.data_module import config_paths, config_funcs
import logging
from codit.data_module.dataset import OceanDataset
import pandas as pd
from argparse import ArgumentParser
from copy import copy
from codit.model_module import CoDiTModel
from jax import numpy as jnp

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--year", default=2020, type=int)
    #parser.add_argument("--half", default=1, type=int, help="1 for first half, 2 for second half")
    parser.add_argument("--delta_coord", default=2, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--num_trajectories", default=-1, type=int)
    parser.add_argument("--prediction_length", default=72, type=int)
    args = parser.parse_args()
    
    dataset_name = 'oceancleanup'
    data_split = f'oceancleanup{args.year}'
    model_name = 'codit_Transformer_open_ocean_conv_adam_K100_hycom_current_copernicus_current_era5_wind_bathymetry'
    np.random.seed(0)
    result_path = os.path.join(
            config_paths.RESULT_PATH,
            dataset_name,
            data_split, 
            model_name,
        )
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    config = config_funcs.load_config(data_split, os.path.join(config_paths.CONFIG_PATH, dataset_name))
    min_lat, max_lat = config['latitudes']
    min_lon, max_lon = config['longitudes']
    name = config['name']

    odf_path = os.path.join(config_paths.DATA_PATH, 'datasets', 'oceancleanup', 'ics_df.csv_0.ods')
    ics_df = pd.read_excel(odf_path, engine='odf')
    ics_df = ics_df.rename(columns={'Unnamed: 0': 'idx'})
    # filter by date, find those in the year
    ics_df['date'] = pd.to_datetime(ics_df['time'])
    # if args.half == 1:
    #     ics_df_year = ics_df[(ics_df['date'] >= f'{args.year}-01-01') & (ics_df['date'] < f'{args.year}-07-01')]
    #     config['start_date'] = f'{args.year}-01-01'
    #     config['end_date'] = f'{args.year}-07-05'
    # else:
    #     ics_df_year = ics_df[(ics_df['date'] >= f'{args.year}-07-01') & (ics_df['date'] < f'{args.year+1}-01-01')]
    #     config['start_date'] = f'{args.year}-07-01'
    #     config['end_date'] = f'{args.year+1}-01-05'
    ics_df_year = ics_df[(ics_df['date'] >= f'{args.year}-01-01') & (ics_df['date'] < f'{args.year+1}-01-01')]
    # dataset = OceanDataset(config, length=args.prediction_length)
    export_path = os.path.join(config_paths.MODEL_PATH, 'trained_models/')
    model = CoDiTModel.import_trained_model(model_name, import_path=export_path)
    if args.num_trajectories > 0:
        indices = np.random.choice(np.arange(len(ics_df_year)), size=args.num_trajectories, replace=False)
    else:
        indices = np.arange(len(ics_df_year))
    for i, idx in enumerate(indices):
        traj_df = ics_df_year.iloc[idx]
        traj_id = traj_df['id']
        traj_config = copy(config)
        traj_config['start_date'] = (traj_df['date'] - timedelta(days=1)).strftime('%Y-%m-%d')
        traj_config['end_date'] = (traj_df['date'] + timedelta(days=4)).strftime('%Y-%m-%d')
        traj_config['name'] = f"{name}_traj_{traj_id}_{idx}"
        traj_config['latitudes'] = [traj_df['lat'] - args.delta_coord, traj_df['lat'] + args.delta_coord]
        traj_config['longitudes'] = [traj_df['lon'] - args.delta_coord, traj_df['lon'] + args.delta_coord]
        logging.info(f"Processing trajectory {traj_config['name']}: {i+1}/{len(indices)}")
        result_file = os.path.join(result_path, f"{traj_config['name']}_result.npz")
        if os.path.exists(result_file):
            logging.info(f"Result file {result_file} already exists. Skipping...")
            continue
        dataset = OceanDataset(traj_config, length=args.prediction_length)
        start_coord = jnp.array(traj_df[['lat', 'lon']].to_list())
        start_time = traj_df['date'].to_pydatetime()
        end_time = start_time + timedelta(hours=args.prediction_length+2)
        interpolator = dataset.get_interpolator(min_date=start_time, max_date=end_time)
        predictions = model.predict(
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
            predictions=predictions,
            start_coord=start_coord,
            #start_time=jnp.array(traj_df['date'].strftime('%Y-%m-%dT%H:%M:%S')),
            ics_df_idx=traj_df['idx'],            
        )
        logging.info(f"Saved results to {result_file}")