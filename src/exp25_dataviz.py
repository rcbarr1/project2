#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 2026

DATA VIZ FOR EXP25: Attempting to replicate map from Yamamoto et al., 2024
- load in each .nc file (one experiment per .nc file), calculate maximum cumulative additionality
- maximum cumulative additionality =  MAX( (delxCO2 * Ma) / (DIC_added_total * model_vols * rho) )
- save the additionalities to the appropriate place in the OCIM grid
- plot grids

@author: Reese C. Barrett
"""
#%%
from src.utils import project2 as p2
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import PyCO2SYS as pyco2
from tqdm import tqdm
import warnings

# load model architecture
data_path = './data/'
#output_path = './outputs/'
output_path = '/Volumes/LaCie/outputs/exp25/'

# open data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()

model_lat = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()    # ºN
model_lon = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()     # ºE
model_depth = model_data['tz'].isel(longitude=0, latitude=0).to_numpy() # m below sea surface
model_vols = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy() # m^3

model_data.close()
rho = 1025 # seawater density for volume to mass [kg m-3]
Ma = 1.8e26 # number of micromoles of air in atmosphere µmol air]

# rules for saving files
t_per_file = 2000 # number of time steps 

# surpress errors from divide by NaN because it just means land boxes here
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide"
)
#%% pull in all experiments (AT release from an individual grid cell across all grid cells)
experiment_names = []
for i in range(8400, 8601):
    experiment_names.append('exp25_2026-02-05_t-mixed_' + str(i))

# set up array to save maximum cumulative additionality in
max_alphas = np.full(ocnmask[:, :, 0].shape, np.nan)
failed_experiments = []

# calculate max_alpha for each experiment
for exp_idx in tqdm(range(len(experiment_names))):
    experiment_name = experiment_names[exp_idx]
    try:
        ds = xr.open_mfdataset(
            output_path + experiment_name + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True)

        # check that there is data until model year 100
        ds.delxCO2.sel(time=2102).values

        # calculate cumulative additionality (alpha) at each time step
        model_vols_xr = xr.DataArray(model_vols, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})
        cum_DIC_added = ds.DIC_added.cumsum(dim='time')

        # alpha = (µmol air  * (1e-6 mol/µmol) * (µmol CO2 / mol air)) / (µmol CO2 kg-1 * m3 * kg * m-3) * 100%
        alpha = np.divide((Ma * 1e-6 * ds.delxCO2), (cum_DIC_added * model_vols_xr * rho).sum(dim=['lat', 'lon', 'depth'])) * 100

        # find maximum alpha across time
        max_alpha = np.max(alpha)

        # find lat and lon of alkalinity release, store nu in array of nus at correct location
        DIC_location = np.argwhere(ds.DIC_added.isel(time=1).transpose('lat', 'lon', 'depth').values < 0)
        lats, lons, _ = DIC_location[0]
        max_alphas[lats, lons] = max_alpha

        ds.close()

    except Exception as e:
        ds.close()
        print(f"Failed: {experiment_name} -> {e}")
        failed_experiments.append(experiment_name)
        continue

#%% used to combine two separate runs shown above into one output array

#max_alphas_old = np.load(output_path + 'max_alphas.npy')

#max_alphas_full = np.nansum(np.dstack((max_alphas,max_alphas_old)),2)

np.save(output_path + 'max_alphas.npy', max_alphas)
     
#%% plot efficiency
#cmap = plt.get_cmap('viridis')

colors = ['#5d4e9f', '#5e67a2', '#607ba4', '#6393a7', '#64a9ac',
          '#65c0ae', '#8bd0b1', '#b2dfb4', '#daefb7', '#feffbb', 
          '#fee2a3', '#fcc58d', '#faa974', '#f78b5d', '#f56e46', 
          '#e45744', '#d24244', '#c12e43', '#af1843', '#9d0142']
cmap = mpl.colors.ListedColormap(colors, name='yamamoto')

p2.plot_surface2d(model_lat, model_lon, max_alphas, 0, 100, cmap, 'maximum cumulative additionality')

# %% watch what happens with single time step
ds = xr.open_dataset('./outputs/exp25_2026-02-03_t-mixed_290_000.nc')

DIC_location = np.argwhere(ds.DIC_added.isel(time=1).values < 0)
DIC_lon = DIC_location[0][1]
DIC_lat = DIC_location[0][2]

# for t_idx in tqdm(range(0, len(ds.time.values))):
for t_idx in tqdm(range(0, 4)):
    p2.plot_surface2d(model_lat, model_lon, ds.delDIC.isel(time=t_idx, depth=0).values, 0, 100, 'viridis', 'delDIC at t = ' + str(ds.time.isel(time=t_idx).values))
    # p2.plot_longitude3d(model_lat, model_depth, ds.delAT.isel(time=t_idx)+0.0001, AT_lon, 0, 10, 'viridis', 'delAT at t = ' + str(ds.time.isel(time=t_idx).values))


# %%
