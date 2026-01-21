#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 2026

DATA VIZ FOR EXP24: Attempting to replicate map from Zhou et al., 2025
- load in each .nc file (one experiment per .nc file), calculate efficiency
--> off the dome this is total delDIC / total delAT at 5 and 15 years, then there's some sort of standard deviation
- save the efficiencies (nu) and std. deviations to the appropriate place in the OCIM grid
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

# load model architecture
data_path = './data/'
output_path = './outputs/'
#output_path = '/Volumes/LaCie/outputs/'

# open data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].to_numpy()

model_depth = model_data['tz'].to_numpy()[:, 0, 0] # m below sea surface
model_lon = model_data['tlon'].to_numpy()[0, :, 0] # ºE
model_lat = model_data['tlat'].to_numpy()[0, 0, :] # ºN
model_vols = model_data['vol'].to_numpy() # m^3

model_data.close()

# rules for saving files
t_per_file = 2000 # number of time steps 
#%% pull in all experiments (AT release from an individual grid cell across all grid cells)
experiment_names = []
for i in range(2000, 2001):
    experiment_names.append('exp24_2026-01-16_t0_' + str(i))

# set up array to save nu in
nus_5years = np.full(ocnmask[0, :, :].shape, np.nan)
nus_15years = np.full(ocnmask[0, :, :].shape, np.nan)

# calculate nu for each experiment
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)

    delDIC = ds.delDIC.sum(dim=['depth', 'lon', 'lat'], skipna=True)    
    delAT = ds.delAT.sum(dim=['depth', 'lon', 'lat'], skipna=True)    

    nu = delDIC / delAT
    nu_5years = nu.isel(time=5).values
    nu_15years = nu.isel(time=15).values

    # find lat and lon of alkalinity release, store nu in array of nus at correct location
    alk_location = np.argwhere(ds.AT_added.isel(time=1).values > 0)
    nus_5years[alk_location[0][1], alk_location[0][2]] = nu_5years 
    nus_15years[alk_location[0][1], alk_location[0][2]] = nu_15years 
     
#%% plot efficiency
p2.plot_surface2d(model_lon, model_lat, nus_5years, 0, 1, 'viridis', 'efficiency at t = 5 years')
p2.plot_surface2d(model_lon, model_lat, nus_15years, 0, 1, 'viridis', 'efficiency at t = 15 years')

# %%
