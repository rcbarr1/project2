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
#output_path = './outputs/'
output_path = '/Volumes/LaCie/outputs/exp24/'

# open data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].to_numpy()

model_depth = model_data['tz'].to_numpy()[:, 0, 0] # m below sea surface
model_lon = model_data['tlon'].to_numpy()[0, :, 0] # ºE
model_lat = model_data['tlat'].to_numpy()[0, 0, :] # ºN
model_vols = model_data['vol'].to_numpy() # m^3

model_data.close()
rho = 1025 # seawater density for volume to mass [kg m-3]

# rules for saving files
t_per_file = 2000 # number of time steps 
#%% pull in all experiments (AT release from an individual grid cell across all grid cells)
experiment_names = []
for i in range(1110, 1600):
    experiment_names.append('exp24_2026-01-28_t-mixed_' + str(i))

# set up array to save nu in
nus_5years = np.full(ocnmask[0, :, :].shape, np.nan)
nus_15years = np.full(ocnmask[0, :, :].shape, np.nan)

# calculate nu for each experiment
for exp_idx in tqdm(range(len(experiment_names))):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    model_vols_xr = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": ds.depth, "lon": ds.lon, "lat": ds.lat})

    # convert delDIC from µmol kg-1 to mol
    delDIC = ds.delDIC * rho * model_vols_xr * 1e-6 # mol
    delDIC = delDIC.sum(dim=['depth', 'lon', 'lat'], skipna=True)

    # convert delAT from µmol kg-1 to mol
    delAT = ds.delAT * rho * model_vols_xr * 1e-6 # mol
    delAT = delAT.sum(dim=['depth', 'lon', 'lat'], skipna=True)
    
    nu = delDIC / delAT
    nu_5years = nu.sel(time=2020).values
    nu_15years = nu.sel(time=2030).values

    # find lat and lon of alkalinity release, store nu in array of nus at correct location
    alk_location = np.argwhere(ds.AT_added.isel(time=1).values > 0)
    nus_5years[alk_location[0][1], alk_location[0][2]] = nu_5years 
    nus_15years[alk_location[0][1], alk_location[0][2]] = nu_15years

#%% used to combine two separate runs shown above into one output array

#nus_5years_old = np.load(output_path + 'nus5yrs_dt1yr.npy')
#nus_15years_old = np.load(output_path + 'nus15yrs_dt1yr.npy')

#nus_5years_full = np.nansum(np.dstack((nus_5years,nus_5years_old)),2)
#nus_15years_full = np.nansum(np.dstack((nus_15years, nus_15years_old)),2)

#np.save(output_path + 'nus15yrs_dt1yr.npy', nus_15years_full)
#np.save(output_path + 'nus5yrs_dt1yr.npy', nus_5years_full)
     
#%% plot efficiency
cmap = plt.get_cmap("viridis", 11)
p2.plot_surface2d(model_lon, model_lat, nus_5years, 0.3, 0.9, cmap, 'efficiency at t = 5 years')
p2.plot_surface2d(model_lon, model_lat, nus_15years, 0.3, 0.9, cmap, 'efficiency at t = 15 years')

# %% watch what happens with single time step
ds = xr.open_dataset('./outputs/exp24_2026-01-28_t-mixed_1116_000.nc')

alk_location = np.argwhere(ds.AT_added.isel(time=1).values > 0)
AT_lon = alk_location[0][1]
AT_lat = alk_location[0][2]

for t_idx in tqdm(range(0, len(ds.time.values))):
# for t_idx in tqdm(range(0, 2)):
    p2.plot_surface2d(model_lon, model_lat, ds.delDIC.isel(time=t_idx, depth=0)+0.0001, 0, 0.1, 'viridis', 'delDIC at t = ' + str(ds.time.isel(time=t_idx).values))
    # p2.plot_longitude3d(model_lat, model_depth, ds.delAT.isel(time=t_idx)+0.0001, AT_lon, 0, 10, 'viridis', 'delAT at t = ' + str(ds.time.isel(time=t_idx).values))


# %%
