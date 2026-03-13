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
ocnmask = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()

model_lat = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()    # ºN
model_lon = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()     # ºE
model_depth = model_data['tz'].isel(longitude=0, latitude=0).to_numpy() # m below sea surface
model_vols = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy() # m^3

model_data.close()
rho = 1025 # seawater density for volume to mass [kg m-3]

# rules for saving files
t_per_file = 2000 # number of time steps 
#%% pull in all experiments (AT release from an individual grid cell across all grid cells)
experiment_names = []
for i in range(4206, 10442):
    experiment_names.append('exp24_2026-02-12_t-mixed_' + f'{i:05d}')

# set up array to save nu in
nus_5years = np.full(ocnmask[:, :, 0].shape, np.nan)
nus_15years = np.full(ocnmask[:, :, 0].shape, np.nan)
failed_experiments = []

# calculate nu for each experiment
for exp_idx in tqdm(range(len(experiment_names))):
    experiment_name = experiment_names[exp_idx]
    try:
        ds = xr.open_mfdataset(
            output_path + experiment_name + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True)
        
        model_vols_xr = xr.DataArray(model_vols, dims=['lat', 'lon', 'depth'], coords={'lat': ds.lat, 'lon': ds.lon, 'depth': ds.depth})

        # convert delDIC from µmol kg-1 to mol
        delDIC = ds.delDIC * rho * model_vols_xr * 1e-6 # mol
        delDIC = delDIC.sum(dim=['lat', 'lon', 'depth'], skipna=True)

        # convert delAT from µmol kg-1 to mol
        delAT = ds.delAT * rho * model_vols_xr * 1e-6 # mol
        delAT = delAT.sum(dim=['lat', 'lon', 'depth'], skipna=True)
        
        nu = delDIC / delAT
        nu_5years = nu.sel(time=2007).values
        nu_15years = nu.sel(time=2017).values

        # find lat and lon of alkalinity release, store nu in array of nus at correct location
        alk_location = np.argwhere(ds.AT_added.isel(time=1).transpose('lat', 'lon', 'depth').values > 0)
        lats, lons, _ = alk_location[0]
        nus_5years[lats, lons] = nu_5years
        nus_15years[lats, lons] = nu_15years
        ds.close()
    except Exception as e:
        ds.close()
        print(f"Failed: {experiment_name} -> {e}")
        failed_experiments.append(experiment_name)
        continue

#%% used to combine two separate runs shown above into one output array

nus_5years_old = np.load(output_path + 'nus5yrs_dtmixed_PT1.npy')
nus_15years_old = np.load(output_path + 'nus15yrs_dtmixed_PT1.npy')

nus_5years_full = np.nansum(np.dstack((nus_5years,nus_5years_old)),2)
nus_15years_full = np.nansum(np.dstack((nus_15years, nus_15years_old)),2)

np.save(output_path + 'nus15yrs_dtmixed.npy', nus_15years_full)
np.save(output_path + 'nus5yrs_dtmixed.npy', nus_5years_full)
     
#%% plot efficiency to match zhou map
nus_5years_full = np.load(output_path + 'nus5yrs_dtmixed.npy')
nus_15years_full = np.load(output_path + 'nus15yrs_dtmixed.npy')

cmap = plt.get_cmap('viridis')
vmin = 0
vmax = 1
fig, axs = plt.subplots(2, 1, dpi=200, figsize=(6.2, 8))

# rotate lons to start at 20
split_idx = np.where(model_lon >= 20)[0][0]
model_lon_rot = np.concatenate((model_lon[split_idx:], model_lon[:split_idx] + 360))
nus_5years_rot = np.concatenate((nus_5years_full[:, split_idx:], nus_5years_full[:, :split_idx]), axis=1)
nus_15years_rot = np.concatenate((nus_15years_full[:, split_idx:], nus_15years_full[:, :split_idx]), axis=1)

# mask out zero values 
nu_5years_masked = np.ma.masked_where(nus_5years_rot == 0, nus_5years_rot)
nu_15years_masked = np.ma.masked_where(nus_15years_rot == 0, nus_15years_rot)

levels = np.linspace(vmin-0.001, vmax, 50)
cntr0 = axs[0].contourf(model_lon_rot, model_lat, nu_5years_masked, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
cntr1 = axs[1].contourf(model_lon_rot, model_lat, nu_15years_masked, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
c = fig.colorbar(cntr1, ax=axs, orientation='horizontal', pad=0.09)
c.set_ticks(np.round(np.linspace(vmin, vmax, 11),2))
c.set_label('Mean (η)')

# overlay black for land
#zero_mask = (variable == 0).astype(float)
#ax.contourf(lons, lats, zero_mask.T, levels=[0.5, 1.5], colors='black', alpha=1.0)

axs[0].get_xaxis().set_visible(False)

axs[0].set_ylabel('Latitude (ºN)')
axs[1].set_ylabel('Latitude (ºN)')
axs[1].set_xlabel('Longitude (ºE)')

axs[0].set_ylim([-80,80])
axs[1].set_ylim([-80,80])

# %% watch what happens with single time step
ds = xr.open_dataset('./outputs/exp24_TEST_000.nc')

alk_location = np.argwhere(ds.AT_added.isel(time=1).transpose('lat', 'lon', 'depth').values > 0)
AT_lat, AT_lon, _ = alk_location[0]

for t_idx in tqdm(range(0, len(ds.time.values))):
# for t_idx in tqdm(range(0, 2)):
    p2.plot_surface2d(model_lon, model_lat, ds.delDIC.isel(time=t_idx, depth=0)+0.0001, 0, 0.1, 'viridis', 'delDIC at t = ' + str(ds.time.isel(time=t_idx).values))
    # p2.plot_longitude3d(model_lat, model_depth, ds.delAT.isel(time=t_idx)+0.0001, AT_lon, 0, 10, 'viridis', 'delAT at t = ' + str(ds.time.isel(time=t_idx).values))


# %%
