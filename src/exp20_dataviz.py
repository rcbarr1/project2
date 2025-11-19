#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:43:49 2025

DATA VIZ FOR EXP20: Attempting maximum alkalinity calculation with more
efficient memory usage

@author: Reese C. Barrett
"""
#%%
from src.utils import project2 as p2
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import PyCO2SYS as pyco2
from tqdm import tqdm

# load model architecture
data_path = '/Users/Reese_1/Documents/Research Projects/project2/data/'
#output_path = '/Users/Reese_1/Documents/Research Projects/project2/outputs/'
output_path = '/Volumes/LaCie/outputs/'

# load transport matrix (OCIM2-48L, from Holzer et al., 2021)
# transport matrix is referred to as "A" vector in John et al., 2020 (AWESOME OCIM)
TR = p2.loadmat(data_path + 'OCIM2_48L_base/OCIM2_48L_base_transport.mat')
TR = TR['TR']

# open up rest of data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].to_numpy()

model_depth = model_data['tz'].to_numpy()[:, 0, 0] # m below sea surface
model_lon = model_data['tlon'].to_numpy()[0, :, 0] # ºE
model_lat = model_data['tlat'].to_numpy()[0, 0, :] # ºN
model_vols = model_data['vol'].to_numpy() # m^3

# some other important numbers
grid_cell_depth = model_data['wz'].to_numpy() # depth of model layers (need bottom of grid cell, not middle) [m]
z1 = grid_cell_depth[1, 0, 0] # depth of first model layer [m]
ns = int(np.nansum(ocnmask[0,:,:])) # number of surface grid cells
rho = 1025 # seawater density for volume to mass [kg m-3]

# rules for saving files
t_per_file = 2000 # number of time steps 

# calculate when new layers start (for line plots)
new_layer_idx = np.zeros(len(model_depth))
for i in range(len(model_depth)):
    new_layer_idx[i] = int(np.nansum(ocnmask[i,:,:]))
new_layer_idx = np.cumsum(new_layer_idx)

#%% pull in preindustrial baselines

# get GLODAP data
DIC_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC.npy') # dissolved inorganic carbon [µmol kg-1]
AT_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA.npy')   # total alkalinity [µmol kg-1]
T_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy') # temperature [ºC]
S_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy') # salinity [unitless]
Si_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/silicate.npy') # silicate [µmol kg-1]
P_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/PO4.npy') # phosphate [µmol kg-1]

S = p2.flatten(S_3D, ocnmask)
T = p2.flatten(T_3D, ocnmask)
Si = p2.flatten(Si_3D, ocnmask)
P = p2.flatten(P_3D, ocnmask)

# get TRACE data
Canth_2015 = p2.loadmat(data_path + 'TRACEv1/trace_outputs_2015.mat')
Canth_2015 = Canth_2015['trace_outputs_2015']
Canth_2015 = Canth_2015.reshape(len(model_lon), len(model_lat), len(model_depth), order='F')
Canth_2015 = Canth_2015.transpose([2, 0, 1])

# calculate preindustrial DIC by subtracting anthropogenic carbon
DIC_preind_3D = DIC_3D - Canth_2015
DIC_preind = p2.flatten(DIC_preind_3D, ocnmask)

# create "pressure" array by broadcasting depth array
pressure_3D = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[1], ocnmask.shape[2]))
pressure = p2.flatten(pressure_3D, ocnmask)

# calculate preindustrial pH assuming steady state alkalinity
co2sys = pyco2.sys(dic=DIC_preind,
                   alkalinity=p2.flatten(AT_3D, ocnmask),
                   salinity=p2.flatten(S_3D,ocnmask),
                   temperature=p2.flatten(T_3D,ocnmask),
                   pressure=p2.flatten(pressure_3D,ocnmask),
                   total_silicate=p2.flatten(Si_3D,ocnmask),
                   total_phosphate=p2.flatten(P_3D,ocnmask))

pH_preind = co2sys['pH']
pH_preind_3D = p2.make_3D(pH_preind, ocnmask)

#%% set experiments we are interested in plotting
experiment_names = ['exp20_2025-10-09-ssp_none-MLD-all_lat_lon-dt_1yr',
                    'exp20_2025-10-09-ssp_none-MLD-all_lat_lon-dt_1month',]

experiment_names = ['exp20_2025-10-15-ssp_none-MLD-all_lat_lon-dt_1year_co2sys1perc',
                    'exp20_2025-10-15-ssp_none-MLD-all_lat_lon-dt_1month_co2sys1perc',]

experiment_names = ['exp21_TEST']

labels = ['dt = 1 yr', 'dt = 1 month']

experiment_names = ['exp21_2025-10-29_20-08-08_t0_none_0.0',
                    'exp21_2025-10-29_22-58-59_t1_none_0.0',
                    'exp21_2025-10-30_03-40-45_t2_none_0.0',
                    'exp21_2025-10-30_10-26-40_t3_none_0.0',
                    'exp21_2025-10-30_16-11-42_t4_none_0.0',]


experiment_names = ['exp21_2025-11-06_t0_none',
                    'exp21_2025-11-06_t1_none',
                    'exp21_2025-11-06_t2_none',
                    'exp21_2025-11-06_t3_none',
                    'exp21_2025-11-06_t4_none',]
'''
experiment_names = ['exp21_2025-10-29_19-55-01_t0_none_0.01',
                    'exp21_2025-10-29_22-11-53_t1_none_0.01',
                    'exp21_2025-10-30_02-53-39_t2_none_0.01',
                    'exp21_2025-10-30_05-48-49_t3_none_0.01',
                    'exp21_2025-10-30_16-09-02_t4_none_0.01',]

experiment_names = ['exp21_2025-10-29_17-57-38_t0_none_0.05',
                    'exp21_2025-10-29_22-03-02_t1_none_0.05',
                    'exp21_2025-10-30_01-49-28_t2_none_0.05',
                    'exp21_2025-10-30_04-36-42_t3_none_0.05',
                    'exp21_2025-10-30_15-49-33_t4_none_0.05',]
'''
labels = ['dt = 1 yr', 'dt = 1 month', 'dt = 1 day', 'dt = 1 hour', 'dt = mixed']
'''
experiment_names = ['exp21_2025-11-05_t2_none',
                    'exp21NN_2025-11-05_t2_none']
labels=['dt = 1 day (pyco2sys)', 'dt = 1 day (NN)']
'''
#%% cumulative AT added
fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # broadcast model_vols to convert ∆AT from per kg to total
    model_vols_xr = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": ds.depth, "lon": ds.lon, "lat": ds.lat})
    
    AT_added = ds['AT_added'] * model_vols_xr * rho * 1e-6
    AT_added = AT_added.sum(dim=['depth', 'lon', 'lat'], skipna=True)
    AT_added_cum = AT_added.cumsum(dim='time')
    
    # only actually pull values into memory needed for plotting
    ax.plot(ds['time'].values, AT_added_cum.compute().values, label=labels[exp_idx])
    
plt.legend()
plt.xlabel('year')
plt.ylabel('cumulative AT added to mixed layer (mol)')
plt.ylim([0, 6.5e16])
#%% normal AT added
fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # broadcast model_vols to convert ∆AT from per kg to total
    model_vols_xr = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": ds.depth, "lon": ds.lon, "lat": ds.lat})
    
    AT_added = ds['AT_added'] * model_vols_xr * rho * 1e-6
    AT_added = AT_added.sum(dim=['depth', 'lon', 'lat'], skipna=True)
    
    ax.plot(ds['time'].values, AT_added.compute().values, label=labels[exp_idx])
    
plt.legend()
plt.xlabel('year')
plt.ylabel('AT added to mixed layer (mol)') 

#%% change in atmospheric CO2
fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # only actually pull values into memory needed for plotting
    ax.plot(ds['time'].values, ds['delxCO2'].values, label=labels[exp_idx])
    
plt.legend()
plt.xlabel('year')
plt.ylabel('change in atmospheric CO2 (ppm)')
plt.ylim([-90, 0])
#%% change in DIC (surface)
fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

# plot preindustrial baseline
ax.axhline(np.average(DIC_preind[0:ns], weights=p2.flatten(model_vols,ocnmask)[0:ns]), c='black', linestyle='--', label='preindustrial DIC')

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # wrap GLODAP DIC in xarray dataset to convert ∆DIC to total DIC over time
    DIC_ds = xr.DataArray(DIC_3D, dims=["depth", "lon", "lat"], coords={"depth": ds.depth, "lon": ds.lon, "lat": ds.lat})
    DIC_modeled_3D = ds['delDIC'] + DIC_ds
    
    # wrap model_vols in xarray dataset to convert from concentration to amount or use in weighted average
    model_vols_ds = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": ds.depth, "lon": ds.lon, "lat": ds.lat})
    DIC_weighted_mean = DIC_modeled_3D.isel(depth=0).weighted(model_vols_xr.isel(depth=0)).mean(dim=['lon', 'lat'])

    ax.plot(ds['time'].values, DIC_weighted_mean.values, label=labels[exp_idx])

plt.legend()
plt.xlabel('year')
plt.ylabel('weighted average surface ocean DIC (µmol kg$^{-1}$)')

#%% change in DIC (full ocean)
fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

# plot preindustrial baseline
ax.axhline(np.average(DIC_preind, weights=p2.flatten(model_vols,ocnmask)), c='black', linestyle='--', label='preindustrial DIC')

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # wrap GLODAP DIC in xarray dataset to convert ∆DIC to total DIC over time
    DIC_ds = xr.DataArray(DIC_3D, dims=["depth", "lon", "lat"], coords={"depth": ds.depth, "lon": ds.lon, "lat": ds.lat})
    DIC_modeled_3D = ds['delDIC'] + DIC_ds
    
    # wrap model_vols in xarray dataset to convert from concentration to amount or use in weighted average
    model_vols_ds = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": ds.depth, "lon": ds.lon, "lat": ds.lat})
    DIC_weighted_mean = DIC_modeled_3D.weighted(model_vols_xr).mean(dim=['depth','lon', 'lat'])

    ax.plot(ds['time'].values, DIC_weighted_mean.values, label=labels[exp_idx])

plt.legend()
plt.xlabel('year')
plt.ylabel('weighted average ocean DIC (µmol kg$^{-1}$)')


#%% change in pH (surface)
fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

# plot preindustrial baseline
ax.axhline(np.average(pH_preind[0:ns], weights=p2.flatten(model_vols,ocnmask)[0:ns]), c='black', linestyle='--', label='preindustrial pH')

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # wrap GLODAP DIC in xarray dataset to convert ∆DIC to total DIC over time
    DIC_ds = xr.DataArray(DIC_3D, dims=["depth", "lon", "lat"], coords={"depth": ds.depth, "lon": ds.lon, "lat": ds.lat})
    DIC_modeled_3D = ds['delDIC'] + DIC_ds
    
    # same for ∆AT to AT
    AT_ds = xr.DataArray(AT_3D, dims=["depth", "lon", "lat"], coords={"depth": ds.depth, "lon": ds.lon, "lat": ds.lat})
    AT_modeled_3D = ds['delAT'] + AT_ds

    avg_pH_modeled_surf = np.zeros(len(ds['time']))

    for idx in tqdm(range(len(ds['time']))):
        
        AT_modeled = AT_modeled_3D.isel(time=idx).isel(depth=0).values[ocnmask[0,:,:] == 1].flatten(order='F')
        DIC_modeled = DIC_modeled_3D.isel(time=idx).isel(depth=0).values[ocnmask[0,:,:] == 1].flatten(order='F')

        #  call co2sys to calculate pH
        co2sys = pyco2.sys(
            alkalinity=AT_modeled,
            dic=DIC_modeled,
            salinity=S[0:ns],
            temperature=T[0:ns],
            pressure=pressure[0:ns],
            total_silicate=Si[0:ns],
            total_phosphate=P[0:ns])
    
        avg_pH_modeled_surf[idx] = np.average(co2sys['pH'], weights=p2.flatten(model_vols,ocnmask)[0:ns])
        
    ax.plot(ds['time'].values, avg_pH_modeled_surf, label=labels[exp_idx])

plt.legend()
plt.xlabel('year')
plt.ylabel('weighted average surface ocean pH (µmol kg$^{-1}$)')

#%% change in pH (full ocean)
fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

# plot preindustrial baseline
ax.axhline(np.average(pH_preind, weights=p2.flatten(model_vols,ocnmask)), c='black', linestyle='--', label='preindustrial pH')

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # wrap GLODAP DIC in xarray dataset to convert ∆DIC to total DIC over time
    DIC_ds = xr.DataArray(DIC_3D, dims=["depth", "lon", "lat"], coords={"depth": ds.depth, "lon": ds.lon, "lat": ds.lat})
    DIC_modeled_3D = ds['delDIC'] + DIC_ds
    
    # same for ∆AT to AT
    AT_ds = xr.DataArray(AT_3D, dims=["depth", "lon", "lat"], coords={"depth": ds.depth, "lon": ds.lon, "lat": ds.lat})
    AT_modeled_3D = ds['delAT'] + AT_ds
    
    avg_pH_modeled = np.zeros(len(ds['time']))
    
    for idx in tqdm(range(len(ds['time']))):
        
        AT_modeled = p2.flatten(AT_modeled_3D.isel(time=idx).values, ocnmask)
        DIC_modeled = p2.flatten(DIC_modeled_3D.isel(time=idx).values, ocnmask)

        #  call co2sys to calculate pH
        co2sys = pyco2.sys(
            alkalinity=AT_modeled,
            dic=DIC_modeled,
            salinity=S,
            temperature=T,
            pressure=pressure,
            total_silicate=Si,
            total_phosphate=P)
        
        avg_pH_modeled[idx] = np.average(co2sys['pH'], weights=p2.flatten(model_vols,ocnmask))
    
    ax.plot(ds['time'].values, avg_pH_modeled, label=labels[exp_idx])

plt.legend()
plt.xlabel('year')
plt.ylabel('weighted average ocean pH (µmol kg$^{-1}$)')
    
#%% line plot of pressure by index
vmin = -50
vmax = 6000
plt.plot(p2.flatten(pressure_3D,ocnmask), c='gray')
plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
plt.title('pressure (dbar)')
plt.xlim([-1000, np.sum(ocnmask)+1000])
plt.ylim([vmin, vmax])
plt.show()

#%% line plot of salinity by index
vmin = 14
vmax = 41
plt.plot(p2.flatten(S_3D,ocnmask), c='skyblue')
plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
plt.title('salinity')
plt.xlim([-1000, np.sum(ocnmask)+1000])
plt.ylim([vmin, vmax])
plt.show()

#%% line plot of temperature by index
vmin = -5
vmax = 35
plt.plot(p2.flatten(T_3D,ocnmask), c='salmon')
plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
plt.title('temperature (ºC)')
plt.xlim([-1000, np.sum(ocnmask)+1000])
plt.ylim([vmin, vmax])
plt.show()

#%% line plot of silicate by index
vmin = -10
vmax = 300
plt.plot(p2.flatten(Si_3D,ocnmask), c='plum')
plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
plt.title('silicate (µmol kg-1)')
plt.xlim([-1000, np.sum(ocnmask)+1000])
plt.ylim([vmin, vmax])
plt.show()

#%% line plot of phosphate by index
vmin = -0.5
vmax = 3.6
plt.plot(p2.flatten(P_3D,ocnmask), c='mediumaquamarine')
plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
plt.title('phosphate (µmol kg-1)')
plt.xlim([-1000, np.sum(ocnmask)+1000])
plt.ylim([vmin, vmax])
plt.show()

#%% line plots of DIC by index
vmin = -500
vmax = 2500
for t_idx in range(0,len(ds.time)):
    plt.plot(p2.flatten(ds.isel(time=t_idx).delDIC.values,ocnmask) + p2.flatten(DIC_3D,ocnmask), c='steelblue')
    plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
    plt.title('DIC (µmol kg-1) at t = ' + str(t_idx))
    plt.xlim([-1000, np.sum(ocnmask)+1000])
    plt.ylim([vmin, vmax])
    plt.show()

#%% line plots of AT by index
vmin = -6000
vmax = 4000
for t_idx in range(0,len(ds.time)):
    plt.plot(p2.flatten(ds.isel(time=t_idx).delAT.values,ocnmask) + p2.flatten(AT_3D,ocnmask), c='goldenrod')
    plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
    plt.title('AT (µmol kg-1) at t = ' + str(t_idx))
    plt.xlim([-1000, np.sum(ocnmask)+1000])
    plt.ylim([vmin, vmax])
    plt.show()
    
#%% line plots of pH by index
vmin = 2
vmax = 10
for t_idx in range(0,len(ds.time)):
    co2sys = pyco2.sys(
        alkalinity=p2.flatten(ds.isel(time=t_idx).delAT.values,ocnmask) + p2.flatten(AT_3D,ocnmask),
        dic=p2.flatten(ds.isel(time=t_idx).delDIC.values,ocnmask) + p2.flatten(DIC_3D,ocnmask),
        salinity=S,
        temperature=T,
        pressure=pressure,
        total_silicate=Si,
        total_phosphate=P)
    
    pH_plot = co2sys['pH']
    
    plt.plot(pH_plot, c='lightpink')
    plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
    plt.title('pH at t = ' + str(t_idx))
    plt.xlim([-1000, np.sum(ocnmask)+1000])
    plt.ylim([vmin, vmax])
    plt.show()

#%% plot surface pH
co2sys = pyco2.sys(
    alkalinity=p2.flatten(ds.isel(time=1).delAT.values,ocnmask) + p2.flatten(AT_3D,ocnmask),
    dic=p2.flatten(ds.isel(time=1).delDIC.values,ocnmask) + p2.flatten(DIC_3D,ocnmask),
    salinity=S,
    temperature=T,
    pressure=pressure,
    total_silicate=Si,
    total_phosphate=P)

pH_plot = co2sys['pH']
pH_plot_3D = p2.make_3D(pH_plot,ocnmask)
p2.plot_surface2d(model_lon, model_lat, pH_plot_3D[0,:,:], 2, 10, 'viridis', 'pH (t = 15, depth = ' + str(np.round(model_depth[4], 2)) + ' m)')
#plt.ylim([-500, 2500])
plt.show()

#%% line plots of AT added by index
vmin = -50
vmax = 250
for t_idx in range(0,len(ds.time)):
    plt.plot(p2.flatten(ds.isel(time=t_idx).AT_added.values,ocnmask), c='darkgoldenrod')
    plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
    plt.title('AT added (µmol kg-1) at t = ' + str(t_idx))
    plt.xlim([-1000, np.sum(ocnmask)+1000])
    plt.ylim([vmin, vmax])
    plt.show()





