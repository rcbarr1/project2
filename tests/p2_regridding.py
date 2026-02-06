#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using to evaluate success of regridding and inpainting techniques.

NOTE: for 3D, need to make sure you're actually comparing depth in (m) and not
indicies (i.e. depth_idx = 10 is not the same depth in meters between OCIM and
other grids!)

@author: Reese C. Barrett
"""
#%%
from src.utils import project2 as p2
import numpy as np
import xarray as xr

# load model architecture
data_path = './data/'
output_path = './outputs/'
cobalt_path = '/Volumes/LaCie/data/OM4p25_cobalt_v3/'

# load transport matrix (OCIM2-48L, from Holzer et al., 2021)
# transport matrix is referred to as "A" vector in John et al., 2020 (AWESOME OCIM)
TR = p2.loadmat(data_path + 'OCIM2_48L_base/OCIM2_48L_base_transport.mat')
TR = TR['TR']

# open up rest of data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()

model_lat = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()    # ºN
model_lon = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()     # ºE
model_depth = model_data['tz'].isel(longitude=0, latitude=0).to_numpy() # m below sea surface
model_vols = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy() # m^3

# grid cell z-dimension for converting from surface area to volume
grid_cell_depth = model_data['wz'].transpose('latitude', 'longitude', 'depth').to_numpy() # depth of model layers (need bottom of grid cell, not middle) [m]
rho = 1025 # seawater density for volume to mass [kg m-3]
#%% WOA data

# regrid WOA18 data with new inpainting
p2.regrid_woa(data_path, 'S', model_lat, model_lon, model_depth, ocnmask)
p2.regrid_woa(data_path, 'T', model_lat, model_lon, model_depth, ocnmask)
p2.regrid_woa(data_path, 'Si', model_lat, model_lon, model_depth, ocnmask)
p2.regrid_woa(data_path, 'P', model_lat, model_lon, model_depth, ocnmask)

# upload regridded WOA18 data

# directly from WOA (not regridded, for comparison)
S_3D_WOA = xr.open_dataset(data_path + 'WOA18/woa18_decav81B0_s00_01.nc', decode_times=False).s_an
S_3D_WOA['lon'] = (S_3D_WOA['lon'] + 360) % 360 # convert
S_3D_WOA = S_3D_WOA.sortby('lon') # resort

T_3D_WOA = xr.open_dataset(data_path + 'WOA18/woa18_decav81B0_t00_01.nc', decode_times=False).t_an
T_3D_WOA['lon'] = (T_3D_WOA['lon'] + 360) % 360 # convert
T_3D_WOA = T_3D_WOA.sortby('lon') # resort

Si_3D_WOA = xr.open_dataset(data_path + 'WOA18/woa18_all_i00_01.nc', decode_times=False).i_an
Si_3D_WOA['lon'] = (Si_3D_WOA['lon'] + 360) % 360 # convert
Si_3D_WOA = Si_3D_WOA.sortby('lon') # resort

P_3D_WOA = xr.open_dataset(data_path + 'WOA18/woa18_all_p00_01.nc', decode_times=False).p_an
P_3D_WOA['lon'] = (P_3D_WOA['lon'] + 360) % 360 # convert
P_3D_WOA = P_3D_WOA.sortby('lon') # resort

# # regridded and inpainted with old inpainting
# S_3D_OCIM_OLD = np.load(data_path + 'WOA18/S_AO_OLD.npy')   # salinity [unitless]
# T_3D_OCIM_OLD = np.load(data_path + 'WOA18/T_AO_OLD.npy')   # temperature [ºC]
# Si_3D_OCIM_OLD = np.load(data_path + 'WOA18/Si_AO_OLD.npy') # silicate [µmol kg-1]
# P_3D_OCIM_OLD = np.load(data_path + 'WOA18/P_AO_OLD.npy')   # phosphate [µmol kg-1]

# regridded and inpainted with new inpainting
S_3D_OCIM = np.load(data_path + 'WOA18/S.npy')   # salinity [unitless]
T_3D_OCIM = np.load(data_path + 'WOA18/T.npy')   # temperature [ºC]
Si_3D_OCIM = np.load(data_path + 'WOA18/Si.npy') # silicate [µmol kg-1]
P_3D_OCIM = np.load(data_path + 'WOA18/P.npy')   # phosphate [µmol kg-1]

# plot data for comparison
p2.plot_surface3d(S_3D_WOA.lat.values, S_3D_WOA.lon.values, np.transpose(S_3D_WOA.values[0, :, :, :], (1, 2, 0)), 0, 25, 38, 'magma', 'WOA salinity distribution WOA GRID')
p2.plot_surface3d(model_lat, model_lon, S_3D_OCIM, 0, 25, 38, 'magma', 'WOA salinity distribution NEW')

p2.plot_surface3d(T_3D_WOA.lat.values, T_3D_WOA.lon.values, np.transpose(T_3D_WOA.values[0, :, :, :], (1, 2, 0)), 0, -10, 35, 'magma', 'WOA temp distribution WOA GRID')
p2.plot_surface3d(model_lat, model_lon, T_3D_OCIM, 0, -10, 35, 'magma', 'WOA temp distribution NEW')

p2.plot_surface3d(Si_3D_WOA.lat.values, Si_3D_WOA.lon.values, np.transpose(Si_3D_WOA.values[0, :, :, :], (1, 2, 0)), 0, 0, 30, 'magma', 'WOA silicate distribution WOA GRID')
p2.plot_surface3d(model_lat, model_lon, Si_3D_OCIM, 0, 0, 30, 'magma', 'WOA silicate distribution NEW')

p2.plot_surface3d(P_3D_WOA.lat.values, P_3D_WOA.lon.values, np.transpose(P_3D_WOA.values[0, :, :, :], (1, 2, 0)), 0, 0, 2.5, 'magma', 'WOA phosphate distribution WOA GRID')
p2.plot_surface3d(model_lat, model_lon, P_3D_OCIM, 0, 0, 2.5, 'magma', 'WOA phosphate distribution NEW')

#%% NCEP/DOE reanalysis II data & NOAA SST reconstruction

# regrid NCEP & NOAA data with new inpainting
p2.regrid_ncep_noaa(data_path, 'icec', model_lat, model_lon, ocnmask)
p2.regrid_ncep_noaa(data_path, 'wspd', model_lat, model_lon, ocnmask)
p2.regrid_ncep_noaa(data_path, 'sst', model_lat, model_lon, ocnmask)

# upload regridded NCEP & NOAA data

# directly from NCEP/NOAA (not regridded, for comparison)
f_ice_2D_NCEP = xr.open_dataset(data_path + 'NCEP_DOE_Reanalysis_II/icec.sfc.mon.ltm.1991-2020.nc').icec.mean(dim='time', skipna=True) # average across all months, pull out values from NCEP
wspd_2D_NCEP = xr.open_dataset(data_path + 'NCEP_DOE_Reanalysis_II/wspd.10m.mon.mean.nc').wspd.mean(dim='time', skipna=True) # average across all months, pull out values from NCEP
sst_2D_NOAA = xr.open_dataset(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst.mon.ltm.1991-2020.nc').sst.mean(dim='time', skipna=True) # average across all months, pull out values from NOAA

# # regridded and inpainted with old inpainting
# f_ice_2D_OCIM_OLD = np.load(data_path + 'NCEP_DOE_Reanalysis_II/icec_AO_OLD.npy') # annual mean ice fraction from 0 to 1 in each grid cell
# uwnd_2D_OCIM_OLD = np.load(data_path + 'NCEP_DOE_Reanalysis_II/uwnd_AO_OLD.npy') # annual mean of forecast of U-wind at 10 m [m/s]
# sst_2D_OCIM_OLD = np.load(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst_AO_OLD.npy') # annual mean sea surface temperature [ºC]

# regridded and inpainted with new inpainting
f_ice_2D_OCIM = np.load(data_path + 'NCEP_DOE_Reanalysis_II/icec.npy') # annual mean ice fraction from 0 to 1 in each grid cell
wspd_2D_OCIM = np.load(data_path + 'NCEP_DOE_Reanalysis_II/wspd.npy') # annual mean of forecast of U-wind at 10 m [m/s]
sst_2D_OCIM = np.load(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst.npy') # annual mean sea surface temperature [ºC]

# plot data for comparison
p2.plot_surface2d(f_ice_2D_NCEP.lat.values, f_ice_2D_NCEP.lon.values, f_ice_2D_NCEP.values, 0, 1, 'magma', 'NCEP ice fraction WOA GRID')
p2.plot_surface2d(model_lat, model_lon, f_ice_2D_OCIM, 0, 1, 'magma', 'NCEP ice fraction NEW')

p2.plot_surface2d(wspd_2D_NCEP.lat.values, wspd_2D_NCEP.lon.values, wspd_2D_NCEP.values, -12, 12, 'RdBu', 'NCEP wspd WOA GRID')
p2.plot_surface2d(model_lat, model_lon, wspd_2D_OCIM, -12, 12, 'RdBu', 'NCEP wspd NEW')

p2.plot_surface2d(sst_2D_NOAA.lat.values, sst_2D_NOAA.lon.values, sst_2D_NOAA.values, -5, 35, 'magma', 'NOAA sst WOA grid')
p2.plot_surface2d(model_lat, model_lon, sst_2D_OCIM, -5, 35, 'magma', 'NOAA sst NEW')

#%% GLODAP data

# regrid GLODAP data
p2.regrid_glodap(data_path, 'temperature', model_lat, model_lon, model_depth, ocnmask)
p2.regrid_glodap(data_path, 'salinity', model_lat, model_lon, model_depth, ocnmask)
p2.regrid_glodap(data_path, 'TCO2', model_lat, model_lon, model_depth, ocnmask)
p2.regrid_glodap(data_path, 'TAlk', model_lat, model_lon, model_depth, ocnmask)
p2.regrid_glodap(data_path, 'silicate', model_lat, model_lon, model_depth, ocnmask)
p2.regrid_glodap(data_path, 'PO4', model_lat, model_lon, model_depth, ocnmask)

# upload regridded GLODAP data

# directly from GLODAP (not regridded, for comparison)
T_3D_GLODAP = xr.open_dataset(data_path + 'GLODAPv2.2016b.MappedProduct/GLODAPv2.2016b.temperature.nc').temperature
S_3D_GLODAP = xr.open_dataset(data_path + 'GLODAPv2.2016b.MappedProduct/GLODAPv2.2016b.salinity.nc').salinity
DIC_3D_GLODAP = xr.open_dataset(data_path + 'GLODAPv2.2016b.MappedProduct/GLODAPv2.2016b.TCO2.nc').TCO2
AT_3D_GLODAP = xr.open_dataset(data_path + 'GLODAPv2.2016b.MappedProduct/GLODAPv2.2016b.TAlk.nc').TAlk
Si_3D_GLODAP = xr.open_dataset(data_path + 'GLODAPv2.2016b.MappedProduct/GLODAPv2.2016b.silicate.nc').silicate
P_3D_GLODAP = xr.open_dataset(data_path + 'GLODAPv2.2016b.MappedProduct/GLODAPv2.2016b.PO4.nc').PO4

# # regridded and inpainted with old inpainting
# DIC_3D_OCIM_OLD = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC_AO_OLD.npy') # dissolved inorganic carbon [µmol kg-1]
# AT_3D_OCIM_OLD = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA_AO_OLD.npy')   # total alkalinity [µmol kg-1]

# regridded and inpainted with new inpainting
T_3D_OCIM = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy') # temperature [ºC]
S_3D_OCIM = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy')   # salinity [unitless]
DIC_3D_OCIM = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC.npy') # dissolved inorganic carbon [µmol kg-1]
AT_3D_OCIM = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA.npy')   # total alkalinity [µmol kg-1]
Si_3D_OCIM = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/silicate.npy') # silicate [µmol kg-1]
P_3D_OCIM = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/PO4.npy')   # phosphate [µmol kg-1]

# plot data for comparison
#model_lon[model_lon < 20] += 360 # make ocim coordinates match glodap coordinates temporarily

p2.plot_surface3d(T_3D_GLODAP.lat.values, T_3D_GLODAP.lon.values, np.transpose(T_3D_GLODAP.values, (1, 2, 0)), 0, -5, 30, 'magma', 'T GLODAP GRID', lon_lims=[20, 380])
p2.plot_surface3d(model_lat, model_lon, T_3D_OCIM, 0, -5, 30, 'magma', 'T NEW', lon_lims=[0, 360])

p2.plot_surface3d(S_3D_GLODAP.lat.values, S_3D_GLODAP.lon.values, np.transpose(S_3D_GLODAP.values, (1, 2, 0)), 0, 10, 40, 'magma', 'S GLODAP GRID', lon_lims=[20, 380])
p2.plot_surface3d(model_lat, model_lon, S_3D_OCIM, 0, 10, 40, 'magma', 'S NEW', lon_lims=[0, 360])

p2.plot_surface3d(DIC_3D_GLODAP.lat.values, DIC_3D_GLODAP.lon.values, np.transpose(DIC_3D_GLODAP.values, (1, 2, 0)), 0, 1800, 2500, 'magma', 'DIC GLODAP GRID', lon_lims=[20, 380])
p2.plot_surface3d(model_lat, model_lon, DIC_3D_OCIM, 0, 1800, 2500, 'magma', 'DIC NEW', lon_lims=[0, 360])

p2.plot_surface3d(AT_3D_GLODAP.lat.values, AT_3D_GLODAP.lon.values, np.transpose(AT_3D_GLODAP.values, (1, 2, 0)), 0, 1800, 2500, 'magma', 'AT GLODAP GRID', lon_lims=[20, 380])
p2.plot_surface3d(model_lat, model_lon, AT_3D_OCIM, 0, 1800, 2500, 'magma', 'AT NEW', lon_lims=[0, 360])

p2.plot_surface3d(Si_3D_GLODAP.lat.values, Si_3D_GLODAP.lon.values, np.transpose(Si_3D_GLODAP.values, (1, 2, 0)), 0, 0, 30, 'magma', 'Si GLODAP GRID', lon_lims=[20, 380])
p2.plot_surface3d(model_lat, model_lon, Si_3D_OCIM, 0, 0, 30, 'magma', 'Si NEW', lon_lims=[0, 360])

p2.plot_surface3d(P_3D_GLODAP.lat.values, P_3D_GLODAP.lon.values, np.transpose(P_3D_GLODAP.values, (1, 2, 0)), 0, 0, 2.5, 'magma', 'P GLODAP GRID', lon_lims=[20, 380])
p2.plot_surface3d(model_lat, model_lon, P_3D_OCIM, 0, 0, 2.5, 'magma', 'P NEW', lon_lims=[0, 360])

#model_lon[model_lon > 360] -= 360 # revert ocim coordinates back

#%% COBALT data

cobalt = xr.open_dataset(cobalt_path + '19580101.ocean_cobalt_fluxes_int.nc', decode_cf=False)

q_diss_arag_plus_btm = cobalt.jdiss_cadet_arag_plus_btm # [mol CACO3 m-2 s-1]
q_diss_calc_plus_btm = cobalt.jdiss_cadet_calc_plus_btm # [mol CACO3 m-2 s-1]
#q_diss_arag = cobalt.jdiss_cadet_arag # [mol CACO3 m-2 s-1]
q_diss_calc = cobalt.jdiss_cadet_calc # [mol CACO3 m-2 s-1]
q_prod_arag = cobalt.jprod_cadet_arag # [mol CACO3 m-2 s-1]
q_prod_calc = cobalt.jprod_cadet_calc # [mol CACO3 m-2 s-1]

data = [q_diss_calc, q_prod_arag, q_prod_calc] # skipping diss_arag for now because I don't have it and plus_btm for now because they're wrong

# regrid COBALT data
for d in data:
    p2.regrid_cobalt(d, model_lat, model_lon, model_depth, ocnmask, output_path)

# pull out cobalt data, make plottable
PROD_CALC_3D_COBALT = q_prod_calc.copy()
PROD_CALC_3D_COBALT = PROD_CALC_3D_COBALT.where(PROD_CALC_3D_COBALT != 1e20) # replace 1e+20 values with np.NaN
PROD_CALC_3D_COBALT = PROD_CALC_3D_COBALT.mean(dim='time', skipna=True) # average across time
PROD_CALC_3D_COBALT['xh'] = (PROD_CALC_3D_COBALT['xh'] + 360) % 360 # convert
PROD_CALC_3D_COBALT = PROD_CALC_3D_COBALT.sortby('xh') # resort

PROD_ARAG_3D_COBALT = q_prod_arag.copy()
PROD_ARAG_3D_COBALT = PROD_ARAG_3D_COBALT.where(PROD_ARAG_3D_COBALT != 1e20) # replace 1e+20 values with np.NaN
PROD_ARAG_3D_COBALT = PROD_ARAG_3D_COBALT.mean(dim='time', skipna=True) # average across time
PROD_ARAG_3D_COBALT['xh'] = (PROD_ARAG_3D_COBALT['xh'] + 360) % 360 # convert
PROD_ARAG_3D_COBALT = PROD_ARAG_3D_COBALT.sortby('xh') # resort

DISS_CALC_3D_COBALT = q_diss_calc.copy()
DISS_CALC_3D_COBALT = DISS_CALC_3D_COBALT.where(DISS_CALC_3D_COBALT != 1e20) # replace 1e+20 values with np.NaN
DISS_CALC_3D_COBALT = DISS_CALC_3D_COBALT.mean(dim='time', skipna=True) # average across time
DISS_CALC_3D_COBALT['xh'] = (DISS_CALC_3D_COBALT['xh'] + 360) % 360 # convert
DISS_CALC_3D_COBALT = DISS_CALC_3D_COBALT.sortby('xh') # resort

cobalt_lat = PROD_CALC_3D_COBALT['yh'].to_numpy()     # ºN (-80 to +90)
cobalt_lon = PROD_CALC_3D_COBALT['xh'].to_numpy()     # ºE (originally -300 to +60, now 0 to 360)
cobalt_depth = PROD_CALC_3D_COBALT['zl'].to_numpy() # m below sea surface
       
# # regridded and inpainted with old inpainting
# PROD_CALC_3D_OCIM_OLD = np.load(data_path + 'COBALT_regridded/jprod_cadet_calc_OLD.npy')
# PROD_ARAG_3D_OCIM_OLD = np.load(data_path + 'COBALT_regridded/jprod_cadet_arag_OLD.npy')
# DISS_CALC_3D_OCIM_OLD = np.load(data_path + 'COBALT_regridded/jdiss_cadet_calc_OLD.npy')

# regridded and inpainted with new inpainting
PROD_CALC_3D_OCIM = np.load(data_path + 'COBALT_regridded/jprod_cadet_calc.npy')
PROD_ARAG_3D_OCIM = np.load(data_path + 'COBALT_regridded/jprod_cadet_arag.npy')
DISS_CALC_3D_OCIM = np.load(data_path + 'COBALT_regridded/jdiss_cadet_calc.npy')

# plot data for comparison
p2.plot_surface3d(cobalt_lat, cobalt_lon, np.transpose(PROD_CALC_3D_COBALT.values, (1, 2, 0)), 2, 1e-17, 1e-10, 'magma', 'PROD CALC COBALT', logscale=True, lon_lims=[0, 360])
p2.plot_surface3d(model_lat, model_lon, PROD_CALC_3D_OCIM, 0, 1e-17, 1e-10, 'magma', 'PROD CALC OCIM', logscale=True, lon_lims=[0, 360])

# plot data for comparison
p2.plot_surface3d(cobalt_lat, cobalt_lon, np.transpose(PROD_ARAG_3D_COBALT.values, (1, 2, 0)), 37, 1e-17, 1e-10, 'magma', 'PROD ARAG COBALT', logscale=True, lon_lims=[0, 360])
p2.plot_surface3d(model_lat, model_lon, PROD_ARAG_3D_OCIM, 10, 1e-17, 1e-10, 'magma', 'PROD ARAG OCIM', logscale=True, lon_lims=[0, 360])

# plot data for comparison
p2.plot_surface3d(cobalt_lat, cobalt_lon, np.transpose(DISS_CALC_3D_COBALT.values, (1, 2, 0)), 68, 1e-15, 1e-9, 'magma', 'DISS CALC COBALT', logscale=True, lon_lims=[0, 360])
p2.plot_surface3d(model_lat, model_lon, DISS_CALC_3D_OCIM, 35, 1e-15, 1e-9, 'magma', 'DISS CALC OCIM', logscale=True, lon_lims=[0, 360])

# %%
