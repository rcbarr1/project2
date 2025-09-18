#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:33:22 2025

EXP16: Attempting maximum alkalinity calculation
- Start with 10 year simulations, will probably want to increase to ~100 later
- Import anthropogenic carbon prediction at each grid cell using TRACEv1 to be initial ∆DIC conditions
- From ∆DIC and gridded GLODAP pH and DIC, calculate how much ∆pH or ∆pCO2 to get back to preindustrial surface conditions
- With this, calculate ∆AT required to offset
- Add this ∆AT as a surface perturbation (globally for now, but eventually do this for each large marine ecosystem)
- Also, add the relevant ∆xCO2 for the time step given the emissions scenario of interest
- Repeat this for 100 years, calculate alkalinity added each year to get back to preindustrial
- Repeat for each LMES & maybe a few combinations of LMES?

exp16_2025-09-08-b.nc
- set up to add maximum amount of AT that can be added to each surface ocean
box at each time step without exceeding preindustrial pH
- recalculating carbonate system every 25 years
- running into average ocean pH exceeds average preindustrial ocean pH (recalculating pH every 25 years, max 1000 years before simulation ends)
- no emissions scenario

SSPs of interest
- From Meinshausen et al. (2020), https://gmd.copernicus.org/articles/13/3571/2020/
- Data downloaded from https://greenhousegases.science.unimelb.edu.au/#!/ghg?mode=yearly-gmnhsh
- 5 high priority scenarios for IPCC AR6 + some tier 2 that are of interest for CDR
- SSP1-2.6 -> 2ºC pathway
- SSP2-4.5 -> "middle of the road" scenario
- SSP3-7.0 -> medium-high with "regional rivalry"
- SSP3-7.0-NTCF -> same as SSP3-7.0, but with reduced near-term climate forcers (i.e. methane)
- SSP4-3.4 -> moderate mitigation
- SSP4-6.0 -> "inequality" dominated world
- SSP5-3.4-OS -> "overshoot scenario", follows SSP5-8.5, then steep emissions cuts and negative emissions
- SSP5-8.5 -> highly fossil-fuel developed world
- STARTING WITH SSP3-7.0

Governing equations (based on my own derivation + COBALT governing equations)
1. d(xCO2)/dt = ∆q_sea-air,xCO2 --> [µatm CO2 (µatm air)-1 yr-1] or [µmol CO2 (µmol air)-1 yr-1]
2. d(∆DIC)/dt = TR * ∆DIC + ∆q_air-sea,DIC + ∆q_hard,DIC + ∆q_CDR,DIC --> [µmol DIC (kg seawater)-1 yr-1]
3. d(∆AT)/dt = TR * ∆AT + ∆q_hard,AT + ∆q_CDR,AT --> [µmol AT (kg seawater)-1 yr-1]

where ∆q_hard,DIC = ∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc (FROM COBALT)
      ∆q_hard,AT = 2 * (∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc) (FROM COBALT)

*NOTE: burial is included in 'diss' in 'plus_btm' versions of calcium and
aragonite dissolution, but for some reason these arrays were all equal to zero
in files Rui sent me -> should investigate further soon

*NOTE: this is assuming no changes to biology, could modulate this (i.e.
production/respiration changes) in the future (see COBALT governing equations
for how this affects alkalinity/DIC in that model)
                                               
Air-sea gas exchange fluxes have to be multiplied by "c" vector because they
rely on ∆c's, which means they are incorporated with the transport matrix into
vector "A"

∆q_sea-air,xCO2 = k * V * (1 - f_ice) / Ma / z1 * (rho * R_DIC * del_DIC / beta_DIC + rho * R_AT * del_AT / beta_AT - K0 * Patm * del_xCO2)
∆q_air-sea,DIC = - k * (1 - f_ice) / z1 * (R_DIC * del_DIC / beta_DIC + R_AT * del_AT / beta_AT - K0 * Patm / rho * del_xCO2)

simplify with parameter "gamma"
gammax = k * V * (1 - f_ice) / Ma / z1
gammaC = - k * (1 - fice) / z1

∆q_sea-air,xCO2 = gammax * (rho * R_DIC * del_DIC / beta_DIC + rho * R_AT * del_AT / beta_AT - K0 * Patm * del_xCO2)
∆q_air-sea,DIC = gammaC * (R_DIC * del_DIC / beta_DIC + R_AT * del_AT / beta_AT - K0 * Patm / rho * del_xCO2)

Note about transport matrix set-up
- This was designed in matlab, which uses "fortran-style" aka column major ordering
- This means that "c" and "b" vectors must be constructed in this order
- This is complicated by the fact that the land boxes are excluded from the transport matrix
- The length of "c" and "b" vectors, as well as the length and width of the
  transport operator, are equal to the total number of ocean boxes in the model
- Best practices: create "c" and "b" vectors in three dimensions, flatten and mask out land boxes simultaneously 

Naming convention for saving model runs (see .txt file for explanation of experiments)
    exp##__YYYY-MM-DD-a.nc (where expXX corresponds to the python file used to
    run the experiment; a, b, c etc. represent model runs from the same day)

@author: Reese C. Barrett
"""
import project2 as p2
import xarray as xr
import cftime
import numpy as np
import PyCO2SYS as pyco2
from scipy import sparse
from tqdm import tqdm
from scipy.sparse.linalg import spilu, LinearOperator, lgmres
from time import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

data_path = '/Users/Reese_1/Documents/Research Projects/project2/data/'
output_path = '/Users/Reese_1/Documents/Research Projects/project2/outputs/'

#%% load transport matrix (OCIM2-48L, from Holzer et al., 2021)
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

ns10 = np.sum(ocnmask[0, :, :]==1) # number of surface ocean grid cells (~10 m)
ns30 = np.sum(ocnmask[0:2, :, :]==1) # number of ocean grid cells in top two layers of ocean (~30 m)
ns50 = np.sum(ocnmask[0:3, :, :]==1) # number of ocean grid cells in top three layers of ocean (~50 m)

# seawater density for volume to mass [kg m-3]
rho = 1025 

# depth of first model layer (need bottom of grid cell, not middle) [m]
grid_cell_height = model_data['wz'].to_numpy()
z1 = grid_cell_height[1, 0, 0]

# to help with conversions
sec_per_year = 60 * 60 * 24 * 365.25 # seconds in a year

#%% set up time stepping

dt = 1 # 1 year
t = np.arange(0, 1001, dt) # 1000 years after year 0 (for now)
t = np.arange(0, 8, dt) # 7 years after year 0 (for now)
nt = len(t)

#%% pulling emissions concentration scenarios
# accessed from https://greenhousegases.science.unimelb.edu.au/#!/ghg?mode=downloads
historical = xr.open_dataset(data_path + 'carbon-dioxide/historical/CMIP6GHGConcentrationHistorical_1_2_0/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_CMIP_UoM-CMIP-1-2-0_gr1-GMNHSH_0000-2014.nc', decode_times=False)
ssp126 = xr.open_dataset(data_path + 'carbon-dioxide/future/CMIP6GHGConcentrationProjections_1_2_1/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-IMAGE-ssp126-1-2-1_gr1-GMNHSH_2015-2500.nc') # 2ºC pathway
ssp245 = xr.open_dataset(data_path + 'carbon-dioxide/future/CMIP6GHGConcentrationProjections_1_2_1/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-MESSAGE-GLOBIOM-ssp245-1-2-1_gr1-GMNHSH_2015-2500.nc') # "middle of the road" scenario
ssp370 = xr.open_dataset(data_path + 'carbon-dioxide/future/CMIP6GHGConcentrationProjections_1_2_1/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-AIM-ssp370-1-2-1_gr1-GMNHSH_2015-2500.nc') # medium-high with "regional rivalry"
ssp370_NTCF = xr.open_dataset(data_path + 'carbon-dioxide/future/CMIP6GHGConcentrationProjections_1_2_1/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-AIM-ssp370-1-2-1_gr1-GMNHSH_2015-2500.nc')# same as SSP3-7.0, but with reduced near-term climate forcers (i.e. methane)
ssp434 = xr.open_dataset(data_path + 'carbon-dioxide/future/CMIP6GHGConcentrationProjections_1_2_1/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-GCAM4-ssp434-1-2-1_gr1-GMNHSH_2015-2500.nc') # moderate mitigation
ssp460 = xr.open_dataset(data_path + 'carbon-dioxide/future/CMIP6GHGConcentrationProjections_1_2_1/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-GCAM4-ssp460-1-2-1_gr1-GMNHSH_2015-2500.nc') # "inequality" dominated world
ssp534_OS = xr.open_dataset(data_path + 'carbon-dioxide/future/CMIP6GHGConcentrationProjections_1_2_1/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-REMIND-MAGPIE-ssp534-over-1-2-1_gr1-GMNHSH_2015-2500.nc')# "overshoot scenario", follows SSP5-8.5, then steep emissions cuts and negative emissions
ssp585 = xr.open_dataset(data_path + 'carbon-dioxide/future/CMIP6GHGConcentrationProjections_1_2_1/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-REMIND-MAGPIE-ssp585-1-2-1_gr1-GMNHSH_2015-2500.nc') # high emissions scenario 

# pull out historical times in decimal years
#historical_time = np.asarray(historical['time'].values, dtype=float)
#was weird uplaoding from .nc file, so doing this manually (checked .csv files for correct times)
historical_time = np.arange(0, 2015)

# pull out future time in decimal years
ssp_time = []
for timestamp in ssp126.time.values:
    # start and end of the year in NoLeap calendar
    year_start = cftime.DatetimeNoLeap(timestamp.year, 1, 1)
    year_end   = cftime.DatetimeNoLeap(timestamp.year + 1, 1, 1)
    
    # 365 days in NoLeap
    year_length = (year_end - year_start).days
    fraction = (timestamp - year_start).days / year_length
    
    ssp_time.append(timestamp.year + fraction)

ssp_time = np.array(ssp_time)

# pull out emissions over time, convert to xCO2 [mol CO2 (mol air)-1] from ppm
historical = historical.mole_fraction_of_carbon_dioxide_in_air.values[:,0] * 1e-6 # [mol CO2 (mol air)-1]
ssp126 = ssp126.mole_fraction_of_carbon_dioxide_in_air.values[:,0] * 1e-6 # [mol CO2 (mol air)-1]
ssp245 = ssp245.mole_fraction_of_carbon_dioxide_in_air.values[:,0] * 1e-6 # [mol CO2 (mol air)-1]
ssp370 = ssp370.mole_fraction_of_carbon_dioxide_in_air.values[:,0] * 1e-6 # [mol CO2 (mol air)-1]
ssp370_NTCF = ssp370_NTCF.mole_fraction_of_carbon_dioxide_in_air.values[:,0] * 1e-6 # [mol CO2 (mol air)-1]
ssp434 = ssp434.mole_fraction_of_carbon_dioxide_in_air.values[:,0] * 1e-6 # [mol CO2 (mol air)-1]
ssp460 = ssp460.mole_fraction_of_carbon_dioxide_in_air.values[:,0] * 1e-6 # [mol CO2 (mol air)-1]
ssp534_OS = ssp534_OS.mole_fraction_of_carbon_dioxide_in_air.values[:,0] * 1e-6 # [mol CO2 (mol air)-1]
ssp585 = ssp585.mole_fraction_of_carbon_dioxide_in_air.values[:,0] * 1e-6 # [mol CO2 (mol air)-1]

# above is cumulative change in xCO2 in atmosphere, calculate ∆q_xCO2 (perturbation at each time step)
q_historical = np.diff(historical, prepend=0) # [mol CO2 (mol air)-1]
q_ssp126 = np.diff(ssp126, prepend=0) # [mol CO2 (mol air)-1]
q_ssp245 = np.diff(ssp245, prepend=0) # [mol CO2 (mol air)-1]
q_ssp370 = np.diff(ssp370, prepend=0) # [mol CO2 (mol air)-1]
q_ssp370_NTCF = np.diff(ssp370_NTCF, prepend=0) # [mol CO2 (mol air)-1]
q_ssp434 = np.diff(ssp434, prepend=0) # [mol CO2 (mol air)-1]
q_ssp460 = np.diff(ssp460, prepend=0) # [mol CO2 (mol air)-1]
q_ssp534_OS = np.diff(ssp534_OS, prepend=0) # [mol CO2 (mol air)-1]
q_ssp585 = np.diff(ssp585, prepend=0) # [mol CO2 (mol air)-1]

#%% get masks for each large marine ecosystem (LME)

lme_id_grid, lme_masks, lme_id_to_name = p2.build_lme_masks(data_path + 'LMES/LMEs66.shp', ocnmask, model_lat, model_lon)
#p2.plot_lmes(lme_masks, ocnmask, model_lat, model_lon) # note: only 62 of 66 can be represented on OCIM grid

# to plot single mask
#idx = 58
#lme_mask = {idx: lme_masks[idx]}
#p2.plot_lmes(lme_mask, ocnmask, model_lat, model_lon)

#%% calculate mixed layer depth at each lat/lon following Holte et al. montly
# climatology, then create mask of ocean cells that are at or below the mixed
# layer depth

# to use dynamic mixed layer depth
monthly_clim = p2.loadmat(data_path + 'monthlyclim.mat')
MLD_da_max = monthly_clim['mld_da_max']
MLD_da_mean = monthly_clim['mld_da_mean']
latm = monthly_clim['latm']
lonm = monthly_clim['lonm']

maxMLDs = p2.find_MLD(model_lon, model_lat, ocnmask, MLD_da_max, latm, lonm, 0)

p2.plot_surface2d(model_lon, model_lat, maxMLDs.T, 0, 399, 'viridis_r', 'maximum annual mixed layer depth')

# create 3D mask where for each grid cell, mask is set to 1 if the depth in the
# grid cell depths array is less than the mixed layer depth for that column
MLDmask = (grid_cell_height < maxMLDs[None, :, :]).astype(int) 

#%% getting initial ∆DIC conditions from TRACEv1
# note, doing set up with Fortran ordering for consistency

# create list of longitudes (ºE), latitudes (ºN), and depths (m) in TRACE format
# this order is required for TRACE
lon, lat, depth = np.meshgrid(model_lon, model_lat, model_depth, indexing='ij')

# reshape meshgrid points into a list of coordinates to interpolate to
output_coordinates = np.array([lon.ravel(order='F'), lat.ravel(order='F'), depth.ravel(order='F'), ]).T

# create required input of dates
# first simulation year will be 2015 (I think), so do then 
dates_2015 = 2015 * np.ones([output_coordinates.shape[0],1])
#dates_2025 = 2025 * np.ones([output_coordinates.shape[0],1])

# get temperature and salinity predictor measurements from GLODAP data
#p2.regrid_glodap(data_path, 'temperature', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'salinity', model_depth, model_lat, model_lon, ocnmask)

# or, upload regridded glodap data
T_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy')
S_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy')

# transpose to match requirements for TRACEv1
T_3D = T_3D.transpose([1, 2, 0])
S_3D = S_3D.transpose([1, 2, 0])
predictor_measurements = np.vstack([S_3D.flatten(order='F'), T_3D.flatten(order='F')]).T

# combine all into .csv file to export for use with TRACEv1 in MATLAB (on the edge of my seat for pyTRACE clearly)
#trace_data = np.hstack([output_coordinates, dates_2015, predictor_measurements])
#np.savetxt(data_path + 'TRACEv1/trace_inputs_2015.txt', trace_data, delimiter = ',')

# transpose temperature and salinity back
T_3D = T_3D.transpose([2, 0, 1])
S_3D = S_3D.transpose([2, 0, 1])

# load in TRACE data
Canth_2015 = p2.loadmat(data_path + 'TRACEv1/trace_outputs_2015.mat')
Canth_2015 = Canth_2015['trace_outputs_2015']
Canth_2015 = Canth_2015.reshape(len(model_lon), len(model_lat), len(model_depth), order='F')
Canth_2015 = Canth_2015.transpose([2, 0, 1])

#p2.plot_surface3d(model_lon, model_lat, Canth_2015, 0, -1, 82, 'viridis', 'anthropogenic carbon')

#%% calculate preindustrial pH from GLODAP DIC minus Canth to get preindustrial DIC and GLODAP TA, assuming steady state

# regrid GLODAP data
#p2.regrid_glodap(data_path, 'TCO2', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'TAlk', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'pHtsinsitutp', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'temperature', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'salinity', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'silicate', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'PO4', model_depth, model_lat, model_lon, ocnmask)

# upload regridded GLODAP data
DIC_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC.npy') # dissolved inorganic carbon [µmol kg-1]
AT_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA.npy')   # total alkalinity [µmol kg-1]
pH_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/pHtsinsitutp.npy') # pH on total scale at in situ temperature and pressure 
T_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy') # temperature [ºC]
S_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy') # salinity [unitless]
Si_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/silicate.npy') # silicate [µmol kg-1]
P_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/PO4.npy') # phosphate [µmol kg-1]

DIC = p2.flatten(DIC_3D, ocnmask)
AT = p2.flatten(AT_3D, ocnmask)
pH = p2.flatten(pH_3D, ocnmask)
T = p2.flatten(T_3D, ocnmask)
S = p2.flatten(S_3D, ocnmask)
Si = p2.flatten(Si_3D, ocnmask)
P = p2.flatten(P_3D, ocnmask)

# calculate preindustrial pH by subtracting anthropogenic carbon
DIC_preind_3D = DIC_3D - Canth_2015
DIC_preind = p2.flatten(DIC_preind_3D, ocnmask)

# create "pressure" array by broadcasting depth array
pressure_3D = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[1], ocnmask.shape[2]))
pressure = pressure_3D[ocnmask == 1].flatten(order='F')

# calculate preindustrial pH from DIC in 2015 minus Canth in 2015 AND TA in 2015 (assuming steady state)

# is it okay to use modern-day temperatures for this?? probably not, but not
# sure if there's a TRACE for this and trying to stick with data-based, not
# model-based
# pyCO2SYS v2
co2sys = pyco2.sys(dic=DIC_preind, alkalinity=AT, salinity=S, temperature=T,
                   pressure=pressure, total_silicate=Si, total_phosphate=P)

pH_preind = co2sys['pH']
avg_pH_preind = np.nanmean(pH_preind)

pH_preind_3D = p2.make_3D(pH_preind, ocnmask)
#p2.plot_surface3d(model_lon, model_lat, pH_preind_3D, 0, 7.9, 8.4, 'viridis_r', 'Surface pH in Year ~1790 (TRACE)')

# calculate AT needed to offset pH drop using present-day DIC & previously calculated preindustrial pH
#co2sys = pyco2.sys(dic=DIC, pH=pH_preind, salinity=S, temperature=T,
#                   pressure=pressure, total_silicate=Si, total_phosphate=P)
#AT_to_offset = co2sys['alkalinity']
#AT_to_offset_3D = p2.make_3D(AT_to_offset, ocnmask)

#p2.plot_surface2d(model_lon, model_lat, AT_to_offset_3D[0, :, :].T, 2000, 2500, 'viridis_r', 'AT needed to offset pH calculated from TRACE only')
#p2.plot_surface2d(model_lon, model_lat, AT_3D[0, :, :].T, 2000, 2500, 'viridis_r', '2015 AT from GLODAP')

#p2.plot_surface2d(model_lon, model_lat, AT_to_offset_3D[0, :, :].T - AT_3D[0, :, :].T, -100, 100, 'RdBu', 'Change in AT needed to offset pH decline (with TRACE only)')

#%% set up air-sea gas exchange (Wanninkhof, 2014)

# regrid NCEP/DOE reanalysis II data
#p2.regrid_ncep_noaa(data_path, 'icec', model_lat, model_lon, ocnmask)
#p2.regrid_ncep_noaa(data_path, 'uwnd', model_lat, model_lon, ocnmask)
#p2.regrid_ncep_noaa(data_path, 'sst', model_lat, model_lon, ocnmask)

# upload regridded NCEP/DOE reanalysis II data
f_ice_2D = np.load(data_path + 'NCEP_DOE_Reanalysis_II/icec.npy') # annual mean ice fraction from 0 to 1 in each grid cell
uwnd_2D = np.load(data_path + 'NCEP_DOE_Reanalysis_II/wspd.npy') # annual mean of forecast of wind speed at 10 m [m/s]
sst_2D = np.load(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst.npy') # annual mean sea surface temperature [ºC]

# calculate Schmidt number using Wanninkhof 2014 parameterization
vec_schmidt = np.vectorize(p2.schmidt)
Sc_2D = vec_schmidt('CO2', sst_2D)

# solve for k (gas transfer velocity) for each ocean cell
a = 0.251 # from Wanninkhof 2014
k_2D = a * uwnd_2D**2 * (Sc_2D/660)**-0.5 # [cm/h] from Yamamoto et al., 2024, adapted from Wanninkhof 2014

#p2.plot_surface2d(model_lon, model_lat, k.T, 0, 20, 'magma', 'Gas transfer velocity (k, cm/hr)')

k_2D *= (24*365.25/100) # [m/yr] convert units

#p2.plot_surface2d(model_lon, model_lat, uwnd_3D.T, -15, 15, 'seismic', 'U-wind at 10 m (m/s)')
#p2.plot_surface2d(model_lon, model_lat, sst_3D.T, -2, 40, 'magma', 'sst (ºC)')

# set up linearized CO2 system (Nowicki et al., 2024)

# upload (or regrid) glodap data for use as initial conditions for marine carbonate system 

# calculate Nowicki et al. parameters
Ma = 1.8e26 # number of micromoles of air in atmosphere [µmol air]

Patm = 1e6 # atmospheric pressure [µatm]
V = p2.flatten(model_vols, ocnmask) # volume of first layer of model [m^3]

# add layers of "np.NaN" for all subsurface layers in k, f_ice, then flatten
k_3D = np.full(ocnmask.shape, np.nan)
k_3D[0, :, :] = k_2D
k = p2.flatten(k_3D, ocnmask)

f_ice_3D = np.full(ocnmask.shape, np.nan)
f_ice_3D[0, :, :] = f_ice_2D
f_ice = p2.flatten(f_ice_3D, ocnmask)

gammax = k * V * (1 - f_ice) / Ma / z1
gammaC = -1 * k * (1 - f_ice) / z1

#%% construct matrix C
# matrix form:
#  dc/dt = A * c + q
#  c = variable(s) of interest
#  A = transport matrix (TR) plus any processes with dependence on c 
#    = source/sink vector (processes not dependent on c)
    
# UNITS NOTE: all xCO2 units are mol CO2 (mol air)-1 all AT units are µmol AT kg-1, all DIC units are µmol DIC kg-1
# see comment at top for more info

# m = # ocean grid cells
# nt = # time steps

m = TR.shape[0]

# c = [ ∆xCO2 ] --> 1 * nt
#     [ ∆DIC  ] --> m * nt
#     [ ∆AT   ] --> m * nt

c = np.zeros((1 + 2*m, nt))

#%% construct initial q vector (it is going to change every iteration)
# q = [ 0                                                                           ] --> 1 * nt, q[0]
#     [ ∆q_CDR,DIC + ∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc      ] --> m * nt, q[1:(m+1)]
#     [ ∆q_CDR,AT + 2 * (∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc) ] --> m * nt, q[(m+1):(2*m+1)]

# which translates to...
# q = [ 0                                      ] --> 1 * nt, q[0]
#     [ ∆q_CDR,DIC + ∆q_diss,DIC - ∆q_prod,DIC ] --> m * nt, q[1:(m+1)]
#     [ ∆q_CDR,AT + ∆q_diss,AT - ∆q_prod,AT    ] --> m * nt, q[(m+1):(2*m+1)]

q = np.zeros((1 + 2*m, nt))
q[0,:] = 0 # no perturbation in xCO2 (for now, will likely add emissions scenario)
q[1:(m+1),:] = 0 # no perturbation in DIC (for now, could do other types of additions besides NaOH)

#%% time stepping simulation forward

for idx in tqdm(range(0,nt)):
    
    # recalculate carbonate system every 25 years
    if idx%25 == 0:
        #AT and DIC are equal to initial AT and DIC + whatever the change in AT and DIC are
        AT_current = AT + c[(m+1):(2*m+1), idx]
        DIC_current = DIC + c[1:(m+1), idx]
        
        # use CO2SYS with GLODAP data to solve for carbonate system at each grid cell
        # do this for only ocean grid cells
        # this is PyCO2SYSv2
        co2sys = pyco2.sys(dic=DIC_current, alkalinity=AT_current, salinity=S, temperature=T,
                           pressure=pressure, total_silicate=Si, total_phosphate=P)
    
        # extract key results arrays
        pCO2 = co2sys['pCO2'] # pCO2 [µatm]
        aqueous_CO2 = co2sys['CO2'] # aqueous CO2 [µmol kg-1]
        R_C = co2sys['revelle_factor'] # revelle factor w.r.t. DIC [unitless]
    
        # calculate revelle factor w.r.t. AT [unitless]
        # must calculate manually, R_AT defined as (dpCO2/pCO2) / (dAT/AT)
        co2sys_000001 = pyco2.sys(dic=DIC_current, alkalinity=AT_current+0.000001, salinity=S,
                               temperature=T, pressure=pressure, total_silicate=Si,
                               total_phosphate=P)
    
        pCO2_000001 = co2sys_000001['pCO2']
        R_A = ((pCO2_000001 - pCO2)/pCO2) / (0.000001/AT)
        
        # see if average pH has exceeded preindustrial average pH
        avg_pH = np.nanmean(co2sys['pH'])
        if avg_pH > avg_pH_preind:
            print('\n\naverage pH exceeded average preindustrial pH at time step ' + str(idx))
            break
        else:
            print('\n\naverage pH at time step ' + str(idx) + ' = ' + str(round(avg_pH,8)))
    
        # calculate rest of Nowicki et al. parameters
        beta_C = DIC/aqueous_CO2 # [unitless]
        beta_A = AT/aqueous_CO2 # [unitless]
        K0 = aqueous_CO2/pCO2*rho # [µmol CO2 m-3 (µatm CO2)-1], in derivation this is defined in per volume units so used density to get there
        
        print('carbonate system recalculated (year ' + str(idx) + ')')
    
        # calculate "A" matrix based on new carbonate system
    
        # dimensions
        # A = [1 x 1][1 x m][1 x m] --> total size 2m + 1 x 2m + 1
        #     [m x 1][m x m][m x m]
        #     [m x 1][m x m][m x m]
    
        # what acts on what
        # A = [THIS BOX * ∆xCO2][THIS BOX * ∆DIC][THIS BOX * ∆AT] --> to calculate new ∆xCO2 (still need q)
        #     [THIS BOX * ∆xCO2][THIS BOX * ∆DIC][THIS BOX * ∆AT] --> to calculate new ∆DIC (still need q)
        #     [THIS BOX * ∆xCO2][THIS BOX * ∆DIC][THIS BOX * ∆AT] --> to calculate new ∆AT (still need q)
    
        # math in each box (note: air-sea gas exchange terms only operate in surface boxes, they are set as main diagonal of identity matrix)
        # A = [-gammax * K0 * Patm      ][gammax * rho * R_DIC / beta_DIC][gammax * rho * R_AT / beta_AT]
        #     [-gammaC * K0 * Patm / rho][TR + gammaC * R_DIC / beta_DIC ][gammaC * R_AT / beta_AT      ]
        #     [0                        ][0                              ][TR                           ]
    
        # notation for setup
        # A = [A00][A01][A02]
        #     [A10][A11][A12]
        #     [A20][A21][A22]
    
        # to solve for ∆xCO2
        A00 = -1 * Patm * np.nansum(gammax * K0) # using nansum because all subsurface boxes are NaN, we only want surface
        A01 = np.nan_to_num(gammax * rho * R_C / beta_C) # nan_to_num sets all NaN = 0 (subsurface boxes, no air-sea gas exchange)
        A02 = np.nan_to_num(gammax * rho * R_A / beta_A)
    
        # combine into A0 row
        A0_ = np.full(1 + 2*m, np.nan)
        A0_[0] = A00
        A0_[1:(m+1)] = A01
        A0_[(m+1):(2*m+1)] = A02
    
        del A00, A01, A02
    
        # to solve for ∆DIC
        A10 = np.nan_to_num(-1 * gammaC * K0 * Patm / rho) # is csc the most efficient format? come back to this
        A11 = TR + sparse.diags(np.nan_to_num(gammaC * R_C / beta_C), format='csc')
        A12 = sparse.diags(np.nan_to_num(gammaC * R_A / beta_A))
    
        A1_ = sparse.hstack((sparse.csc_matrix(np.expand_dims(A10,axis=1)), A11, A12))
    
        del A10, A11, A12
    
        # to solve for ∆AT
        A20 = np.zeros(m)
        A21 = 0 * TR
        A22 = TR
    
        A2_ = sparse.hstack((sparse.csc_matrix(np.expand_dims(A20,axis=1)), A21, A22))
    
        del A20, A21, A22
    
        # build into one mega-array!!
        A = sparse.vstack((sparse.csc_matrix(np.expand_dims(A0_,axis=0)), A1_, A2_))
    
        del A0_, A1_, A2_
            
        # perform time stepping using Euler backward
        LHS = sparse.eye(A.shape[0], format="csc") - dt * A
    
        # test condition number of matrix
        est = sparse.linalg.onenormest(LHS)
        print('estimated 1-norm condition number LHS: ' + str(round(est,1)))
    
        start = time()
        ilu = spilu(LHS.tocsc(), drop_tol=1e-5, fill_factor=20)
        stop = time()
        print('ilu calculations: ' + str(stop - start) + ' s\n')
    
        M = LinearOperator(LHS.shape, ilu.solve)
    
    # not calculating delAT/delDIC/delxCO2 at time = 0 (this time step is initial conditions only)
    if idx >= 1:
        # add CDR perturbation (construct q vector, it is going to change every iteration in this experiment)
        # for now, assuming NaOH (no change in DIC)
        
        # calculate AT required to return to preindustrial pH
        # using DIC at previous time step (initial DIC + modeled change in DIC) and preindustrial pH
        DIC_new = DIC + c[1:(m+1), idx-1]
        AT_new = AT + c[(m+1):(2*m+1), idx-1]
        AT_to_offset = p2.calculate_AT_to_add(pH_preind, DIC_new, AT_new, T, S, pressure, Si, P, low=0, high=200, tol=1e-6, maxiter=50)
        AT_to_offset_3D = p2.make_3D(AT_to_offset, ocnmask)

        # make sure there are no negative values
        if len(AT_to_offset[AT_to_offset<0]) != 0:
            print('error: AT offset is negative')
            break

        # set CDR perturbation equal to this AT in mixed layer
        # ∆q_CDR,AT (change in alkalinity due to CDR addition) - final units: [µmol AT kg-1 yr-1]
        #del_q_CDR_AT = np.zeros(m)
        #del_q_CDR_AT[0:ns50] = AT_to_offset[0:ns50] # calculated from CO2SYS above, only apply in surface
        
        del_q_CDR_AT = p2.flatten(AT_to_offset_3D * MLDmask, ocnmask) # apply in maximum annual mixed layer depth
    
        # add in source/sink vectors for ∆AT to q vector
        q[(m+1):(2*m+1), idx] = del_q_CDR_AT
        
        # add in emissions scenario (∆xCO2) to q vector as well
        # start with SSP2-4.5
        #q[0,idx] = q_ssp245[idx] # [mol CO2 (mol air)-1]
        
        # add starting guess after first time step
        if idx > 1:
            c0 = c[:,idx-1]
        else:
            c0=None
    
        # calculate right hand side and perform time stepping
        RHS = c[:,idx-1] + np.squeeze(dt*q[:,idx])
        #start = time()
        c[:,idx], info = lgmres(LHS, RHS, M=M, x0=c0, rtol = 1e-5, atol=0)
        #stop = time()
        #print('t = ' + str(idx) + ', solve time: ' + str(stop - start) + ' s')
       
        if info != 0:
            if info > 0:
                print(f'did not converge in {info} iterations.')
            else:
                print('illegal input or breakdown')

# rebuild 3D concentrations from 1D array used for solving matrix equation
    
# partition "c" into xCO2, DIC, and AT
c_delxCO2 = c[0, :]
c_delDIC  = c[1:(m+1), :]
c_delAT   = c[(m+1):(2*m+1), :]

# partition "q" into xCO2, DIC, and AT
q_delxCO2 = q[0, :]
q_delDIC  = q[1:(m+1), :]
q_delAT   = q[(m+1):(2*m+1), :]

# convert delxCO2 units from unitless [µatm CO2 / µatm air] or [µmol CO2 / µmol air] to ppm
c_delxCO2 *= 1e6
q_delxCO2 *= 1e6

# reconstruct 3D arrays for DIC and AT
c_delDIC_3D = np.full([len(t), ocnmask.shape[0], ocnmask.shape[1], ocnmask.shape[2]], np.nan) # make 3D vector full of nans
c_delAT_3D = np.full([len(t), ocnmask.shape[0], ocnmask.shape[1], ocnmask.shape[2]], np.nan) # make 3D vector full of nans

q_delDIC_3D = np.full([len(t), ocnmask.shape[0], ocnmask.shape[1], ocnmask.shape[2]], np.nan) # make 3D vector full of nans
q_delAT_3D = np.full([len(t), ocnmask.shape[0], ocnmask.shape[1], ocnmask.shape[2]], np.nan) # make 3D vector full of nans

# for each time step, reshape 1D array into 3D array, then save to larger 4D array output (time, depth, longitude, latitude)
for idx in range(0, len(t)):
    c_delDIC_reshaped = np.full(ocnmask.shape, np.nan)
    c_delAT_reshaped = np.full(ocnmask.shape, np.nan)
    
    q_delDIC_reshaped = np.full(ocnmask.shape, np.nan)
    q_delAT_reshaped = np.full(ocnmask.shape, np.nan)

    c_delDIC_reshaped[ocnmask == 1] = np.reshape(c_delDIC[:, idx], (-1,), order='F')
    c_delAT_reshaped[ocnmask == 1] = np.reshape(c_delAT[:, idx], (-1,), order='F')
    
    q_delDIC_reshaped[ocnmask == 1] = np.reshape(q_delDIC[:, idx], (-1,), order='F')
    q_delAT_reshaped[ocnmask == 1] = np.reshape(q_delAT[:, idx], (-1,), order='F')
    
    c_delDIC_3D[idx, :, :, :] = c_delDIC_reshaped
    c_delAT_3D[idx, :, :, :] = c_delAT_reshaped
    
    q_delDIC_3D[idx, :, :, :] = q_delDIC_reshaped
    q_delAT_3D[idx, :, :, :] = q_delAT_reshaped

# save model output in netCDF format
global_attrs = {'description': 'attempt at calculating max alkalinity to be added at each time step - with ssp245 emissions scenario - global application - calculated with pyCO2sys recalculating carbonate system every 25 years - recalculating and applying new AT at each time step'}
# save model output
p2.save_model_output(
    'exp16_2025-09-17-c.nc', 
    t, 
    model_depth, 
    model_lon,
    model_lat, 
    tracers=[c_delxCO2, c_delDIC_3D, c_delAT_3D, q_delxCO2, q_delDIC_3D, q_delAT_3D,], 
    tracer_dims=[('time',), ('time', 'depth', 'lon', 'lat'), ('time', 'depth', 'lon', 'lat'), ('time',), ('time', 'depth', 'lon', 'lat'), ('time', 'depth', 'lon', 'lat')],
    tracer_names=['delxCO2', 'delDIC', 'delAT', 'xCO2_added', 'DIC_added', 'AT_added'], 
    tracer_units=['ppm', 'umol kg-3', 'umol kg-3', 'ppm', 'umol kg-3', 'umol kg-3'],
    global_attrs=global_attrs
)

#%% open and analyze outputs
# calculate change in surface pH at each time step
data = xr.open_dataset(output_path + 'exp16_2025-09-17-c.nc')
t = data['time'].values
nt = len(t)

DIC_broadcasted = xr.DataArray(DIC_3D, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast DIC to convert ∆DIC to total DIC over time
AT_broadcasted = xr.DataArray(AT_3D, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast DIC to convert ∆DIC to total DIC over time

DIC_modeled_3D = data.delDIC + DIC_broadcasted
AT_modeled_3D = data.delAT + AT_broadcasted

#DIC_modeled_3D = data.DIC_added + DIC_broadcasted
#AT_modeled_3D = data.AT_added + AT_broadcasted

pH_modeled = []
avg_pH_modeled = np.zeros(nt)
avg_pH_modeled_surf = np.zeros(nt)

for idx in range(nt):
    DIC_modeled = p2.flatten(DIC_modeled_3D.isel(time=idx).values, ocnmask)
    AT_modeled = p2.flatten(AT_modeled_3D.isel(time=idx).values, ocnmask)
        
    co2sys = pyco2.sys(dic=DIC_modeled, alkalinity=AT_modeled, salinity=S, temperature=T,
                       pressure=pressure, total_silicate=Si, total_phosphate=P)
    
    pH_modeled.append(co2sys['pH'])
    avg_pH_modeled[idx] = np.nanmean(co2sys['pH'])
    avg_pH_modeled_surf[idx] = np.nanmean(co2sys['pH'][0:ns10])

    #print(np.nanmean(co2sys['pH']))
    pH_modeled_3D = p2.make_3D(co2sys['pH'], ocnmask)
    print(np.nanmean(pH_modeled_3D[0,:,:]))
    p2.plot_surface3d(data.lon, data.lat, pH_preind_3D - pH_modeled_3D, 0, -0.5, 0.5, 'RdBu', 'pH difference from preindustrial at year ' + str(data['time'].isel(time=idx).values))

for idx in range(nt-1):
    p2.plot_surface3d(data.lon, data.lat, data['AT_added'].isel(time=idx).values, 0, 0, 100, 'viridis', 'AT (µmol kg-1) added at year ' + str(data['time'].isel(time=idx).values))


#%% make figure of annual alkalinity change each year vs. average ocean pH
years = np.arange(start=2015, stop=2015 + nt)
model_vols_xr = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast model_vols to convert ∆AT from per kg to total

AT_added = data['AT_added'] * model_vols_xr * rho * 1e-6
AT_added = AT_added.sum(dim=['depth', 'lon', 'lat'], skipna=True).values
AT_added = np.cumsum(AT_added)

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.plot(years, avg_pH_modeled, label='pH under max OAE')
ax.axhline(np.nanmean(pH_preind), c='black', linestyle='--', label='preindustrial pH') # add line showing preindustrial surface pH

# set up secondary axis "years"
year_to_AT = interp1d(years, AT_added, kind='linear', fill_value="extrapolate")
AT_to_year = interp1d(AT_added, years, kind='linear', fill_value="extrapolate")
secax = ax.secondary_xaxis('top', functions=(year_to_AT, AT_to_year))
secax.set_xlabel('total amount of AT added (mol)')


ax.set_ylabel('average ocean pH')
ax.set_xlabel('year')
ax.set_xlim([2015, 2040])
ax.set_ylim([7.9, 8])
plt.legend(loc = 'lower right')
plt.show()

#%% same thing, but for surface ocean
years = np.arange(start=2015, stop=2015 + nt)

AT_added = data['AT_added'] * model_vols_xr * rho * 1e-6
AT_added = AT_added.sum(dim=['depth', 'lon', 'lat'], skipna=True).values
AT_added = np.cumsum(AT_added)

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.plot(years, avg_pH_modeled_surf, label='pH under max OAE')
ax.axhline(np.nanmean(pH_preind[0:ns10]), c='black', linestyle='--', label='preindustrial pH') # add line showing preindustrial surface pH

# set up secondary axis "years"
year_to_AT = interp1d(years, AT_added, kind='linear', fill_value="extrapolate")
AT_to_year = interp1d(AT_added, years, kind='linear', fill_value="extrapolate")
secax = ax.secondary_xaxis('top', functions=(year_to_AT, AT_to_year))
secax.set_xlabel('total amount of AT added (mol)')


ax.set_ylabel('average surface ocean pH')
ax.set_xlabel('year')
ax.set_xlim([2015, 2040])
ax.set_ylim([8, 8.3])
plt.legend(loc = 'lower right')
plt.show()

#%% at each time step, does AT_added + delAT + GLODAP AT equal preindustrial AT?



