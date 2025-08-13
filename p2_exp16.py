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
import gsw
import PyCO2SYS as pyco2
from scipy import sparse
from tqdm import tqdm
from scipy.sparse.linalg import spilu, LinearOperator, lgmres
from scipy.interpolate import RegularGridInterpolator
from time import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

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

# grid cell z-dimension for converting from surface area to volume
grid_z = model_vols / model_data['area'].to_numpy()
rho = 1025 # seawater density for volume to mass [kg m-3]

# to help with conversions
sec_per_year = 60 * 60 * 24 * 365.25 # seconds in a year

#%% pulling emissions concentration scenarios
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
historical_time = np.asarray(historical['time'].values, dtype=float)

# pull out future time in decimal years
ssp_time = []
for t in ssp126.time.values:
    # start and end of the year in NoLeap calendar
    year_start = cftime.DatetimeNoLeap(t.year, 1, 1)
    year_end   = cftime.DatetimeNoLeap(t.year + 1, 1, 1)
    
    # 365 days in NoLeap
    year_length = (year_end - year_start).days
    fraction = (t - year_start).days / year_length
    
    ssp_time.append(t.year + fraction)

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

#%% get preindustrial pH from Jiang et al. (2019)
# data is https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0206289

# get average pH in 1770
data = xr.open_dataset(data_path + 'pH_1770/Surface_pH_1770_2000.nc')

# regrid to match OCIM grid
#p2.regrid_pH(data_path, data, model_lat, model_lon, ocnmask)

# or, open regridded data
pH_1770 = np.load(data_path + 'pH_1770/pH_1770_AO.npy')   # pH [unitless]

# plot regridded data
p2.plot_surface2d(model_lon, model_lat, pH_1770.T, 8, 8.25, 'viridis_r', 'Surface pH in Year 1770')

#%% getting initial ∆DIC conditions from TRACEv1
# note, doing set up with Fortran ordering for consistency

# create list of longitudes (ºE), latitudes (ºN), and depths (m) in TRACE format
# this order is required for TRACE
lon, lat, depth = np.meshgrid(model_lon, model_lat, model_depth, indexing='ij')

# reshape meshgrid points into a list of coordinates to interpolate to
output_coordinates = np.array([lon.ravel(order='F'), lat.ravel(order='F'), depth.ravel(order='F'), ]).T

# create required input of dates
# first simulation year will be 2015 (I think), so do then (also should do 2025 option)
dates_2015 = 2015 * np.ones([output_coordinates.shape[0],1])
dates_2025 = 2025 * np.ones([output_coordinates.shape[0],1])

# get temperature and salinity predictor measurements from OCIM data
ptemperature0 = model_data['ptemp'].values.flatten(order='F')
salinity0 = model_data['salt'].values.flatten(order='F')

# convert temperature from potential temperature to standard temperature
pressure0 = gsw.p_from_z(output_coordinates[:,2]*-1, output_coordinates[:,1]) # convert depth [m, positive up] to pressure [dbar] using latitude [-90º to +90º]
asalinity0 = gsw.SA_from_SP(salinity0, pressure0, output_coordinates[:,0], output_coordinates[:,1])  # convert practical salinity [unitless] to absolute salinity [g/kg] with pressure [dbar], latitude [-90º to +90º], and longitude [-360º to 360º]
ctemperature0 = gsw.CT_from_pt(asalinity0, ptemperature0) # convert potential temperature [ºC] to conservative temperature [ºC] using absolute salinity [g/kg]
temperature0 = gsw.t_from_CT(asalinity0, ctemperature0, pressure0) # convert conservative temperature [ºC[ to in-situ temperature [ºC] using absolute salinity [g/kg] and pressure [dbar] (GSW python toolbox does not have direct conversion)

predictor_measurements = np.vstack([salinity0, temperature0]).T

# combine all into .csv file to export for use with TRACEv1 in MATLAB (on the edge of my seat for pyTRACE clearly)
#trace_data = np.hstack([output_coordinates, dates_2015, predictor_measurements])
#np.savetxt(data_path + 'TRACEv1/trace_inputs_2015.txt', trace_data, delimiter = ',')

#trace_data = np.hstack([output_coordinates, dates_2025, predictor_measurements])
#np.savetxt(data_path + 'TRACEv1/trace_inputs_2025.txt', trace_data, delimiter = ',')

# load in TRACE data
Canth_2015 = p2.loadmat(data_path + 'TRACEv1/trace_outputs_2015.mat')
Canth_2015 = Canth_2015['trace_outputs_2015']

#Canth_2025 = p2.loadmat(data_path + 'TRACEv1/trace_outputs_2025.mat')
#Canth_2025 = Canth_2025['trace_outputs_2025']

#%% get masks for each large marine ecosystem (LME)

lme_id_grid, lme_masks, lme_id_to_name = p2.build_lme_masks(data_path + 'LMES/LMEs66.shp', ocnmask, model_lat, model_lon)
p2.plot_lmes(lme_masks, ocnmask, model_lat, model_lon) # note: only 62 of 66 can be represented on OCIM grid

# to plot single mask
#lme_mask = {64: lme_masks[64]}
#plot_lmes(lme_mask, ocnmask, model_lat, model_lon)


