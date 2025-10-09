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
import numpy as np
import PyCO2SYS as pyco2
from scipy import sparse
from tqdm import tqdm
from scipy.sparse.linalg import spilu, LinearOperator, lgmres
from time import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# load model architecture
data_path = '/Users/Reese_1/Documents/Research Projects/project2/data/'
output_path = '/Users/Reese_1/Documents/Research Projects/project2/outputs/'

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

#%% SET EXPERIMENTAL VARIABLES: WEEKEND RUN 0
# - length of time of experiment/time stepping
# - depth of addition
# - location of addition
# - emissions scenarios
# - experiment names
# in this experiment, amount of addition is set as the maximum amount of AT
# that can be added to a grid cell before exceeding preindustrial pH, so it is
# not treated as a variable

# TIME
dt0 = 1/8640 # 1 hour
dt1 = 1/360 # 1 day
dt2 = 1/12 # 1 month
dt3 = 1 # 1 year

# just year time steps
exp0_t = np.arange(0,3,dt3)

# an experiment with dt = 1/360 (1 day) for the first year, then dt = 1/12 (1 month) for the second year
t1 = np.arange(0, 1, dt1) # use a 1 day time step for the first year
t2 = np.arange(1, 2+dt2, dt2) # use a 1 month time step until the 2nd year
exp1_t = np.concatenate((t1, t2))

# another experiment with dt = 1/360 (1 day) for the first year, then dt = 1 (1 year) for the second year
t1 = np.arange(0, 1, dt1) # use a 1 day time step for the first year
t3 = np.arange(1, 2+dt3, dt3) # use a 1 year time step until the 2nd year
exp2_t = np.concatenate((t1, t3))

# another with dt = 1/360 (1 day) for the first month, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year
t1 = np.arange(0, 1/12, dt1) # use a 1 day time step for the first month
t2 = np.arange(1/12, 1, dt2) # use a 1 month time step until the first year
t3 = np.arange(1, 2+dt3, dt3) # use a 1 year time step until the 2nd year
exp3_t = np.concatenate((t1, t2, t3))

# another with dt = 1/8640 (1 hour) for the first day, then dt = 1/360 (1 day) for the next 29 days, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year
t0 = np.arange(0, 1/360, dt0) # use a 1 day time step for the first month
t1 = np.arange(1/360, 1/12, dt1) # use a 1 day time step for the first month
t2 = np.arange(1/12, 1, dt2) # use a 1 month time step until the first year
t3 = np.arange(1, 2+dt3, dt3) # use a 1 year time step until the 2nd year
exp4_t = np.concatenate((t0, t1, t2, t3))

exp_t = [exp0_t, exp1_t, exp2_t, exp3_t, exp4_t]

# DEPTHS OF ADDITION

# testing with built-in MLD (see p2_exp19.py)

mldmask = np.load(data_path + 'mld mask tests/mldmask_built_in.npy') 
q_AT_depths = p2.make_3D(mldmask, ocnmask)

# to do addition in mixed layer...
# pull mixed layer depth at each lat/lon from OCIM model data, then create mask
# of ocean cells that are at or below the mixed layer depth
mld = model_data.mld.values # [m]
# create 3D mask where for each grid cell, mask is set to 1 if the depth in the
# grid cell depths array is less than the mixed layer depth for that column
# note: this does miss cells where the MLD is close but does not reach the
# depth of the next grid cell below (i.e. MLD = 40 m, grid cell depths are at
# 30 m and 42 m, see lon_idx, lat_idx = 20, 30). I am intentionally leaving
# this for now to ensure what enters the ocean stays mostly within the mixed
# layer, but the code could be changed to a different method if needed.exp2_t

mldmask = (grid_cell_depth < mld[None, :, :]).astype(int)
q_AT_depths = mldmask

# to do addition in first (or first two, or first three, etc.) model layer(s)
#q_AT_depths = ocnmask.copy()
#q_AT_depths[1::, :, :] = 0 # all ocean grid cells in surface layer (~10 m) are 1, rest 0
#q_AT_depths[2::, :, :] = 0 # all ocean grid cells in top 2 surface layers (~30 m) are 1, rest 0
#q_AT_depths[3::, :, :] = 0 # all ocean grid cells in top 3 surface layers (~50 m) are 1, rest 0

# to do all lat/lons
q_AT_latlons = ocnmask[0,:,:].copy()

# to constrain lat/lon of addition to LME(s)
# get masks for each large marine ecosystem (LME)
#lme_masks, lme_id_to_name = p2.build_lme_masks(data_path + 'LMES/LMEs66.shp', ocnmask, model_lat, model_lon)
#p2.plot_lmes(lme_masks, ocnmask, model_lat, model_lon) # note: only 62 of 66 can be represented on OCIM grid
#lme_idx = [22,52] # subset of LMEs
#lme_idx = list(lme_masks.keys()) # all LMES
#q_AT_latlons = sum(lme_masks[idx] for idx in lme_idx)

# COMBINE DEPTH + LAT/LON OF ADDITION
q_AT_locations_mask = q_AT_depths * q_AT_latlons

# EMISSIONS SCENARIOS
# no emissions scenario
#q_emissions = np.zeros(nt)

# with emissions scenario
scenarios = ['none', 'none', 'none', 'none', 'none']

# EXPERIMENT NAMES AND DESCRIPTIONS

experiment_names = ['exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping4.nc',]

experiment_attrs = ['adding max AT to built-in MLD before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1 (1 year) for two years',
                    'adding max AT to built-in MLD before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first year, then dt = 1/12 (1 month) for the second year',
                    'adding max AT to built-in MLD before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first year, then dt = 1 (1 year) for the second year',
                    'adding max AT to built-in MLD before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first month, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year',
                    'adding max AT to built-in MLD before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/8640 (1 hour) for the first day, then dt = 1/360 (1 day) for the next 29 days, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year']

#%% SET EXPERIMENTAL VARIABLES: WEEKEND RUN 1
# - length of time of experiment/time stepping
# - depth of addition
# - location of addition
# - emissions scenarios
# - experiment names
# in this experiment, amount of addition is set as the maximum amount of AT
# that can be added to a grid cell before exceeding preindustrial pH, so it is
# not treated as a variable

# TIME
dt0 = 1/8640 # 1 hour
dt1 = 1/360 # 1 day
dt2 = 1/12 # 1 month
dt3 = 1 # 1 year

# just year time steps
exp0_t = np.arange(0,3,dt3)

# an experiment with dt = 1/360 (1 day) for the first year, then dt = 1/12 (1 month) for the second year
t1 = np.arange(0, 1, dt1) # use a 1 day time step for the first year
t2 = np.arange(1, 2+dt2, dt2) # use a 1 month time step until the 2nd year
exp1_t = np.concatenate((t1, t2))

# another experiment with dt = 1/360 (1 day) for the first year, then dt = 1 (1 year) for the second year
t1 = np.arange(0, 1, dt1) # use a 1 day time step for the first year
t3 = np.arange(1, 2+dt3, dt3) # use a 1 year time step until the 2nd year
exp2_t = np.concatenate((t1, t3))

# another with dt = 1/360 (1 day) for the first month, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year
t1 = np.arange(0, 1/12, dt1) # use a 1 day time step for the first month
t2 = np.arange(1/12, 1, dt2) # use a 1 month time step until the first year
t3 = np.arange(1, 2+dt3, dt3) # use a 1 year time step until the 2nd year
exp3_t = np.concatenate((t1, t2, t3))

# another with dt = 1/8640 (1 hour) for the first day, then dt = 1/360 (1 day) for the next 29 days, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year
t0 = np.arange(0, 1/360, dt0) # use a 1 day time step for the first month
t1 = np.arange(1/360, 1/12, dt1) # use a 1 day time step for the first month
t2 = np.arange(1/12, 1, dt2) # use a 1 month time step until the first year
t3 = np.arange(1, 2+dt3, dt3) # use a 1 year time step until the 2nd year
exp4_t = np.concatenate((t0, t1, t2, t3))

exp_t = [exp0_t, exp1_t, exp2_t, exp3_t, exp4_t]

# DEPTHS OF ADDITION

# testing with alt MLD 1% threshold (see p2_exp19.py)

mldmask = np.load(data_path + 'mld mask tests/mldmask_alt_1percentthresh.npy') 
q_AT_depths = p2.make_3D(mldmask, ocnmask)

# to do addition in first (or first two, or first three, etc.) model layer(s)
#q_AT_depths = ocnmask.copy()
#q_AT_depths[1::, :, :] = 0 # all ocean grid cells in surface layer (~10 m) are 1, rest 0
#q_AT_depths[2::, :, :] = 0 # all ocean grid cells in top 2 surface layers (~30 m) are 1, rest 0
#q_AT_depths[3::, :, :] = 0 # all ocean grid cells in top 3 surface layers (~50 m) are 1, rest 0

# to do all lat/lons
q_AT_latlons = ocnmask[0,:,:].copy()

# to constrain lat/lon of addition to LME(s)
# get masks for each large marine ecosystem (LME)
#lme_masks, lme_id_to_name = p2.build_lme_masks(data_path + 'LMES/LMEs66.shp', ocnmask, model_lat, model_lon)
#p2.plot_lmes(lme_masks, ocnmask, model_lat, model_lon) # note: only 62 of 66 can be represented on OCIM grid
#lme_idx = [22,52] # subset of LMEs
#lme_idx = list(lme_masks.keys()) # all LMES
#q_AT_latlons = sum(lme_masks[idx] for idx in lme_idx)

# COMBINE DEPTH + LAT/LON OF ADDITION
q_AT_locations_mask = q_AT_depths * q_AT_latlons

# EMISSIONS SCENARIOS
# no emissions scenario
#q_emissions = np.zeros(nt)

# with emissions scenario
scenarios = ['none', 'none', 'none', 'none', 'none']

# EXPERIMENT NAMES AND DESCRIPTIONS

experiment_names = ['exp16_2025-10-08-ssp_none-MLDalt1-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-08-ssp_none-MLDalt1-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-10-08-ssp_none-MLDalt1-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-10-08-ssp_none-MLDalt1-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-10-08-ssp_none-MLDalt1-all_lat_lon-time_stepping4.nc',]

experiment_attrs = ['adding max AT to alt MLD with 1 percent threshold before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1 (1 year) for two years',
                    'adding max AT to alt MLD with 1 percent threshold before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first year, then dt = 1/12 (1 month) for the second year',
                    'adding max AT to alt MLD with 1 percent threshold before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first year, then dt = 1 (1 year) for the second year',
                    'adding max AT to alt MLD with 1 percent threshold before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first month, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year',
                    'adding max AT to alt MLD with 1 percent threshold before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/8640 (1 hour) for the first day, then dt = 1/360 (1 day) for the next 29 days, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year']

#%% SET EXPERIMENTAL VARIABLES: WEEKEND RUN 2
# - length of time of experiment/time stepping
# - depth of addition
# - location of addition
# - emissions scenarios
# - experiment names
# in this experiment, amount of addition is set as the maximum amount of AT
# that can be added to a grid cell before exceeding preindustrial pH, so it is
# not treated as a variable

# TIME
dt0 = 1/8640 # 1 hour
dt1 = 1/360 # 1 day
dt2 = 1/12 # 1 month
dt3 = 1 # 1 year

# just year time steps
exp0_t = np.arange(0,3,dt3)

# an experiment with dt = 1/360 (1 day) for the first year, then dt = 1/12 (1 month) for the second year
t1 = np.arange(0, 1, dt1) # use a 1 day time step for the first year
t2 = np.arange(1, 2+dt2, dt2) # use a 1 month time step until the 2nd year
exp1_t = np.concatenate((t1, t2))

# another experiment with dt = 1/360 (1 day) for the first year, then dt = 1 (1 year) for the second year
t1 = np.arange(0, 1, dt1) # use a 1 day time step for the first year
t3 = np.arange(1, 2+dt3, dt3) # use a 1 year time step until the 2nd year
exp2_t = np.concatenate((t1, t3))

# another with dt = 1/360 (1 day) for the first month, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year
t1 = np.arange(0, 1/12, dt1) # use a 1 day time step for the first month
t2 = np.arange(1/12, 1, dt2) # use a 1 month time step until the first year
t3 = np.arange(1, 2+dt3, dt3) # use a 1 year time step until the 2nd year
exp3_t = np.concatenate((t1, t2, t3))

# another with dt = 1/8640 (1 hour) for the first day, then dt = 1/360 (1 day) for the next 29 days, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year
t0 = np.arange(0, 1/360, dt0) # use a 1 day time step for the first month
t1 = np.arange(1/360, 1/12, dt1) # use a 1 day time step for the first month
t2 = np.arange(1/12, 1, dt2) # use a 1 month time step until the first year
t3 = np.arange(1, 2+dt3, dt3) # use a 1 year time step until the 2nd year
exp4_t = np.concatenate((t0, t1, t2, t3))

exp_t = [exp0_t, exp1_t, exp2_t, exp3_t, exp4_t]

#exp_t = [np.concatenate((np.arange(0, 5*dt0, dt0), np.arange(5*dt0, 5*dt1, dt1), np.arange(5*dt1, 12*dt2, dt2)))]

# DEPTHS OF ADDITION

# testing with alt MLD 1% threshold (see p2_exp19.py)

mldmask = np.load(data_path + 'mld mask tests/mldmask_alt_point1percentthresh.npy') 
q_AT_depths = p2.make_3D(mldmask, ocnmask)

# to do addition in first (or first two, or first three, etc.) model layer(s)
#q_AT_depths = ocnmask.copy()
#q_AT_depths[1::, :, :] = 0 # all ocean grid cells in surface layer (~10 m) are 1, rest 0
#q_AT_depths[2::, :, :] = 0 # all ocean grid cells in top 2 surface layers (~30 m) are 1, rest 0
#q_AT_depths[3::, :, :] = 0 # all ocean grid cells in top 3 surface layers (~50 m) are 1, rest 0

# to do all lat/lons
q_AT_latlons = ocnmask[0,:,:].copy()

# to constrain lat/lon of addition to LME(s)
# get masks for each large marine ecosystem (LME)
#lme_masks, lme_id_to_name = p2.build_lme_masks(data_path + 'LMES/LMEs66.shp', ocnmask, model_lat, model_lon)
#p2.plot_lmes(lme_masks, ocnmask, model_lat, model_lon) # note: only 62 of 66 can be represented on OCIM grid
#lme_idx = [22,52] # subset of LMEs
#lme_idx = list(lme_masks.keys()) # all LMES
#q_AT_latlons = sum(lme_masks[idx] for idx in lme_idx)

# COMBINE DEPTH + LAT/LON OF ADDITION
q_AT_locations_mask = q_AT_depths * q_AT_latlons

# EMISSIONS SCENARIOS
# no emissions scenario
#q_emissions = np.zeros(nt)

# with emissions scenario
scenarios = ['none', 'none', 'none', 'none', 'none']

# EXPERIMENT NAMES AND DESCRIPTIONS

experiment_names = ['exp16_2025-10-08-ssp_none-MLDalt2-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-08-ssp_none-MLDalt2-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-10-08-ssp_none-MLDalt2-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-10-08-ssp_none-MLDalt2-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-10-08-ssp_none-MLDalt2-all_lat_lon-time_stepping4.nc',]

experiment_attrs = ['adding max AT to alt MLD with 0.1 percent threshold before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1 (1 year) for two years',
                    'adding max AT to alt MLD with 0.1 percent threshold before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first year, then dt = 1/12 (1 month) for the second year',
                    'adding max AT to alt MLD with 0.1 percent threshold before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first year, then dt = 1 (1 year) for the second year',
                    'adding max AT to alt MLD with 0.1 percent threshold before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first month, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year',
                    'adding max AT to alt MLD with 0.1 percent threshold before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/8640 (1 hour) for the first day, then dt = 1/360 (1 day) for the next 29 days, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year']

#%% SET EXPERIMENTAL VARIABLES: WEEKEND RUN 3
# - length of time of experiment/time stepping
# - depth of addition
# - location of addition
# - emissions scenarios
# - experiment names
# in this experiment, amount of addition is set as the maximum amount of AT
# that can be added to a grid cell before exceeding preindustrial pH, so it is
# not treated as a variable

# TIME
#nyears = 5
#dt = 1 # 1 year
#t = np.arange(0, nyears, dt) # 200 years after year 0 (for now)
#nt = len(t)

#nyears = 2
#dt = 1/360 # 1 day
#t = np.arange(0, nyears, dt) # 5 years after year 0 (for now)
#nt = len(t)

dt0 = 1/8640 # 1 hour
dt1 = 1/360 # 1 day
dt2 = 1/12 # 1 month
dt3 = 1 # 1 year

# just year time steps
exp0_t = np.arange(0,50+dt3,dt3)

# experiment with dt = 1/12 (1 month) time steps
exp1_t = np.arange(0,15+dt2,dt2)

# experiment with dt = 1/360 (1 day) time steps
exp2_t = np.arange(0,1+dt1,dt1)

# another with dt = 1/8640 (1 hour) time steps
exp3_t = np.arange(0,0.05+dt0,dt0)

exp_t = [exp0_t, exp1_t, exp2_t, exp3_t]
exp_t = [exp1_t, exp2_t, exp3_t]

#exp_t = [exp1_t]

# DEPTHS OF ADDITION

# to do addition in mixed layer...
# pull mixed layer depth at each lat/lon from OCIM model data, then create mask
# of ocean cells that are at or below the mixed layer depth
mld = model_data.mld.values # [m]
# create 3D mask where for each grid cell, mask is set to 1 if the depth in the
# grid cell depths array is less than the mixed layer depth for that column
# note: this does miss cells where the MLD is close but does not reach the
# depth of the next grid cell below (i.e. MLD = 40 m, grid cell depths are at
# 30 m and 42 m, see lon_idx, lat_idx = 20, 30). I am intentionally leaving
# this for now to ensure what enters the ocean stays mostly within the mixed
# layer, but the code could be changed to a different method if needed.exp2_t

mldmask = (grid_cell_depth < mld[None, :, :]).astype(int)
q_AT_depths = mldmask

plot_mlds = mldmask.sum(axis=0)

#p2.plot_surface2d(model_lon, model_lat, plot_mlds, 0, 30, 'viridis', 'how many depth layers deep does mld go')

# to do addition in first (or first two, or first three, etc.) model layer(s)
#q_AT_depths = ocnmask.copy()
#q_AT_depths[1::, :, :] = 0 # all ocean grid cells in surface layer (~10 m) are 1, rest 0
#q_AT_depths[2::, :, :] = 0 # all ocean grid cells in top 2 surface layers (~30 m) are 1, rest 0
#q_AT_depths[3::, :, :] = 0 # all ocean grid cells in top 3 surface layers (~50 m) are 1, rest 0

# to do all lat/lons
q_AT_latlons = ocnmask[0,:,:].copy()

# to constrain lat/lon of addition to LME(s)
# get masks for each large marine ecosystem (LME)
#lme_masks, lme_id_to_name = p2.build_lme_masks(data_path + 'LMES/LMEs66.shp', ocnmask, model_lat, model_lon)
#p2.plot_lmes(lme_masks, ocnmask, model_lat, model_lon) # note: only 62 of 66 can be represented on OCIM grid
#lme_idx = [22,52] # subset of LMEs
#lme_idx = list(lme_masks.keys()) # all LMES
#q_AT_latlons = sum(lme_masks[idx] for idx in lme_idx)

# COMBINE DEPTH + LAT/LON OF ADDITION
q_AT_locations_mask = q_AT_depths * q_AT_latlons

# EMISSIONS SCENARIOS
# no emissions scenario
#q_emissions = np.zeros(nt)

# with emissions scenario
scenarios = ['none', 'none', 'none', 'none']

# EXPERIMENT NAMES AND DESCRIPTIONS

experiment_names = ['exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1yr_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1month_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1day_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1hr_LONG.nc']

experiment_attrs = ['adding max AT before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1 (1 year) for two years',
                    'adding max AT before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/12 (1 month) for the first year, then dt = 1/12 (1 month) for the second year',
                    'adding max AT before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first year, then dt = 1 (1 year) for the second year',
                    'adding max AT before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first month, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year']

experiment_names = ['exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1month_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1day_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1hr_LONG.nc']

experiment_attrs = ['adding max AT before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/12 (1 month) for the first year, then dt = 1/12 (1 month) for the second year',
                    'adding max AT before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first year, then dt = 1 (1 year) for the second year',
                    'adding max AT before reaching preind pH to all cells within max annual mixed layer across full ocean surface using no emissions scenario and dt = 1/360 (1 day) for the first month, then dt = 1/12 (1 month) for 11 months, then dt = 1 (1 year) for the second year']

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

#%% set up air-sea gas exchange (Wanninkhof, 2014)

# regrid NCEP/DOE reanalysis II data
#p2.regrid_ncep_noaa(data_path, 'icec', model_lat, model_lon, ocnmask)
#p2.regrid_ncep_noaa(data_path, 'wspd', model_lat, model_lon, ocnmask)
#p2.regrid_ncep_noaa(data_path, 'sst', model_lat, model_lon, ocnmask)

# upload regridded NCEP/DOE reanalysis II data
f_ice_2D = np.load(data_path + 'NCEP_DOE_Reanalysis_II/icec.npy') # annual mean ice fraction from 0 to 1 in each grid cell
wspd_2D = np.load(data_path + 'NCEP_DOE_Reanalysis_II/wspd.npy') # annual mean of forecast of wind speed at 10 m [m/s]
sst_2D = np.load(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst.npy') # annual mean sea surface temperature [ºC]

# calculate Schmidt number using Wanninkhof 2014 parameterization
vec_schmidt = np.vectorize(p2.schmidt)
Sc_2D = vec_schmidt('CO2', sst_2D)

# solve for k (gas transfer velocity) for each ocean cell
a = 0.251 # from Wanninkhof 2014
k_2D = a * wspd_2D**2 * (Sc_2D/660)**-0.5 # [cm/h] from Yamamoto et al., 2024, adapted from Wanninkhof 2014

k_2D *= (24*365.25/100) # [m/yr] convert units

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

#%% loop through multiple experiments

for exp_idx in range(len(experiment_names)):
    t = exp_t[exp_idx] # time steps (starting from zero) [yr]
    nt = len(t) # total number of time steps
    dt = np.diff(t, prepend=np.nan) # difference between each time step [yr]
    
    print('\nnow running experiment ' + experiment_names[exp_idx] + '\n')

    # set up emissions scenario
    
    # get annual emissions
    q_emissions = np.zeros(nt)

    if scenarios[exp_idx] != 'none':
    
        _, _, emissions_annual = p2.get_emissions_scenario(data_path, scenarios[exp_idx])
        emissions_annual = emissions_annual[2015::]
        
        nyears_emissions = len(emissions_annual)
        integrated_emissions = np.zeros(nt, dtype=float)
        
        # for each timestep, compute overlap with all simulation years that
        # intersect the emissions scenario interval
        #for idx in range(0,1):
        for idx in range(0,nt-1):
            a = t[idx] # interval start
            b = t[idx+1] # interval end
            
            # find if there are years that could overlap in this interval (i.e.
            # if an interval spans more than one year)
            y_start = int(np.floor(a))
            y_end = int(np.floor(b - 1e-12)) # inclusive last year index that starts before b
            
            # clamp to available annual emissions indices (i.e. assume
            # emissions beyond data available are zero)
            y0 = max(0, y_start)
            y1 = min(nyears_emissions - 1, y_end)
            
            # for all year indicies "y" that might overlap with a timestep,
            # calculate the length of the interval, get integrated value, and 
            # accumulate emissions (if > 1 year time step)
            for y in range(y0, y1 + 1):
                year_start = float(y)
                year_end = float(y + 1)
                overlap_start = max(a, year_start)
                overlap_end = min(b, year_end)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > 0:
                    integrated_emissions[idx+1] += emissions_annual[y] * overlap # emissions_annual [mol CO2 (mol air)-1 yr-1] * overlap [yr] -> mol CO2 (mol air)-1
       
        # calculate rate of emissions from integraetd emissions
        q_emissions[1::] = integrated_emissions[1::] / dt[1::]

    # construct matrix C
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
    
    # construct initial q vector (it is going to change every iteration)
    # q = [ 0                                                                           ] --> 1 * nt, q[0]
    #     [ ∆q_CDR,DIC + ∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc      ] --> m * nt, q[1:(m+1)]
    #     [ ∆q_CDR,AT + 2 * (∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc) ] --> m * nt, q[(m+1):(2*m+1)]
    
    # which translates to...
    # q = [ 0                                      ] --> 1 * nt, q[0]
    #     [ ∆q_CDR,DIC + ∆q_diss,DIC - ∆q_prod,DIC ] --> m * nt, q[1:(m+1)]
    #     [ ∆q_CDR,AT + ∆q_diss,AT - ∆q_prod,AT    ] --> m * nt, q[(m+1):(2*m+1)]
    
    q = np.zeros((1 + 2*m, nt))
    
    # time stepping simulation forward
    
    # set initial baseline to evaluate if co2sys recalculation is needed
    AT_at_last_calc = AT.copy()
    DIC_at_last_calc = DIC.copy()
    AT_current = AT.copy()
    DIC_current = DIC.copy()
  
    # not calculating delAT/delDIC/delxCO2 at time = 0 (this time step is initial conditions only)
    for idx in tqdm(range(1,nt)):
        # AT and DIC are equal to initial AT and DIC + whatever the change
        # in AT and DIC seen in previous time step are
        AT_current = AT + c[(m+1):(2*m+1), idx-1]
        DIC_current = DIC + c[1:(m+1), idx-1]
        
        # recalculate carbonate system every time >5% of grid cells see change
        # in AT or DIC >5% since last recalculation
        frac_AT = np.mean(np.abs(AT_current - AT_at_last_calc) > 0.05 * np.abs(AT_at_last_calc)) # calculate fraction of grid cells with change in AT above 10%
        frac_DIC = np.mean(np.abs(DIC_current - DIC_at_last_calc) > 0.05 * np.abs(DIC_at_last_calc)) # calculate fraction of grid cells with change in DIC above 10%
        print('\nfraction of cells with change in AT >5%: ' + str(frac_AT))
        print('fraction of cells with change in DIC >5%: ' + str(frac_DIC)) 
 
        # (re)calculate carbonate system if it has not yet been calculated or
        # needs to be recalculated
        # not calculating delAT/delDIC/delxCO2 at time = 0 (this time step is initial conditions only)
        if idx == 1 or frac_AT > 0.05 or frac_DIC > 0.05:
            AT_at_last_calc = AT_current.copy()
            DIC_at_last_calc = DIC_current.copy()
            # use CO2SYS with GLODAP data to solve for carbonate system at each grid cell
            # do this for only surface ocean grid cells
            # this is PyCO2SYSv2
            co2sys = pyco2.sys(dic=DIC_current, alkalinity=AT_current,
                               salinity=S, temperature=T, pressure=pressure,
                               total_silicate=Si, total_phosphate=P)
        
            # extract key results arrays
            pCO2 = co2sys['pCO2'] # pCO2 [µatm]
            aqueous_CO2 = co2sys['CO2'] # aqueous CO2 [µmol kg-1]
            R_C = co2sys['revelle_factor'] # revelle factor w.r.t. DIC [unitless]
        
            # calculate revelle factor w.r.t. AT [unitless]
            # must calculate manually, R_AT defined as (dpCO2/pCO2) / (dAT/AT)
            # to speed up, only calculating this in surface
            co2sys_000001 = pyco2.sys(dic=DIC_current[0:ns], alkalinity=AT_current[0:ns]+0.000001, salinity=S[0:ns],
                                   temperature=T[0:ns], pressure=pressure[0:ns], total_silicate=Si[0:ns],
                                   total_phosphate=P[0:ns])
        
            pCO2_000001 = co2sys_000001['pCO2']
            R_A_surf = ((pCO2_000001 - pCO2[0:ns])/pCO2[0:ns]) / (0.000001/AT[0:ns])
            R_A = np.full(R_C.shape, np.nan)
            R_A[0:ns] = R_A_surf
         
            # see if average pH has exceeded preindustrial average pH
            avg_pH = np.average(co2sys['pH'], weights=p2.flatten(model_vols,ocnmask))
            if avg_pH > avg_pH_preind:
                print('average pH exceeded average preindustrial pH at time step ' + str(idx))
                break
            else:
                print('average pH at time step ' + str(idx) + ' = ' + str(round(avg_pH,8)))
        
            # calculate rest of Nowicki et al. parameters
            beta_C = DIC/aqueous_CO2 # [unitless]
            beta_A = AT/aqueous_CO2 # [unitless]
            K0 = aqueous_CO2/pCO2*rho # [µmol CO2 m-3 (µatm CO2)-1], in derivation this is defined in per volume units so used density to get there
            
            print('carbonate system recalculated (t = ' + str(t[idx]) + ')')
        
        # must (re)calculate A matrix if 1. it has not yet been calculated
        # 2. the carbonate system needs to be recalculated or 3. the time
        # step interval (dt) has changed
        if idx == 1 or frac_AT > 0.05 or frac_DIC > 0.05 or np.round(dt[idx],10) != np.round(dt[idx-1],10):
            
            if frac_AT > 0.05:
                print('frac_AT > 0.05')
                
            if frac_DIC > 0.05:
                print('frac_DIC > 0.05')
                
            if dt[idx] != dt[idx-1]:
                print('dt[' + str(idx) + '] = ' + str(dt[idx]))
                print('dt[' + str(idx-1) + '] = ' + str(dt[idx-1]))
        
            # calculate "A" matrix
        
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
            LHS = sparse.eye(A.shape[0], format="csc") - dt[idx] * A
    
            # test condition number of matrix
            est = sparse.linalg.onenormest(LHS)
            print('estimated 1-norm condition number LHS: ' + str(round(est,1)))
        
            start = time()
            ilu = spilu(LHS.tocsc(), drop_tol=1e-5, fill_factor=20)
            stop = time()
            print('ilu calculations: ' + str(stop - start) + ' s\n')
        
            M = LinearOperator(LHS.shape, ilu.solve)
        
        # add CDR perturbation (construct q vector, it is going to change every iteration in this experiment)
        # for now, assuming NaOH (no change in DIC)
        
        # calculate AT required to return to preindustrial pH
        # using DIC at previous time step (initial DIC + modeled change in DIC) and preindustrial pH
        DIC_new = DIC + c[1:(m+1), idx-1]
        AT_new = AT + c[(m+1):(2*m+1), idx-1]
        AT_to_offset = p2.calculate_AT_to_add(pH_preind, DIC_new, AT_new, T, S, pressure, Si, P, low=0, high=200, tol=1e-6, maxiter=50)

        # make sure there are no negative values
        if len(AT_to_offset[AT_to_offset<0]) != 0:
            print('error: AT offset is negative')
            break

        # from this offset, calculate rate at which AT must be applied
        # by solving discretized equation for q(t)
        # OCIM manual eqn. 40: (I - dt * TR) * c_t = c_(t-dt) + dt * q(t)
        # solve for q(t): q(t) = [(I - dt * TR) * c_t - c_(t-dt)] / dt
        # --> define c_t as modern DIC + whatever AT required to get to
        #     preind pH
        # --> define c_(t-1) as preindustrial DIC + preindustrial AT (which
        #     we are saying is the same as GLODAP AT)
        # set CDR perturbation equal to this RATE in mixed layer only
        # ∆q_CDR,AT (change in alkalinity due to CDR addition) - final units: [µmol AT kg-1 yr-1]
        del_q_CDR_AT = (((sparse.eye(TR.shape[0], format="csc") - dt[idx] * TR) * (AT + AT_to_offset) - AT)) / dt[idx]
        del_q_CDR_AT *= p2.flatten(q_AT_locations_mask, ocnmask) # apply in mixed layer only
 
        # add in source/sink vectors for ∆AT to q vector
        q[(m+1):(2*m+1), idx] = del_q_CDR_AT
        
        # add in emissions scenario (∆xCO2) to q vector as well
        q[0,idx] = q_emissions[idx] # [mol CO2 (mol air)-1]
        
        # add starting guess after first time step
        if idx > 1:
            c0 = c[:,idx-1]
        else:
            c0=None
    
        # calculate right hand side and perform time stepping
        RHS = c[:,idx-1] + np.squeeze(dt[idx] * q[:,idx])
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
    # convert from flux (amount yr-1) to amount by multiplying by dt [yr]
    q_delxCO2 = q[0, :] * dt
    q_delDIC  = q[1:(m+1), :] * dt
    q_delAT   = q[(m+1):(2*m+1), :] * dt
    
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
    global_attrs = {'description': experiment_attrs[exp_idx]}
    # save model output
    p2.save_model_output(
        experiment_names[exp_idx], 
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
data = xr.open_dataset(output_path + 'exp16_2025-09-26-ssp126-MLD-all_lat_lon.nc')
scenario = 'ssp126'
t = data['time'].values

nt = len(t)
nt_plot = 5
nyears=2

#nyears = 2
#dt = 1/360 # 1 day
#t = np.arange(0, nyears, dt) # 5 years after year 0 (for now)
#nt = len(t)
#nt_plot = nt

DIC_broadcasted = xr.DataArray(DIC_3D, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast DIC to convert ∆DIC to total DIC over time
AT_broadcasted = xr.DataArray(AT_3D, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast DIC to convert ∆DIC to total DIC over time

DIC_modeled_3D = data.delDIC + DIC_broadcasted
AT_modeled_3D = data.delAT + AT_broadcasted

#DIC_modeled_3D = data.DIC_added + DIC_broadcasted
#AT_modeled_3D = data.AT_added + AT_broadcasted

pH_modeled = []
del_pH_modeled = []
del_pH_preind_modeled = []
avg_pH_modeled = np.zeros(nt)
avg_pH_modeled_surf = np.zeros(nt)

avg_AT_modeled = np.zeros(nt)
avg_AT_modeled_surf = np.zeros(nt)

avg_DIC_modeled = np.zeros(nt)
avg_DIC_modeled_surf = np.zeros(nt)

for idx in range(nt):
    DIC_modeled = p2.flatten(DIC_modeled_3D.isel(time=idx).values, ocnmask)
    AT_modeled = p2.flatten(AT_modeled_3D.isel(time=idx).values, ocnmask)        
    co2sys = pyco2.sys(dic=DIC_modeled, alkalinity=AT_modeled, salinity=S, temperature=T,
                       pressure=pressure, total_silicate=Si, total_phosphate=P)
    
    pH_modeled.append(co2sys['pH'])
    del_pH_modeled.append(co2sys['pH'] - pH_preind)
    del_pH_preind_modeled.append(co2sys['pH'] - pH_preind)
    
    avg_pH_modeled[idx] = np.average(co2sys['pH'], weights=p2.flatten(model_vols,ocnmask))
    avg_pH_modeled_surf[idx] = np.average(co2sys['pH'][0:ns], weights=p2.flatten(model_vols,ocnmask)[0:ns])
    
    avg_AT_modeled[idx] = np.average(AT_modeled, weights=p2.flatten(model_vols,ocnmask))
    avg_AT_modeled_surf[idx] = np.average(AT_modeled[0:ns], weights=p2.flatten(model_vols,ocnmask)[0:ns])
    
    avg_DIC_modeled[idx] = np.average(DIC_modeled, weights=p2.flatten(model_vols,ocnmask))
    avg_DIC_modeled_surf[idx] = np.average(DIC_modeled[0:ns], weights=p2.flatten(model_vols,ocnmask)[0:ns])
    
    #print(np.nanmean(co2sys['pH']))
    pH_modeled_3D = p2.make_3D(co2sys['pH'], ocnmask)
    #print(np.nanmean(pH_modeled_3D[0,:,:]))
    #print(np.nanmax(pH_modeled_3D[0,:,:]))
    #p2.plot_surface3d(data.lon, data.lat, pH_preind_3D - pH_modeled_3D, 0, -0.5, 0.5, 'RdBu', 'pH difference from preindustrial at year ' + str(data['time'].isel(time=idx).values))

#for idx in range(nt-1):
#    p2.plot_surface3d(data.lon, data.lat, data['AT_added'].isel(time=idx).values, 0, 0, 100, 'viridis', 'AT (µmol kg-1) added at year ' + str(data['time'].isel(time=idx).values))


#%% make figure of annual alkalinity change each year vs. average ocean pH
years = t + 2015
model_vols_xr = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast model_vols to convert ∆AT from per kg to total

AT_added = data['AT_added'] * model_vols_xr * rho * 1e-6
AT_added = AT_added.sum(dim=['depth', 'lon', 'lat'], skipna=True).values
AT_added_cum = np.cumsum(AT_added)

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.plot(years[0:nt_plot], avg_pH_modeled[0:nt_plot], label='pH under max OAE')
ax.axhline(np.nanmean(pH_preind), c='black', linestyle='--', label='preindustrial pH') # add line showing preindustrial surface pH

# set up secondary axis "years"
year_to_AT = interp1d(years, AT_added_cum, kind='linear', fill_value="extrapolate")
AT_to_year = interp1d(AT_added_cum, years, kind='linear', fill_value="extrapolate")
secax = ax.secondary_xaxis('top', functions=(year_to_AT, AT_to_year))
secax.set_xlabel('total amount of AT added (mol)')

ax.set_ylabel('average ocean pH (weighted by grid cell volume)')
ax.set_xlabel('year')
ax.set_xlim([2015, 2020])
ax.set_ylim([7.8, 8.1])
plt.legend(loc = 'lower right')
plt.show()

# same thing, but for surface ocean

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.plot(years[0:nt_plot], avg_pH_modeled_surf[0:nt_plot], label='pH under max OAE')
ax.axhline(np.nanmean(pH_preind[0:ns]), c='black', linestyle='--', label='preindustrial pH') # add line showing preindustrial surface pH

secax = ax.secondary_xaxis('top', functions=(year_to_AT, AT_to_year))
secax.set_xlabel('total amount of AT added (mol)')

ax.set_ylabel('average surface ocean pH (weighted by grid cell volume)')
ax.set_xlabel('year')
ax.set_xlim([2015, 2020])
ax.set_ylim([8, 8.3])
plt.legend(loc = 'lower right')
plt.show()

#%% plot DIC over time

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.plot(years[0:nt_plot], avg_DIC_modeled_surf[0:nt_plot], label='DIC under max OAE')
ax.axhline(np.nanmean(DIC_preind[0:ns]), c='black', linestyle='--', label='preindustrial DIC') # add line showing preindustrial surface DIC

secax = ax.secondary_xaxis('top', functions=(year_to_AT, AT_to_year))
secax.set_xlabel('total amount of AT added (mol)')

ax.set_ylabel('average surface ocean DIC (weighted by grid cell volume)')
ax.set_xlabel('year')
ax.set_xlim([2015, 2020])
ax.set_ylim([1950, 2350])
plt.legend(loc = 'upper right')
plt.show()

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.plot(years[0:nt_plot], avg_DIC_modeled[0:nt_plot], label='DIC under max OAE')
ax.axhline(np.nanmean(DIC_preind), c='black', linestyle='--', label='preindustrial DIC') # add line showing preindustrial surface DIC

secax = ax.secondary_xaxis('top', functions=(year_to_AT, AT_to_year))
secax.set_xlabel('total amount of AT added (mol)')

ax.set_ylabel('average ocean DIC (weighted by grid cell volume)')
ax.set_xlabel('year')
ax.set_xlim([2015, 2020])
ax.set_ylim([1950, 2350])
plt.legend(loc = 'lower right')
plt.show()

#%% plot AT over time

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.plot(years[0:nt_plot], avg_AT_modeled_surf[0:nt_plot], label='AT under max OAE')
ax.axhline(np.nanmean(AT[0:ns]), c='black', linestyle='--', label='glodap AT') # add line showing preindustrial surface DIC

secax = ax.secondary_xaxis('top', functions=(year_to_AT, AT_to_year))
secax.set_xlabel('total amount of AT added (mol)')

ax.set_ylabel('average surface ocean AT (weighted by grid cell volume)')
ax.set_xlabel('year')
ax.set_xlim([2015, 2020])
ax.set_ylim([2200, 2550])
plt.legend(loc = 'lower right')
plt.show()

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.plot(years[0:nt_plot], avg_AT_modeled[0:nt_plot], label='AT under max OAE')
ax.axhline(np.nanmean(AT), c='black', linestyle='--', label='glodap AT') # add line showing preindustrial surface DIC

secax = ax.secondary_xaxis('top', functions=(year_to_AT, AT_to_year))
secax.set_xlabel('total amount of AT added (mol)')

ax.set_ylabel('average ocean AT (weighted by grid cell volume)')
ax.set_xlabel('year')
ax.set_xlim([2015, 2020])
ax.set_ylim([2200, 2550])
plt.legend(loc = 'lower right')
plt.show()

#%% plot atmospheric CO2 over time

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.plot(years[0:nt_plot], data['delxCO2'].values[0:nt_plot] + 400, label='atmospheric CO2 under max OAE')

_, emissions_cumulative, _ = p2.get_emissions_scenario(data_path, scenario)
emissions_cumulative = emissions_cumulative[2015:nt+2015] * 1e6

# set up emissions scenario (right now, can only do 1 dt per simulation, but dt can be anything)
_, _, emissions_annual = p2.get_emissions_scenario(data_path, scenario)
q_emissions = np.zeros(nt)

steps_per_year = int(1 / dt)

for year in range(nyears):
    annual_val = emissions_annual[year]
    per_step_val = annual_val / steps_per_year
    start_idx = year * steps_per_year
    end_idx = start_idx + steps_per_year
    q_emissions[start_idx:end_idx] = per_step_val
    
emissions_cumulative = np.cumsum(q_emissions) + 400

# this isn't correct, because emissions will still equilibrate without OAE --> need to run actual counterfactual with no OAE
#ax.plot(years[0:nt_plot], emissions_cumulative[0:nt_plot], label='atmospheric CO2 without OAE') 

ax.axhline(280, c='black', linestyle='--', label='preindustrial atmospheric CO2') # add line showing preindustrial surface pH

secax = ax.secondary_xaxis('top', functions=(year_to_AT, AT_to_year))
secax.set_xlabel('total amount of AT added (mol)')

ax.set_ylabel('average atmospheric CO2 (ppm)')
ax.set_xlabel('year')
ax.set_xlim([2015, 2020])
ax.set_ylim([270, 410])
plt.legend(loc = 'center right')
plt.show()


#%% plot AT added over time

years = t + 2015

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.plot(years[0:nt_plot], AT_added[0:nt_plot], label='AT added')
ax.plot(years[0:nt_plot], np.cumsum(AT_added)[0:nt_plot], label='cumulative AT added')

ax.set_ylabel('AT (mol)')
ax.set_xlabel('year')
ax.set_xlim([2015, 2020])
ax.set_ylim([-1e17, 9e17])
plt.legend(loc = 'upper right')
plt.show()

#%% make clip of surface AT over time
colorbar_label = 'Change in Total Alkalinity (µmol kg$^{-1}$)'
p2.make_surf_animation(data['delAT'], colorbar_label, model_lon, model_lat, t, nt_plot, -300, 300, 'RdBu', 'surface_delAT.mp4')

colorbar_label = 'Total Alkalinity (µmol kg$^{-1}$)'
p2.make_surf_animation(data['delAT'] + p2.make_3D(AT,ocnmask), colorbar_label, model_lon, model_lat, t, nt_plot, 2100, 2700, 'viridis', 'surface_AT.mp4')

#%% make clip of surface DIC over time
colorbar_label = 'Change in Dissolved Inorganic Carbon (µmol kg$^{-1}$)'
p2.make_surf_animation(data['delDIC'], colorbar_label, model_lon, model_lat, t, nt_plot, -300, 300, 'RdBu', 'surface_delDIC.mp4')

colorbar_label = 'Dissolved Inorganic Carbon (µmol kg$^{-1}$)'
p2.make_surf_animation(data['delDIC'] + p2.make_3D(DIC,ocnmask), colorbar_label, model_lon, model_lat, t, nt_plot, 1850, 2350, 'viridis', 'surface_DIC.mp4')

#%% make clip of surface pH over time
colorbar_label = 'pH'
p2.make_surf_animation_pH(pH_modeled, colorbar_label, model_lon, model_lat, t, nt_plot, ocnmask, 7.5, 8.5, 'viridis', 'surface_pH.mp4')

colorbar_label = 'deviation in pH from preindustrial'
p2.make_surf_animation_pH(del_pH_preind_modeled, colorbar_label, model_lon, model_lat, t, nt_plot, ocnmask, -0.5, 0.5, 'RdBu', 'surface_delpH_preind.mp4')

#%% make clip of change in section of AT over time
colorbar_label = 'Change in Total Alkalinity (µmol kg$^{-1}$)'
p2.make_section_animation(data['delAT'], colorbar_label, model_depth, model_lat, t, nt_plot, -300, 300, 'RdBu', 'section_delAT.mp4')

#%% make clip of change in section of DIC over time
colorbar_label = 'Change in Dissolved Inorganic Carbon (µmol kg$^{-1}$)'
p2.make_section_animation(data['delDIC'], colorbar_label, model_depth, model_lat, t, nt_plot, -300, 300, 'RdBu', 'section_delDIC.mp4')

#%% make clip of change in section of pH over time
colorbar_label = 'pH'
p2.make_section_animation_pH(pH_modeled, pH_preind, colorbar_label, model_depth, model_lat, t, nyears, ocnmask, 7.5, 8.5, 'viridis', 'section_pH.mp4')

colorbar_label = 'deviation in pH from preindustrial'
p2.make_section_animation_pH(del_pH_preind_modeled, colorbar_label, model_depth, model_lat, t, nyears, ocnmask, -0.5, 0.5, 'RdBu', 'section_delpH_preind.mp4')

#%% testing thresholding for co2sys calcs
c = np.zeros((1 + 2*m, nt))

for idx in range(nt):
    c[0, idx] = data['delxCO2'].isel(time=idx).values
    c[1:(m+1), idx] = p2.flatten(data['delDIC'].isel(time=idx).values, ocnmask)
    c[(m+1):(2*m+1), idx] = p2.flatten(data['delAT'].isel(time=idx).values, ocnmask)

# set initial baseline to evaluate if co2sys recalculation is needed
AT_at_last_calc = AT.copy()
DIC_at_last_calc = DIC.copy()
AT_current = AT.copy()
DIC_current = DIC.copy()
  
for idx in range(0,186):
    if idx > 0:
        # AT and DIC are equal to initial AT and DIC + whatever the change in AT and DIC seen in previous time step are
        AT_current = AT + c[(m+1):(2*m+1), idx-1]
        DIC_current = DIC + c[1:(m+1), idx-1]
    
    # recalculate carbonate system every time >5% of grid cells see change in AT or DIC >5% since last recalculation
    frac_AT = np.mean(np.abs(AT_current - AT_at_last_calc) > 0.05 * np.abs(AT_at_last_calc)) # calculate fraction of grid cells with change in AT above 10%
    frac_DIC = np.mean(np.abs(DIC_current - DIC_at_last_calc) > 0.05 * np.abs(DIC_at_last_calc)) # calculate fraction of grid cells with change in DIC above 10%
    #print('\n idx = ' + str(idx))
    #print('fraction of cells with change in AT >5%: ' + str(frac_AT))
    #print('fraction of cells with change in DIC >5%: ' + str(frac_DIC))
 
    if idx == 0 or frac_AT > 0.05 or frac_DIC > 0.05:
        AT_at_last_calc = AT_current.copy()
        DIC_at_last_calc = DIC_current.copy()
        print('carbonate system (re)calculated at year: ' + str(idx+2015))

#%% make figure of amount of AT added over time, comparing across scenarios

labels = ['time stepping 0', 'time stepping 1', 'time stepping 2', 'time stepping 3', 'time stepping 4']

experiment_names = ['exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping1.nc']
'''

labels = ['dt = 1 yr', 'dt = 1 month', 'dt = 1 day', 'dt = 1 hr']

experiment_names = ['exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1yr_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1month_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1day_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1hr_LONG.nc']
'''


fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

for exp_idx in range(0,len(experiment_names)):
    
    data = xr.open_dataset(output_path + experiment_names[exp_idx])
    t = data['time'].values
    nt = len(t)
        
    model_vols_xr = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast model_vols to convert ∆AT from per kg to total

    AT_added = data['AT_added'] * model_vols_xr * rho * 1e-6
    AT_added = AT_added.sum(dim=['depth', 'lon', 'lat'], skipna=True).values
    AT_added_cum = np.cumsum(AT_added)
    
    ax.plot(t[0:nt] + 2015, AT_added_cum[0:nt], label=labels[exp_idx])
    
plt.legend()
#plt.xlim([2015, 2017])
#plt.ylim([-0.25e17, 5e17])
#ax.set_xticks([2015, 2016, 2017])
plt.xlabel('year')
plt.ylabel('AT added to mixed layer (mol)')
    
#%% make figure of amount of xCO2 removed added over time, comparing across scenarios

labels = ['time stepping 0', 'time stepping 1', 'time stepping 2', 'time stepping 3', 'time stepping 4']

experiment_names = ['exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-08-ssp_none-MLDalt2-all_lat_lon-time_steppingTEST1.nc']

experiment_names = ['exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping1.nc']
'''

labels = ['dt = 1 yr', 'dt = 1 month', 'dt = 1 day', 'dt = 1 hr']

experiment_names = ['exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1yr_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1month_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1day_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1hr_LONG.nc']
'''

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

for exp_idx in range(0,len(experiment_names)):
    
    data = xr.open_dataset(output_path + experiment_names[exp_idx])
    t = data['time'].values
    nt=len(t)
    
    delxCO2 = data['delxCO2'].values
    ax.plot(t[0:nt] + 2015, delxCO2[0:nt], label=labels[exp_idx])
    
plt.legend()
#plt.xlim([2015, 2017])
#ax.set_xticks([2015, 2016, 2017])
#ax.set_ylim([-260, 10])
plt.xlabel('year')
plt.ylabel('change in atmospheric CO2 (ppm)')
    
#%% make figure of average ocean pH change each year, comparing across scenarios

labels = ['time stepping 0', 'time stepping 1', 'time stepping 2', 'time stepping 3', 'time stepping 4']

experiment_names = ['exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-08-ssp_none-MLDalt2-all_lat_lon-time_steppingTEST1.nc']

experiment_names = ['exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping1.nc']


labels = ['dt = 1 yr', 'dt = 1 month', 'dt = 1 day', 'dt = 1 hr']

experiment_names = ['exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1yr_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1month_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1day_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1hr_LONG.nc']


model_vols_xr = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast model_vols to convert ∆AT from per kg to total
DIC_broadcasted = xr.DataArray(DIC_3D, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast DIC to convert ∆DIC to total DIC over time
AT_broadcasted = xr.DataArray(AT_3D, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast DIC to convert ∆DIC to total DIC over time

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.axhline(np.average(pH_preind, weights=p2.flatten(model_vols,ocnmask)), c='black', linestyle='--', label='preindustrial pH') # add line showing preindustrial surface pH

for exp_idx in range(0,len(experiment_names)):
    
    data = xr.open_dataset(output_path + experiment_names[exp_idx])
    t = data['time'].values
    nt=len(t)
    
    DIC_modeled_3D = data.delDIC + DIC_broadcasted
    AT_modeled_3D = data.delAT + AT_broadcasted

    avg_pH_modeled = np.zeros(nt)
    
    for idx in range(nt):
        DIC_modeled = p2.flatten(DIC_modeled_3D.isel(time=idx).values, ocnmask)
        AT_modeled = p2.flatten(AT_modeled_3D.isel(time=idx).values, ocnmask)        
        co2sys = pyco2.sys(dic=DIC_modeled, alkalinity=AT_modeled, salinity=S, temperature=T,
                           pressure=pressure, total_silicate=Si, total_phosphate=P)
        
        avg_pH_modeled[idx] = np.average(co2sys['pH'], weights=p2.flatten(model_vols,ocnmask))
        
    ax.plot(t + 2015, avg_pH_modeled[0:nt], label=labels[exp_idx])

ax.set_ylabel('average ocean pH (weighted by grid cell volume)')
ax.set_xlabel('year')
ax.set_xlim([2015, 2015.1])
#ax.set_xticks([2015, 2016, 2017])
#ax.set_ylim([7.838, 7.874])
plt.legend()
plt.show()

#%% same thing, but for surface ocean

labels = ['time stepping 0', 'time stepping 1', 'time stepping 2', 'time stepping 3', 'time stepping 4']

experiment_names = ['exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-08-ssp_none-MLDalt2-all_lat_lon-time_steppingTEST1.nc']

experiment_names = ['exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping1.nc']


labels = ['dt = 1 yr', 'dt = 1 month', 'dt = 1 day', 'dt = 1 hr']

experiment_names = ['exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1yr_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1month_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1day_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1hr_LONG.nc']


model_vols_xr = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast model_vols to convert ∆AT from per kg to total
DIC_broadcasted = xr.DataArray(DIC_3D, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast DIC to convert ∆DIC to total DIC over time
AT_broadcasted = xr.DataArray(AT_3D, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast DIC to convert ∆DIC to total DIC over time

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.axhline(np.average(pH_preind[0:ns], weights=p2.flatten(model_vols,ocnmask)[0:ns]), c='black', linestyle='--', label='preindustrial pH') # add line showing preindustrial surface pH

for exp_idx in range(0,len(experiment_names)):
    
    data = xr.open_dataset(output_path + experiment_names[exp_idx])
    t = data['time'].values
    nt=len(t)
    
    DIC_modeled_3D = data.delDIC + DIC_broadcasted
    AT_modeled_3D = data.delAT + AT_broadcasted

    avg_pH_modeled_surf = np.zeros(nt)
    
    for idx in range(nt):
        DIC_modeled = p2.flatten(DIC_modeled_3D.isel(time=idx).values, ocnmask)
        AT_modeled = p2.flatten(AT_modeled_3D.isel(time=idx).values, ocnmask)        
        co2sys = pyco2.sys(dic=DIC_modeled[0:ns], alkalinity=AT_modeled[0:ns],
                           salinity=S[0:ns], temperature=T[0:ns],
                           pressure=pressure[0:ns], total_silicate=Si[0:ns],
                           total_phosphate=P[0:ns])
        
        avg_pH_modeled_surf[idx] = np.average(co2sys['pH'], weights=p2.flatten(model_vols,ocnmask)[0:ns])
        
    ax.plot(t + 2015, avg_pH_modeled_surf[0:nt], label=labels[exp_idx])

ax.set_ylabel('average surface ocean pH (weighted by grid cell volume)')
ax.set_xlabel('year')
#ax.set_xlim([2015, 2015.01])
#ax.set_xticks([2015, 2016, 2017])
#ax.set_ylim([7.45, 8.25])
plt.legend()
plt.show()

#%% make figure of DIC change each year, comparing across scenarios

labels = ['time stepping 0', 'time stepping 1', 'time stepping 2', 'time stepping 3', 'time stepping 4']

experiment_names = ['exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-08-ssp_none-MLDalt2-all_lat_lon-time_steppingTEST.nc']

experiment_names = ['exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping1.nc']
'''

labels = ['dt = 1 yr', 'dt = 1 month', 'dt = 1 day', 'dt = 1 hr']

experiment_names = ['exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1yr_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1month_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1day_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1hr_LONG.nc']
'''


model_vols_xr = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast model_vols to convert ∆AT from per kg to total
DIC_broadcasted = xr.DataArray(DIC_3D, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast DIC to convert ∆DIC to total DIC over time
AT_broadcasted = xr.DataArray(AT_3D, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast DIC to convert ∆DIC to total DIC over time

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.axhline(np.average(DIC_preind, weights=p2.flatten(model_vols,ocnmask)), c='black', linestyle='--', label='preindustrial DIC') # add line showing preindustrial surface pH

for exp_idx in range(0,len(experiment_names)):
    
    data = xr.open_dataset(output_path + experiment_names[exp_idx])
    t = data['time'].values
    nt=len(t)
    
    DIC_modeled_3D = data.delDIC + DIC_broadcasted
    AT_modeled_3D = data.delAT + AT_broadcasted

    avg_DIC_modeled = np.zeros(nt)
    
    for idx in range(nt):
        DIC_modeled = p2.flatten(DIC_modeled_3D.isel(time=idx).values, ocnmask)
        avg_DIC_modeled[idx] = np.average(DIC_modeled, weights=p2.flatten(model_vols,ocnmask))
        
    ax.plot(t + 2015, avg_DIC_modeled[0:nt], label=labels[exp_idx])

ax.set_ylabel('average ocean DIC (weighted by grid cell volume)')
ax.set_xlabel('year')
#ax.set_xlim([2015, 2017])
#ax.set_xticks([2015, 2016, 2017])
#ax.set_ylim([2235, 2255])
plt.legend()
plt.show()

#%% same thing, but for surface ocean

labels = ['time stepping 0', 'time stepping 1', 'time stepping 2', 'time stepping 3', 'time stepping 4']

experiment_names = ['exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-09-26-ssp_none-MLD-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-10-06-ssp_none-MLD_builtin-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-07-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-07-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping1.nc']

experiment_names = ['exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping1.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping2.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping3.nc',
                    'exp16_2025-10-07-ssp_none-MLDalt2-all_lat_lon-time_stepping4.nc',]

experiment_names = ['exp16_2025-10-08-ssp_none-MLDalt2-all_lat_lon-time_steppingTEST.nc']

experiment_names = ['exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping0.nc',
                    'exp16_2025-10-08-ssp_none-MLD_builtinOLD-all_lat_lon-time_stepping1.nc']
'''

labels = ['dt = 1 yr', 'dt = 1 month', 'dt = 1 day', 'dt = 1 hr']

experiment_names = ['exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1yr_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1month_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1day_LONG.nc',
                    'exp16_2025-10-08-ssp_none-MLD-all_lat_lon-dt_1hr_LONG.nc']
'''


model_vols_xr = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast model_vols to convert ∆AT from per kg to total
DIC_broadcasted = xr.DataArray(DIC_3D, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast DIC to convert ∆DIC to total DIC over time
AT_broadcasted = xr.DataArray(AT_3D, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast DIC to convert ∆DIC to total DIC over time

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.axhline(np.nanmean(DIC_preind[0:ns]), c='black', linestyle='--', label='preindustrial DIC') # add line showing preindustrial surface pH

for exp_idx in range(0,len(experiment_names)):
    
    data = xr.open_dataset(output_path + experiment_names[exp_idx])
    t = data['time'].values
    nt=len(t)
    
    DIC_modeled_3D = data.delDIC + DIC_broadcasted
    AT_modeled_3D = data.delAT + AT_broadcasted

    avg_DIC_modeled_surf = np.zeros(nt)
    
    for idx in range(nt):
        DIC_modeled = p2.flatten(DIC_modeled_3D.isel(time=idx).values, ocnmask)        
        avg_DIC_modeled_surf[idx] = np.average(DIC_modeled[0:ns], weights=p2.flatten(model_vols,ocnmask)[0:ns])
        
    ax.plot(t + 2015, avg_DIC_modeled_surf[0:nt], label=labels[exp_idx])

ax.set_ylabel('average surface ocean DIC (weighted by grid cell volume)')
ax.set_xlabel('year')
#ax.set_xlim([2015, 2017])
#ax.set_xticks([2015, 2016, 2017])
#ax.set_ylim([1980, 2065])
plt.legend()
plt.show()
