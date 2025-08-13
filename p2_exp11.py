#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:15:02 2025

EXP11: Trying to replicate Yamamoto et al., 2024 results ("instantaneous OAE")
- Assuming that ∆q_diss term is equal to 0.
exp11_2025-7-21-a.nc
- Testing ILU + GMRES
- Perturbation at a point in top ocean layer for the first 30 days
- Shortened time steps after
exp11_2025-7-21-b.nc
- Testing ILU + LGMRES
- Perturbation of at a point in top ocean layer for the first 30 days
- Shortened time steps after
exp11_2025-7-31-a.nc
- RUNNING SOMETHING TO COMPARE WITH KANA RESULTS
- Perturbation of -1 µmol kg-1 yr-1 DIC at (-39.5, 101) in top ocean layer for the first 30 days
- This turned out to have a units error, hopefully that's why its' efficiency was too low
exp11_2025-8-1-a.nc
- Kana comparison a: perturbation of -1 µmol kg-1 yr-1 DIC at (-39.5, 101) in top ocean layer for the first 30 days
exp11_2025-8-1-b.nc
- Kana comparison b: perturbation of -1 µmol kg-1 yr-1 DIC at (-39.5, 101) in top ocean layer for the first 30 days
exp11_2025-8-1-c.nc
- Kana comparison c: perturbation of -1 µmol kg-1 yr-1 DIC at (-39.5, 101) in top ocean layer for the first 30 days
exp11_2025-8-1-d.nc
- Kana comparison d: perturbation of -1 µmol kg-1 yr-1 DIC at (-39.5, 101) in top ocean layer for the first 30 days
exp11_2025-8-6-a.nc
- Same simulation as exp11_2025-8-1-a.nc except only running for 5 years and getting rid of AT factor change to see if I get the same results as kana
exp11_2025-8-6-b.nc
- Same simulation as exp11_2025-8-1-a.nc except only running for 5 years and getting rid of AT factor change to see if I get the same results as kana
- ALSO using OCIM1 to see if results change
exp11_2025-8-6-a.nc
- Same simulation as exp11_2025-8-1-a.nc except only running for 5 years
- ALSO setting rtol = 1e-7 in fgmres solver
exp11_2025-8-6-b.nc
- Same simulation as exp11_2025-8-1-a.nc except only running for 5 years
- ALSO setting rtol = 1e-7 in fgmres solver AND drop_tol = 1e-7 in spilu decomposition


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
#from tqdm import tqdm
from scipy.sparse.linalg import spilu, LinearOperator, lgmres
from time import time
import matplotlib.pyplot as plt

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
model_areas = model_data['area'].to_numpy() # m^2

# try OCIM1
#model_data = p2.loadmat(data_path + 'OCIM2_48L_base/OCIM1_CTL.mat')

#TR = model_data['output']['TR']
#ocnmask = np.asfortranarray(model_data['output']['M3d'])
#ocnmask = np.transpose(ocnmask, axes=(2, 1, 0))

#model_depth = np.array(model_data['output']['grid']['zt']) # m below sea surface
#model_lon = np.array(model_data['output']['grid']['xt']) # ºE
#model_lat = np.array(model_data['output']['grid']['yt']) # ºN
#model_vols = np.asfortranarray(model_data['output']['grid']['DXT3d']) * np.asfortranarray(model_data['output']['grid']['DYT3d']) * np.asfortranarray(model_data['output']['grid']['DZT3d']) # m^3
#model_vols = np.transpose(model_vols, axes=(2, 1, 0))

#model_vols = np.asfortranarray(model_data['output']['grid']['DXT3d']) * np.asfortranarray(model_data['output']['grid']['DYT3d']) * np.asfortranarray(model_data['output']['grid']['DZT3d']) # m^3
#model_vols = np.transpose(model_vols, axes=(2, 1, 0))

#model_areas = np.asfortranarray(model_data['output']['grid']['DXT3d']) * np.asfortranarray(model_data['output']['grid']['DYT3d']) # m^2
#model_areas = np.transpose(model_areas, axes=(2, 1, 0))

# grid cell z-dimension for converting from surface area to volume
grid_z = model_vols / model_areas
rho = 1025 # seawater density for volume to mass [kg m-3]

# to help with conversions
sec_per_year = 60 * 60 * 24 * 365.25 # seconds in a year


#%% set up air-sea gas exchange (Wanninkhof 2014)

# upload (or regrid) woa18 data for use in CO2 system calculations

# regrid WOA18 data
#p2.regrid_woa(data_path, 'S', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_woa(data_path, 'T', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_woa(data_path, 'Si', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_woa(data_path, 'P', model_depth, model_lat, model_lon, ocnmask)

# upload regridded WOA18 data
S_3D = np.load(data_path + 'WOA18/S_AO.npy')   # salinity [unitless]
T_3D = np.load(data_path + 'WOA18/T_AO.npy')   # temperature [ºC]
Si_3D = np.load(data_path + 'WOA18/Si_AO.npy') # silicate [µmol kg-1]
P_3D = np.load(data_path + 'WOA18/P_AO.npy')   # phosphate [µmol kg-1]

# flatten data
S = p2.flatten(S_3D, ocnmask)
T = p2.flatten(T_3D, ocnmask)
Si = p2.flatten(Si_3D, ocnmask)
P = p2.flatten(P_3D, ocnmask)

#p2.plot_surface3d(model_lon, model_lat, S_3D, 0, 25, 38, 'magma', 'WOA salinity distribution')
#p2.plot_surface3d(model_lon, model_lat, T_3D, 0, -10, 35, 'magma', 'WOA temp distribution')
#p2.plot_surface3d(model_lon, model_lat, Si_3D, 0, 0, 30, 'magma', 'WOA silicate distribution')
#p2.plot_surface3d(model_lon, model_lat, P_3D, 0, 0, 2.5, 'magma', 'WOA phosphate distribution')

# regrid NCEP/DOE reanalysis II data
#p2.regrid_ncep_noaa(data_path, 'icec', model_lat, model_lon, ocnmask)
#p2.regrid_ncep_noaa(data_path, 'uwnd', model_lat, model_lon, ocnmask)
#p2.regrid_ncep_noaa(data_path, 'sst', model_lat, model_lon, ocnmask)

# upload regridded NCEP/DOE reanalysis II data
f_ice_2D = np.load(data_path + 'NCEP_DOE_Reanalysis_II/icec_AO.npy') # annual mean ice fraction from 0 to 1 in each grid cell
uwnd_2D = np.load(data_path + 'NCEP_DOE_Reanalysis_II/uwnd_AO.npy') # annual mean of forecast of U-wind at 10 m [m/s]
sst_2D = np.load(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst_AO.npy') # annual mean sea surface temperature [ºC]

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

# regrid GLODAP data
#p2.regrid_glodap(data_path, 'TCO2', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'TAlk', model_depth, model_lat, model_lon, ocnmask)

# upload regridded GLODAP data
DIC_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC_AO.npy') # dissolved inorganic carbon [µmol kg-1]
AT_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA_AO.npy')   # total alkalinity [µmol kg-1]

# flatten data
DIC = p2.flatten(DIC_3D, ocnmask)
AT = p2.flatten(AT_3D, ocnmask)

# create "pressure" array by broadcasting depth array
pressure_3D = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[1], ocnmask.shape[2]))
pressure = pressure_3D[ocnmask == 1].flatten(order='F')

# use CO2SYS with GLODAP and WOA data to solve for carbonate system at each grid cell
# do this for only ocean grid cells
# this is PyCO2SYSv2
co2sys = pyco2.sys(dic=DIC, alkalinity=AT, salinity=S, temperature=T,
                   pressure=pressure, total_silicate=Si, total_phosphate=P)

# extract key results arrays
pCO2 = co2sys['pCO2'] # pCO2 [µatm]
aqueous_CO2 = co2sys['CO2'] # aqueous CO2 [µmol kg-1]
R_C = co2sys['revelle_factor'] # revelle factor w.r.t. DIC [unitless]

# calculate revelle factor w.r.t. AT [unitless]
# must calculate manually, R_AT defined as (dpCO2/pCO2) / (dAT/AT)
co2sys_000001 = pyco2.sys(dic=DIC, alkalinity=AT+0.000001, salinity=S,
                       temperature=T, pressure=pressure, total_silicate=Si,
                       total_phosphate=P)

pCO2_000001 = co2sys_000001['pCO2']
R_A = ((pCO2_000001 - pCO2)/pCO2) / (0.000001/AT)

# calculate Nowicki et al. parameters
Ma = 1.8e26 # number of micromoles of air in atmosphere [µmol air]
beta_C = DIC/aqueous_CO2 # [unitless]
beta_A = AT/aqueous_CO2 # [unitless]
K0 = aqueous_CO2/pCO2*rho # [µmol CO2 m-3 (µatm CO2)-1], in derivation this is defined in per volume units so used density to get there
Patm = 1e6 # atmospheric pressure [µatm]
z1 = model_depth[0] # depth of first layer of model [m]
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

#%% plot pCO2, Revelle factor, K0, and k

pCO2_3D = p2.make_3D(pCO2, ocnmask) # µatm CO2
R_C_3D = p2.make_3D(R_C, ocnmask) # unitless
K0_3D = p2.make_3D(K0, ocnmask) # µmol CO2 m-3 (µatm CO2)-1

p2.plot_surface3d(model_lon, model_lat, pCO2_3D, 0, 220, 500, 'jet', 'pCO2 (ppm) calculated with pyCO2SYS from regridded GLODAP DIC & AT (OCIM grid)', lon_lims=[0, 360])
p2.plot_surface3d(model_lon, model_lat, R_C_3D, 0, 6, 18, 'viridis', 'Revelle Factor calculated with pyCO2SYS from regridded GLODAP DIC & AT (OCIM grid)', lon_lims=[0, 360])
p2.plot_surface3d(model_lon, model_lat, K0_3D / rho, 0, 0, .1, 'magma', 'Calculated solubility of CO2 (mol kg-1 atm-1)', lon_lims=[0, 360])
p2.plot_surface3d(model_lon, model_lat, k_3D / (24*365.25/100), 0, 0, 25, 'magma', 'Calculated gas transfer velocity k (cm/hr)', lon_lims=[0, 360])

p2.plot_surface3d(model_lon, model_lat, k_3D / (24*365.25/100) * K0_3D / rho / 11.1371, 0, 0, .15, 'magma', 'k (cm/hr) * K0 (mol/kg/atm) / 11.1371', lon_lims=[0, 360])

fig = plt.figure(figsize=(6,5), dpi=200)
ax = fig.gca()
ax.scatter(uwnd_2D.flatten(), k_2D.flatten() / (24*365.25/100))
ax.set_xlabel('annual mean of forecast of U-wind at 10 m [m/s]')
ax.set_ylabel('gas transfer velocity $k_{660}$ [cm/hr]')
ax.set_xlim([-20, 20])
ax.set_ylim([0, 200])

fig = plt.figure(figsize=(6,5), dpi=200)
ax = fig.gca()
ax.scatter(sst_2D.flatten() + 273.15, K0_3D[0, :, :].flatten() / rho)
ax.set_xlabel('sea surface temperature [K]')
ax.set_ylabel('solubility of CO2 $K_{0}$ [mol kg-1 atm-1]')
#ax.set_xlim([270, 310])
#ax.set_ylim([0.02, 0.07])

#%% set up time stepping

# more complicated time stepping
# set up time domain
dt1 = 1/360 # 1 day
dt2 = 1/12 # 1 month
dt3 = 1 # 1 year
dt4 = 10 # 10 years
dt5 = 100 # 100 years

# Kana's time stepping
t1 = np.arange(0, 90/360, dt1) # use a 1 day time step for the first 90 days
t2 = np.arange(90/360, 5, dt2) # use a 1 month time step until the 5th year
t3 = np.arange(5, 100, dt3) # use a 1 year time step until the 100th year
t4 = np.arange(100, 500, dt4) # use a 10 year time step until the 500th year
t5 = np.arange(500, 1000+dt5, dt5) # use a 100 year time step until the 1000th year

# shortened time stepping to test solvers#
#t1 = np.arange(0, 30/360, dt1) # use a 1 day time step for the first 30 days
#t2 = np.arange(30/360, 1, dt2) # use a 1 month time step until the 1st year
#t3 = np.arange(1, 3, dt3) # use a 1 year time step until the 3rd year
#t4 = np.arange(3, 23, dt4) # use a 10 year time step until the 23rd year
#t5 = np.arange(23, 123+dt5, dt5) # use a 100 year time step until the 1000th year

t = np.concatenate((t1, t2, t3, t4, t5))
#t = np.concatenate((t1, t2)) # for shortened sim
#t = np.concatenate((t1, t2, t3))

#%% run multiple experiments
experiment_names = ['exp11_2025-8-11-a.nc', 'exp11_2025-8-11-b.nc', 'exp11_2025-8-11-c.nc', 'exp11_2025-8-11-d.nc']
experiment_attrs = ['Attempting to repeat Yamamoto et al 2024 experiment - instantaneous OAE - location is model_lon[50] model_lat[25]',
                    'Attempting to repeat Yamamoto et al 2024 experiment - instantaneous OAE',
                    'Attempting to repeat Yamamoto et al 2024 experiment - instantaneous OAE',
                    'Attempting to repeat Yamamoto et al 2024 experiment - instantaneous OAE']

# Applying perturbations at (-39.5, 101), which is (model_lat[25], model_lon[50])
#                           (5.9, -99), which is (model_lat[47], model_lon[130])
#                           (61.3, -175), which is (model_lat[76], model_lon[92]) 
#                           (61.3, -27), which is (model_lat[76], model_lon[166])

experiment_lons_idx = [50, 130, 92, 166]
experiment_lats_idx = [25, 47, 76, 76]

# for shortened sim
experiment_names = ['exp11_2025-8-6-a.nc']
#experiment_attrs = ['Attempting to repeat Yamamoto et al 2024 experiment - 5 years only - rtol=1e-7 and drop_tol=1e-7 - instantaneous OAE - location is model_lon[50] model_lat[25]']
experiment_lons_idx = [50]
experiment_lats_idx = [25]
#%%
for exp_idx in range(0, len(experiment_names)):
    experiment_name = experiment_names[exp_idx]
    experiment_attr = experiment_attrs[exp_idx]
    experiment_lon_idx = experiment_lons_idx[exp_idx]
    experiment_lat_idx = experiment_lats_idx[exp_idx]

    # add CDR perturbation
    # Add surface ocean perturbation of -1 µmol kg-1 yr-1 in DIC, no change in AT
    # Goal: compare results with Yamamoto et al., 2024 supplemental figures
    
    # ∆q_CDR,AT (change in alkalinity due to CDR addition) - final units: [µmol AT kg-1 yr-1]
    del_q_CDR_AT_3D = np.full(ocnmask.shape, np.nan)
    del_q_CDR_AT_3D[ocnmask == 1] = 0
    del_q_CDR_AT = p2.flatten(del_q_CDR_AT_3D, ocnmask)
    
    # ∆q_CDR,DIC (change in DIC due to CDR addition) - final units: [µmol DIC kg-1 yr-1]
    del_q_CDR_DIC_3D = np.full(ocnmask.shape, np.nan)
    del_q_CDR_DIC_3D[ocnmask == 1] = 0
    del_q_CDR_DIC_3D[0, experiment_lon_idx, experiment_lat_idx] = -1 # µmol kg-1 yr-1
    del_q_CDR_DIC = p2.flatten(del_q_CDR_DIC_3D, ocnmask)
    
    # construct matricies
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
    nt = len(t)
    
    # c = [ ∆xCO2 ] --> 1 * nt
    #     [ ∆DIC  ] --> m * nt
    #     [ ∆AT   ] --> m * nt
    
    c = np.zeros((1 + 2*m, nt))
    
    # q = [ 0                                                                           ] --> 1 * nt, q[0]
    #     [ ∆q_CDR,DIC + ∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc      ] --> m * nt, q[1:(m+1)]
    #     [ ∆q_CDR,AT + 2 * (∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc) ] --> m * nt, q[(m+1):(2*m+1)]
    
    # which translates to...
    # q = [ 0                                      ] --> 1 * nt, q[0]
    #     [ ∆q_CDR,DIC + ∆q_diss,DIC - ∆q_prod,DIC ] --> m * nt, q[1:(m+1)]
    #     [ ∆q_CDR,AT + ∆q_diss,AT - ∆q_prod,AT    ] --> m * nt, q[(m+1):(2*m+1)]
    
    q = np.zeros((1 + 2*m, nt))
    
    # add in source/sink vectors for ∆AT, only add perturbation for time step 0
    
    # for ∆DIC, add perturbation for first 30 days
    q[1:(m+1),0:30] = np.tile(del_q_CDR_DIC[:, np.newaxis], (1,30))
    
    # for ∆AT, no change
    
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
    
    #del TR
    
    # perform time stepping using Euler backward
    LHS1 = sparse.eye(A.shape[0], format="csc") - dt1 * A
    LHS2 = sparse.eye(A.shape[0], format="csc") - dt2 * A
    LHS3 = sparse.eye(A.shape[0], format="csc") - dt3 * A
    LHS4 = sparse.eye(A.shape[0], format="csc") - dt4 * A
    LHS5 = sparse.eye(A.shape[0], format="csc") - dt5 * A
    
    # test condition number of matrix
    est1 = sparse.linalg.onenormest(LHS1)
    print("Estimated 1-norm condition number LHS1: ", est1)
    est2 = sparse.linalg.onenormest(LHS2)
    print("Estimated 1-norm condition number LHS2: ", est2)
    est3 = sparse.linalg.onenormest(LHS3)
    print("Estimated 1-norm condition number LHS3: ", est3)
    est4 = sparse.linalg.onenormest(LHS4)
    print("Estimated 1-norm condition number LHS4: ", est4)
    est5 = sparse.linalg.onenormest(LHS5)
    print("Estimated 1-norm condition number LHS5: ", est5)
    
    start = time()
    ilu1 = spilu(LHS1.tocsc(), drop_tol=1e-5, fill_factor=20)
    stop = time()
    print('ilu1 calculations: ' + str(stop - start) + ' s')
    
    start = time()
    ilu2 = spilu(LHS2.tocsc(), drop_tol=1e-5, fill_factor=20)
    stop = time()
    print('ilu2 calculations: ' + str(stop - start) + ' s')
    
    '''
    start = time()
    ilu3 = spilu(LHS3.tocsc(), drop_tol=1e-3, fill_factor=10)
    stop = time()
    print('ilu3 calculations: ' + str(stop - start) + ' s')
    
    start = time()
    ilu4 = spilu(LHS4.tocsc(), drop_tol=1e-5, fill_factor=20)
    stop = time()
    print('ilu4 calculations: ' + str(stop - start) + ' s')
    
    start = time()
    ilu5 = spilu(LHS5.tocsc())
    stop = time()
    print('ilu5 calculations: ' + str(stop - start) + ' s')
    '''
    
    M1 = LinearOperator(LHS1.shape, ilu1.solve)
    M2 = LinearOperator(LHS2.shape, ilu2.solve)
    '''
    M3 = LinearOperator(LHS3.shape, ilu3.solve)
    M4 = LinearOperator(LHS4.shape, ilu4.solve)
    M5 = LinearOperator(LHS5.shape, ilu5.solve)
    '''
    
    #for idx in tqdm(range(1, len(t))):
    for idx in range(1, len(t)):
        
        # add starting guess after first time step
        if idx > 1:
            c0 = c[:,idx-1]
        else:
            c0=None
        
        if t[idx] <= 90/360: # 1 day time step
        #if t[idx] <= 30/360: # 1 day time step
            RHS = c[:,idx-1] + np.squeeze(dt1*q[:,idx-1])
            start = time()
            c[:,idx], info = lgmres(LHS1, RHS, M=M1, x0=c0, rtol = 1e-5, atol=0)
            stop = time()
            print('t = ' + str(idx) + ', solve time: ' + str(stop - start) + ' s')
       
        elif (t[idx] > 90/360) & (t[idx] <= 5): # 1 month time step
        #elif (t[idx] > 30/360) & (t[idx] <= 1): # 1 month time step
            RHS = c[:,idx-1] + np.squeeze(dt2*q[:,idx-1])
            start = time()
            c[:,idx], info = lgmres(LHS2, RHS, M=M2, x0=c0, rtol = 1e-5, atol=0)
            stop = time()
            print('t = ' + str(idx) + ', solve time: ' + str(stop - start) + ' s')
        '''
        elif (t[idx] > 5) & (t[idx] <= 100): # 1 year time step
        #elif (t[idx] > 1) & (t[idx] <= 3): # 1 year time step
            start = time()
            RHS = c[:,idx-1] + np.squeeze(dt3*q[:,idx-1])
            c[:,idx], info = lgmres(LHS3, RHS, M=M3, x0=c0, rtol = 1e-5, atol=0)
            stop = time()
            print('t = ' + str(idx) + ', solve time: ' + str(stop - start) + ' s')
    
        elif (t[idx] > 100) & (t[idx] <= 500): # 10 year time step
        #elif (t[idx] > 3) & (t[idx] <= 23): # 10 year time step
            start = time()
            RHS = c[:,idx-1] + np.squeeze(dt4*q[:,idx-1])
            c[:,idx], info = lgmres(LHS4, RHS, M=M4, x0=c0, rtol = 1e-5, atol=0)
            stop = time()
            print('t = ' + str(idx) + ', solve time: ' + str(stop - start) + ' s')
    
        else: # 100 year time step
            start = time()
            RHS = c[:,idx-1] + np.squeeze(dt5*q[:,idx-1])
            c[:,idx], info = lgmres(LHS5, RHS, M=M5, x0=c0, rtol = 1e-5, atol=0)
            stop = time()
            print('t = ' + str(idx) + ', solve time: ' + str(stop - start) + ' s')
        '''
            
        if info != 0:
            if info > 0:
                print(f'did not converge in {info} iterations.')
            else:
                print('illegal input or breakdown')
    
    # rebuild 3D concentrations from 1D array used for solving matrix equation
    
    # partition "x" into xCO2, DIC, and AT
    c_xCO2 = c[0, :]
    c_DIC  = c[1:(m+1), :]
    c_AT   = c[(m+1):(2*m+1), :]
    
    # convert xCO2 units from unitless [µatm CO2 / µatm air] or [µmol CO2 / µmol air] to ppm
    c_xCO2 *= 1e6
    
    # reconstruct 3D arrays for DIC and AT
    c_DIC_3D = np.full([len(t), ocnmask.shape[0], ocnmask.shape[1], ocnmask.shape[2]], np.nan) # make 3D vector full of nans
    c_AT_3D = np.full([len(t), ocnmask.shape[0], ocnmask.shape[1], ocnmask.shape[2]], np.nan) # make 3D vector full of nans
    
    # for each time step, reshape 1D array into 3D array, then save to larger 4D array output (time, depth, longitude, latitude)
    for idx in range(0, len(t)):
        c_DIC_reshaped = np.full(ocnmask.shape, np.nan)
        c_AT_reshaped = np.full(ocnmask.shape, np.nan)
    
        c_DIC_reshaped[ocnmask == 1] = np.reshape(c_DIC[:, idx], (-1,), order='F')
        c_AT_reshaped[ocnmask == 1] = np.reshape(c_AT[:, idx], (-1,), order='F')
        
        c_DIC_3D[idx, :, :, :] = c_DIC_reshaped
        c_AT_3D[idx, :, :, :] = c_AT_reshaped
    
    # save model output in netCDF format
    global_attrs = {'description': experiment_attr}
    
    # save model output
    p2.save_model_output(
        experiment_name, 
        t, 
        model_depth, 
        model_lon,
        model_lat, 
        tracers=[c_xCO2, c_DIC_3D, c_AT_3D], 
        tracer_dims=[('time',), ('time', 'depth', 'lon', 'lat'), ('time', 'depth', 'lon', 'lat')],
        tracer_names=['delxCO2', 'delDIC', 'delAT'], 
        tracer_units=['ppm', 'umol kg-3', 'umol kg-3'],
        global_attrs=global_attrs
    )


#%% open and plot model output
data = xr.open_dataset(output_path + 'exp11_2025-8-6-a.nc')

#test = data['delDIC'].isel(lon=50).isel(lat=25).isel(depth=0).values
#for x in test:
#    print(x)
    
t = data.time
model_lon = data.lon.data
model_lat = data.lat.data
model_depth = data.depth.data

nt = len(t)

for idx in range(0, nt):
    print(idx)
    p2.plot_surface3d(model_lon, model_lat, data['delDIC'].isel(time=idx).values, 0, -6e-5, 6e-5, 'RdBu', 'Surface ∆DIC (µmol kg-1) at t=' + str(np.round(t[idx].values,3)) + ' yr')

#for idx in range(0, nt):
#    p2.plot_longitude3d(model_lat, model_depth, data['delDIC'].isel(time=idx).values, 97, 0, 5e-5, 'plasma', ' ∆DIC (µmol kg-1) at t=' +str(t[idx]) + ' along 165ºW longitude')


#%% calculate kana's "alpha" metric to compare with supplemental figure S2
# alpha = ∆C_atm / ∆C_CDR

# forgot to save this out oops, recalculating here
experiment_del_q_CDRs = []
for exp_idx in range(0, len(experiment_names)):
    del_q_CDR_DIC_3D = np.full(ocnmask.shape, np.nan)
    del_q_CDR_DIC_3D[ocnmask == 1] = 0
    del_q_CDR_DIC_3D[0, experiment_lons_idx[exp_idx], experiment_lats_idx[exp_idx]] = -1 # µmol kg-1 yr-1
    del_q_CDR_DIC = p2.flatten(del_q_CDR_DIC_3D, ocnmask)
    experiment_del_q_CDRs.append(del_q_CDR_DIC)

del_C_atm = np.full((nt), np.nan)
del_C_CDR = np.full((nt), np.nan)

for idx in range(0, nt):
    del_C_CDR[idx] = np.nansum(del_q_CDR_DIC_3D * dt1 * idx * rho * model_vols) # [µmol CO2]
    del_C_atm[idx] = data['delxCO2'].isel(time=idx).values * 1e-6 * Ma # [µmol CO2]
    
alpha = del_C_atm / del_C_CDR * 100 # unitless

print('alpha at t = 5 yr: ' + str(round(alpha[146], 2)) + ' %')
#print('alpha at t = 5 yr: ' + str(round(alpha[147], 2)) + ' %')
#print('alpha at t = 20 yr: ' + str(round(alpha[162], 2)) + ' %')
#print('alpha at t = 50 yr: ' + str(round(alpha[192], 2)) + ' %')
#print('alpha at t = 100 yr: ' + str(round(alpha[242], 2)) + ' %')

data.close()