#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Try to build a model using Rui's outputs!

Governing equations (based on my own derivation + COBALT governing equations)
1. d(∆q_xCO2)/dt = ∆q_xCO2,sea-air
2. d(∆DIC)/dt = TR * ∆DIC + ∆q_air-sea,DIC + ∆q_CDR,DIC + ∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc
3. d(∆AT)/dt = TR * ∆AT + ∆q_CDR,AT + 2 * [∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc]

*NOTE: burial is included in dissolution in 'plus_btm' versions of calcium and
aragonite dissolution, but for some reason these arrays were all equal to zero
in files Rui sent me -> should investigate further soon

*NOTE: this is assuming no changes to biology, could modulate this (i.e.
production/respiration changes) in the future (see COBALT governing equations
for how this affects alkalinity/DIC in that model)

Created on Tue Jul  8 12:24:04 2025

@author: Reese Barrett
"""

import project2 as p2
import xarray as xr
import numpy as np
import PyCO2SYS as pyco2

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

#%% load in regridded COBALT data (or, regrid COBALT data)
cobalt_path = data_path + 'COBALT_regridded/'

#cobalt = xr.open_dataset('/Volumes/LaCie/data/OM4p25_cobalt_v3/19580101.ocean_cobalt_fluxes_int.nc', decode_cf=False)
#p2.regrid_cobalt(cobalt.jdiss_cadet_arag, model_depth, model_lat, model_lon, ocnmask, cobalt_path)
#p2.regrid_cobalt(cobalt.jdiss_cadet_calc, model_depth, model_lat, model_lon, ocnmask, cobalt_path)
#p2.regrid_cobalt(cobalt.jprod_cadet_arag, model_depth, model_lat, model_lon, ocnmask, cobalt_path)
#p2.regrid_cobalt(cobalt.jprod_cadet_calc, model_depth, model_lat, model_lon, ocnmask, cobalt_path)

#q_diss_arag_3D = np.load(data_path + 'COBALT_regridded/jdiss_cadet_arag.npy') # [mol CACO3 m-2 s-1]
q_diss_calc_3D = np.load(cobalt_path + 'jdiss_cadet_calc.npy') # [mol CACO3 m-2 s-1]
q_prod_arag_3D = np.load(cobalt_path + 'jprod_cadet_arag.npy') # [mol CACO3 m-2 s-1]
q_prod_calc_3D = np.load(cobalt_path + 'jprod_cadet_calc.npy') # [mol CACO3 m-2 s-1]

#%% make assumptions to calculate "delta" for dissolution and production
# right now, assuming linear relationship with arbitrary factor of 0.1 to see
# if this even sort of works

# ∆q_diss,arag [mol m-2 s-1]
#del_q_diss_arag = p2.flatten(0.1 * q_diss_arag_3D, ocnmask)

# ∆q_diss,calc [mol m-2 s-1]
del_q_diss_calc = p2.flatten(0.1 * q_diss_calc_3D, ocnmask)

# ∆q_prod,arag [mol m-2 s-1]
del_q_prod_arag = p2.flatten(0.1 * q_prod_arag_3D, ocnmask)

# ∆q_prod,calc [mol m-2 s-1]
del_q_prod_calc = p2.flatten(0.1 * q_prod_calc_3D, ocnmask)


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

# solve for Kw (gas transfer velocity) for each ocean cell
a = 0.251 # from Wanninkhof 2014
Kw_2D = a * uwnd_2D**2 * (Sc_2D/660)**-0.5 # [cm/h] from Yamamoto et al., 2024, adapted from Wanninkhof 2014

#p2.plot_surface2d(model_lon, model_lat, Kw.T, 0, 20, 'magma', 'Gas transfer velocity (Kw, cm/hr)')

Kw_2D *= (24*365.25/100) # [m/yr] convert units

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
pCO2 = co2sys['pCO2'] # pCO2 [atm]
aqueous_CO2 = co2sys['CO2'] # aqueous CO2 [µmol kg-1]
R_DIC = co2sys['revelle_factor'] # revelle factor w.r.t. DIC [unitless]

# calculate revelle factor w.r.t. AT [unitless]
# must calculate manually, R_AT defined as (dpCO2/pCO2) / (dAT/AT)
co2sys_000001 = pyco2.sys(dic=DIC, alkalinity=AT+0.000001, salinity=S,
                       temperature=T, pressure=pressure, total_silicate=Si,
                       total_phosphate=P)

pCO2_000001 = co2sys_000001['pCO2']
R_AT = ((pCO2_000001 - pCO2)/pCO2) / (0.000001/AT)

# calculate Nowicki et al. parameters
rho = 1025 # seawater density [kg m-3]
Ma = 1.8e26 # number of micromoles of air in atmosphere
beta_DIC = DIC/aqueous_CO2 # [unitless]
beta_AT = AT/aqueous_CO2 # [unitless]
K0 = aqueous_CO2/pCO2*rho # [µmol m-3 atm-1], in derivation this is defined in per volume units so used density to get there
Patm = 1 # atmospheric pressure [atm]
z1 = model_depth[0] # depth of first layer of model [m]
#V = # volume of first layer of model [m3]

# add layers of "np.NaN" for all subsurface layers in Kw, f_ice, then flatten
Kw_3D = np.full(ocnmask.shape, np.nan)
Kw_3D[0, :, :] = Kw_2D
Kw = p2.flatten(Kw_3D, ocnmask)

f_ice_3D = np.full(ocnmask.shape, np.nan)
f_ice_3D[0, :, :] = f_ice_2D
f_ice = p2.flatten(f_ice_3D, ocnmask)

#%% plug in values to calculate ∆q_air-sea,DIC and ∆q_sea-air,xCO2

# ∆q_air-sea,DIC [µmol kg-1 s-1]
del_q_air_sea_DIC = -1 * Kw * (1 - f_ice) / z1 * (R_DIC * del_DIC / beta_DIC + R_AT * del_AT / beta_AT - K0 * Patm / rho * del_xCO2)

# ∆q_sea-air,xCO2 [DOUBLE CHECK UNITS]
del_q_sea_air_xCO2 = V * Kw * (1 - f_ice) / Ma / z1 * (rho * R_DIC * del_DIC / beta_DIC + rho * R_AT * del_AT / beta_AT - K0 * Patm * del_xCO2)


#%% add CDR perturbation
# as a test, add point source of 4 µmol m-2 s-1 NaOH (this is what they did in
# Wang et al., 2022 Bering Sea paper)
# with NaOH, no alkalinity change, 1:1 AT:NaOH change

# depth = 0, latitude = ~54, longitude = ~-165
# in model, this approximately corresponds to model_depth[0], model_lat[73], model_lon[97]

# ∆q_CDR,AT (change in alkalinity due to CDR addition) [µmol m-2 s-1]
del_q_CDR_AT_3D = np.full(ocnmask.shape, 0)
del_q_CDR_AT_3D[0, 97, 73] = 4
del_q_CDR_AT = p2.flatten(del_q_CDR_AT_3D, ocnmask)

# ∆q_CDR,DIC (change in DIC due to CDR addition) [µmol m-2 s-1]
del_q_CDR_DIC_3D = np.full(ocnmask.shape, 0)
del_q_CDR_DIC = p2.flatten(del_q_CDR_DIC_3D, ocnmask)





















