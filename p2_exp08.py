#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Try to build a model using Rui's outputs!

Governing equations (based on my own derivation + COBALT governing equations)
1. d(∆q_xCO2)/dt = ∆q_xCO2,sea-air
2. d(∆DIC)/dt = T * ∆DIC + ∆q_air-sea,DIC + ∆q_CDR,DIC + ∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc
3. d(∆AT)/dt = T * ∆AT + ∆q_CDR,AT + 2 * [∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc]

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

#%% load in regridded COBALT data (or, regrid COBALT data)
cobalt_path = data_path + 'COBALT_regridded/'

#cobalt = xr.open_dataset('/Volumes/LaCie/data/OM4p25_cobalt_v3/19580101.ocean_cobalt_fluxes_int.nc', decode_cf=False)
#p2.regrid_cobalt(cobalt.jdiss_cadet_arag, model_depth, model_lat, model_lon, ocnmask, cobalt_path)
#p2.regrid_cobalt(cobalt.jdiss_cadet_calc, model_depth, model_lat, model_lon, ocnmask, cobalt_path)
#p2.regrid_cobalt(cobalt.jprod_cadet_arag, model_depth, model_lat, model_lon, ocnmask, cobalt_path)
#p2.regrid_cobalt(cobalt.jprod_cadet_calc, model_depth, model_lat, model_lon, ocnmask, cobalt_path)

#q_diss_arag = np.load(data_path + 'COBALT_regridded/jdiss_cadet_arag.npy') # [mol CACO3 m-2 s-1]
q_diss_calc = np.load(cobalt_path + 'jdiss_cadet_calc.npy') # [mol CACO3 m-2 s-1]
q_prod_arag = np.load(cobalt_path + 'jprod_cadet_arag.npy') # [mol CACO3 m-2 s-1]
q_prod_calc = np.load(cobalt_path + 'jprod_cadet_calc.npy') # [mol CACO3 m-2 s-1]

#%% make assumptions to calculate "delta" for dissolution and production
# right now, assuming linear relationship with arbitrary factor of 0.1 to see
# if this even sort of works

#del_q_diss_arag = p2.flatten(0.1 * q_diss_arag, ocnmask)
del_q_diss_calc = p2.flatten(0.1 * q_diss_calc, ocnmask)
del_q_prod_arag = p2.flatten(0.1 * q_prod_arag, ocnmask)
del_q_prod_calc = p2.flatten(0.1 * q_prod_calc, ocnmask)

#%% upload (or regrid) woa18 data for use in CO2 system calculations

# regrid WOA18 data
#p2.regrid_woa(data_path, 'S', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_woa(data_path, 'T', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_woa(data_path, 'Si', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_woa(data_path, 'P', model_depth, model_lat, model_lon, ocnmask)

# upload regridded WOA18 data
S = np.load(data_path + 'WOA18/S_AO.npy')   # salinity [unitless]
T = np.load(data_path + 'WOA18/T_AO.npy')   # temperature [ºC]
Si = np.load(data_path + 'WOA18/Si_AO.npy') # silicate [µmol kg-1]
P = np.load(data_path + 'WOA18/P_AO.npy')   # phosphate [µmol kg-1]

#p2.plot_surface3d(model_lon, model_lat, S, 0, 25, 38, 'magma', 'WOA salinity distribution')
#p2.plot_surface3d(model_lon, model_lat, T, 0, -10, 35, 'magma', 'WOA temp distribution')
#p2.plot_surface3d(model_lon, model_lat, Si, 0, 0, 30, 'magma', 'WOA silicate distribution')
#p2.plot_surface3d(model_lon, model_lat, P, 0, 0, 2.5, 'magma', 'WOA phosphate distribution')

#%% set up air-sea gas exchange (Wanninkhof 2014)

# regrid NCEP/DOE reanalysis II data
#p2.regrid_ncep_noaa(data_path, 'icec', model_lat, model_lon, ocnmask)
#p2.regrid_ncep_noaa(data_path, 'uwnd', model_lat, model_lon, ocnmask)
#p2.regrid_ncep_noaa(data_path, 'sst', model_lat, model_lon, ocnmask)

# upload regridded NCEP/DOE reanalysis II data
f_ice = np.load(data_path + 'NCEP_DOE_Reanalysis_II/icec_AO.npy') # annual mean ice fraction from 0 to 1 in each grid cell
uwnd = np.load(data_path + 'NCEP_DOE_Reanalysis_II/uwnd_AO.npy') # annual mean of forecast of U-wind at 10 m [m/s]
sst = np.load(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst_AO.npy') # annual mean sea surface temperature [ºC]

# calculate Schmidt number using Wanninkhof 2014 parameterization
vec_schmidt = np.vectorize(p2.schmidt)
Sc = vec_schmidt('CO2', sst)

# solve for Kw (gas transfer velocity) for each ocean cell
a = 0.251 # from Wanninkhof 2014
Kw = a * uwnd**2 * (Sc/660)**-0.5 # [cm/h] from Yamamoto et al., 2024, adapted from Wanninkhof 2014

#p2.plot_surface2d(model_lon, model_lat, Kw.T, 0, 20, 'magma', 'Gas transfer velocity (Kw, cm/hr)')

Kw *= (24*365.25/100) # [m/yr] convert units

#p2.plot_surface2d(model_lon, model_lat, uwnd.T, -15, 15, 'seismic', 'U-wind at 10 m (m/s)')
#p2.plot_surface2d(model_lon, model_lat, sst.T, -2, 40, 'magma', 'sst (ºC)')

#%% set up linearized CO2 system (Nowicki et al., 2024)

# upload (or regrid) glodap data for use as initial conditions for marine carbonate system 

# regrid GLODAP data
#p2.regrid_glodap(data_path, 'TCO2', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'TAlk', model_depth, model_lat, model_lon, ocnmask)

# upload regridded GLODAP data
DIC = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC_AO.npy') # dissolved inorganic carbon [µmol kg-1]
TA = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA_AO.npy')   # total alkalinity [µmol kg-1]

# create "pressure" array by broadcasting depth array
pressure = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[1], ocnmask.shape[2]))

# use CO2sys with GLODAP and WOA data to solve for carbonate system at each grid cell
# do this for only ocean grid cells
co2sys_results = pyco2.sys(par1=TA[ocnmask == 1].flatten(order='F'),
                    par2=DIC[ocnmask == 1].flatten(order='F'),
                    par1_type=1, par2_type=2,
                    salinity=S[ocnmask == 1].flatten(order='F'),
                    temperature=T[ocnmask == 1].flatten(order='F'),
                    pressure=pressure[ocnmask == 1].flatten(order='F'),
                    total_silicate=Si[ocnmask == 1].flatten(order='F'),
                    total_phosphate=P[ocnmask == 1].flatten(order='F'))

# extract key results arrays, make 3D

# pCO2 [µatm]
pCO2 = np.full(ocnmask.shape, np.nan)
pCO2[ocnmask == 1] = np.reshape(co2sys_results['pCO2'], (-1,), order='F')

# aqueous CO2 [µmol kg-1]
aqueous_CO2 = np.full(ocnmask.shape, np.nan)
aqueous_CO2[ocnmask == 1] = np.reshape(co2sys_results['aqueous_CO2'], (-1,), order='F')

# revelle factor [unitless]
R = np.full(ocnmask.shape, np.nan)
R[ocnmask == 1] = np.reshape(co2sys_results['revelle_factor'], (-1,), order='F')

# calculate Nowicki et al. parameters
rho = 1025 # seawater density [kg m-3]
Ma = 1.8e26 # number of micromoles of air in atmosphere
beta = DIC/aqueous_CO2 # [unitless]
K0 = aqueous_CO2/pCO2*rho # [µmol m-3 atm-1], in derivation this is defined in per volume units so used density to get there
del_z1 = model_depth[0] # depth of first layer of model [m]
tau_CO2 = (del_z1 * beta[0, :, :]) / (Kw * R[0, :, :]) # timescale of air-sea CO2 equilibration [yr]

p2.plot_surface2d(model_lon, model_lat, R[0,:,:].T, 8, 18, 'magma', 'Revelle factor (unitless)') # this is correct
p2.plot_surface2d(model_lon, model_lat, DIC[0,:,:].T, 1800, 2300, 'magma', 'DIC (µmol kg-1)') # this is correct
p2.plot_surface2d(model_lon, model_lat, aqueous_CO2[0,:,:].T, 0, 80, 'magma', 'Aqueous CO2 (µmol kg-1)') # I think this is correct

p2.plot_surface2d(model_lon, model_lat, tau_CO2.T, 0, 1.6, 'magma', 'tau_co2 (yr)') # this should be right? comparing to Nowicki 2024 supplemental, the general pattern seems correct, but the gradients here are more dramatic? note: those units are in days, this is in years

#%% add air-sea gas exchange



#%% add CDR perturbation
# as a test, add ?? THINK ABOUT UNITS?






















