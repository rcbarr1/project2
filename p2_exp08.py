#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Try to build a model using Rui's outputs!

Governing equations (based on my own derivation + COBALT governing equations)
1. d(xCO2)/dt = ∆q_sea-air,xCO2
2. d(∆DIC)/dt = TR * ∆DIC + ∆q_air-sea,DIC + ∆q_CDR,DIC + ∆q_diss,DIC - ∆q_prod,DIC
3. d(∆AT)/dt = TR * ∆AT + ∆q_CDR,AT + ∆q_diss,AT - ∆q_prod,AT

where ∆q_diss,DIC = ∆q_diss,arag + ∆q_diss,calc
      ∆q_prod,DIC = ∆q_prod,arag + ∆q_prod,calc
      ∆q_diss,AT = 2 * (∆q_diss,arag + ∆q_diss,calc)
      ∆q_prod,AT = 2 * (∆q_prod,arag + ∆q_prod,calc)

*NOTE: burial is included in 'diss' in 'plus_btm' versions of calcium and
aragonite dissolution, but for some reason these arrays were all equal to zero
in files Rui sent me -> should investigate further soon

*NOTE: this is assuming no changes to biology, could modulate this (i.e.
production/respiration changes) in the future (see COBALT governing equations
for how this affects alkalinity/DIC in that model)
                                               
Air-sea gas exchange fluxes have to be multiplied by "x" vector because they
rely on ∆x's, which means they are incorporated with the transport matrix into
vector "A"

units: µmol kg-1 s-1
∆q_sea-air,xCO2 = V * Kw * (1 - f_ice) / Ma / z1 * (rho * R_DIC * del_DIC / beta_DIC + rho * R_AT * del_AT / beta_AT - K0 * Patm * del_xCO2)
∆q_air-sea,DIC = -1 * Kw * (1 - f_ice) / z1 * (R_DIC * del_DIC / beta_DIC + R_AT * del_AT / beta_AT - K0 * Patm / rho * del_xCO2)

simplify with parameter "gamma"
gamma1 = V * Kw * (1 - f_ice) / Ma / z1
gamma2 = -Kw * (1 - fice) / z1

∆q_sea-air,xCO2 = gamma1 * (rho * R_DIC * del_DIC / beta_DIC + rho * R_AT * del_AT / beta_AT - K0 * Patm * del_xCO2)
∆q_air-sea,DIC = gamma2 * (R_DIC * del_DIC / beta_DIC + R_AT * del_AT / beta_AT - K0 * Patm / rho * del_xCO2)

Created on Tue Jul  8 12:24:04 2025

@author: Reese Barrett
"""

import project2 as p2
import xarray as xr
import numpy as np
import PyCO2SYS as pyco2
from scipy import sparse

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

#%% add CDR perturbation
# as a test, add point source of 4 µmol m-2 s-1 NaOH (this is what they did in
# Wang et al., 2022 Bering Sea paper)
# with NaOH, no alkalinity change, 1:1 AT:NaOH change

# depth = 0, latitude = ~54, longitude = ~-165
# in model, this approximately corresponds to model_depth[0], model_lat[73], model_lon[97]

# ∆q_CDR,AT (change in alkalinity due to CDR addition) - final units: [µmol AT kg-1 s-1]
del_q_CDR_AT_3D = np.full(ocnmask.shape, np.nan)
del_q_CDR_AT_3D[ocnmask == 1] = 0
del_q_CDR_AT_3D[0, 97, 73] = 4 # [µmol m-2 s-1]
del_q_CDR_AT_3D = del_q_CDR_AT_3D * grid_z / rho # convert from [µmol AT m-2 s-1] to [µmol AT kg-1 s-1]
del_q_CDR_AT = p2.flatten(del_q_CDR_AT_3D, ocnmask)

# ∆q_CDR,DIC (change in DIC due to CDR addition) - final units: [µmol DIC kg-1 s-1]
del_q_CDR_DIC_3D = np.full(ocnmask.shape, np.nan)
del_q_CDR_DIC_3D[ocnmask == 1] = 0
del_q_CDR_DIC_3D = del_q_CDR_DIC_3D * grid_z / rho # convert from [µmol DIC m-2 s-1] to [µmol DIC kg-1 s-1]
del_q_CDR_DIC = p2.flatten(del_q_CDR_DIC_3D, ocnmask)

#%% load in regridded COBALT data (or, regrid COBALT data)
cobalt_path = data_path + 'COBALT_regridded/'

#cobalt = xr.open_dataset('/Volumes/LaCie/data/OM4p25_cobalt_v3/19580101.ocean_cobalt_fluxes_int.nc', decode_cf=False)
#p2.regrid_cobalt(cobalt.jdiss_cadet_arag, model_depth, model_lat, model_lon, ocnmask, cobalt_path)
#p2.regrid_cobalt(cobalt.jdiss_cadet_calc, model_depth, model_lat, model_lon, ocnmask, cobalt_path)
#p2.regrid_cobalt(cobalt.jprod_cadet_arag, model_depth, model_lat, model_lon, ocnmask, cobalt_path)
#p2.regrid_cobalt(cobalt.jprod_cadet_calc, model_depth, model_lat, model_lon, ocnmask, cobalt_path)

# final units: [µmol DIC kg-1 s-1]
#q_diss_arag_3D = np.load(cobalt_path + 'jdiss_cadet_arag.npy') # [mol CACO3 m-2 s-1]
q_diss_calc_3D = np.load(cobalt_path + 'jdiss_cadet_calc.npy') # [mol CACO3 m-2 s-1]
#q_diss_DIC_3D = (q_diss_arag_3D + q_diss_calc_3D) * 1e-6 * grid_z / rho # [µmol DIC kg-1 s-1]
q_diss_DIC_3D = (q_diss_calc_3D) * 1e-6 * grid_z / rho # [µmol DIC m-2 s-1]
q_diss_AT_3D = q_diss_DIC_3D * 2 # [µmol DIC m-2 s-1]

q_prod_arag_3D = np.load(cobalt_path + 'jprod_cadet_arag.npy') # [mol CACO3 m-2 s-1]
q_prod_calc_3D = np.load(cobalt_path + 'jprod_cadet_calc.npy') # [mol CACO3 m-2 s-1]
q_prod_DIC_3D = (q_prod_arag_3D + q_prod_calc_3D) * 1e-6 * grid_z / rho # [µmol DIC kg-1 s-1]
q_prod_AT_3D = q_prod_DIC_3D * 2 # [µmol AT kg-1 s-1]

#%% make assumptions to calculate "delta" for dissolution and production
# right now, assuming linear relationship with arbitrary factor of 0.1 to see
# if this even sort of works

# ∆q_diss,DIC [µmol DIC kg-1 s-1]
q_diss_DIC = p2.flatten(0.1 * q_diss_DIC_3D, ocnmask)

# ∆q_diss,AT [µmol AT kg-1 s-1 s-1]
q_diss_AT = p2.flatten(0.1 * q_diss_AT_3D, ocnmask)

# ∆q_prod,DIC [µmol DIC kg-1 s-1]
q_prod_DIC = p2.flatten(0.1 * q_prod_DIC_3D, ocnmask)

# ∆q_prod,AT [µmol AT kg-1 s-1]
q_prod_AT = p2.flatten(0.1 * q_prod_AT_3D, ocnmask)

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
Ma = 1.8e26 # number of micromoles of air in atmosphere
beta_DIC = DIC/aqueous_CO2 # [unitless]
beta_AT = AT/aqueous_CO2 # [unitless]
K0 = aqueous_CO2/pCO2*rho # [µmol m-3 atm-1], in derivation this is defined in per volume units so used density to get there
Patm = 1 # atmospheric pressure [atm]
z1 = model_depth[0] # depth of first layer of model [m]
V = p2.flatten(model_vols, ocnmask) # volume of first layer of model [m^3]

# add layers of "np.NaN" for all subsurface layers in Kw, f_ice, then flatten
Kw_3D = np.full(ocnmask.shape, np.nan)
Kw_3D[0, :, :] = Kw_2D
Kw = p2.flatten(Kw_3D, ocnmask)

f_ice_3D = np.full(ocnmask.shape, np.nan)
f_ice_3D[0, :, :] = f_ice_2D
f_ice = p2.flatten(f_ice_3D, ocnmask)

gamma1 = V * Kw * (1 - f_ice) / Ma / z1
gamma2 = -Kw * (1 - f_ice) / z1

#%% set up time stepping
# see p2_exp04.py for more advanced time stepping, test for now

t = np.arange(0,4,1)

#%% construct matricies
# matrix form:
#  dx/dt = A * x + b
#  x = variable(s) of interest
#  A = transport matrix (TR) plus any processes with dependence on x   
#  b = source/sink vector (processes not dependent on x)
    
# m = # ocean grid cells
# nt = # time steps

m = TR.shape[0]
nt = len(t)

# x = [ ∆xCO2 ] --> 1 * nt
#     [ ∆DIC  ] --> m * nt
#     [ ∆AT   ] --> m * nt

x = np.zeros((1 + 2*m, nt))

# b = [ 0                                                                           ] --> 1 * ns, b[0]
#     [ ∆q_CDR,AT + 2 * (∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc) ] --> m * ns, b[1:(m+1)]
#     [ ∆q_CDR,DIC + ∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc      ] --> m * ns, b[(m+1):(2*m+1)]

# which translates to...
# b = [ 0                                      ] --> 1 * ns, b[0]
#     [ ∆q_CDR,AT + ∆q_diss,AT - ∆q_prod,AT    ] --> m * ns, b[1:(m+1)]
#     [ ∆q_CDR,DIC + ∆q_diss,DIC - ∆q_prod,DIC ] --> m * ns, b[(m+1):(2*m+1)]

b = np.zeros((1 + 2*m, nt))

# add in source/sink vectors for ∆AT, only add perturbation for time step 0

# for ∆AT
b[1:(m+1),0] = del_q_CDR_AT + q_diss_AT - q_prod_AT
b[1:(m+1),1:nt] = np.tile((q_diss_AT - q_prod_AT)[:, np.newaxis], (1, 3))

# for ∆DIC
b[(m+1):(2*m+1),0] = del_q_CDR_DIC + q_diss_DIC - q_prod_DIC
b[(m+1):(2*m+1),1:nt] = np.tile((q_diss_DIC - q_prod_DIC)[:, np.newaxis], (1, 3))

# dimensions
# A = [1 x 1][1 x m][1 x m] --> total size 2m + 1 x 2m + 1
#     [m x 1][m x m][m x m]
#     [m x 1][m x m][m x m]

# what acts on what
# A = [THIS BOX * ∆xCO2][THIS BOX * ∆DIC][THIS BOX * ∆AT] --> to calculate new ∆xCO2 (still need b)
#     [THIS BOX * ∆xCO2][THIS BOX * ∆DIC][THIS BOX * ∆AT] --> to calculate new ∆DIC (still need b)
#     [THIS BOX * ∆xCO2][THIS BOX * ∆DIC][THIS BOX * ∆AT] --> to calculate new ∆AT (still need b)

# math in each box (note: air-sea gas exchange terms only operate in surface boxes, they are set as main diagonal of identity matrix)
# A = [-gamma1 * K0 * Patm      ][gamma1 * rho * R_DIC / beta_DIC][gamma1 * rho * R_AT / beta_AT]
#     [-gamma2 * K0 * Patm / rho][TR + gamma2 * R_DIC / beta_DIC ][gamma2 * R_AT / beta_AT      ]
#     [0                        ][0                              ][TR                           ]

# notation for setup
# A = [A00][A01][A02]
#     [A10][A11][A12]
#     [A20][A21][A22]

# to solve for ∆xCO2
A00 = -1 * Patm * np.nansum(gamma1 * K0) # using nansum because all subsurface boxes are NaN, we only want surface
A01 = np.nan_to_num(gamma1 * rho * R_DIC / beta_DIC) # nan_to_num sets all NaN = 0 (subsurface boxes, no air-sea gas exchange)
A02 = np.nan_to_num(gamma1 * rho * R_AT / beta_AT)

# combine into A0 row
A0_ = np.full(1 + 2*m, np.nan)
A0_[0] = A00
A0_[1:(m+1)] = A01
A0_[(m+1):(2*m+1)] = A02

del A00, A01, A02

# to solve for ∆DIC
A10 = np.nan_to_num(-1 * gamma2 * K0 * Patm / rho)# is csc the most efficient format? come back to this
A11 = TR + sparse.diags(np.nan_to_num(gamma2 * R_DIC / beta_DIC), format='csc')
A12 = sparse.diags(np.nan_to_num(gamma2 * R_AT / beta_AT))

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

#%% perform time stepping











