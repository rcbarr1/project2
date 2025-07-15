#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:13:48 2025

EXP10: Creating zero-sensitivity run for simulations.
- Assuming that ∆q_diss and ∆q_prod terms are equal to 0.
exp10_2025-7-15-a.nc
- NaOH (4 µmol NaOH m-2 s-1, which is 4 µmol AT m-2 s-1) added before first time step ONLY at model_depth[0], model_lat[73], model_lon[97]
exp10_2025-7-15-b.nc
- CaCO3 (2 µmol CaCO3 m-2 s-1, which is 2 µmol DIC m-2 s-1 and 4 µmol AT m-2 s-1) added before first time step ONLY at model_depth[0], model_lat[73], model_lon[97]

Governing equations (based on my own derivation + COBALT governing equations)
1. d(xCO2)/dt = ∆q_sea-air,xCO2 --> [atm CO2 (atm air)-1 yr-1] or [µmol CO2 (µmol air)-1 yr-1]
2. d(∆DIC)/dt = TR * ∆DIC + ∆q_air-sea,DIC + ∆q_CDR,DIC + ∆q_diss,DIC - ∆q_prod,DIC --> [µmol DIC (kg seawater)-1 yr-1]
3. d(∆AT)/dt = TR * ∆AT + ∆q_CDR,AT + ∆q_diss,AT - ∆q_prod,AT --> [µmol AT (kg seawater)-1 yr-1]

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

Note about transport matrix set-up
- This was designed in matlab, which uses "fortran-style" aka column major ordering
- This means that "e" and "b" vectors (see John et al., 2020) must be constructed in this order
- This is complicated by the fact that the land boxes are excluded from the transport matrix
- The length of e and b vectors, as well as the length and width of the
  transport operator, are equal to the total number of ocean boxes in the model
- Best practices: create "e" and "b" vectors in three dimensions, flatten and mask out land boxes simultaneously 

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
from scipy.sparse.linalg import spsolve
import os

# set fewer threads to see if I can avoid running out of memory and crashing
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

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

#%% add CDR perturbation
# for experiment "a", add point source of 4 µmol m-2 s-1 NaOH (this is what
# they did in Wang et al., 2022 Bering Sea paper)
#  --> with NaOH, no DIC change, 1:1 AT:NaOH change
# for experiment "b", add point source of 2 µmol m-2 s-1 CaCO3
#  --> per mole of CaCO3, DIC will change by 1 unit, AT will change by 2 units

# depth = 0, latitude = ~54, longitude = ~-165
# in model, this approximately corresponds to model_depth[0], model_lat[73], model_lon[97]

# ∆q_CDR,AT (change in alkalinity due to CDR addition) - final units: [µmol AT kg-1 yr-1]
del_q_CDR_AT_3D = np.full(ocnmask.shape, np.nan)
del_q_CDR_AT_3D[ocnmask == 1] = 0
del_q_CDR_AT_3D[0, 97, 73] = 4 # [µmol m-2 s-1]
del_q_CDR_AT_3D = del_q_CDR_AT_3D * grid_z / rho * sec_per_year # convert from [µmol AT m-2 s-1] to [µmol AT kg-1 yr-1]
del_q_CDR_AT = p2.flatten(del_q_CDR_AT_3D, ocnmask)

# ∆q_CDR,DIC (change in DIC due to CDR addition) - final units: [µmol DIC kg-1 yr-1]
del_q_CDR_DIC_3D = np.full(ocnmask.shape, np.nan)
del_q_CDR_DIC_3D[ocnmask == 1] = 0
del_q_CDR_DIC_3D[0, 97, 73] = 2 # [µmol m-2 s-1], turn this off for NaOH run but on for CaCO3 run
del_q_CDR_DIC_3D = del_q_CDR_DIC_3D * grid_z / rho * sec_per_year # convert from [µmol DIC m-2 s-1] to [µmol DIC kg-1 yr-1]
del_q_CDR_DIC = p2.flatten(del_q_CDR_DIC_3D, ocnmask)

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

# simple for now
t = np.arange(0,4,1)
dt = 1 # time step length

#%% construct matricies
# matrix form:
#  dx/dt = A * x + b
#  x = variable(s) of interest
#  A = transport matrix (TR) plus any processes with dependence on x   
#  b = source/sink vector (processes not dependent on x)
    
# UNITS NOTE: all xCO2 units are yr-1 all AT units are µmol AT kg-1 s-2, all DIC units are µmol DIC kg-1 s-2
# see comment at top for more info

# m = # ocean grid cells
# nt = # time steps

m = TR.shape[0]
nt = len(t)

# x = [ ∆xCO2 ] --> 1 * nt
#     [ ∆DIC  ] --> m * nt
#     [ ∆AT   ] --> m * nt

x = np.zeros((1 + 2*m, nt))

# b = [ 0                                                                           ] --> 1 * nt, b[0]
#     [ ∆q_CDR,DIC + ∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc      ] --> m * nt, b[1:(m+1)]
#     [ ∆q_CDR,AT + 2 * (∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc) ] --> m * nt, b[(m+1):(2*m+1)]

# which translates to...
# b = [ 0                                      ] --> 1 * nt, b[0]
#     [ ∆q_CDR,DIC + ∆q_diss,DIC - ∆q_prod,DIC ] --> m * nt, b[1:(m+1)]
#     [ ∆q_CDR,AT + ∆q_diss,AT - ∆q_prod,AT    ] --> m * nt, b[(m+1):(2*m+1)]

# EXCEPT, ASSUMING ALL ∆q_diss & ∆q_prod = 0 FOR ZERO-SENSITIVITY RUN
b = np.zeros((1 + 2*m, nt))

# add in source/sink vectors for ∆AT, only add CDR perturbation for time step 0

# for ∆DIC
b[1:(m+1),0] = del_q_CDR_DIC

# for ∆AT
b[(m+1):(2*m+1),0] = del_q_CDR_AT

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

del TR

#%% perform time stepping using Euler backward
LHS = sparse.eye(A.shape[0], format="csc") - dt * A
del A

for idx in tqdm(range(1, len(t))):
    RHS = x[:,idx-1] + np.squeeze(dt*b[:,idx-1])
        
    x[:,idx] = spsolve(LHS,RHS) # time step with backwards Euler

#%% rebuild 3D concentrations from 1D array used for solving matrix equation

# partition "x" into xCO2, DIC, and AT
x_xCO2 = x[0, :]
x_DIC  = x[1:(m+1), :]
x_AT   = x[(m+1):(2*m+1), :]

# convert xCO2 units from unitless [atm CO2 / atm air] or [mol CO2 / mol air] to ppm
x_xCO2 *= 1e6

# reconstruct 3D arrays for DIC and AT
x_DIC_3D = np.full([len(t), ocnmask.shape[0], ocnmask.shape[1], ocnmask.shape[2]], np.nan) # make 3D vector full of nans
x_AT_3D = np.full([len(t), ocnmask.shape[0], ocnmask.shape[1], ocnmask.shape[2]], np.nan) # make 3D vector full of nans

# for each time step, reshape 1D array into 3D array, then save to larger 4D array output (time, depth, longitude, latitude)
for idx in range(0, len(t)):
    x_DIC_reshaped = np.full(ocnmask.shape, np.nan)
    x_AT_reshaped = np.full(ocnmask.shape, np.nan)

    x_DIC_reshaped[ocnmask == 1] = np.reshape(x_DIC[:, idx], (-1,), order='F')
    x_AT_reshaped[ocnmask == 1] = np.reshape(x_AT[:, idx], (-1,), order='F')
    
    x_DIC_3D[idx, :, :, :] = x_DIC_reshaped
    x_AT_3D[idx, :, :, :] = x_AT_reshaped

#%% save model output in netCDF format
global_attrs = {'description':'Zero-sensitivity run b - all del_q_prod and del_q_diss are set to equal zero - addition of 2 umol m-2 s-1 of CaCO3 in Bering Strait'}

# save model output
p2.save_model_output(
    'exp10_2025-7-15-b.nc', 
    t, 
    model_depth, 
    model_lon,
    model_lat, 
    tracers=[x_xCO2, x_DIC_3D, x_AT_3D], 
    tracer_dims=[('time',), ('time', 'depth', 'lon', 'lat'), ('time', 'depth', 'lon', 'lat')],
    tracer_names=['delxCO2', 'delDIC', 'delAT'], 
    tracer_units=['ppm', 'umol kg-3', 'umol kg-3'],
    global_attrs=global_attrs
)

#%% open and plot model output
data = xr.open_dataset(output_path + 'exp10_2025-7-15-a.nc')

model_time = data.time
nt = len(model_time)

model_lon = data.lon.data
model_lat = data.lat.data
model_depth = data.depth.data

for idx in range(0, nt):
    p2.plot_surface3d(model_lon, model_lat, data['delAT'].isel(time=idx).values, 0, 0, 5e-5, 'plasma', 'surface ∆AT (µmol kg-1) at t=' + str(model_time[idx].values))

for idx in range(0, nt):
    p2.plot_longitude3d(model_lat, model_depth, data['delAT'].isel(time=idx).values, 97, 0, 5e-5, 'plasma', ' ∆AT (µmol kg-1) at t=' +str(model_time[idx].values) + ' along 165ºW longitude')

for idx in range(0, nt):
    p2.plot_surface3d(model_lon, model_lat, data['delDIC'].isel(time=idx).values, 0, 0, 5e-5, 'plasma', 'surface ∆DIC (µmol kg-1) at t=' + str(model_time[idx].values))
    
for idx in range(0, nt):
    p2.plot_longitude3d(model_lat, model_depth, data['delDIC'].isel(time=idx).values, 97, 0, 5e-5, 'plasma', ' ∆DIC (µmol kg-1) at t=' +str(model_time[idx].values) + ' along 165ºW longitude')
    
print('total delAT:')
for idx in range(0, nt):
    print('\t' + str(np.round(np.nansum(data['delAT'].isel(time=idx).values),4)) + ' µmol kg-1')
    
print('total delDIC:')
for idx in range(0, nt):
    print('\t' + str(np.round(np.nansum(data['delDIC'].isel(time=idx).values),4)) + ' µmol kg-1')
    
print('delxCO2:')
for idx in range(0, nt):
    print('\t' + str(np.round(data['delxCO2'].isel(time=idx).values,10)) + ' ppm')
