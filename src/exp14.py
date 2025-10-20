#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:40:11 2025

EXP14: Trying to replicate Wang et al., 2022 results (OAE in Bering Sea)
- TA flux applied in a single grid cell located in the middle of Unimak Pass
  with a rate of 4 μmol m−2 s−1 for 10 years (or, overall domain-wide TA
  addition rate is 1.67e10 mol yr−1)
  
  1.67 × 1010 mol yr−1
- Simulation is run for only 10 years
- TA addition coordinates are approx. 54.333724, 195.062015 in Wang model, the
  closest we can get with this model is 55.38, 195 (which is model_lat[73],
  model_lon[97])
- This grid cell has a volume of 2.7422e11 m3 and a mass of 2.8107e14 kg, which
  means we will add TA as a flux of 59.41375 µmol kg-1 yr-1
  (calculation: 1.67e10 / (model_vols[0, 95, 73] * rho) * 1e6)
- Assuming no biotic calcification feedback

exp14_2025-08-04-a.nc
- Running 10 years, 1 year time steps, with addition of 59.41375 µmol kg-1 yr-1
  in surface ocean at model_lat[73], model_lon[95] (wrong lon)
exp14_2025-08-28-a.nc
- WITH FIXED WIND SPEED & z1 !
- Running 10 years, 1 year time steps, with addition of 59.41375 µmol kg-1 yr-1
  in surface ocean at model_lat[73], model_lon[95] (wrong lon)
exp14_2025-08-29-a.nc
- WITH FIXED WIND SPEED & z1 !
- Running 10 years, 1 year time steps, with addition of 59.41375 µmol kg-1 yr-1
  in surface ocean at model_lat[73], model_lon[97]
exp14_2025-09-03-a.nc
- WITH FIXED WIND SPEED & z1 !
- Running 10 years, 1 year time steps, with addition of 59.41375 µmol kg-1 yr-1
  in surface ocean at model_lat[73], model_lon[97] ONLY IN FIRST TIME STEP

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
    exp##_YYYY-MM-DD-a.nc (where expXX corresponds to the python file used to
    run the experiment; a, b, c etc. represent model runs from the same day)

@author: Reese C. Barrett
"""

import project2 as p2
import xarray as xr
import numpy as np
import PyCO2SYS as pyco2
from scipy import sparse
from scipy.sparse.linalg import spilu, LinearOperator, lgmres
from time import time
from tqdm import tqdm
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

# seawater density for volume to mass [kg m-3]
rho = 1025 

# depth of first model layer (need bottom of grid cell, not middle) [m]
z1 = model_data['wz'].to_numpy()[1, 0, 0]

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
S_3D = np.load(data_path + 'WOA18/S.npy')   # salinity [unitless]
T_3D = np.load(data_path + 'WOA18/T.npy')   # temperature [ºC]
Si_3D = np.load(data_path + 'WOA18/Si.npy') # silicate [µmol kg-1]
P_3D = np.load(data_path + 'WOA18/P.npy')   # phosphate [µmol kg-1]

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
f_ice_2D = np.load(data_path + 'NCEP_DOE_Reanalysis_II/icec.npy') # annual mean ice fraction from 0 to 1 in each grid cell
uwnd_2D = np.load(data_path + 'NCEP_DOE_Reanalysis_II/wspd.npy') # annual mean of forecast of U-wind at 10 m [m/s]
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

# regrid GLODAP data
#p2.regrid_glodap(data_path, 'TCO2', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'TAlk', model_depth, model_lat, model_lon, ocnmask)

# upload regridded GLODAP data
DIC_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC.npy') # dissolved inorganic carbon [µmol kg-1]
AT_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA.npy')   # total alkalinity [µmol kg-1]

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

#%% set up time stepping

# set up time domain
dt = 1 # 1 year

# 10 years after year 0
t = np.arange(0, 11, dt)

#%% add CDR perturbation
# Add surface ocean perturbation of -1 µmol kg-1 yr-1 in DIC, no change in AT
# Goal: compare results with Yamamoto et al., 2024 supplemental figures

# ∆q_CDR,AT (change in alkalinity due to CDR addition) - final units: [µmol AT kg-1 yr-1]
del_q_CDR_AT_3D = np.full(ocnmask.shape, np.nan)
del_q_CDR_AT_3D[ocnmask == 1] = 0
del_q_CDR_AT_3D[0, 97, 73] = 1.67e10 / (model_vols[0, 97, 73] * rho) * 1e6 # µmol kg-1 yr-1
del_q_CDR_AT = p2.flatten(del_q_CDR_AT_3D, ocnmask)

# ∆q_CDR,DIC (change in DIC due to CDR addition) - final units: [µmol DIC kg-1 yr-1]
del_q_CDR_DIC_3D = np.full(ocnmask.shape, np.nan)
del_q_CDR_DIC_3D[ocnmask == 1] = 0
del_q_CDR_DIC = p2.flatten(del_q_CDR_DIC_3D, ocnmask)
    
#%% construct matricies
# matrix form:
#  dc/dt = A * c + q
#  c = variable(s) of interest
#  A = transport matrix (TR) plus any processes with dependence on c 
#    = source/sink vector (processes not dependent on c)
    
# UNITS NOTE: all xCO2 units are yr-1 all AT units are µmol AT kg-1, all DIC units are µmol DIC kg-1
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

# for ∆DIC, no change

# for ∆AT, add perturbation for first 10 years
#q[(m+1):(2*m+1), 0:10] = np.tile(del_q_CDR_AT[:, np.newaxis], (1,10))
q[(m+1):(2*m+1), 0] = del_q_CDR_AT

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
A10 = np.nan_to_num(-1 * gammaC * K0 * Patm / rho)
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
    
#%% perform time stepping using Euler backward
LHS = sparse.eye(A.shape[0], format="csc") - dt * A

# test condition number of matrix
est = sparse.linalg.onenormest(LHS)
print("Estimated 1-norm condition number LHS1: ", est)

start = time()
ilu = spilu(LHS.tocsc(), drop_tol=1e-5, fill_factor=20)
stop = time()
print('ilu1 calculations: ' + str(stop - start) + ' s')

M = LinearOperator(LHS.shape, ilu.solve)

start = time()

for idx in tqdm(range(1, len(t))):
#for idx in range(1, len(t)):
    
    # add starting estimate after first time step
    if idx > 1:
        c0 = c[:,idx-1]
    else:
        c0=None
    
    RHS = c[:,idx-1] + np.squeeze(dt*q[:,idx-1])
    #start = time()
    c[:,idx], info = lgmres(LHS, RHS, M=M, x0=c0, rtol = 1e-5, atol=0)
    #stop = time()
    #print('t = ' + str(idx) + ', solve time: ' + str(stop - start) + ' s')
    
    if info != 0:
        if info > 0:
            print(f'did not converge in {info} iterations.')
        else:
            print('illegal input or breakdown')

stop = time()
print('time stepping total time: ' + str(stop - start) + ' s')

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
global_attrs = {'description': 'Different version of Wang et al., 2022 where only first year sees perturbation- running 10 years, 1 year time steps, with addition of 59.41375 umol kg-1 yr-1 in surface ocean at model_lat[73], model_lon[97] - with fixed wind speed and z1'}

# save model output
p2.save_model_output(
    'exp14_2025-09-03-a.nc', 
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

data = xr.open_dataset(output_path + 'exp14_2025-08-29-a.nc')

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
    p2.plot_surface3d(model_lon, model_lat, data['delAT'].isel(time=idx).values, 0, -8e-2, 8e-2, 'RdBu', 'Surface ∆DIC (µmol kg-1) at t=' + str(np.round(t[idx].values,3)) + ' yr')

'''   
for idx in range(0, nt):
    p2.plot_longitude3d(model_lat, model_depth, data['delDIC'].isel(time=idx).values, 97, 0, 5e-5, 'plasma', ' ∆DIC (µmol kg-1) at t=' +str(t[idx]) + ' along 165ºW longitude')
'''

#%% calculate total DIC sequestered

del_CO2_sw = np.nansum(data['delDIC'].isel(time=10).values * model_vols * rho) * 1e-6 * 44.0095 / 1000 / 1000 # metric tons CO2 in ocean due to AT addition (µmol -> mol -> g -> kg -> metric ton)

print('\ntotal change in DIC due to OAE by year 10: ' + str(np.round(del_CO2_sw, 10)) + ' metric tons')

# calculate figure 8 metrics: whole model
DIC_metric_tons = data['delDIC'].copy() * xr.DataArray(model_vols, dims=("depth", "lon", "lat"), coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) * rho * 1e-6 # mol DIC at each grid cell at each time
DIC_metric_tons = DIC_metric_tons.sum(dim=['depth', 'lon', 'lat'], skipna=True) # cumulative moles of DIC increased

AT_metric_tons = data['delAT'].copy() * xr.DataArray(model_vols, dims=("depth", "lon", "lat"), coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) * rho * 1e-6 # mol AT at each grid cell at each time
AT_metric_tons = AT_metric_tons.sum(dim=['depth', 'lon', 'lat'], skipna=True) # cumulative moles of DIC increased
#%%
fig = plt.figure(figsize=(5,4))
ax = fig.gca()

ax.plot(AT_metric_tons, label='TA', c='blue')
ax.plot(DIC_metric_tons, label='DIC', c='red')
ax.set_xlabel('Year')
ax.set_ylabel('Accumulation in Entire Model in mol')

#%% calculate figure 8 metrics: approx. bering sea
DIC_metric_tons = data['delDIC'].copy() * xr.DataArray(model_vols, dims=("depth", "lon", "lat"), coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) * rho * 1e-6 # mol DIC at each grid cell at each time
DIC_metric_tons = DIC_metric_tons.isel(lon=slice(90, 105), lat=slice(70, 80))
DIC_metric_tons = DIC_metric_tons.sum(dim=['depth', 'lon', 'lat'], skipna=True) # cumulative moles of DIC increased

AT_metric_tons = data['delAT'].copy() * xr.DataArray(model_vols, dims=("depth", "lon", "lat"), coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) * rho * 1e-6 # mol AT at each grid cell at each time
AT_metric_tons = AT_metric_tons.isel(lon=slice(90, 105), lat=slice(70, 80))
AT_metric_tons = AT_metric_tons.sum(dim=['depth', 'lon', 'lat'], skipna=True) # cumulative moles of DIC increased
#%%
fig = plt.figure(figsize=(5,4))
ax = fig.gca()

ax.plot(AT_metric_tons, label='TA', c='blue')
ax.plot(DIC_metric_tons, label='DIC', c='red')
ax.set_xlabel('Year')
ax.set_ylabel('Accumulation in Bering Sea (ish) in mol')

plt.legend()

#%% looking total carbon content over time (mass balance)
# actually, this doesn't work for this scenario. we need to see if C in the ocean + atmosphere stays constant after mCDR perturbations stop

# calculate total C added to system
# for this study, no C is added (only NaOH)
C_tot = np.zeros(nt)
for idx in range(nt):
    C_tot[idx] = np.nansum(p2.make_3D(q[1:(m+1), idx], ocnmask) * model_vols * rho * 1e-6) #  µmol kg-1 DIC to mol TA to mol C added
C_tot = np.cumsum(C_tot)

# open output array
data = xr.open_dataset(output_path + 'exp14_2025-09-03-a.nc')

# calculate change in carbon stored in ocean 
model_vols_xr = xr.DataArray(model_vols, dims=["depth", "lon", "lat"], coords={"depth": data.depth, "lon": data.lon, "lat": data.lat}) # broadcast model_vols to convert ∆DIC from per kg to total
C_ocean = data['delDIC'] * model_vols_xr * rho * 1e-6 #  µmol kg-1 DIC to mol C
C_ocean = np.squeeze(np.apply_over_axes(np.nansum, C_ocean, [1, 2, 3])) # sum over depth, lon, lat

# calculate change in carbon stored in air
C_air = data['delxCO2'].values * 1e-6 * (Ma * 1e-6) # ppm xCO2 (µmol CO2 per mol air) to mol C

print('carbon added by CDR perturbation at each time step (mol C):')
print(C_tot)
print('\nchange in carbon stored in ocean at each time step (mol C):')
print(C_ocean)
print('\nchange in carbon stored in air at each time step (mol C):')
print(C_air)
print('\nsum of change in carbon stored in ocean and air at each time step (mol C):')
print(C_ocean + C_air)

fig = plt.figure(figsize=(5,4), dpi=200)
ax = fig.gca()

ax.plot(t, C_ocean, label='C$_{ocean}$')
ax.plot(t, C_air, label='C$_{air}$')
ax.plot(t, C_ocean + C_air, label='C$_{ocean}$ + C$_{air}$')
ax.plot(t, C_tot, label='C$_{CDR}$', c='black', linestyle=':')
ax.legend()
ax.set_xlabel('Time (yr)')
ax.set_ylabel('Change in Carbon in CDR Scenario (mol C)')




