#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:10:43 2025

EXP09: Creating control run for simulations.
- Assuming that all ∆q terms are equal to 0.

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
from scipy import sparse
from tqdm import tqdm
from scipy.sparse.linalg import spsolve

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
    
# UNITS NOTE: all xCO2 units are yr-1, all AT units are µmol AT kg-1 s-2, all DIC units are µmol DIC kg-1 s-2
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

# EXCEPT, ASSUMING ALL = 0 FOR CONTROL RUN
b = np.zeros((1 + 2*m, nt))

# add in source/sink vectors for ∆AT, only add perturbation for time step 0

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

# BUT, WITH NO AIR-SEA GAS EXCHANGE (∆q_air-sea = 0), looks like this:
# A = [0      ][0      ][0      ]
#     [0      ][TR     ][0      ]
#     [0      ][0      ][TR     ]
    
# notation for setup
# A = [A00][A01][A02]
#     [A10][A11][A12]
#     [A20][A21][A22]

# to solve for ∆xCO2
A00 = 0
A01 = np.zeros(m)
A02 = np.zeros(m)

# combine into A0 row
A0_ = np.full(1 + 2*m, np.nan)
A0_[0] = A00
A0_[1:(m+1)] = A01
A0_[(m+1):(2*m+1)] = A02

del A00, A01, A02

# to solve for ∆DIC
A10 = np.zeros(m)
A11 = TR
A12 = 0 * TR

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
global_attrs = {'description':'Control run - all del_q are set to 0'}

# save model output
p2.save_model_output(
    'exp09_2025-7-15-a.nc', 
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
data = xr.open_dataset(output_path + 'exp09_2025-7-15-a.nc')

model_time = data.time
model_lon = data.lon.data
model_lat = data.lat.data
model_depth = data.depth.data

for idx in range(0, nt):
    print(idx)
    p2.plot_surface3d(model_lon, model_lat, data['delAT'].isel(time=idx).values, 0, 0, 5e-5, 'plasma', 'surface ∆DIC (µmol kg-1) at t=' + str(t[idx]))
   
for idx in range(0, nt):
    p2.plot_longitude3d(model_lat, model_depth, data['delAT'].isel(time=idx).values, 97, 0, 5e-5, 'plasma', ' ∆DIC (µmol kg-1) at t=' +str(t[idx]) + ' along 165ºW longitude')
    
# test: sum across time steps 1, 2, 3
for idx in range(0, nt):
    print(np.nansum(data['delAT'].isel(time=idx).values))
    print(np.nansum(data['delDIC'].isel(time=idx).values))
    print(data['delxCO2'].isel(time=idx).values)

for idx in range(0, nt):
    print(np.nanmax(data['delAT'].isel(time=idx).values))
    print(np.nanmax(data['delDIC'].isel(time=idx).values))




