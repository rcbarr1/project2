#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:15:02 2025

EXP12: Testing conservation with ILU + LGMRES, exploring different tolerances
exp11_2025-7-21-a.nc
- ILU + LGMRES with starting value of 100 µmol kg-1 of arbitrary tracer at
  arbitrary location (-39.5, 101), which is (model_lat[25], model_lon[50])
- default tolerances (rtol = 1e-5, atol = 0.0, maxiter = 1000)

Governing equations (based on my own derivation + COBALT governing equations)
1. d(xCO2)/dt = ∆q_sea-air,xCO2 --> [atm CO2 (atm air)-1 yr-1] or [mol CO2 (mol air)-1 yr-1]
2. d(∆DIC)/dt = TR * ∆DIC + ∆q_air-sea,DIC + ∆q_hard,DIC + ∆q_CDR,DIC --> [mol DIC (kg seawater)-1 yr-1]
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

units: mol kg-1 yr-1
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
from scipy import sparse
#from tqdm import tqdm
from scipy.sparse.linalg import spilu, LinearOperator, lgmres
from time import time

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

# shortened time stepping to test solvers
t1 = np.arange(0, 30/360, dt1) # use a 1 day time step for the first 30 days
t2 = np.arange(30/360, 1, dt2) # use a 1 month time step until the 1st year
t3 = np.arange(1, 3, dt3) # use a 1 year time step until the 3rd year
t4 = np.arange(3, 23, dt4) # use a 10 year time step until the 23rd year
t5 = np.arange(23, 123+dt5, dt5) # use a 100 year time step until the 1000th year

t = np.concatenate((t1, t2, t3, t4, t5))

#%% construct matricies
# matrix form:
#  dc/dt = A * c + q
#  c = variable(s) of interest
#  A = transport matrix (TR) plus any processes with dependence on c  
#  q = source/sink vector (processes not dependent on c)
    
# UNITS NOTE: all xCO2 units are yr-1 all AT units are mol AT kg-1, all DIC units are mol DIC kg-1
# see comment at top for more info

# m = # ocean grid cells
# nt = # time steps

m = TR.shape[0]
nt = len(t)

# c = [ c ] --> m * nt

c = np.zeros((m, nt))

# q = [ q ] --> m * nt

q = np.zeros((m, nt))

#%% add in intial value for x to test conservation
# x represents an arbitrary conserved tracer with units mol kg-1

c0_3D = np.zeros(ocnmask.shape)
c0_3D[0, 50, 25] = 100 # mol m-3

c0 = p2.flatten(c0_3D, ocnmask)

c[:, 0] = c0

#%% perform time stepping using Euler backward
LHS1 = sparse.eye(TR.shape[0], format="csc") - dt1 * TR
LHS2 = sparse.eye(TR.shape[0], format="csc") - dt2 * TR
LHS3 = sparse.eye(TR.shape[0], format="csc") - dt3 * TR
LHS4 = sparse.eye(TR.shape[0], format="csc") - dt4 * TR
LHS5 = sparse.eye(TR.shape[0], format="csc") - dt5 * TR

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

M1 = LinearOperator(LHS1.shape, ilu1.solve)
M2 = LinearOperator(LHS2.shape, ilu2.solve)
M3 = LinearOperator(LHS3.shape, ilu3.solve)
M4 = LinearOperator(LHS4.shape, ilu4.solve)
M5 = LinearOperator(LHS5.shape, ilu5.solve)

for idx in range(1, len(t)):
    
    # add starting guess after first time step
    if idx > 1:
        c0 = c[:,idx-1]
    else:
        c0=None
    
    #if t[idx] <= 90/360: # 1 day time step
    if t[idx] <= 30/360: # 1 day time step
        RHS = c[:,idx-1] + np.squeeze(dt1*q[:,idx-1])
        start = time()
        c[:,idx], info = lgmres(LHS1, RHS, M=M1, x0=c0, rtol = 1e-5, atol=0)
        stop = time()
        #print('t = ' + str(idx) + ', solve time: ' + str(stop - start) + ' s')
        print(stop - start)
   
    #elif (t[idx] > 90/360) & (t[idx] <= 5): # 1 month time step
    elif (t[idx] > 30/360) & (t[idx] <= 1): # 1 month time step
        RHS = c[:,idx-1] + np.squeeze(dt2*q[:,idx-1])
        start = time()
        c[:,idx], info = lgmres(LHS2, RHS, M=M2, x0=c0, rtol = 1e-5, atol=0)
        stop = time()
        #print('t = ' + str(idx) + ', solve time: ' + str(stop - start) + ' s')
        print(stop - start)
    
    #elif (t[idx] > 5) & (t[idx] <= 100): # 1 year time step
    elif (t[idx] > 1) & (t[idx] <= 3): # 1 year time step
        start = time()
        RHS = c[:,idx-1] + np.squeeze(dt3*q[:,idx-1])
        c[:,idx], info = lgmres(LHS3, RHS, M=M3, x0=c0, rtol = 1e-5, atol=0)
        stop = time()
        #print('t = ' + str(idx) + ', solve time: ' + str(stop - start) + ' s')
        print(stop - start)

    #elif (t[idx] > 100) & (t[idx] <= 500): # 10 year time step
    elif (t[idx] > 3) & (t[idx] <= 23): # 10 year time step
        start = time()
        RHS = c[:,idx-1] + np.squeeze(dt4*q[:,idx-1])
        c[:,idx], info = lgmres(LHS4, RHS, M=M4, x0=c0, rtol = 1e-5, atol=0)
        stop = time()
        #print('t = ' + str(idx) + ', solve time: ' + str(stop - start) + ' s')
        print(stop - start)

    else: # 100 year time step
        start = time()
        RHS = c[:,idx-1] + np.squeeze(dt5*q[:,idx-1])
        c[:,idx], info = lgmres(LHS5, RHS, M=M5, x0=c0, rtol = 1e-5, atol=0)
        stop = time()
        #print('t = ' + str(idx) + ', solve time: ' + str(stop - start) + ' s')
        print(stop - start)
        
if info != 0:
    if info > 0:
        print(f'did not converge in {info} iterations.')
    else:
        print('illegal input or breakdown')

#%% rebuild 3D concentrations from 1D array used for solving matrix equation

# reconstruct 3D array
c_3D = np.full([len(t), ocnmask.shape[0], ocnmask.shape[1], ocnmask.shape[2]], np.nan) # make 3D vector full of nans

# for each time step, reshape 1D array into 3D array, then save to larger 4D array output (time, depth, longitude, latitude)
for idx in range(0, len(t)):
    c_reshaped = np.full(ocnmask.shape, np.nan)

    c_reshaped[ocnmask == 1] = np.reshape(c[:, idx], (-1,), order='F')
    
    c_3D[idx, :, :, :] = c_reshaped

#%% save model output in netCDF format
global_attrs = {'description':'testing conservation with ilu and lgmres - default tolerances'}

# save model output
p2.save_model_output(
    'exp11_2025-7-28-a.nc', 
    t, 
    model_depth, 
    model_lon,
    model_lat, 
    tracers=[c_3D,], 
    tracer_dims=[('time', 'depth', 'lon', 'lat')],
    tracer_names=['c'], 
    tracer_units=['umol m-3'],
    global_attrs=global_attrs
)

#%% test for conservation

data = xr.open_dataset(output_path + 'exp11_2025-7-28-a.nc')

test = data['c'].isel(lon=50).isel(lat=25).isel(depth=0).values
    
t = data.time
model_lon = data.lon.data
model_lat = data.lat.data
model_depth = data.depth.data

nt = len(t)

print('\n\ntracer amount (µmol):')

# test: sum tracer concentration at each time step (starting at t = 6 when addition is over) to see if conserved
# currently it is not conserved!! why!
for i in range(0, nt):
    # multiply mol m^-3 * m^3 to see if AMOUNT is conserved
    c_amount = data['x'].isel(time=i) * model_vols
    
    print(str(np.nansum(c_amount)))
   
data.close()


















