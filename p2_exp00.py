#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp00 is just a tracer transport experiment (proof-of-concept that I can
meaningfully time step OCIM forward). There is not much physical meaning here.
 
 Note about transport matrix set-up
 - This was designed in matlab, which uses "fortran-style" aka column major ordering
 - This means that "e" and "b" vectors (see John et al., 2020) must be constructed in this order
 - This is complicated by the fact that the land boxes are excluded from the transport matrix
 - The length of e and b vectors, as well as the length and width of the
   transport operator, are equal to the total number of ocean boxes in the model
 - Best practices: create "e" and "b" vectors in three dimensions, flatten and mask out land boxes simultaneously 
 
 - To mask and flatten simultaneously, call: 
     e_flat = e_3D[ocnmask == 1].flatten(order='F')
   
 - To unflatten and unmask simultaneously (i.e. back to 3D grid), call:
     e_3D = np.full(ocnmask.shape, np.nan)
     e_3D[ocnmask == 1] = np.reshape(e_flat, (-1,), order='F')
     
Naming convention for saving model runs (see .txt file for explanation of experiments)
    exp##__YYYY-MM-DD-a.nc (where expXX corresponds to the python file used to
    run the experiment; a, b, c etc. represent model runs from the same day)

Created on Mon Oct  7 13:55:39 2024

@author: Reese Barrett
"""

import project2 as p2
import xarray as xr
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

data_path = '/Users/Reese_1/Documents/Research Projects/project2/data/'

#%% load transport matrix (OCIM-48L, from Holzer et al., 2021)
# transport matrix is referred to as "A" vector in John et al., 2020 (AWESOME OCIM)
TR = p2.loadmat(data_path + 'OCIM2_48L_base/OCIM2_48L_base_transport.mat')
TR = TR['TR']

# open up rest of data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].to_numpy()

model_depth = model_data['tz'].to_numpy()[:, 0, 0] # m below sea surface
model_lon = model_data['tlon'].to_numpy()[0, :, 0] # ºE
model_lat = model_data['tlat'].to_numpy()[0, 0, :] # ºN

#%% apply a tracer of concentration c µmol kg-1 per year, start with matrix
# of zeros, plot change in temperature anomaly with time
num_years = 15

c_anomaly = np.zeros(ocnmask.shape) # potential temperature [ºC]
c_anomaly = c_anomaly[ocnmask == 1].flatten(order='F') # reshape b vector

c_anomaly_3D = np.full(ocnmask.shape, np.nan)
c_anomaly_3D[ocnmask == 1] = np.reshape(c_anomaly, (-1,), order='F') # reshape e vector

c_anomaly_atm = np.zeros(ocnmask.shape)
c_anomaly_atm[0, 90:100, 30:40] = 1 # create boundary condition of 0.001 (change in surface forcing of 0.001 kg-1 yr-1)
p2.plot_surface3d(model_lon, model_lat, c_anomaly_atm, 0, 0, 1.5, 'plasma', 'surface forcing')
c_anomaly_atm = c_anomaly_atm[0, :, :]

c_anomaly_3D = np.full(ocnmask.shape, np.nan)
c_anomaly_3D[ocnmask == 1] = np.reshape(c_anomaly, (-1,), order='F')

c_anomalies = np.zeros([num_years, model_depth, model_lon, model_lat])
c_anomalies [0, :, :, :] = c_anomaly_3D

for t in range(1, num_years):
    print(t)
    # assuming constant b here, don't need to recalculate in this loop
    
    c_anomaly_surf = c_anomaly_3D[0, :, :]
    
    # turn off surface forcing after 5 years
    b_c = np.zeros(ocnmask.shape) # make an array of zeros the size of the grid
    if t <= 5:
        b_c[0, :, :] = 1/(30/365.25) * (c_anomaly_atm - c_anomaly_surf) # create boundary condition/forcing for top model layer
    b_c = b_c[ocnmask == 1].flatten(order='F') # reshape b vector
    
    c_anomaly = spsolve(eye(len(b_c)) - TR, c_anomaly + b_c) 
    
    new_c_anomaly_3D = np.full(ocnmask.shape, np.nan)
    new_c_anomaly_3D[ocnmask == 1] = np.reshape(c_anomaly, (-1,), order='F')
    c_anomalies[t, :, :, :] = new_c_anomaly_3D
    
#%% save model output   
filename = '/Users/Reese_1/Documents/Research Projects/project2/outputs/exp00_2024-12-12-a.nc'
global_attrs = {'description':'exp00: conservative test tracer (could be conservative temperature) moving from patch in ocean. Patch is forced from t = 1 through t = 5, after that, boundary condition is zero. Forcing applied is c_anomaly_atm[0, 90:100, 30:40] = 1; b_c[0, :, :] = 1/(30/365.25) * (c_anomaly_atm - c_anomaly_surf)'}

p2.save_model_output(filename, model_depth, model_lon, model_lat, np.array(range(0, num_years)), c_anomalies, tracer_names='tracer_concentration', tracer_units=None, global_attrs=global_attrs)

#%% open and plot model output
print(xr.open_dataset(filename))

for t in range(0, num_years):
    p2.plot_surface3d(model_lon, model_lat, c_anomalies[t, :, :, :], 0, 0, 1.5, 'plasma', 'surface potential temperature anomaly at t=' + str(t))
    
for t in range(0, num_years):
    p2.plot_longitude3d(model_lat, model_depth, c_anomalies[t, :, :, :], 100, 0, 1.5, 'plasma', 'potential temperature anomaly at t=' +str(t) + ' along 201ºE longitude')
    