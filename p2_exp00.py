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

#%% apply a tracer of concentration c µmol kg-1 per year, start with matrix
# of zeros, plot change in temperature anomaly with time
# using notation in OCIM user manual from Kana
num_years = 3
delt = 1 # time step of simulation [years]

# c = tracer concentration
c = np.zeros(ocnmask.shape) # start with concentration = 0 everywhere
c = c[ocnmask == 1].flatten(order='F') # reshape c vector to be flat, only include ocean boxes
c[0] = 100

c_3D = np.full(ocnmask.shape, np.nan) # make 3D vector full of nans
c_3D[ocnmask == 1] = np.reshape(c, (-1,), order='F') # reshape flat c vector into 3D vector with nans in land boxes

# c_sat = saturation (atmospheric) concentration of c --> we are going to simulate air-sea gas exchange
# c_sat = 1 # create boundary condition of 1 (change in surface forcing of 1 yr-1). this is assuming a well-mixed atmosphere with tracer concentration = 1 throughout
#p2.plot_surface3d(model_lon, model_lat, c_sat, 0, 0, 1.5, 'plasma', 'surface forcing')

# save c at each time step in this array
c_out = np.zeros([num_years, len(model_depth), len(model_lon), len(model_lat)])
c_out[0, :, :, :] = c_3D

for t in range(1, num_years):
    print(t)
    # assuming constant b here, don't need to recalculate in this loop
    #c_surf = c_out[t - 1, 0 , :, :]
    
    # q = source/sink vector (using air-sea gas sexchange parameterization)
    q = np.zeros(ocnmask.shape) # make an array of zeros the size of the grid
    #if t <= 1:
    #    q[0, 90:100, 30:40] = 1 # simple source/sink = 1 in this box
    q = q[ocnmask == 1].flatten(order='F') # reshape b vector
    
    c = spsolve((eye(len(q)) - TR), (c + q)) 
    
    c_3D = np.full(ocnmask.shape, np.nan)
    c_3D[ocnmask == 1] = np.reshape(c, (-1,), order='F')
    c_out[t, :, :, :] = c_3D

# test: sum tracer concentration at each time step (starting at t = 6 when addition is over) to see if conserved
# currently it is not conserved!! why!
for i in range(0, num_years):
    print('t = ' + str(i) + '\t c = ' + str(np.nansum(c_out[i,:,:,:])))    
    
#%% save model output   
global_attrs = {'description':'exp00: conservative test tracer (could be conservative temperature) moving from point-source in ocean. Except it should be conserved at each time step and is not.'}

p2.save_model_output(output_path + 'exp00_2025-1-6-c.nc', model_depth, model_lon, model_lat, np.array(range(0, num_years)), [c_out], tracer_names=['tracer_concentration'], tracer_units=None, global_attrs=global_attrs)

#%% open and plot model output
c_anomalies = xr.open_dataset(output_path + 'exp00_2025-1-6-b.nc')

for t in range(0, num_years):
    p2.plot_surface3d(model_lon, model_lat, c_anomalies['tracer_concentration'].isel(time=t), 0, 0, 1.5, 'plasma', 'surface tracer concentration anomaly at t=' + str(t))
    
for t in range(0, num_years):
    p2.plot_longitude3d(model_lat, model_depth, c_anomalies['tracer_concentration'].isel(time=t), 100, 0, 1.5, 'plasma', 'tracer concentration anomaly at t=' +str(t) + ' along 201ºE longitude')
    
