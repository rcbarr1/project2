#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMPLEMENTING DEVRIES EXAMPLE FROM SECTION 10 OF OCIM USER MANUAL IN PYTHON TO
MAKE SURE MY EULER BACKWARDS IS CORRECT
 
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
    exp##_YYYY-MM-DD-a.nc (where expXX corresponds to the python file used to
    run the experiment; a, b, c etc. represent model runs from the same day)

Created on Mon Oct  7 13:55:39 2024

@author: Reese Barrett
"""

import project2 as p2
import xarray as xr
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
import h5netcdf
import datetime as dt

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

m = TR.shape[0]

# load in SST anomaly data
sstanom = np.loadtxt('/Users/Reese_1/Documents/Research Projects/project2/examples/devries/sstanom.txt', skiprows=3, delimiter=',')
tan = sstanom[:,0]
Tan = sstanom[:,1]

#%% partition transport matrix

# get land & sea masks
iocn = np.squeeze(np.argwhere(ocnmask.flatten(order='F'))) # 2D indicies of ocean points
iland = np.setdiff1d(range(0,len(ocnmask.flatten(order='F'))), iocn) # 2D indicies of land points
 
# get indicies of surface & interior points
tmp = np.zeros(ocnmask.shape)
tmp[0,:,:] = 1
tmp = tmp.flatten(order='F') 
iint = np.squeeze(np.argwhere(tmp[iocn]==0)) # interior points
isurf = np.squeeze(np.argwhere(tmp[iocn]==1)) # surface points
nint = len(iint) # number of interior points
nsurf = len(isurf) # number of surface points

# do matrix partitioning
TRss = TR[isurf, :]
TRss = TRss[:,isurf]

TRsi = TR[isurf, :]
TRsi = TRsi[:,iint]

TRis = TR[iint, :]
TRis = TRis[:,isurf]

TRii = TR[iint, :]
TRii = TRii[:,iint]

# initial steady state
n = len(Tan) # number of years
ci = np.zeros([len(iint), n])

# set up matricies for Euler backwards
dt = 1 # 1 year time step
M = eye(nint) - dt * TRii

#%% question 1: temperature anomalies
# time-step using equation (45)
#for i in range (1, n):
for i in range (1, 50):
    print(i)
    cs = np.zeros(len(isurf)) + Tan[i] # uniform temperature anomaly at surface
    ci[:,i] = spsolve(M,ci[:,i - 1] + dt * TRis * cs) # time step with backwards Euler - equation (45)

# horizontal integral operator
#VOL = 

#%% save model output   
filename = '/Users/Reese_1/Documents/Research Projects/project2/outputs/exp00_2024-12-10-a.nc'
with h5netcdf.File(filename, 'w', invalid_netcdf=True) as ncfile:
    # create dimensions
    nc_time = ncfile.dimensions['time'] = num_years
    nc_depth = ncfile.dimensions['depth'] = len(model_depth)
    nc_lon = ncfile.dimensions['lon'] = len(model_lon)
    nc_lat = ncfile.dimensions['lat'] = len(model_lat)
    
    # create variables
    times = ncfile.create_variable('time', ('time',), dtype='f8')
    depths = ncfile.create_variable('depth', ('depth',), dtype='f8')
    lons = ncfile.create_variable('lon', ('lon',), dtype='f8')
    lats = ncfile.create_variable('lat', ('lat',), dtype='f8')
    tracer_conc = ncfile.create_variable('tracer_concentration', ('time', 'depth', 'lon', 'lat'), dtype='f8')
    
    # create units and variable descriptions
    times.attrs['units'] = 'years'
    depths.attrs['units'] = 'meters'
    lons.attrs['units'] = 'degrees_east'
    lats.attrs['units'] = 'degrees_north'
    tracer_conc.attrs['description'] = 'Sample tracer (i.e. conservative temperature) spreading, test case so no meaningful units'
    
    # write data
    depths[:] = model_depth
    lons[:] = model_lon
    lats[:] = model_lat
    times[:] = np.array(range(0, num_years))
    for i in range(0, num_years):
        tracer_conc[0, :, :, :] = c_anomalies[i]
    
    # add global attributes, close dataset
    ncfile.attrs['description'] = 'exp00: conservative test tracer (could be conservative temeprature) moving from patch in ocean. Patch is forced from t = 1 through t = 5, after that, boundary condition is zero. Forcing applied is c_anomaly_atm[0, 90:100, 30:40] = 1; b_c[0, :, :] = 1/(30/365.25) * (c_anomaly_atm - c_anomaly_surf)'
    ncfile.attrs['history'] = 'Created ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ncfile.attrs['source'] = 'Python script project2_main.py (will likely be renamed to exp00_conservative_tracer_test.py)'
    
print(xr.open_dataset(filename))

    