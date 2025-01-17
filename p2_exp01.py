#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TESTING PARTITIONING BETWEEN INTERIOR AND SURFACE POINTS (FOLLOWING srun_dac_sim.m AND MANUAL CODE)
 
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
#from scipy.sparse.linalg import spsolve
from scikits.umfpack import spsolve
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import spilu, LinearOperator
import time

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

#%% partition transport matrix

# get land & sea masks
iocn = np.squeeze(np.argwhere(ocnmask.flatten(order='F'))) # 2D indicies of ocean points
iland = np.setdiff1d(range(0,len(ocnmask.flatten(order='F'))), iocn) # 2D indicies of land points
 
# get indicies of surface & interior points
tmp = np.zeros(ocnmask.shape)
tmp[0,:,:] = 1 # set all surface points = 1
tmp = tmp.flatten(order='F') 
iint = np.squeeze(np.argwhere(tmp[iocn]==0)) # interior points
isurf = np.squeeze(np.argwhere(tmp[iocn]==1)) # surface points
nint = len(iint) # number of interior points
nsurf = len(isurf) # number of surface points

# get index of interior ocean point for testing
tmp = np.zeros(ocnmask.shape)
tmp[24,118,38] = 1 # random interior point in middle of pacific ocean (away from possible boundary conditions)
tmp = tmp.flatten(order='F') 
itracer = np.squeeze(np.argwhere(tmp[iocn]==1)) # surface points

# do matrix partitioning
TRss = TR[isurf, :]
TRss = TRss[:,isurf]

TRsi = TR[isurf, :]
TRsi = TRsi[:,iint]

TRis = TR[iint, :]
TRis = TRis[:,isurf]

TRii = TR[iint, :]
TRii = TRii[:,iint]

#%% set surface concentration to 100 at a particular point for first time step,
# then set to 0 for following time steps

# set simulation time
dt = 1 # 1 year time steps
num_years = 10 # start with 3 year simulation

# create initial state
ci = np.zeros([num_years, len(iint)])
cs = np.zeros([num_years, len(isurf)])

c_1d = np.zeros(len(iint) + len(isurf)) # initial c in 1d (for t = 0)
c_1d[itracer] = 100 # interior ocean point source tracer
ci[0, :] = c_1d[iint]
cs[0, :] = c_1d[isurf]

#ci[0, 0] = 100 # I don't think this is effectively a point source?

#%% create M = I - dt * TR for interior points (see section 10 of OCIM manual, this is backwards Euler)

#A = eye(nint) - dt * TRii # csr format
A = eye(nint, format="csc") - dt * TRii # csc format

# eliminate zeros
#A.eliminate_zeros()

# use RCM fill-reducing ordering to minimize the fill-in during factorization
#perm = reverse_cuthill_mckee(A)
#inv_perm = np.argsort(perm)
#A = A[perm, :][:, perm]

# LU factorization
#lu = splu(A)

# incomplete LU preconditioning for grmes
print('preconditioning')
start_time = time.time()
ilu = spilu(A)
M = LinearOperator(A.shape, ilu.solve)
end_time = time.time()
print(end_time - start_time) # elapsed time of preconditioning in seconds

# perform time-stepping
for t in range(1,num_years):
    print(t) # see where the simulation is

    b = ci[t - 1,:] + dt * TRis * cs[t - 1,:]   
    #b = b[perm] # make sure b is permutated too when using RCM
    
    # right now I am only solving for interior points and set the surface points as the boundary condition
    start_time = time.time()
    
    # just spsolve
    #ci[t,:] = spsolve(A,b) # time step with backwards Euler 
    
    # for RCM (also see b above), with or without umfpack (depends which package loaded at top)
    #x = spsolve(A,b) # time step with backwards Euler
    #ci[t,:] = x[inv_perm] # reverse the permutation
    
    # for LU factorization
    #ci[t,:] = lu.solve(b)
    
    # for RCM + LU factorizaton (also see b above)
    #x = lu.solve(b) # time step with backwards Euler
    #ci[t,:] = x[inv_perm] # reverse the permutation
    
    # for gmres (iterative solver)
    x, exit_code = gmres(A, b, x0=ci[t-1,:], M=M)
    ci[t,:] = x
    
    end_time = time.time()
    
    # also for gmres
    if exit_code == 0:
        print("solution converged")
    else:
        print("gmres did not converge")
        
    print(end_time - start_time) # elapsed time of solve in seconds

#%% rebuild 3D concentrations from 1D array used for solving matrix equation
c = np.full([num_years, ocnmask.shape[0], ocnmask.shape[1], ocnmask.shape[2]], np.nan) # make 3D vector full of nans

for t in range(0, num_years):
    # put surface points in the right place
    c_1d = np.zeros(len(iint) + len(isurf))
    c_1d[iint] = ci[t, :]
    c_1d[isurf] = cs[t, :]
    
    c_3d = np.full(ocnmask.shape, np.nan)
    c_3d[ocnmask == 1] = np.reshape(c_1d, (-1,), order='F')
    
    c[t, :, :, :] = c_3d
    
    # I think I can maybe understand why this is not conserved? because you're
    # using the surface layer as the boundary condition for solving? but also
    # there are no sources/sinks? need to think about what the boundary conditions
    # are and what the sources/sinks are

#%% save model output in netCDR format
#global_attrs = {'description':'exp01: conservative test tracer moving from point-source in ocean. Attempting to use partitioning to impose boundary conditions to make the tracer actually conserved. Added a test tracer(?) in the middle (mid-depth) of the pacific ocean to try to see if boundary conditions were the problem.'}
global_attrs = {'description':'testing how to speed up spsolve (GMRES with ILU preconditioning)'}

p2.save_model_output(output_path + 'exp01_2025-1-17-d.nc', model_depth, model_lon, model_lat, np.array(range(0, num_years)), [c], tracer_names=['tracer_concentration'], tracer_units=None, global_attrs=global_attrs)

#%% open and plot model output
c_anomalies = xr.open_dataset(output_path + 'exp01_2025-1-17-d.nc')

num_years = len(c_anomalies.time)
model_lon = c_anomalies.lon.data
model_lat = c_anomalies.lat.data
model_depth = c_anomalies.depth.data

for t in range(0, num_years):
    p2.plot_surface3d(model_lon, model_lat, c_anomalies['tracer_concentration'].isel(time=t), 0, 0, 0.1, 'plasma', 'surface tracer concentration anomaly at t=' + str(t))
    
for t in range(0, num_years):
    p2.plot_longitude3d(model_lat, model_depth, c_anomalies['tracer_concentration'].isel(time=t), 100, 0, 0.1, 'plasma', 'tracer concentration anomaly at t=' +str(t) + ' along 201ºE longitude')
    
# test: sum tracer concentration at each time step (starting at t = 6 when addition is over) to see if conserved
# currently it is not conserved!! why!
for i in range(0, num_years):
    print('t = ' + str(i) + '\t c = ' + str(np.nansum(c_anomalies['tracer_concentration'].isel(time=i))))  
    
c_anomalies.close()
