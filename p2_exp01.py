#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TESTING PARTITIONING BETWEEN INTERIOR AND SURFACE POINTS (FOLLOWING srun_dac_sim.m AND MANUAL CODE)
Big question: can biological carbonate compensation act as both a positive and 
negative feedback on climate change (atmospheric pCO2 and/or temperature)?

Hypothesis:
    (+) warming-led/dominated events cause TA increase (omega increase) in
        surface ocean, increased TA export, and decreased CO2 stored in the
        ocean (AMPLIFY atmospheric pCO2)
    (-) CO2-led/dominated events cause TA decrease (omega decrease) in surface
        ocean, decreased TA export, and increased CO2 stored in the ocean 
        (MODERATE) atmospheric CO2
        
To model this: distribution of TA (shown in first project that HCO3- and CO32-
               affect CaCO3 production), CO2, and temperature

- Initial conditions of TA, pCO2, and T based on GLODAP
- Experimental forcings of pCO2 and T based on case studies of paleoclimate
- Experimental forcing of TA based on carbon dioxide removal scenario
- Natural forcings of T are atmospheric/sea surface conditions, natural
  forcings of pCO2 are atmosphere input/output of pCO2 (also base on sea
  surface?), natural forcings of TA are riverine inputs and export/burial (we
  are missing a lot of things here, do any of them matter?)
                                                                           
*Between each time step (where forcings are applied), re-equilibrate carbonate
 chemistry with CO2SYS
 
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

#%% set surface concentration to 100 at a particular point for first time step,
# then set to 0 for following time steps

# set simulation time
dt = 1 # 1 year time steps
num_years = 3 # start with 3 year simulation

# create initial state
ci = np.zeros([num_years, len(iint)])
cs = np.zeros([num_years, len(isurf)])
cs[0, 0] = 100 # I don't think this is effectively a point source?

#%% create M = I - dt * TR for interior points (see section 10 of OCIM manual, this is backwards Euler)
M = eye(nint) - dt * TRii

# perform time-stepping
for t in range(1,num_years):
    print(t) # see where the simulation is
    
    # right now I am only solving for interior points and set the surface points as the boundary condition
    ci[t,:] = spsolve(M,ci[t - 1,:] + dt * TRis * cs[t - 1,:]) # time step with backwards Euler
    

#%% rebuild 3D concentrations
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

#%% save model output   
global_attrs = {'description':'exp01: conservative test tracer (could be conservative temperature) moving from point-source in ocean. Attempting to use partitioning to impose boundary conditions to make the tracer actually conserved'}

p2.save_model_output(output_path + 'exp01_2025-1-9-a.nc', model_depth, model_lon, model_lat, np.array(range(0, num_years)), [c], tracer_names=['tracer_concentration'], tracer_units=None, global_attrs=global_attrs)

#%% open and plot model output
c_anomalies = xr.open_dataset(output_path + 'exp01_2025-1-9-a.nc')

for t in range(0, num_years):
    p2.plot_surface3d(model_lon, model_lat, c_anomalies['tracer_concentration'].isel(time=t), 0, 0, 0.1, 'plasma', 'surface tracer concentration anomaly at t=' + str(t))
    
for t in range(0, num_years):
    p2.plot_longitude3d(model_lat, model_depth, c_anomalies['tracer_concentration'].isel(time=t), 100, 0, 0.1, 'plasma', 'tracer concentration anomaly at t=' +str(t) + ' along 201ºE longitude')
    
# test: sum tracer concentration at each time step (starting at t = 6 when addition is over) to see if conserved
# currently it is not conserved!! why!
for i in range(0, num_years):
    print('t = ' + str(i) + '\t c = ' + str(np.nansum(c_anomalies['tracer_concentration'].isel(time=i))))  
    
c_anomalies.close()