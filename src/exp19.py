#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 15:22:48 2025

EXP19: MLD experiment–trying to find the mixed layer manually. Adding a tracer
to the surface ocean, propagate with a small time step, and then see where it
ends up.

@author: Reese C. Barrett
"""

import project2 as p2
import xarray as xr
import numpy as np
from scipy import sparse
from time import time
from scipy.sparse.linalg import spilu, LinearOperator, lgmres
from tqdm import tqdm
import matplotlib.pyplot as plt

# load model architecture
data_path = '/Users/Reese_1/Documents/Research Projects/project2/data/'
output_path = '/Users/Reese_1/Documents/Research Projects/project2/outputs/'

# load transport matrix (OCIM2-48L, from Holzer et al., 2021)
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

# some other important numbers
grid_cell_depth = model_data['wz'].to_numpy() # depth of model layers (need bottom of grid cell, not middle) [m]
ns = int(np.nansum(ocnmask[0,:,:])) # number of surface grid cells
rho = 1025 # seawater density for volume to mass [kg m-3]

#%% set up time stepping
dt = 1/1000 # small time step
nt = 2
t = np.arange(0,nt*dt,dt)

#%% construct matrix C
# matrix form:
#  dc/dt = A * c + q
#  c = variable(s) of interest
#  A = transport matrix (TR) plus any processes with dependence on c 
#    = source/sink vector (processes not dependent on c)
    
# UNITS NOTE: all xCO2 units are mol CO2 (mol air)-1 all AT units are µmol AT kg-1, all DIC units are µmol DIC kg-1
# see comment at top for more info

# m = # ocean grid cells
# nt = # time steps

m = TR.shape[0]

# c = [ ∆xCO2 ] --> 1 * nt
#     [ ∆DIC  ] --> m * nt
#     [ ∆AT   ] --> m * nt

c = np.zeros((m, nt))
c[0:ns,0] = 1 / p2.flatten(model_vols,ocnmask)[0:ns] # add "1" unit of tracer to each surface grid cell [amount]

# construct initial q vector (it is going to change every iteration)
# q = [ 0                                                                           ] --> 1 * nt, q[0]
#     [ ∆q_CDR,DIC + ∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc      ] --> m * nt, q[1:(m+1)]
#     [ ∆q_CDR,AT + 2 * (∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc) ] --> m * nt, q[(m+1):(2*m+1)]

# which translates to...
# q = [ 0                                      ] --> 1 * nt, q[0]
#     [ ∆q_CDR,DIC + ∆q_diss,DIC - ∆q_prod,DIC ] --> m * nt, q[1:(m+1)]
#     [ ∆q_CDR,AT + ∆q_diss,AT - ∆q_prod,AT    ] --> m * nt, q[(m+1):(2*m+1)]

q = np.zeros((m, nt))

#%% calculate "A" matrix and perform time stepping

# dimensions
# A = [m x m]

A = TR # just transport, no air-sea gas exchange, etc.
    
# perform time stepping using Euler backward
LHS = sparse.eye(A.shape[0], format="csc") - dt * A

# test condition number of matrix
est = sparse.linalg.onenormest(LHS)
print('estimated 1-norm condition number LHS: ' + str(round(est,1)))

start = time()
ilu = spilu(LHS.tocsc(), drop_tol=1e-5, fill_factor=20)
stop = time()
print('ilu calculations: ' + str(stop - start) + ' s\n')

M = LinearOperator(LHS.shape, ilu.solve)

for idx in tqdm(range(1,nt)):
    
    # add starting guess
    c0 = c[:,idx-1]
    
    # calculate right hand side and perform time stepping
    RHS = c[:,idx-1] + np.squeeze(dt * q[:,idx])
    c[:,idx], info = lgmres(LHS, RHS, M=M, x0=c0, rtol = 1e-5, atol=0)
   
    if info != 0:
        if info > 0:
            print(f'did not converge in {info} iterations.')
        else:
            print('illegal input or breakdown')
    
#%% calculate amount of tracer from concentration
amount = np.zeros(c.shape)

for idx in np.arange(0,nt):
    amount[:, idx] = c[:,idx] * p2.flatten(model_vols,ocnmask)

#%% compare where tracer ended up with current mixed layer mask

# to do addition in mixed layer...
# pull mixed layer depth at each lat/lon from OCIM model data, then create mask
# of ocean cells that are at or below the mixed layer depth
mld = model_data.mld.values # [m]
# create 3D mask where for each grid cell, mask is set to 1 if the depth in the
# grid cell depths array is less than the mixed layer depth for that column
# note: this does miss cells where the MLD is close but does not reach the
# depth of the next grid cell below (i.e. MLD = 40 m, grid cell depths are at
# 30 m and 42 m, see lon_idx, lat_idx = 20, 30). I am intentionally leaving
# this for now to ensure what enters the ocean stays mostly within the mixed
# layer, but the code could be changed to a different method if needed.exp2_t

mldmask = (grid_cell_depth < mld[None, :, :]).astype(int)

plt.fill_between(np.arange(0, amount.shape[0]), p2.flatten(mldmask,ocnmask), label='mixed layer')
plt.fill_between(np.arange(0, amount.shape[0]), amount[:,0], label='tracer at t = 0')
plt.fill_between(np.arange(0, amount.shape[0]), amount[:,1], label = 'tracer at t = 1')
#plt.ylim([0,10])
plt.xlim([0,c.shape[0]])
plt.xlabel('index of grid cell')
plt.ylabel('amount of tracer')
plt.legend()
plt.show()

#%% create alternate MLD mask where MLD is where >= 0.1% of amount in surface remains
mldmask_alt_idx1 = [i for i,v in enumerate(amount[:,1]) if v > 0.01]
mldmask_alt1 = np.zeros(amount.shape[0]).astype(int)
mldmask_alt1[mldmask_alt_idx1] = 1

mldmask_alt_idx2 = [i for i,v in enumerate(amount[:,1]) if v > 0.001]
mldmask_alt2 = np.zeros(amount.shape[0]).astype(int)
mldmask_alt2[mldmask_alt_idx2] = 1

#%% save alt MLD masks to test in other code
np.save(data_path + 'mld mask tests/mldmask_built_in.npy', p2.flatten(mldmask,ocnmask))
np.save(data_path + 'mld mask tests/mldmask_alt_1percentthresh.npy', mldmask_alt1)
np.save(data_path + 'mld mask tests/mldmask_alt_point1percentthresh.npy', mldmask_alt2)


#%% plot what mld masks look like
plt.fill_between(np.arange(0, amount.shape[0]), p2.flatten(mldmask,ocnmask), label='mixed layer')
plt.title('mask based on built-in MLD')
plt.show()
plt.fill_between(np.arange(0, amount.shape[0]), mldmask_alt1, label='mixed layer')
plt.title('mask based on 1% threshold')
plt.show()
plt.fill_between(np.arange(0, amount.shape[0]), mldmask_alt2, label='mixed layer')
plt.title('mask based on 0.1% threshold')
plt.show()

#%% plot what mld masks look like spatially

mldmask_3D = p2.make_3D(mldmask_alt1, ocnmask)
plot_mlds = np.nansum(mldmask_3D, axis=0)

p2.plot_surface2d(model_lon, model_lat, plot_mlds, 0, 20, 'viridis', 'how many depth layers deep does 1% threshold mld go')




