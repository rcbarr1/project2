#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:53:54 2024

@author: Reese Barrett

Translating DeVries OCIM1 example code from MATLAB to python to teach myself
how to work with this model
https://tdevries.eri.ucsb.edu/models-and-data-products/

"""

import numpy as np
import devries_example as dv_p2
import matplotlib.pyplot as plt
import cmocean.cm as cmo

#%% load in control OCIM1 file (downloaded from website above, 8-8-2024)
CTL = dv_p2.loadmat('./CTL.mat')

#%% extract data from mat file to more usable variables
output = CTL['output']

# 3D mask of ocean and land, 1 = ocean, 0 = land
M3d = np.array(output['M3d'])

# "grid metrics", not sure what that means exactly yet
grid = output['grid']
grid = {key: np.array(value) for key, value in grid.items()} # convert dict values from lists to np arrays

# masks for each ocean basin
MSKS = output['MSKS']
MSKS = {key: np.array(value) for key, value in MSKS.items()} # convert dict values from lists to np arrays

# get number of ocean boxes
m = len(M3d[M3d==1])

# transport operator [yr^-1], not sure what this means yet either
TR = output['TR'] # this is a weird datatype, scipy compressed sparse column matrix
TR = TR.astype(np.cfloat)
#%% example 1: set surface boundary condition to track waters ventilated in S. Ocean (S of -65º)
# REG = 1 everywhere S of -65º, REG = 0 elsewhere

# define region
REG = 0 * M3d
REG[grid['YT3d'] <=-65] = 1

# "solve for ventilation fractions" -> not entirely sure what this does? I think TR is the transport matrix, and we are moving one time step using it?
F = dv_p2.eq_wmfrac(TR, REG, M3d, grid)

# "zonally average for Atlantic and Pacific basins"
vol = grid['DXT3d'] * grid['DYT3d'] * grid['DZT3d'] * M3d # not sure what DXT, etc. is, but i think vol is "volume"
Fatl = F * MSKS['ATL'] * vol # atlantic only
Fatl = np.nansum(Fatl, 1) / np.nansum(MSKS['ATL']*vol, 1) # average over all longitudes (get zonal means)
Fpac = F * MSKS['PAC'] * vol # pacific only
Fpac = np.nansum(Fpac, 1) / np.nansum(MSKS['PAC']*vol, 1) # average over all longitudes (get zonal means)

# plot
fig = plt.figure(figsize=(8,8))
ax = fig.gca()
cmap = cmo.dense
plt.contourf(grid['yt'], grid['zt'], Fpac.T, levels=np.arange(0.0, 1.05, 0.05),cmap=cmap)
ax.set_title('Fraction of AABW in Pacific Ocean')
ax.set_xlabel('Latitude (ºN)')
ax.set_ylabel('Depth (m)')
plt.colorbar()
ax.invert_yaxis()

#%% calculate water age
age, _, _, _ = dv_p2.eqage(TR, grid, M3d) # age in years
age = np.real(age)

# zonally average for Atlantic and Pacific basins
ageatl = age * MSKS['ATL'] * vol
ageatl = np.nansum(ageatl, axis=1) / np.nansum(MSKS['ATL'] * vol, axis = 1)
agepac = age * MSKS['PAC'] * vol
agepac = np.nansum(agepac, axis=1) / np.nansum(MSKS['PAC'] * vol, axis = 1)

#%% plot
fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.invert_yaxis()
cmap = cmo.tempo
plt.contourf(grid['yt'], grid['zt'], agepac.T, levels=np.arange(0, 1500, 100),cmap=cmap)
ax.set_title('Zonally Averaged Age of Waters in Pacific Ocean')
ax.set_xlabel('Latitude (ºN)')
ax.set_ylabel('Depth (m)')
plt.colorbar(label='Age (Years)')




