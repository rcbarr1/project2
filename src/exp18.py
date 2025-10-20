#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 13:02:11 2025

EXP18: Trying to solve for q_AT using eqn. 40 from OCIM manual. This would be
the amount of alkalinity to add in the very first time step to get back to
preindustrial pH... we are not trying to get back to preindustrial AT.

Eqn. 40:
    (I - dt * TR) * c_t = c_(t-dt) + dt * q(t)
    
solve for q(t):
    q(t) = [(I - dt * TR) * c_t - c_(t-dt)] / dt
    
--> define c_t as modern DIC + whatever AT required to get to preind pH
--> define c_(t-1) as preindustrial DIC + preindustrial AT (which we are saying
    is the same as GLODAP AT)
--> test different dt to see if we get the same q{t}, which is a RATE
    
@author: Reese C. Barrett
"""

import project2 as p2
import xarray as xr
import numpy as np
import PyCO2SYS as pyco2
from scipy import sparse
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

# depth of first model layer (need bottom of grid cell, not middle) [m]
grid_cell_depth = model_data['wz'].to_numpy()
z1 = grid_cell_depth[1, 0, 0]
ns = int(np.nansum(ocnmask[0,:,:])) # number of surface grid cells
rho = 1025 # seawater density for volume to mass [kg m-3]

#%% getting initial ∆DIC conditions from TRACEv1
# note, doing set up with Fortran ordering for consistency

# create list of longitudes (ºE), latitudes (ºN), and depths (m) in TRACE format
# this order is required for TRACE
lon, lat, depth = np.meshgrid(model_lon, model_lat, model_depth, indexing='ij')

# reshape meshgrid points into a list of coordinates to interpolate to
output_coordinates = np.array([lon.ravel(order='F'), lat.ravel(order='F'), depth.ravel(order='F'), ]).T

# create required input of dates
# first simulation year will be 2015 (I think), so do then 
dates_2015 = 2015 * np.ones([output_coordinates.shape[0],1])
#dates_2025 = 2025 * np.ones([output_coordinates.shape[0],1])

# get temperature and salinity predictor measurements from GLODAP data
#p2.regrid_glodap(data_path, 'temperature', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'salinity', model_depth, model_lat, model_lon, ocnmask)

# or, upload regridded glodap data
T_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy')
S_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy')

# transpose to match requirements for TRACEv1
T_3D = T_3D.transpose([1, 2, 0])
S_3D = S_3D.transpose([1, 2, 0])
predictor_measurements = np.vstack([S_3D.flatten(order='F'), T_3D.flatten(order='F')]).T

# combine all into .csv file to export for use with TRACEv1 in MATLAB (on the edge of my seat for pyTRACE clearly)
#trace_data = np.hstack([output_coordinates, dates_2015, predictor_measurements])
#np.savetxt(data_path + 'TRACEv1/trace_inputs_2015.txt', trace_data, delimiter = ',')

# transpose temperature and salinity back
T_3D = T_3D.transpose([2, 0, 1])
S_3D = S_3D.transpose([2, 0, 1])

# load in TRACE data
Canth_2015 = p2.loadmat(data_path + 'TRACEv1/trace_outputs_2015.mat')
Canth_2015 = Canth_2015['trace_outputs_2015']
Canth_2015 = Canth_2015.reshape(len(model_lon), len(model_lat), len(model_depth), order='F')
Canth_2015 = Canth_2015.transpose([2, 0, 1])

#%% calculate preindustrial pH from GLODAP DIC minus Canth to get preindustrial DIC and GLODAP TA, assuming steady state

# regrid GLODAP data
#p2.regrid_glodap(data_path, 'TCO2', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'TAlk', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'pHtsinsitutp', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'temperature', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'salinity', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'silicate', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'PO4', model_depth, model_lat, model_lon, ocnmask)

# upload regridded GLODAP data
DIC_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC.npy') # dissolved inorganic carbon [µmol kg-1]
AT_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA.npy')   # total alkalinity [µmol kg-1]
pH_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/pHtsinsitutp.npy') # pH on total scale at in situ temperature and pressure 
T_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy') # temperature [ºC]
S_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy') # salinity [unitless]
Si_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/silicate.npy') # silicate [µmol kg-1]
P_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/PO4.npy') # phosphate [µmol kg-1]

DIC = p2.flatten(DIC_3D, ocnmask) # this is c_t for DIC
AT = p2.flatten(AT_3D, ocnmask) # this is c_(t-1) for AT
pH = p2.flatten(pH_3D, ocnmask)
T = p2.flatten(T_3D, ocnmask)
S = p2.flatten(S_3D, ocnmask)
Si = p2.flatten(Si_3D, ocnmask)
P = p2.flatten(P_3D, ocnmask)

# calculate preindustrial pH by subtracting anthropogenic carbon
DIC_preind_3D = DIC_3D - Canth_2015
DIC_preind = p2.flatten(DIC_preind_3D, ocnmask) # this is c_(t-1) for DIC

# create "pressure" array by broadcasting depth array
pressure_3D = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[1], ocnmask.shape[2]))
pressure = pressure_3D[ocnmask == 1].flatten(order='F')

# calculate preindustrial pH from DIC in 2015 minus Canth in 2015 AND TA in 2015 (assuming steady state)

# is it okay to use modern-day temperatures for this?? probably not, but not
# sure if there's a TRACE for this and trying to stick with data-based, not
# model-based
# pyCO2SYS v2
co2sys = pyco2.sys(dic=DIC_preind, alkalinity=AT, salinity=S, temperature=T,
                   pressure=pressure, total_silicate=Si, total_phosphate=P)

pH_preind = co2sys['pH']
avg_pH_preind = np.nanmean(pH_preind)

pH_preind_3D = p2.make_3D(pH_preind, ocnmask)

#%% calculate c_t for AT using iterative solve
AT_perturbed = AT + p2.calculate_AT_to_add(pH_preind, DIC, AT, T, S, pressure, Si, P, low=0, high=200, tol=1e-6, maxiter=50)

#%% solve for q_AT
# q(t) = [(I - dt * TR) * c_t - c_(t-dt)] / dt

dt1 = 1/360 # 1 day
dt2 = 1/12 # 1 month
dt3 = 1 # 1 yr
dt4 = 100 # 100 yr

q_dt1 = ((sparse.eye(TR.shape[0], format="csc") - dt1 * TR) * AT_perturbed - AT) / dt1 
q_dt2 = ((sparse.eye(TR.shape[0], format="csc") - dt2 * TR) * AT_perturbed - AT) / dt2 
q_dt3 = ((sparse.eye(TR.shape[0], format="csc") - dt3 * TR) * AT_perturbed - AT) / dt3
q_dt4 = ((sparse.eye(TR.shape[0], format="csc") - dt4 * TR) * AT_perturbed - AT) / dt4

plt.plot(q_dt1)
plt.title('1 day timestep')
plt.xlabel('index')
plt.ylabel('value of q (µmol AT kg-1 yr-1)')
plt.show()

plt.plot(q_dt2)
plt.title('1 month timestep')
plt.xlabel('index')
plt.ylabel('value of q (µmol AT kg-1 yr-1)')
plt.show()

plt.plot(q_dt3)
plt.title('1 year timestep')
plt.xlabel('index')
plt.ylabel('value of q (µmol AT kg-1 yr-1)')
plt.show()

plt.plot(q_dt4)
plt.title('100 year timestep')
plt.xlabel('index')
plt.ylabel('value of q (µmol AT kg-1 yr-1)')
plt.show()


