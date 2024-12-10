#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

Created on Mon Oct  7 13:55:39 2024

@author: Reese Barrett
"""

import project2 as p2
import xarray as xr
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

model_path = '/Users/Reese_1/Documents/Research Projects/project2/OCIM2_48L_base/'
glodap_path = '/Users/Reese_1/Documents/Research Projects/project2/GLODAPv2.2016b.MappedProduct/'
woa_path = '/Users/Reese_1/Documents/Research Projects/project2/woa18/'
#%% load transport matrix (OCIM-48L, from Holzer et al., 2021)
# transport matrix is referred to as "A" vector in John et al., 2020 (AWESOME OCIM)
TR = p2.loadmat(model_path + 'OCIM2_48L_base_transport.mat')
TR = TR['TR']

# open up rest of data associated with transport matrix
model_data = xr.open_dataset(model_path + 'OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].to_numpy()

model_depth = model_data['tz'].to_numpy()[:, 0, 0] # m below sea surface
model_lon = model_data['tlon'].to_numpy()[0, :, 0] # ºE
model_lat = model_data['tlat'].to_numpy()[0, 0, :] # ºN

#%% load and regrid GLODAP data (https://glodap.info/index.php/mapped-data-product/)
#DIC_data = xr.open_dataset(glodap_path + 'GLODAPv2.2016b.TCO2.nc')
#TA_data = xr.open_dataset(glodap_path + 'GLODAPv2.2016b.TAlk.nc')

# pull out arrays of depth, latitude, and longitude from GLODAP
#glodap_depth = DIC_data['Depth'].to_numpy() # m below sea surface
#glodap_lon = DIC_data['lon'].to_numpy()     # ºE
#glodap_lat = DIC_data['lat'].to_numpy()     # ºN

# pull out values of DIC and TA from GLODAP
#DIC = DIC_data['TCO2'].values
#TA = TA_data['TAlk'].values

# switch order of GLODAP dimensions to match OCIM dimensions
#DIC = np.transpose(DIC, (0, 2, 1))
#TA = np.transpose(TA, (0, 2, 1))

# plot surface & longitude transect straight from glodap
#p2.plot_surface3d(glodap_lon, glodap_lat, DIC, 0, 960, 2400, 'plasma', 'glodap DIC surface distribution')
#p2.plot_surface3d(glodap_lon, glodap_lat, TA, 0, 1040, 2640, 'viridis', 'glodap DIC surface distribution')
#p2.plot_longitude3d(glodap_lat, glodap_depth, DIC, 320, 1920, 2280, 'plasma', 'glodap DIC distribution along 340.5ºE longitude')
#p2.plot_longitude3d(glodap_lat, glodap_depth, TA, 320, 2080, 2460, 'viridis', 'glodap TA distribution along 340.5ºE longitude')

#DIC = p2.regrid_glodap(DIC, glodap_depth, glodap_lat, glodap_lon, model_depth, model_lat, model_lon, ocnmask)
#TA = p2.regrid_glodap(TA, glodap_depth, glodap_lat, glodap_lon, model_depth, model_lat, model_lon, ocnmask)

# save regridded data
#np.save(glodap_path + 'DIC_AO.npy', DIC)
#np.save(glodap_path + 'TA_AO.npy', TA)

# upload regridded data
DIC = np.load(glodap_path + 'DIC_AO.npy')
TA = np.load(glodap_path + 'TA_AO.npy')

# visualize regridded data
#p2.plot_surface3d(model_lon, model_lat, DIC, 0, 960, 2400, 'plasma', 'regridded glodap DIC surface distribution')
#p2.plot_surface3d(model_lon, model_lat, TA, 0, 1040, 2640, 'viridis', 'regridded glodap TA surface distribution')
#p2.plot_longitude3d(model_lat, model_depth, DIC, 170, 1920, 2280, 'plasma', 'regridded glodap DIC distribution along 341ºE longitude')
#p2.plot_longitude3d(model_lat, model_depth, TA, 170, 2080, 2460, 'viridis', 'regridded glodap TA distribution along 341ºE longitude')

#%% get tracer distributions (called "e" vectors in John et al., 2020)
# POTENTIAL TEMPERATURE (θ)
# open up .nc dataset included with this model to pull out potential temperature
ptemp = model_data['ptemp'].to_numpy() # potential temperature [ºC]
ptemp = ptemp[ocnmask == 1].flatten(order='F') # flatten only ocean boxes in column-major form ("E" vector format)

# DIC
DIC = DIC[ocnmask == 1].flatten(order='F') # flatten only ocean boxes in column-major form ("E" vector format)

# ALKALINITY
TA = TA[ocnmask == 1].flatten(order='F') # flatten only ocean boxes in column-major form ("E" vector format)

#%% load in data to make "b" vector for each tracer (source/sink vector) --> will need to create vectors fully at each time step because dependent on previous time step

# DATA FOR POTENTIAL TEMPERATURE (θ)
# in top model layer, you have exchange with atmosphere. In other model layers,
# there is no source/sink. This is explained more fully in DeVries, 2014

# atmospheric "restoring" potential temperature comes from World Ocean Atlas 13 temperature data
# don't need to convert potential temperature to temperature because reference point is sea level, so they're equivalent at the surface
# I could only download WOA 18 data, from https://www.ncei.noaa.gov/access/world-ocean-atlas-2018/bin/woa18.pl
woa_data = xr.open_dataset(woa_path + '1_woa18_decav_t00_01.nc', decode_times=False)
woa_data = woa_data.isel(time=0).isel(depth=0)

# transform WOA longitude to be 0 to 360, reorder to increase from 0 to 360
woa_data = woa_data.assign(**{'lon': np.mod(woa_data['lon'], 360)})
woa_data = woa_data.reindex({ 'lon' : np.sort(woa_data['lon'])})

# export to numpy
ptemp_atm = woa_data['t_an'].to_numpy()
woa_lat = woa_data['lat'].to_numpy()
woa_lon = woa_data['lon'].to_numpy()

# regrid surface woa data to be same shape as ptemp_surf
#p2.plot_surface2d(woa_lon, woa_lat, ptemp_atm, 0, -30, 40, 'magma', 'WOA temp surface distribution')
#ptemp_atm = p2.regrid_woa(ptemp_atm.T, woa_lat, woa_lon, model_lat, model_lon, ocnmask[0, :, :])
#p2.plot_surface2d(model_lon, model_lat, ptemp_atm.T, 0, -5, 32, 'magma', 'WOA temp surface distribution')
 
   
# DATA FOR DIC

# DATA FOR ALKALINITY

#%% new attempt:
# okay so what if I make ptemp_atm what it has to be to do restoring? should be able to back-calculate out of system of equations
ptemp_3D = np.full(ocnmask.shape, np.nan)
ptemp_3D[ocnmask == 1] = np.reshape(ptemp, (-1,), order='F')
ptemp_surf = ptemp_3D[0, :, :]

# assuming steady state, de/dt = 0, so A*θ = b = 1/(30/365.25) * (θ_atm - θ)
# each time step is one year? how were they doing one month in DeVries, 2022?
# taking out the 365.25 factor makes the backed out temperature make NO sense
TR_ptemp = TR*ptemp
TR_ptemp_3d = np.full(ocnmask.shape, np.nan)
TR_ptemp_3d[ocnmask == 1] = np.reshape(TR_ptemp, (-1,), order='F')
ptemp_atm = TR_ptemp_3d[0, :, :] / (1/(30/365.25)) + ptemp_surf

p2.plot_surface2d(model_lon, model_lat, ptemp_atm.T, 0, -5, 32, 'magma', 'back-calculated temp surface distribution')

# THIS SORT OF WORKS!! except numerical errors really start to compound after even one time step --> how to account for this?

#%% move time steps
# format of equation is de/dt + A*e = b (or dc/dt + TR*c = s, see DeVries, 2014)
# c = concentration of tracer, s = source/sink term
# therefore, to solve for next time step (t2) from current time step (t1), need to do
#   de/dt = -A*e1 + b
#   e2 = e1 + de/dt
# DeVries 2022 has it backwards for OCIM2-48L 
#   de/dt = A * e - b

# CHECK IF THIS IS LOGICAL ONCE I HAVE BETTER SURFACE TEMP DATA IN
# del_ptemp should be essentially zero --> this should be running at steady state

ptemp_3D = np.full(ocnmask.shape, np.nan)
ptemp_3D[ocnmask == 1] = np.reshape(ptemp, (-1,), order='F')
p2.plot_surface3d(model_lon, model_lat, ptemp_3D, 0, -4, 32, 'plasma', 'surface potential temperature at t=0')
p2.plot_longitude3d(model_lat, model_depth, ptemp_3D, 170, -4, 32, 'plasma', 'potential temperature at t=0 along 341ºE longitude')

for t in range(1, 3):
    print(t)
    # POTENTIAL TEMPERATURE:
    # calculate b (source/sink vector)
    # all of b should be zero, except surface ocean boxes should equal s = k / ∆z1 * (alpha * θ_atm - θ) (see DeVries, 2014)
    # alpha = 1
    # k = ∆z1 * (30 days)^-1 --> I think this means k / ∆z1 = 1/(30 days) = 1/(30/365.25 years)
    
    ptemp_3D = np.full(ocnmask.shape, np.nan)
    ptemp_3D[ocnmask == 1] = np.reshape(ptemp, (-1,), order='F')
    ptemp_surf = ptemp_3D[0, :, :]
    
    b_ptemp = np.zeros(ocnmask.shape) # make an array of zeros the size of the grid
    b_ptemp[0, :, :] = 1/(30/365.25) * (ptemp_atm - ptemp_surf) # create boundary condition/forcing for top model layer
    b_ptemp = b_ptemp[ocnmask == 1].flatten(order='F') # reshape b vector
    
    del_ptemp = TR * ptemp - b_ptemp
    ptemp += del_ptemp
    
    new_ptemp_3D = np.full(ocnmask.shape, np.nan)
    new_ptemp_3D[ocnmask == 1] = np.reshape(ptemp, (-1,), order='F')

    p2.plot_surface3d(model_lon, model_lat, new_ptemp_3D, 0, -4, 32, 'plasma', 'surface potential temperature at t=' + str(t))
    p2.plot_longitude3d(model_lat, model_depth, new_ptemp_3D, 170, -4, 32, 'plasma', 'potential temperature at t=' +str(t) + ' along 341ºE longitude')

#%% new attempt 2: apply a tracer of concentration c µmol kg-1 per year, start with matrix
# of zeros, plot change in temperature anomaly with time
num_years = 15

c_anomaly = np.zeros(ocnmask.shape) # potential temperature [ºC]
c_anomaly = c_anomaly[ocnmask == 1].flatten(order='F') # reshape b vector

c_anomaly_3D = np.full(ocnmask.shape, np.nan)
c_anomaly_3D[ocnmask == 1] = np.reshape(c_anomaly, (-1,), order='F') # reshape e vector

p2.plot_surface3d(model_lon, model_lat, c_anomaly_3D, 0, 0, 1, 'plasma', 'surface potential temperature anomaly at t=0')
p2.plot_longitude3d(model_lat, model_depth, c_anomaly_3D, 100, 0, 1, 'plasma', 'potential temperature anomaly at t=0 along 201ºE longitude')

c_anomaly_atm = np.zeros(ocnmask.shape)
c_anomaly_atm[0, 90:100, 30:40] = 1 # create boundary condition of 0.001 (change in surface forcing of 0.001 kg-1 yr-1)
p2.plot_surface3d(model_lon, model_lat, c_anomaly_atm, 0, 0, 1.5, 'plasma', 'surface forcing')
c_anomaly_atm = c_anomaly_atm[0, :, :]

c_anomaly_3D = np.full(ocnmask.shape, np.nan)
c_anomaly_3D[ocnmask == 1] = np.reshape(c_anomaly, (-1,), order='F')

c_anomalies = [c_anomaly_3D]

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
    c_anomalies.append(new_c_anomaly_3D)

for t in range(0, num_years):
    p2.plot_surface3d(model_lon, model_lat, c_anomalies[t], 0, 0, 1.5, 'plasma', 'surface potential temperature anomaly at t=' + str(t))
    
for t in range(0, num_years):
    p2.plot_longitude3d(model_lat, model_depth, c_anomalies[t], 100, 0, 1.5, 'plasma', 'potential temperature anomaly at t=' +str(t) + ' along 201ºE longitude')
    
    
    
    
    
    
    
    
    