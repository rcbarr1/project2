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
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

#from scipy.interpolate import griddata

model_path = '/Users/Reese_1/Documents/Research Projects/project2/OCIM2_48L_base/'
glodap_path = '/Users/Reese_1/Documents/Research Projects/project2/GLODAPv2.2016b.MappedProduct/'

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
DIC_data = xr.open_dataset(glodap_path + 'GLODAPv2.2016b.TCO2.nc')
TA_data = xr.open_dataset(glodap_path + 'GLODAPv2.2016b.TAlk.nc')

# pull out arrays of depth, latitude, and longitude from GLODAP
glodap_depth = DIC_data['Depth'].to_numpy() # m below sea surface
glodap_lon = DIC_data['lon'].to_numpy()     # ºE
glodap_lat = DIC_data['lat'].to_numpy()     # ºN

# pull out values of DIC and TA from GLODAP
DIC = DIC_data['TCO2'].values
TA = TA_data['TAlk'].values

#%% plot surface distribution
fig = plt.figure(figsize=(10,7))
ax = fig.gca()
cntr = plt.contourf(glodap_lon, glodap_lat, DIC[0, :, :], levels=20, cmap='plasma', vmin=960, vmax=2400)
c = plt.colorbar(cntr, ax=ax)
plt.xlabel('longitude (ºE)')
plt.ylabel('latitude (ºN)')
plt.title('glodap DIC surface distribution')
plt.xlim([20, 380]), plt.ylim([-90,90])

fig = plt.figure(figsize=(10,7))
ax = fig.gca()
cntr = plt.contourf(glodap_lon, glodap_lat, TA[0, :, :], levels=20, cmap='viridis', vmin=1040, vmax=2640)
c = plt.colorbar(cntr, ax=ax)
plt.xlabel('longitude (ºE)')
plt.ylabel('latitude (ºN)')
plt.title('glodap TA surface distribution')
plt.xlim([20, 380]), plt.ylim([-90,90])

# plot distribution along 340.5ºE longitude
fig = plt.figure(figsize=(10,7))
ax = fig.gca()
cntr = plt.contourf(glodap_lat, glodap_depth, DIC[:, :, 320], levels=20, cmap='plasma', vmin=1920, vmax=2280)
c = plt.colorbar(cntr, ax=ax)
ax.invert_yaxis()
plt.xlabel('longitude (ºE)')
plt.ylabel('depth (m)')
plt.title('glodap DIC distribution along 340.5ºE longitude')
plt.xlim([-90, 90]), plt.ylim([5500, 0])

fig = plt.figure(figsize=(10,7))
ax = fig.gca()
cntr = plt.contourf(glodap_lat, glodap_depth, TA[:, :, 320], levels=20, cmap='viridis', vmin=2080, vmax=2460)
c = plt.colorbar(cntr, ax=ax)
ax.invert_yaxis()
plt.xlabel('longitude (ºE)')
plt.ylabel('depth (m)')
plt.title('glodap TA distribution along 340.5ºE longitude')
plt.xlim([-90, 90]), plt.ylim([5500, 0])

#%% switch order of GLODAP dimensions to match OCIM dimensions
DIC = np.transpose(DIC, (0, 2, 1))
TA = np.transpose(TA, (0, 2, 1))

# create interpolator
interpDIC = RegularGridInterpolator((glodap_depth, glodap_lon, glodap_lat), DIC, bounds_error=False, fill_value=None)
interpTA = RegularGridInterpolator((glodap_depth, glodap_lon, glodap_lat), TA, bounds_error=False, fill_value=None)

# transform model_lon for anything < 20 (because GLODAP goes from 20ºE - 380ºE)
model_lon[model_lon < 20] += 360

# create meshgrid for OCIM grid
depth, lon, lat = np.meshgrid(model_depth, model_lon, model_lat, indexing='ij')

# reshape meshgrid points into a list of coordinates to interpolate to
query_points = np.array([depth.ravel(), lon.ravel(), lat.ravel()]).T

# perform interpolation (regrid GLODAP data to match OCIM grid)
DIC = interpDIC(query_points)
TA = interpTA(query_points)

# transform results back to model grid shape
DIC = DIC.reshape(depth.shape)
TA = TA.reshape(depth.shape)

# inpaint nans
TA = p2.inpaint_nans(TA, mask=ocnmask.astype(bool))
DIC = p2.inpaint_nans(DIC, mask=ocnmask.astype(bool))

# transform model_lon and meshgrid back for anything > 360
model_lon[model_lon > 360] -= 360
depth, lon, lat = np.meshgrid(model_depth, model_lon, model_lat, indexing='ij')

#%% visualize regridded glodap distributions
fig = plt.figure(figsize=(10,7))
ax = fig.gca()
cntr = plt.contourf(model_lon, model_lat, DIC[0, :, :].T, levels=20, cmap='plasma', vmin=960, vmax=2400)
c = plt.colorbar(cntr, ax=ax)
plt.xlabel('longitude (ºE)')
plt.ylabel('latitude (ºN)')
plt.title('regridded glodap DIC surface distribution')
plt.xlim([0, 360]), plt.ylim([-90,90])

fig = plt.figure(figsize=(10,7))
ax = fig.gca()
cntr = plt.contourf(model_lon, model_lat, TA[0, :, :].T, levels=20, cmap='viridis', vmin=1040, vmax=2640)
c = plt.colorbar(cntr, ax=ax)
plt.xlabel('longitude (ºE)')
plt.ylabel('latitude (ºN)')
plt.title('regridded glodap TA surface distribution')
plt.xlim([0, 360]), plt.ylim([-90,90])

# plot distribution along 340.5ºE longitude
fig = plt.figure(figsize=(10,7))
ax = fig.gca()
cntr = plt.contourf(model_lat, model_depth, DIC[:, 170, :], levels=20, cmap='plasma', vmin=1920, vmax=2280)
c = plt.colorbar(cntr, ax=ax)
ax.invert_yaxis()
plt.xlabel('longitude (ºE)')
plt.ylabel('depth (m)')
plt.title('regridded glodap DIC distribution along 341ºE longitude')
plt.xlim([-90, 90]), plt.ylim([5500, 0])

fig = plt.figure(figsize=(10,7))
ax = fig.gca()
cntr = plt.contourf(model_lat, model_depth, TA[:, 170, :], levels=20, cmap='viridis', vmin=2080, vmax=2460)
c = plt.colorbar(cntr, ax=ax)
ax.invert_yaxis()
plt.xlabel('longitude (ºE)')
plt.ylabel('depth (m)')
plt.title('regridded glodap TA distribution along 341ºE longitude')
plt.xlim([-90, 90]), plt.ylim([5500, 0])
#%% get tracer distributions (called "e" vectors in John et al., 2020)
# POTENTIAL TEMPERATURE (θ)
# open up .nc dataset included with this model to pull out potential temperature
ptemp = model_data['ptemp'].to_numpy() # potential temperature [ºC]
ptemp = ptemp[ocnmask == 1].flatten(order='F') # flatten only ocean boxes in column-major form ("E" vector format)

# DIC
DIC = DIC[ocnmask == 1].flatten(order='F') # flatten only ocean boxes in column-major form ("E" vector format)

# ALKALINITY
TA = TA[ocnmask == 1].flatten(order='F') # flatten only ocean boxes in column-major form ("E" vector format)

#%% create "b" vector for each tracer (source/sink vector) --> will need to repeat at each time step because dependent on previous time step

# POTENTIAL TEMPERATURE (θ)
# in top model layer, you have exchange with atmosphere. In other model layers,
# there is no source/sink. This is explained more fully in DeVries, 2014

# convert ptemp calculated at previous time step to 3D grid, pull surface ptemps
ptemp_3D = np.full(ocnmask.shape, np.nan)
ptemp_3D[ocnmask == 1] = np.reshape(ptemp, (-1,), order='F')
ptemp_surf = ptemp_3D[0, :, :]

# atmospheric "restoring" potential temperature comes from World Ocean Atlas 13 temperature data
# don't need to convert potential temperature to temperature because reference point is sea level, so they're equivalent at the surface
# this is currently down due to hurricane helene :(( when it comes back up do this
    # going to want to grid potential temperature at surface onto 180 x 91 grid (get ptemp_atm)
#ptemp_atm = np.full(ptemp_surf.shape, np.nan)
ptemp_atm = np.zeros(ptemp_surf.shape)
    
# all should be zero, except surface ocean boxes should equal 30^-1 * (θ_atm - θ)
b_ptemp = np.zeros(ocnmask.shape) # make an array of zeros the size of the grid
b_ptemp[0, :, :] = 1/30 * (ptemp_atm - ptemp_surf) # create boundary condition/forcing for top model layer
b_ptemp = b_ptemp[ocnmask == 1].flatten(order='F') # reshape b vector

# DIC

# ALKALINITY

#%% move one time step
# format of equation is de/dt + A*e = b (or dc/dt + TR*c = s, see DeVries, 2014)
# c = concentration of tracer, s = source/sink term
# therefore, to solve for next time step (t2) from current time step (t1), need to do
#   de/dt = -A*e1 + b
#   e2 = e1 + de/dt

del_ptemp = -1 * TR * ptemp + b_ptemp
new_ptemp = ptemp + del_ptemp

# CHECK IF THIS IS LOGICAL ONCE I HAVE BETTER SURFACE TEMP DATA IN
# del_ptemp should be essentially zero --> this should be running at steady state

