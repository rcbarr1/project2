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

datapath = '/Users/Reese_1/Documents/Research Projects/project2/OCIM2_48L_base/'

#%% load transport matrix (OCIM-48L, from Holzer et al., 2021
# transport matirx is referred to as "A" vector in John et al., 2020 (AWESOME OCIM)
TR = p2.loadmat(datapath + 'OCIM2_48L_base_transport.mat')
TR = TR['TR']

# open up rest of data associated with transport matrix
data = xr.open_dataset(datapath + 'OCIM2_48L_base_data.nc')
ocnmask = data['ocnmask'].to_numpy()

#%% get tracer distributions (called "e" vectors in John et al., 2020)
# POTENTIAL TEMPERATURE (θ)
# open up .nc dataset included with this model to pull out potential temperature
ptemp = data['ptemp'].to_numpy() # potential temperature [ºC]
ptemp = ptemp[ocnmask == 1].flatten(order='F') # flatten only ocean boxes in column-major form ("E" vector format)

# DIC


# ALKALINITY


#%% create "b" vector for each tracer (source/sink vector) --> will need to repeat at each time step because dependent on previous time step

# POTENTIAL TEMPERATURE (θ)
# in top model layer, you have exchange with atmosphere. In other model layers,
# there is no source/sink. This is explained more fully in DeVries, 2014

# HOWEVER, I DON'T THINK WE NEED A SOURCE/SINK TERM FOR TEMPERATURE UNLESS WE
# ARE MODIFYING IT (I.E. IN "WARMING-LED" EXPERIMENTS) BECAUSE THE ANNUAL
# TEMPERATURES SHOULD BE THE SAME
# I think if we used the exact same dataset and gridding method for atmospheric
# temp data as they did when formulating the model, we would get no net affect
# (i.e. the model should be optimized so the annual average sea surface
# temperature is ~almost~ the same as the annual average bottom layer of
# atmosphere temperature)

# so, the only "s" term we need is if we are perturbing the temperature from steady state!

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