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
                                                                           
*Assume well-mixed atmospheric reservior exchanging CO2 with ocean
                                                                           
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
    exp##__YYYY-MM-DD-a.nc (where expXX corresponds to the python file used to
    run the experiment; a, b, c etc. represent model runs from the same day)

Created on Mon Feb 24 12:04:44 2025

@author: Reese Barrett
"""

import project2 as p2
import xarray as xr
import numpy as np
from scipy.sparse import eye, diags, csc_matrix, bmat
from scipy.sparse.linalg import spsolve
import PyCO2SYS as pyco2
#import time
from tqdm import tqdm 

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
model_vols = model_data['vol'].to_numpy() # m^3

#%% upload (or regrid) glodap data for use as initial conditions for marine carbonate system 

# regrid GLODAP data
#p2.regrid_glodap(data_path, 'TCO2', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'TAlk', model_depth, model_lat, model_lon, ocnmask)

# upload regridded GLODAP data
DIC = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC_AO.npy') # dissolved inorganic carbon [µmol kg-1]
TA = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA_AO.npy')   # total alkalinity [µmol kg-1]

p2.plot_surface2d(model_lon, model_lat, DIC[0, :, :].T, 1500, 2500, 'magma', 'GLODAP DIC distribution')

#%% upload (or regrid) woa18 data for use in CO2 system calculations

# regrid WOA18 data
#p2.regrid_woa(data_path, 'S', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_woa(data_path, 'T', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_woa(data_path, 'Si', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_woa(data_path, 'P', model_depth, model_lat, model_lon, ocnmask)

# upload regridded WOA18 data
S = np.load(data_path + 'WOA18/S_AO.npy')   # salinity [unitless]
T = np.load(data_path + 'WOA18/T_AO.npy')   # temperature [ºC]
Si = np.load(data_path + 'WOA18/Si_AO.npy') # silicate [µmol kg-1]
P = np.load(data_path + 'WOA18/P_AO.npy')   # phosphate [µmol kg-1]

p2.plot_surface2d(model_lon, model_lat, S[0, :, :].T, 25, 38, 'magma', 'WOA salinity surface distribution')
p2.plot_surface2d(model_lon, model_lat, T[0, :, :].T, -10, 35, 'magma', 'WOA temp surface distribution')
p2.plot_surface2d(model_lon, model_lat, Si[0, :, :].T, 0, 30, 'magma', 'WOA silicate surface distribution')
p2.plot_surface2d(model_lon, model_lat, P[0, :, :].T, 0, 2.5, 'magma', 'WOA phosphate surface distribution')

#%% set up air-sea gas exchange (Wanninkhof 2014)

# regrid NCEP/DOE reanalysis II data
#p2.regrid_ncep_noaa(data_path, 'icec', model_lat, model_lon, ocnmask)
#p2.regrid_ncep_noaa(data_path, 'uwnd', model_lat, model_lon, ocnmask)
#p2.regrid_ncep_noaa(data_path, 'sst', model_lat, model_lon, ocnmask)

# upload regridded NCEP/DOE reanalysis II data
f_ice = np.load(data_path + 'NCEP_DOE_Reanalysis_II/icec_AO.npy') # annual mean ice fraction from 0 to 1 in each grid cell
uwnd = np.load(data_path + 'NCEP_DOE_Reanalysis_II/uwnd_AO.npy') # annual mean of forecast of U-wind at 10 m [m/s]
sst = np.load(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst_AO.npy') # annual mean sea surface temperature [ºC]

# mask out land boxes
f_ice = np.where(ocnmask[0, :, :] == 1, f_ice, np.nan)
uwnd = np.where(ocnmask[0, :, :] == 1, uwnd, np.nan)
sst = np.where(ocnmask[0, :, :] == 1, sst, np.nan)

# calculate Schmidt number using Wanninkhof 2014 parameterization
vec_schmidt = np.vectorize(p2.schmidt)
Sc = vec_schmidt('CO2', sst)

# solve for Kw (gas transfer velocity) for each ocean cell
a = 0.251 # from Wanninkhof 2014
Kw = a * uwnd**2 * (Sc/660)**-0.5 # [cm/h] from Yamamoto et al., 2024, adapted from Wanninkhof 2014

p2.plot_surface2d(model_lon, model_lat, Kw.T, 0, 20, 'magma', 'Gas transfer velocity (Kw, cm/hr)')

Kw *= (24*365.25/100) # [m/yr] convert units

p2.plot_surface2d(model_lon, model_lat, uwnd.T, -15, 15, 'seismic', 'U-wind at 10 m (m/s)')
p2.plot_surface2d(model_lon, model_lat, sst.T, -2, 40, 'magma', 'sst (ºC)')

#%% set up linearized CO2 system (Nowicki et al., 2024)

# create "pressure" array by broadcasting depth array
pressure = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[1], ocnmask.shape[2]))

# use CO2sys with GLODAP and WOA data to solve for carbonate system at each grid cell
# do this for only ocean grid cells
co2sys_results = pyco2.sys(par1=TA[ocnmask == 1].flatten(order='F'),
                    par2=DIC[ocnmask == 1].flatten(order='F'),
                    par1_type=1, par2_type=2,
                    salinity=S[ocnmask == 1].flatten(order='F'),
                    temperature=T[ocnmask == 1].flatten(order='F'),
                    pressure=pressure[ocnmask == 1].flatten(order='F'),
                    total_silicate=Si[ocnmask == 1].flatten(order='F'),
                    total_phosphate=P[ocnmask == 1].flatten(order='F'))

# extract key results arrays, make 3D

# pCO2 [µatm]
pCO2 = np.full(ocnmask.shape, np.nan)
pCO2[ocnmask == 1] = np.reshape(co2sys_results['pCO2'], (-1,), order='F')

# aqueous CO2 [µmol kg-1]
aqueous_CO2 = np.full(ocnmask.shape, np.nan)
aqueous_CO2[ocnmask == 1] = np.reshape(co2sys_results['aqueous_CO2'], (-1,), order='F')

# revelle factor [unitless]
R = np.full(ocnmask.shape, np.nan)
R[ocnmask == 1] = np.reshape(co2sys_results['revelle_factor'], (-1,), order='F')

# calculate Nowicki et al. parameters
rho = 1025 # seawater density [kg m-3]
Ma = 1.8e26 # number of micromoles of air in atmosphere
beta = DIC/aqueous_CO2 # [unitless]
K0 = aqueous_CO2/pCO2*rho # [µmol m-3 atm-1], in derivation this is defined in per volume units so used density to get there
del_z1 = model_depth[0] # depth of first layer of model [m]
tau_CO2 = (del_z1 * beta[0, :, :]) / (Kw * R[0, :, :]) # timescale of air-sea CO2 equilibration [yr]

p2.plot_surface2d(model_lon, model_lat, R[0,:,:].T, 8, 18, 'magma', 'Revelle factor (unitless)') # this is correct
p2.plot_surface2d(model_lon, model_lat, DIC[0,:,:].T, 1800, 2300, 'magma', 'DIC (µmol kg-1)') # this is correct
p2.plot_surface2d(model_lon, model_lat, aqueous_CO2[0,:,:].T, 0, 80, 'magma', 'Aqueous CO2 (µmol kg-1)') # I think this is correct

p2.plot_surface2d(model_lon, model_lat, tau_CO2.T, 0, 1.6, 'magma', 'tau_co2 (yr)') # this should be right? comparing to Nowicki 2024 supplemental, the general pattern seems correct, but the gradients here are more dramatic? note: those units are in days, this is in years

#%% setting up full matrix A (transport matrix + fluxes between reservoirs)
# total matrix will have size (m + 1) x (m + 1)

m = np.size(ocnmask[ocnmask == 1]) # total number of ocean grid cells in model
surfmask = ocnmask[0,:,:]
m_surf = np.size(surfmask[surfmask == 1]) # total number of surface ocean grid cells in model

# A_11: top left m x m of full matrix A, operates on ∆DIC to calculate change in ∆DIC with each time step
# A_11 = TR - Q_gas, represents DIC change in each cell due to movement around ocean (TR) + CO2/carbonate chemistry equilibration in surface layer (Q_gas)
# q_gas = math for carbonate chemistry equilibration in surface ocean, (1 - f_ice)/(rho * tau_CO2) for surface ocean and zero elsewhere
q_gas = np.zeros(ocnmask.shape)
q_gas[0, :, :] = (1 - f_ice)/(tau_CO2) # [yr-1]
Q_gas = diags(q_gas[ocnmask == 1].flatten(order='F'), format='csc') # flatten q_gas and create sparse matrix where q_gas is main diagonal
A_11 = TR - Q_gas # [yr-1]

# A_12: top right m x 1 of full matrix A, operates on ∆xCO2 to represent flux of atmospheric CO2 to ocean grid cells
# A_12 = q_atm, surface ocean boxes only
# A_12 = (1 - f_ice) * K0 * beta * P_atm /(rho * tau_CO2), represents air-sea gas exchange from atmosphere to ocean
P_atm = 1 # atmospheric pressure [atm]
A_12 = ((1 - f_ice) * K0 * beta * P_atm /(rho * tau_CO2 * R))[ocnmask == 1].flatten(order='F') # [yr-1]
# create sparse matrix of m x 1 dimensions, store A_12 in surface boxes
A_12 = csc_matrix((A_12[0:m_surf], (range(0,m_surf), np.zeros(m_surf))), shape=(m,1))

# A_21: bottom left 1 x m of full matrix A, operates on ∆DIC to represent flux of oceanic CO2 to atmosphere at each time step
# A_21 = rho * vol * (1 - f_ice) / (Ma * tau_CO2), represents air-sea gas exchange from ocean to atmosphere
A_21 = (rho * model_vols * (1 - f_ice) / (Ma * tau_CO2))[ocnmask == 1].flatten(order='F') # [kg mol-1 yr-1]
# create sparse matrix of 1 x m dimensions, store A_21 in surface boxes
A_21 = csc_matrix((A_21[0:m_surf], (np.zeros(m_surf), range(0,m_surf))), shape=(1,m))

# A_22: bottom right 1 x 1 of full matrix A, operates on ∆xCO2 to represent change of ∆xCO2 with each time step
# A_22 = -(P_atm/Ma) * sum[(rho * vol) / (tau_CO2 * beta)] in surface ocean boxes only
A_22 = -1 * np.nansum((model_vols[0, :, :] * K0[0, :, :] * beta[0, :, :] * P_atm * (1 - f_ice)) / (Ma * tau_CO2 * R[0, :, :])) # [yr-1]
# create sparse matrix of 1 x 1 dimensions to store A_22
A_22 = csc_matrix([[A_22]])

# compose full A matrix by piecing together submatrices
# A = [A_11   A_12]
#     [A_21   A_22]

A = bmat([[A_11, A_12], [A_21, A_22]], format='csc')

#%% set up time-stepping

# set up time domain
dt1 = 1/360 # 1 day
dt2 = 1/12 # 1 month
dt3 = 1 # 1 year
dt4 = 10 # 10 years
dt5 = 100 # 100 years

t1 = np.arange(0, 90/360, dt1) # use a 1 day time step for the first 90 days
t2 = np.arange(90/360, 5, dt2) # use a 1 month time step until the 5th year
t3 = np.arange(5, 100, dt3) # use a 1 year time step until the 100th year
t4 = np.arange(100, 500, dt4) # use a 10 year time step until the 500th year
t5 = np.arange(500, 1000+dt5, dt5) # use a 100 year time step until the 1000th year

t = np.concatenate((t1, t2, t3, t4, t5))

# shorten ts for testing
#t = t[0:7]
t = np.arange(0,400,100)

#%% perform time-stepping

# preallocate arrays
x = np.full((len(t), m+1), np.nan) # [∆DIC, ∆xCO2] at each time step
x[0, :] = 0 # ∆DIC, ∆xCO2 = 0 at time step 0
b = np.zeros((m+1, 1)) # [-∆J_CDRocn, -∆J_CDRatm], adding perturbation below

# add perturbation for first time step
b[-1] = 10/1e6 # perturbation of x ppm, divide by 10^6 to make units correct
#x[0, -1] = 1 # setting xCO2 at t = 0 to 1

# time step using Euler backward
for idx in tqdm(range(1, len(t))):
    #print(idx)
    '''
    if t[idx] <= 90/360: # 1 day time step
        LHS = eye(A.shape[0], format="csc") - dt1 * A
        RHS = x[idx-1,:] + np.squeeze(dt1*b)
   
    elif (t[idx] > 90/360) & (t[idx] <= 5): # 1 month time step
        LHS = eye(A.shape[0], format="csc") - dt2 * A
        RHS = x[idx-1,:] + np.squeeze(dt2*b)
    
    elif (t[idx] > 5) & (t[idx] <= 100): # 1 year time step
        LHS = eye(A.shape[0], format="csc") - dt3 * A
        RHS = x[idx-1,:] + np.squeeze(dt3*b)
   
    elif (t[idx] > 100) & (t[idx] <= 500): # 10 year time step
        LHS = eye(A.shape[0], format="csc") - dt4 * A
        RHS = x[idx-1,:] + np.squeeze(dt4*b)
    
    else: # 100 year time step
        LHS = eye(A.shape[0], format="csc") - dt5 * A
        RHS = x[idx-1,:] + np.squeeze(dt5*b)
    '''
    
    LHS = eye(A.shape[0], format="csc") - dt5 * A
    RHS = x[idx-1,:] + np.squeeze(dt5*b)
    
    #start_time = time.time()
    
    x[idx,:] = spsolve(LHS,RHS) # time step with backwards Euler
    
    #end_time = time.time()
    #print(str(end_time - start_time) + ' s') # elapsed time of solve in seconds
    
    # remove perturbation 
    b[-1] = 0

 #%% rebuild 3D concentrations from 1D array used for solving matrix equation
delDIC = np.full([len(t), ocnmask.shape[0], ocnmask.shape[1], ocnmask.shape[2]], np.nan) # make 3D vector full of nans

for idx in range(0, len(t)):
    x_reshaped = np.full(ocnmask.shape, np.nan)
    x_reshaped[ocnmask == 1] = np.reshape(x[idx, 0:-1], (-1,), order='F')
    
    delDIC[idx, :, :, :] = x_reshaped # save first m outputs as DIC

delxCO2 = x[:, -1] # last output (m + 1) is atmospheric CO2 (xCO2)

#%% save model output in netCDF format

global_attrs = {'description':'continuing testing my derivation instead of copying kana - working on how to validate model. time step of 100 year. flipped signs of A21 and A22 to match kana'}

# Save model output
p2.save_model_output(
    'exp04_2025-4-14-d.nc', 
    t, 
    model_depth, 
    model_lon,
    model_lat, 
    tracers=[delDIC, delxCO2], 
    tracer_dims=[('time', 'depth', 'lon', 'lat'), ('time')],
    tracer_names=['delDIC', 'delxCO2'], 
    tracer_units=['µmol kg-3', 'ppm'],
    global_attrs=global_attrs
)

#%% open and plot model output
data = xr.open_dataset(output_path + 'exp04_2025-4-14-d.nc')

model_time = data.time
model_lon = data.lon.data
model_lat = data.lat.data
model_depth = data.depth.data

#for idx in range(0, len(model_time)):
for idx in range(0, 4):
    print(idx)
    p2.plot_surface3d(model_lon, model_lat, data['delDIC'].isel(time=idx).values, 0, 0, 1e-7, 'plasma', 'surface ∆DIC (µmol kg-1) at t=' + str(t[idx]))
   
#for idx in range(0, len(model_time)):
for idx in range(0, 4):
    p2.plot_longitude3d(model_lat, model_depth, data['delDIC'].isel(time=idx).values, 100, 0, 1e-7, 'plasma', ' ∆DIC (µmol kg-1) at t=' +str(t[idx]) + ' along 201ºE longitude')
    
# test: sum tracer concentration at each time step (starting at t = 6 when addition is over) to see if conserved
# currently it is not conserved!! why!
#for idx in range(0, len(model_time)):
for idx in range(0, 4):
    # multiply (mol kg^-1) * (kg m^-3) * m^3 to see if AMOUNT is conserved
    DIC_amount = data['delDIC'].isel(time=idx) * rho * model_vols
    
    # multiply by 10^6 to change from unitless to ppm (still technically unitless)
    xCO2_amount = data['delxCO2'].isel(time=idx).values * 1e6
    
    # or, multiply by mass of air to get moles of CO2 in atmosphere
    xCO2_amount_mol = data['delxCO2'].isel(time=idx).values * Ma # amount of xCO2 in atmosphere [µmol]
    total_amount = (np.nansum(DIC_amount) + xCO2_amount_mol) * 1e-6 # [mol]
    
    print('t = ' + str(idx) + '\t\t total ∆DIC = {:.2f} mol'.format((np.nansum(DIC_amount) * 1e-6)) + '\t ∆xCO2 = {:.2f} mol'.format(xCO2_amount_mol * 1e-6))    
    
    print('\t\t\t total CO2 = {:.2f} mol\n'.format(total_amount))
data.close()






