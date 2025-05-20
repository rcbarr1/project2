#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Another (less small) diversion: trying to back out a 3-D gridded flux from OCIM + ESPERs 

Created on Fri May  2 12:42:56 2025

@author: Reese Barrett
"""
import PyESPER.PyESPER as PyESPER
import numpy as np
import project2 as p2
import xarray as xr
import gsw
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import rcParams

data_path = '/Users/Reese_1/Documents/Research Projects/project2/data/'
output_path = '/Users/Reese_1/Documents/Research Projects/project2/outputs/'
main_path = '/Users/Reese_1/Documents/Research Projects/project2'

rcParams['font.family'] = 'Avenir'

#%% load transport matrix (OCIM2-48L, from Holzer et al., 2021)
# transport matrix is referred to as "A" vector in John et al., 2020 (AWESOME OCIM)
TR = p2.loadmat(data_path + 'OCIM2_48L_base/OCIM2_48L_base_transport.mat')
TR = TR['TR']

# open up rest of data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].to_numpy()

model_depth = model_data['tz'].to_numpy() # m below sea surface
model_lon = model_data['tlon'].to_numpy() # ºE
model_lat = model_data['tlat'].to_numpy() # ºN
model_vols = model_data['vol'].to_numpy() # m^3
model_areas = model_data['area'].to_numpy() # m^2
model_thickness = model_vols / model_areas # m

# pull temperature and salinity data from model
ptemp0 = model_data.ptemp.to_numpy() # potential temperature [ºC]
sal0 = model_data.salt.to_numpy() # salinity [psu]

# convert potential temperature to temperature
pres0 = gsw.p_from_z(model_depth*-1, model_lat) # convert depth [m, positive up] to pressure [dbar] using latitude [-90º to +90º]
asal0 = gsw.SA_from_SP(sal0, pres0, model_lon, model_lat)  # convert practical salinity [unitless] to absolute salinity [g/kg] with pressure [dbar], latitude [-90º to +90º], and longitude [-360º to 360º]
ctemp0 = gsw.CT_from_pt(asal0, ptemp0) # convert potential temperature [ºC] to conservative temperature [ºC] using absolute salinity [g/kg]
temp0 = gsw.t_from_CT(asal0, ctemp0, pres0) # convert conservative temperature [ºC[ to in-situ temperature [ºC] using absolute salinity [g/kg] and pressure [dbar] (GSW python toolbox does not have direct conversion)

# calculate density at each grid cell
model_rho = gsw.density.rho_t_exact(asal0, temp0, pres0)

#%% step temperature and salinity forward in time to look for numerical artifacts
dt = 1 # 1 year
LHS = eye(TR.shape[0], format="csc") - dt * TR

T0 = ptemp0[ocnmask == 1].flatten(order='F')
S0 = sal0[ocnmask == 1].flatten(order='F')

RHS_T = T0 + dt
RHS_S = S0 + dt

T1 = spsolve(LHS, RHS_T)
S1 = spsolve(LHS, RHS_S)

# turn T1, S1 into 3D arrays and convert to format needed for espers
ptemp1 = np.full(ocnmask.shape, np.nan)
ptemp1[ocnmask == 1] = np.reshape(T1, (-1,), order='F')

sal1 = np.full(ocnmask.shape, np.nan)
sal1[ocnmask == 1] = np.reshape(S1, (-1,), order='F')

pres1 = gsw.p_from_z(model_depth*-1, model_lat) # convert depth [m, positive up] to pressure [dbar] using latitude [-90º to +90º]
asal1 = gsw.SA_from_SP(sal1, pres1, model_lon, model_lat)  # convert practical salinity [unitless] to absolute salinity [g/kg] with pressure [dbar], latitude [-90º to +90º], and longitude [-360º to 360º]
ctemp1 = gsw.CT_from_pt(asal1, ptemp1) # convert potential temperature [ºC] to conservative temperature [ºC] using absolute salinity [g/kg]
temp1 = gsw.t_from_CT(asal1, ctemp1, pres1) # convert conservative temperature [ºC[ to in-situ temperature [ºC] using absolute salinity [g/kg] and pressure [dbar] (GSW python toolbox does not have direct conversion)

# plot T0, T1
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], ptemp0, 0, -5, 30, 'plasma', 'Potential Temp t = 0 (ºC)')
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], ptemp1, 0, -5, 30, 'plasma', 'Potential Temp t = 1 (ºC)')

# plot S0, S1
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], sal0, 0, 27, 40, 'plasma', 'Salinity t = 0 (PSU)')
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], sal1, 0, 27, 40, 'plasma', 'Salinity t = 1 (PSU)')

#%% set up pyESPERs with model temperature and salinity to estimate alkalinity

# flatten all arrays and turn into lists
longitude = model_lon[ocnmask == 1].flatten(order='F').tolist() # ºE
latitude = model_lat[ocnmask == 1].flatten(order='F').tolist() # ºN
depth = model_depth[ocnmask == 1].flatten(order='F').tolist() # [m, positive down]

salinity0 = sal0[ocnmask == 1].flatten(order='F').tolist() # [unitless]
temperature0 = temp0[ocnmask == 1].flatten(order='F').tolist() # [ºC]

salinity1 = sal1[ocnmask == 1].flatten(order='F').tolist() # [unitless]
temperature1 = temp1[ocnmask == 1].flatten(order='F').tolist() # [ºC]

# create dictionary for output coordinates and predictor measurements
output_coordinates = {}
predictor_measurements0 = {}
predictor_measurements1 = {}

output_coordinates.update({'longitude' : longitude,
                          'latitude' : latitude,
                          'depth' : depth})

predictor_measurements0.update({'salinity' : salinity0,
                              'temperature' : temperature0})

predictor_measurements1.update({'salinity' : salinity1,
                              'temperature' : temperature1})

#%% call NN for total alkalinity [µmol kg-1] at T0
TA_NN, _= PyESPER.PyESPER_NN(['TA'], main_path, output_coordinates, predictor_measurements0)

# reformat and save data
TA_NN = TA_NN.iloc[:,15].to_numpy() # pull out ESPER prediction of TA, convert to numpy array
TA_NN_3D = np.full(ocnmask.shape, np.nan)
TA_NN_3D[ocnmask == 1] = np.reshape(TA_NN, (-1,), order='F') # rebuild 3D array
np.save(str(data_path) + 'TA_ESPER_NN_3D.npy', TA_NN_3D) # export data

# call LIR for total alkalnity [µmol kg-1] at T0
TA_LIR, _, _ = PyESPER.PyESPER_LIR(['TA'], main_path, output_coordinates, predictor_measurements0)

# reformat and save data
TA_LIR = TA_LIR.iloc[:,15].to_numpy() # pull out ESPER prediction of TA, convert to numpy array
TA_LIR_3D = np.full(ocnmask.shape, np.nan)
TA_LIR_3D[ocnmask == 1] = np.reshape(TA_LIR, (-1,), order='F') # rebuild 3D array
np.save(str(data_path) + 'TA_ESPER_LIR_3D.npy', TA_LIR_3D) # export data

#%% call NN for nitrate [µmol kg-1] at T0
NO3_NN, _= PyESPER.PyESPER_NN(['nitrate'], main_path, output_coordinates, predictor_measurements0)

# reformat and save data
NO3_NN = NO3_NN.iloc[:,15].to_numpy() # pull out ESPER prediction of NO3, convert to numpy array
NO3_NN_3D = np.full(ocnmask.shape, np.nan)
NO3_NN_3D[ocnmask == 1] = np.reshape(NO3_NN, (-1,), order='F') # rebuild 3D array
np.save(str(data_path) + 'NO3_ESPER_NN_3D.npy', NO3_NN_3D) # export data

# call LIR for nitrate  [µmol kg-1] at T0
NO3_LIR, _, _ = PyESPER.PyESPER_LIR(['nitrate'], main_path, output_coordinates, predictor_measurements0)

# reformat and save data
NO3_LIR = NO3_LIR.iloc[:,15].to_numpy() # pull out ESPER prediction of TA, convert to numpy array
NO3_LIR_3D = np.full(ocnmask.shape, np.nan)
NO3_LIR_3D[ocnmask == 1] = np.reshape(NO3_LIR, (-1,), order='F') # rebuild 3D array
np.save(str(data_path) + 'NO3_ESPER_LIR_3D.npy', NO3_LIR_3D) # export data

#%% call NN for DIC [µmol kg-1] at T0
DIC_NN, _= PyESPER.PyESPER_NN(['DIC'], main_path, output_coordinates, predictor_measurements0)

# reformat and save data
DIC_NN = DIC_NN.iloc[:,15].to_numpy() # pull out ESPER prediction of NO3, convert to numpy array
DIC_NN_3D = np.full(ocnmask.shape, np.nan)
DIC_NN_3D[ocnmask == 1] = np.reshape(DIC_NN, (-1,), order='F') # rebuild 3D array
np.save(str(data_path) + 'DIC_ESPER_NN_3D.npy', DIC_NN_3D) # export data

# call LIR for DIC [µmol kg-1] at T0
DIC_LIR, _, _ = PyESPER.PyESPER_LIR(['DIC'], main_path, output_coordinates, predictor_measurements0)

# reformat and save data
DIC_LIR = DIC_LIR.iloc[:,15].to_numpy() # pull out ESPER prediction of TA, convert to numpy array
DIC_LIR_3D = np.full(ocnmask.shape, np.nan)
DIC_LIR_3D[ocnmask == 1] = np.reshape(DIC_LIR, (-1,), order='F') # rebuild 3D array
np.save(str(data_path) + 'DIC_ESPER_LIR_3D.npy', DIC_LIR_3D) # export data

#%% open saved TA, nitrate, DIC files, flatten at T0
TA_NN_3D = np.load(str(data_path) + 'TA_ESPER_NN_3D.npy')
TA_NN = TA_NN_3D[ocnmask == 1].flatten(order='F')

NO3_NN_3D = np.load(str(data_path) + 'NO3_ESPER_NN_3D.npy')
NO3_NN = NO3_NN_3D[ocnmask == 1].flatten(order='F')

DIC_NN_3D = np.load(str(data_path) + 'DIC_ESPER_NN_3D.npy')
DIC_NN = DIC_NN_3D[ocnmask == 1].flatten(order='F')

TA_LIR_3D = np.load(str(data_path) + 'TA_ESPER_LIR_3D.npy')
TA_LIR = TA_LIR_3D[ocnmask == 1].flatten(order='F')

NO3_LIR_3D = np.load(str(data_path) + 'NO3_ESPER_LIR_3D.npy')
NO3_LIR = NO3_LIR_3D[ocnmask == 1].flatten(order='F')

DIC_LIR_3D = np.load(str(data_path) + 'DIC_ESPER_LIR_3D.npy')
DIC_LIR = DIC_LIR_3D[ocnmask == 1].flatten(order='F')

#%% call NN for total alkalinity [µmol kg-1] at T1
TA_NN_1, _= PyESPER.PyESPER_NN(['TA'], main_path, output_coordinates, predictor_measurements1)

# reformat and save data
TA_NN_1 = TA_NN_1.iloc[:,15].to_numpy() # pull out ESPER prediction of TA, convert to numpy array
TA_NN_1_3D = np.full(ocnmask.shape, np.nan)
TA_NN_1_3D[ocnmask == 1] = np.reshape(TA_NN_1, (-1,), order='F') # rebuild 3D array
np.save(str(data_path) + 'TA_ESPER_NN_1_3D.npy', TA_NN_1_3D) # export data

# call LIR for total alkalnity [µmol kg-1] at T1
TA_LIR_1, _, _ = PyESPER.PyESPER_LIR(['TA'], main_path, output_coordinates, predictor_measurements1)

# reformat and save data
TA_LIR_1 = TA_LIR_1.iloc[:,15].to_numpy() # pull out ESPER prediction of TA, convert to numpy array
TA_LIR_1_3D = np.full(ocnmask.shape, np.nan)
TA_LIR_1_3D[ocnmask == 1] = np.reshape(TA_LIR_1, (-1,), order='F') # rebuild 3D array
np.save(str(data_path) + 'TA_ESPER_LIR_1_3D.npy', TA_LIR_1_3D) # export data

#%% call NN for nitrate [µmol kg-1] at T1
NO3_NN_1, _= PyESPER.PyESPER_NN(['nitrate'], main_path, output_coordinates, predictor_measurements1)

# reformat and save data
NO3_NN_1 = NO3_NN_1.iloc[:,15].to_numpy() # pull out ESPER prediction of NO3, convert to numpy array
NO3_NN_1_3D = np.full(ocnmask.shape, np.nan)
NO3_NN_1_3D[ocnmask == 1] = np.reshape(NO3_NN_1, (-1,), order='F') # rebuild 3D array
np.save(str(data_path) + 'NO3_ESPER_NN_1_3D.npy', NO3_NN_1_3D) # export data

# call LIR for nitrate  [µmol kg-1] at T1
NO3_LIR_1, _, _ = PyESPER.PyESPER_LIR(['nitrate'], main_path, output_coordinates, predictor_measurements1)

# reformat and save data
NO3_LIR_1 = NO3_LIR_1.iloc[:,15].to_numpy() # pull out ESPER prediction of TA, convert to numpy array
NO3_LIR_1_3D = np.full(ocnmask.shape, np.nan)
NO3_LIR_1_3D[ocnmask == 1] = np.reshape(NO3_LIR_1, (-1,), order='F') # rebuild 3D array
np.save(str(data_path) + 'NO3_ESPER_LIR_1_3D.npy', NO3_LIR_1_3D) # export data

#%% call NN for DIC [µmol kg-1] at T1
DIC_NN_1, _= PyESPER.PyESPER_NN(['DIC'], main_path, output_coordinates, predictor_measurements1)

# reformat and save data
DIC_NN_1 = DIC_NN_1.iloc[:,15].to_numpy() # pull out ESPER prediction of NO3, convert to numpy array
DIC_NN_1_3D = np.full(ocnmask.shape, np.nan)
DIC_NN_1_3D[ocnmask == 1] = np.reshape(DIC_NN_1, (-1,), order='F') # rebuild 3D array
np.save(str(data_path) + 'DIC_ESPER_NN_1_3D.npy', DIC_NN_1_3D) # export data

# call LIR for DIC [µmol kg-1] at T1
DIC_LIR_1, _, _ = PyESPER.PyESPER_LIR(['DIC'], main_path, output_coordinates, predictor_measurements1)

# reformat and save data
DIC_LIR_1 = DIC_LIR_1.iloc[:,15].to_numpy() # pull out ESPER prediction of TA, convert to numpy array
DIC_LIR_1_3D = np.full(ocnmask.shape, np.nan)
DIC_LIR_1_3D[ocnmask == 1] = np.reshape(DIC_LIR_1, (-1,), order='F') # rebuild 3D array
np.save(str(data_path) + 'DIC_ESPER_LIR_1_3D.npy', DIC_LIR_1_3D) # export data

#%% open saved TA, nitrate, DIC files, flatten at T1
TA_NN_1_3D = np.load(str(data_path) + 'TA_ESPER_NN_1_3D.npy')
TA_NN_1 = TA_NN_1_3D[ocnmask == 1].flatten(order='F')

NO3_NN_1_3D = np.load(str(data_path) + 'NO3_ESPER_NN_1_3D.npy')
NO3_NN_1 = NO3_NN_1_3D[ocnmask == 1].flatten(order='F')

DIC_NN_1_3D = np.load(str(data_path) + 'DIC_ESPER_NN_1_3D.npy')
DIC_NN_1 = DIC_NN_1_3D[ocnmask == 1].flatten(order='F')

TA_LIR_1_3D = np.load(str(data_path) + 'TA_ESPER_LIR_1_3D.npy')
TA_LIR_1 = TA_LIR_1_3D[ocnmask == 1].flatten(order='F')

NO3_LIR_1_3D = np.load(str(data_path) + 'NO3_ESPER_LIR_1_3D.npy')
NO3_LIR_1 = NO3_LIR_1_3D[ocnmask == 1].flatten(order='F')

DIC_LIR_1_3D = np.load(str(data_path) + 'DIC_ESPER_LIR_1_3D.npy')
DIC_LIR_1 = DIC_LIR_1_3D[ocnmask == 1].flatten(order='F')

#%% plot DIC & AT
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], DIC_NN_3D, 0, 1700, 2500, 'plasma', 'DIC_NN (µmol kg-1)')
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], DIC_LIR_3D, 0, 1700, 2500, 'plasma', 'DIC_LIR (µmol kg-1)')

p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], TA_NN_3D, 0, 1700, 2500, 'plasma', 'TA_NN (µmol kg-1)')
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], TA_LIR_3D, 0, 1700, 2500, 'plasma', 'TA_LIR (µmol kg-1)')

#%% METHOD 0: solving inverse euler for qs assuming c_t = c_(t-1)
dt = 1 # 1 year

# calculate  nitrate fluxes
q_NO3_NN = -1 * TR * NO3_NN # this is the flux vector (flat)
q_NO3_LIR = -1 * TR * NO3_LIR # this is the flux vector (flat)

# test if equal
#LHS = eye(TR.shape[0], format="csc") - dt * TR
#RHS = NO3_NN + dt * q_NO3
#NO3_NN_2 = spsolve(LHS, RHS)
#should_be_basically_zero_flat = NO3_NN_2 - NO3_NN
#should_be_basically_zero_3D = np.full(ocnmask.shape, np.nan)
#should_be_basically_zero_3D[ocnmask == 1] = np.reshape(should_be_basically_zero_flat, (-1,), order='F') # rebuild 3D array
#p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], should_be_basically_zero_3D, 0, -0.1, 0.1, 'RdBu', 'Surface NO3 Flux (µmol kg-1 yr-1)')

# calculate AT fluxes
q_AT_NN = -1 * TR * TA_NN # this is the flux vector (flat)
q_AT_LIR = -1 * TR * TA_LIR # this is the flux vector (flat)

# test if equal
#LHS = eye(TR.shape[0], format="csc") - dt * TR
#RHS = TA_NN + dt * q_AT
#TA_NN_2 = spsolve(LHS, RHS)
#should_be_basically_zero_flat = TA_NN_2 - TA_NN
#should_be_basically_zero_3D = np.full(ocnmask.shape, np.nan)
#should_be_basically_zero_3D[ocnmask == 1] = np.reshape(should_be_basically_zero_flat, (-1,), order='F') # rebuild 3D array
#p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], should_be_basically_zero_3D, 0, -0.1, 0.1, 'RdBu', 'Surface AT Flux (µmol kg-1 yr-1)')

# total flux
q_AT_LIR_3D = np.full(ocnmask.shape, np.nan)
q_AT_LIR_3D[ocnmask == 1] = np.reshape(q_AT_LIR, (-1,), order='F') # rebuild 3D array

q_AT_NN_3D = np.full(ocnmask.shape, np.nan)
q_AT_NN_3D[ocnmask == 1] = np.reshape(q_AT_NN, (-1,), order='F') # rebuild 3D array

p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_3D, 0, -12500, 12500, 'RdBu', 'Surface AT_NN Flux (µmol kg-1 yr-1)')
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_3D, 0, -12500, 12500, 'RdBu', 'Surface AT_LIR Flux (µmol kg-1 yr-1)')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_AT_NN_3D, 100, -70, 70, 'RdBu', ' AT_NN Flux (µmol kg-1 yr-1) along 201ºE longitude')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_AT_LIR_3D, 100, -70, 70, 'RdBu', ' AT_LIR Flux (µmol kg-1 yr-1) along 201ºE longitude') 

# soft tissue (NO3- * -1.36?)
q_AT_LIR_soft = q_NO3_LIR * -1.36

q_AT_LIR_soft_3D = np.full(ocnmask.shape, np.nan)
q_AT_LIR_soft_3D[ocnmask == 1] = np.reshape(q_AT_LIR_soft, (-1,), order='F') # rebuild 3D array

q_AT_NN_soft = q_NO3_NN * -1.36

q_AT_NN_soft_3D = np.full(ocnmask.shape, np.nan)
q_AT_NN_soft_3D[ocnmask == 1] = np.reshape(q_AT_NN_soft, (-1,), order='F') # rebuild 3D array

p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_soft_3D, 0, -25000, 25000, 'RdBu', 'Surface AT_NN Flux (Soft, µmol kg-1 yr-1)')
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_soft_3D, 0, -25000, 25000, 'RdBu', 'Surface AT_LIR Flux (Soft, µmol kg-1 yr-1)')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_AT_NN_soft_3D, 100, -70, 70, 'RdBu', ' AT_NN Flux (Soft, µmol kg-1 yr-1) along 201ºE longitude')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_AT_LIR_soft_3D, 100, -70, 70, 'RdBu', ' AT_LIR Flux (Soft, µmol kg-1 yr-1) along 201ºE longitude')

# hard tissue AT fluxes
q_AT_LIR_hard = q_AT_LIR - q_AT_LIR_soft
q_AT_NN_hard = q_AT_NN - q_AT_NN_soft

q_AT_LIR_hard_3D = np.full(ocnmask.shape, np.nan)
q_AT_LIR_hard_3D[ocnmask == 1] = np.reshape(q_AT_LIR_hard, (-1,), order='F') # rebuild 3D array

q_AT_NN_hard_3D = np.full(ocnmask.shape, np.nan)
q_AT_NN_hard_3D[ocnmask == 1] = np.reshape(q_AT_NN_hard, (-1,), order='F') # rebuild 3D array

p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_hard_3D, 0, -25000, 25000, 'RdBu', 'Surface AT_NN Flux (Hard, µmol kg-1 yr-1)')
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_hard_3D, 0, -25000, 25000, 'RdBu', 'Surface AT_LIR Flux (Hard, µmol kg-1 yr-1)')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_AT_NN_hard_3D, 100, -70, 70, 'RdBu', ' AT_NN Flux (Hard, µmol kg-1 yr-1) along 201ºE longitude')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_AT_LIR_hard_3D, 100, -70, 70, 'RdBu', ' AT_LIR Flux (Hard, µmol kg-1 yr-1) along 201ºE longitude')

# calculate DIC fluxes

# solve for DIC fluxes
q_DIC_NN = -1 * TR * DIC_NN # this is the flux vector (flat)
q_DIC_LIR = -1 * TR * DIC_LIR # this is the flux vector (flat)

# test if equal
#LHS = eye(TR.shape[0], format="csc") - dt * TR
#RHS = DIC_NN + dt * q_DIC
#DIC_NN_2 = spsolve(LHS, RHS)
#should_be_basically_zero_flat = DIC_NN_2 - DIC_NN
#should_be_basically_zero_3D = np.full(ocnmask.shape, np.nan)
#should_be_basically_zero_3D[ocnmask == 1] = np.reshape(should_be_basically_zero_flat, (-1,), order='F') # rebuild 3D array
#p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], should_be_basically_zero_3D, 0, -0.1, 0.1, 'RdBu', 'Surface DIC Flux (µmol kg-1 yr-1)')

# total flux
q_DIC_NN_3D = np.full(ocnmask.shape, np.nan)
q_DIC_NN_3D[ocnmask == 1] = np.reshape(q_DIC_NN, (-1,), order='F') # rebuild 3D array
q_DIC_LIR_3D = np.full(ocnmask.shape, np.nan)
q_DIC_LIR_3D[ocnmask == 1] = np.reshape(q_DIC_LIR, (-1,), order='F') # rebuild 3D array

p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_3D, 0, -150000, 150000, 'RdBu', 'Surface DIC_NN Flux (µmol kg-1 yr-1)')
#p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_3D, 1, -150000, 150000, 'RdBu', 'Surface DIC Flux (µmol kg-1 yr-1)')
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_3D, 0, -150000, 150000, 'RdBu', 'Surface DIC_LIR Flux (µmol kg-1 yr-1)')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_DIC_NN_3D, 100, -100, 100, 'RdBu', ' DIC_NN Flux (µmol kg-1 yr-1) along 201ºE longitude')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_DIC_LIR_3D, 100, -100, 100, 'RdBu', ' DIC_LIR Flux (µmol kg-1 yr-1) along 201ºE longitude')

# calculate soft tissue DIC fluxes (NO3- * 6.6?)
q_DIC_NN_soft = q_NO3_NN * 6.6
q_DIC_LIR_soft = q_NO3_LIR * 6.6

q_DIC_NN_soft_3D = np.full(ocnmask.shape, np.nan)
q_DIC_NN_soft_3D[ocnmask == 1] = np.reshape(q_DIC_NN_soft, (-1,), order='F') # rebuild 3D array
q_DIC_LIR_soft_3D = np.full(ocnmask.shape, np.nan)
q_DIC_LIR_soft_3D[ocnmask == 1] = np.reshape(q_DIC_LIR_soft, (-1,), order='F') # rebuild 3D array

p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_soft_3D, 0, -100000, 100000, 'RdBu', 'Surface DIC_NN Flux (Soft, µmol kg-1 yr-1)')
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_soft_3D, 0, -100000, 100000, 'RdBu', 'Surface DIC_LIR Flux (Soft, µmol kg-1 yr-1)')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_DIC_NN_soft_3D, 100, -100, 100, 'RdBu', ' DIC_NN Flux (Soft, µmol kg-1 yr-1) along 201ºE longitude')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_DIC_LIR_soft_3D, 100, -100, 100, 'RdBu', ' DIC_LIR Flux (Soft, µmol kg-1 yr-1) along 201ºE longitude')

# calculate hard tissue pump DIC fluxes (NO3- * 6.6?)
q_DIC_NN_hard = 0.5*q_AT_NN_hard
q_DIC_LIR_hard = 0.5*q_AT_LIR_hard

q_DIC_NN_hard_3D = np.full(ocnmask.shape, np.nan)
q_DIC_NN_hard_3D[ocnmask == 1] = np.reshape(q_DIC_NN_hard, (-1,), order='F') # rebuild 3D array
q_DIC_LIR_hard_3D = np.full(ocnmask.shape, np.nan)
q_DIC_LIR_hard_3D[ocnmask == 1] = np.reshape(q_DIC_LIR_hard, (-1,), order='F') # rebuild 3D array

p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_hard_3D, 0, -100000, 100000, 'RdBu', 'Surface DIC_NN Flux (Hard, µmol kg-1 yr-1)')
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_hard_3D, 0, -100000, 100000, 'RdBu', 'Surface DIC_LIR Flux (Hard, µmol kg-1 yr-1)')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_DIC_NN_hard_3D, 100, -100, 100, 'RdBu', ' DIC_NN Flux (Hard, µmol kg-1 yr-1) along 201ºE longitude')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_DIC_LIR_hard_3D, 100, -100, 100, 'RdBu', ' DIC_LIR Flux (Hard, µmol kg-1 yr-1) along 201ºE longitude')

# calculate air-sea gas ex DIC fluxes
q_DIC_NN_air_sea = q_DIC_NN - q_DIC_NN_soft - 0.5*q_AT_NN_hard
q_DIC_LIR_air_sea = q_DIC_LIR - q_DIC_LIR_soft - 0.5*q_AT_LIR_hard

q_DIC_NN_air_sea_3D = np.full(ocnmask.shape, np.nan)
q_DIC_NN_air_sea_3D[ocnmask == 1] = np.reshape(q_DIC_NN_air_sea, (-1,), order='F') # rebuild 3D array
q_DIC_LIR_air_sea_3D = np.full(ocnmask.shape, np.nan)
q_DIC_LIR_air_sea_3D[ocnmask == 1] = np.reshape(q_DIC_LIR_air_sea, (-1,), order='F') # rebuild 3D array

p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_air_sea_3D, 0, -100000, 100000, 'RdBu', 'Surface DIC_NN Flux (Air-sea, µmol kg-1 yr-1)')
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_air_sea_3D, 0, -100000, 100000, 'RdBu', 'Surface DIC_LIR Flux (Air-sea, µmol kg-1 yr-1)')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_DIC_NN_air_sea_3D, 100, -70, 70, 'RdBu', ' DIC_NN Flux (Air-sea, µmol kg-1 yr-1) along 201ºE longitude')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_DIC_LIR_air_sea_3D, 100, -70, 70, 'RdBu', ' DIC_LIR Flux (Air-sea, µmol kg-1 yr-1) along 201ºE longitude')

#%% plot NO3 fluxes
q_NO3_NN_3D = np.full(ocnmask.shape, np.nan)
q_NO3_NN_3D[ocnmask == 1] = np.reshape(q_NO3_NN, (-1,), order='F') # rebuild 3D array

q_NO3_LIR_3D = np.full(ocnmask.shape, np.nan)
q_NO3_LIR_3D[ocnmask == 1] = np.reshape(q_NO3_LIR, (-1,), order='F') # rebuild 3D array

p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_NO3_NN_3D, 0, -20000, 20000, 'RdBu', 'Surface NO3_NN Flux (µmol kg-1 yr-1)')
p2.plot_surface3d(model_lon[0, :, 0], model_lat[0, 0, :], q_NO3_LIR_3D, 0, -20000, 20000, 'RdBu', 'Surface NO3_LIR Flux (µmol kg-1 yr-1)')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_NO3_NN_3D, 100, -70, 70, 'RdBu', ' NO3_NN Flux (µmol kg-1 yr-1) along 201ºE longitude')
p2.plot_longitude3d(model_lat[0, 0, :], model_depth[:, 0, 0],  q_NO3_LIR_3D, 100, -70, 70, 'RdBu', ' NO3_LIR Flux (µmol kg-1 yr-1) along 201ºE longitude')

#%% AT: integrate fluxes from 0-100 m (layers 0 - 8) and (100 m - bottom) (layers 9 - 47)

# to integrate fluxes -> iflux [mol m-2 yr-1] = sum(flux [µmol kg-1 yr-1] * 10^-6 [mol µmol-1] * density [kg m-3] * grid cell thickness [m])

q_AT_NN_surf = np.nansum((q_AT_NN_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]),axis=0)
q_AT_NN_int = np.nansum((q_AT_NN_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]),axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_surf.T, -8, 8, 'RdBu', 'Integrated Surface AT_NN Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_int.T, -15, 15, 'RdBu', 'Integrated Interior AT_NN Flux (48 m to bottom, mol m-2 yr-1)')

q_AT_LIR_surf = np.nansum((q_AT_LIR_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]),axis=0)
q_AT_LIR_int = np.nansum((q_AT_LIR_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]),axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_surf.T, -8, 8, 'RdBu', 'Integrated Surface AT_LIR Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_int.T, -15, 15, 'RdBu', 'Integrated Interior AT_LIR Flux (48 m to bottom, mol m-2 yr-1)')

# soft tissue
q_AT_NN_soft_3D_surf = np.nansum((q_AT_NN_soft_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]), axis=0)
q_AT_NN_soft_3D_int = np.nansum((q_AT_NN_soft_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_soft_3D_surf.T, -5, 5, 'RdBu', 'Integrated Surface AT_NN Soft Tissue Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_soft_3D_int.T, -10, 10, 'RdBu', 'Integrated Interior AT_NN Soft Tissue Flux (48 m to bottom, mol m-2 yr-1)')

q_AT_LIR_soft_3D_surf = np.nansum((q_AT_LIR_soft_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]), axis=0)
q_AT_LIR_soft_3D_int = np.nansum((q_AT_LIR_soft_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_soft_3D_surf.T, -5, 5, 'RdBu', 'Integrated Surface AT_LIR Soft Tissue Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_soft_3D_int.T, -10, 10, 'RdBu', 'Integrated Interior AT_LIR Soft Tissue Flux (48 m to bottom, mol m-2 yr-1)')

# hard tissue
q_AT_NN_hard_3D_surf = np.nansum((q_AT_NN_hard_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]), axis=0)
q_AT_NN_hard_3D_int = np.nansum((q_AT_NN_hard_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_hard_3D_surf.T, -10, 10, 'RdBu', 'Integrated Surface AT_NN Hard Tissue Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_hard_3D_int.T, -20, 20, 'RdBu', 'Integrated Interior AT_NN Hard Tissue Flux (48 m to bottom, mol m-2 yr-1)')

q_AT_LIR_hard_3D_surf = np.nansum((q_AT_LIR_hard_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]), axis=0)
q_AT_LIR_hard_3D_int = np.nansum((q_AT_LIR_hard_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_hard_3D_surf.T, -10, 10, 'RdBu', 'Integrated Surface AT_LIR Hard Tissue Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_hard_3D_int.T, -20, 20, 'RdBu', 'Integrated Interior AT_LIR Hard Tissue Flux (48 m to bottom, mol m-2 yr-1)')

#%% DIC: integrate fluxes from 0-100 m (layers 0 - 8) and (100 m - bottom) (layers 9 - 47)

q_DIC_NN_surf = np.nansum((q_DIC_NN_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]),axis=0)
q_DIC_NN_int = np.nansum((q_DIC_NN_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]),axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_surf.T, -1000, 1000, 'RdBu', 'Integrated Surface DIC_NN Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_int.T, -1000, 1000, 'RdBu', 'Integrated Interior DIC_NN Flux (48 m to bottom, mol m-2 yr-1)')

q_DIC_LIR_surf = np.nansum((q_DIC_LIR_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]),axis=0)
q_DIC_LIR_int = np.nansum((q_DIC_LIR_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]),axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_surf.T, -1000, 1000, 'RdBu', 'Integrated Surface DIC_LIR Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_int.T, -1000, 1000, 'RdBu', 'Integrated Interior DIC_LIR Flux (48 m to bottom, mol m-2 yr-1)')

# air sea gas exchange
q_DIC_NN_air_sea_3D_surf = np.nansum((q_DIC_NN_air_sea_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]), axis=0)
q_DIC_NN_air_sea_3D_int = np.nansum((q_DIC_NN_air_sea_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_air_sea_3D_surf.T, -500, 500, 'RdBu', 'Integrated Surface DIC_NN Air-Sea Gas Ex Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_air_sea_3D_int.T, -500, 500, 'RdBu', 'Integrated Interior DIC_NN Air-Sea Gas Ex Flux (48 m to bottom, mol m-2 yr-1)')

q_DIC_LIR_air_sea_3D_surf = np.nansum((q_DIC_LIR_air_sea_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]), axis=0)
q_DIC_LIR_air_sea_3D_int = np.nansum((q_DIC_LIR_air_sea_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_air_sea_3D_surf.T, -500, 500, 'RdBu', 'Integrated Surface DIC_LIR Air-Sea Gas Ex Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_air_sea_3D_int.T, -500, 500, 'RdBu', 'Integrated Interior DIC_LIR Air-Sea Gas Ex Flux (48 m to bottom, mol m-2 yr-1)')

# soft tissue
q_DIC_NN_soft_3D_surf = np.nansum((q_DIC_NN_soft_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]), axis=0)
q_DIC_NN_soft_3D_int = np.nansum((q_DIC_NN_soft_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_soft_3D_surf.T, -1000, 1000, 'RdBu', 'Integrated Surface DIC_NN Soft Tissue Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_soft_3D_int.T, -1000, 1000, 'RdBu', 'Integrated Interior DIC_NN Soft Tissue Flux (48 m to bottom, mol m-2 yr-1)')

q_DIC_LIR_soft_3D_surf = np.nansum((q_DIC_LIR_soft_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]), axis=0)
q_DIC_LIR_soft_3D_int = np.nansum((q_DIC_LIR_soft_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_soft_3D_surf.T, -1000, 1000, 'RdBu', 'Integrated Surface DIC_LIR Soft Tissue Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_soft_3D_int.T, -1000, 1000, 'RdBu', 'Integrated Interior DIC_LIR Soft Tissue Flux (48 m to bottom, mol m-2 yr-1)')

# hard tissue
q_DIC_NN_hard_3D_surf = np.nansum((q_DIC_NN_hard_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]), axis=0)
q_DIC_NN_hard_3D_int = np.nansum((q_DIC_NN_hard_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_hard_3D_surf.T, -250, 250, 'RdBu', 'Integrated Surface DIC_NN Hard Tissue Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_hard_3D_int.T, -250, 250, 'RdBu', 'Integrated Interior DIC_NN Hard Tissue Flux (48 m to bottom, mol m-2 yr-1)')

q_DIC_LIR_hard_3D_surf = np.nansum((q_DIC_LIR_hard_3D[0:5, :, :] * 10**-6 * model_rho[0:5, :, :] * model_thickness[0:5, :, :]), axis=0)
q_DIC_LIR_hard_3D_int = np.nansum((q_DIC_LIR_hard_3D[5:48, :, :] * 10**-6 * model_rho[5:48, :, :] * model_thickness[5:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_hard_3D_surf.T, -250, 250, 'RdBu', 'Integrated Surface DIC_LIR Hard Tissue Flux (surface to 48 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_hard_3D_int.T, -250, 250, 'RdBu', 'Integrated Interior DIC_LIR Soft Hard Flux (48 m to bottom, mol m-2 yr-1)')

#%% horizontal averaging
q_AT_NN_avg = np.nanmean((q_AT_NN_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))
q_AT_NN_soft_avg = np.nanmean((q_AT_NN_soft_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))
q_AT_NN_hard_avg = np.nanmean((q_AT_NN_hard_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))

q_AT_LIR_avg = np.nanmean((q_AT_LIR_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))
q_AT_LIR_soft_avg = np.nanmean((q_AT_LIR_soft_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))
q_AT_LIR_hard_avg = np.nanmean((q_AT_LIR_hard_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))

data_NN = [q_AT_NN_avg, q_AT_NN_soft_avg, q_AT_NN_hard_avg]
data_LIR = [q_AT_LIR_avg, q_AT_LIR_soft_avg, q_AT_LIR_hard_avg]

labels = ['Average\nAT flux', 'Average AT\nsoft tissue flux', 'Average AT hard\ntissue flux/residual']

depth_upper = 0
depth_lower = 20

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(8,4), dpi=200)
for i, ax in enumerate(axs):
    
    # normal averages
    to_plot_NN = data_NN[i] 
    to_plot_LIR = data_LIR[i] 
    
    # alternatively, absolute values
    #to_plot_NN = np.abs(data_NN[i])
    #to_plot_LIR = np.abs(data_LIR[i])
    
    # alternatively, running averages
    #to_plot_NN = np.cumsum(np.abs(to_plot_NN[::-1]))[::-1]
    #to_plot_LIR = np.cumsum(np.abs(to_plot_LIR[::-1]))[::-1]
    
    ax.scatter(to_plot_LIR[depth_upper:depth_lower], model_depth[depth_upper:depth_lower, 0, 0], marker='o', facecolors='none', edgecolors='tab:orange', label='LIR')
    ax.scatter(to_plot_NN[depth_upper:depth_lower], model_depth[depth_upper:depth_lower, 0, 0], marker='o', facecolors='none', edgecolors='tab:blue', label='NN')
    ax.set_xlabel(labels[i])
    if i == 0:
        ax.legend(loc='lower left')

axs[0].invert_yaxis()
fig.text(0.04, 0.5, 'Depth (m)', va='center', rotation='vertical')
fig.text(0.5, -0.08, '(mol m-2 yr-1)', ha='center')
#%% DIC fluxes
q_DIC_NN_avg = np.nanmean((q_DIC_NN_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))
q_DIC_NN_soft_avg = np.nanmean((q_DIC_NN_soft_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))
q_DIC_NN_hard_avg = np.nanmean((q_DIC_NN_hard_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))
q_DIC_NN_residual_avg = np.nanmean((q_DIC_NN_air_sea_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))

q_DIC_LIR_avg = np.nanmean((q_DIC_LIR_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))
q_DIC_LIR_soft_avg = np.nanmean((q_DIC_LIR_soft_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))
q_DIC_LIR_hard_avg = np.nanmean((q_DIC_LIR_hard_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))
q_DIC_LIR_residual_avg = np.nanmean((q_DIC_LIR_air_sea_3D * 10**-6 * model_rho * model_thickness),axis=(1,2))

data_NN = [q_DIC_NN_avg, q_DIC_NN_soft_avg, q_DIC_NN_hard_avg, q_DIC_NN_residual_avg]
data_LIR = [q_DIC_LIR_avg, q_DIC_LIR_soft_avg, q_DIC_LIR_hard_avg, q_DIC_LIR_residual_avg]

labels = ['Average\nDIC flux', 'Average DIC\nsoft tissue flux', 'Average DIC\nhard tissue flux', 'Average DIC flux\nresidual/air-sea gas ex']

depth_upper = 0
depth_lower = 20

fig, axs = plt.subplots(1, 4, sharey=True, figsize=(8,4), dpi=200)
for i, ax in enumerate(axs):
    
    # normal averages
    to_plot_NN = data_NN[i] 
    to_plot_LIR = data_LIR[i] 
    
    # alternatively, absolute values
    #to_plot_NN = np.abs(data_NN[i])
    #to_plot_LIR = np.abs(data_LIR[i])
    
    # alternatively, running averages
    to_plot_NN = np.cumsum((to_plot_NN))
    to_plot_LIR = np.cumsum((to_plot_LIR))
    
    ax.scatter(to_plot_LIR[depth_upper:depth_lower], model_depth[depth_upper:depth_lower, 0, 0], marker='o', facecolors='none', edgecolors='tab:orange', label='LIR')
    ax.scatter(to_plot_NN[depth_upper:depth_lower], model_depth[depth_upper:depth_lower, 0, 0], marker='o', facecolors='none', edgecolors='tab:blue', label='NN')
    ax.set_xlabel(labels[i])
    if i == 0:
        ax.legend(loc='lower left')

axs[0].invert_yaxis()
fig.text(0.04, 0.5, 'Depth (m)', va='center', rotation='vertical')
fig.text(0.5, -0.08, '(mol m-2 yr-1)', ha='center')


#%% METHOD 1: using T1 and T0 to calculate qs -> WE DECIDED THIS METHOD IS BS
# solving inverse euler for q(t) assuming that c_t != c_(t-1)

dt = 1 # yr

q_AT_NN_1 = (TA_NN_1 - TA_NN) / dt - TR * TA_NN_1
q_AT_LIR_1 = (TA_LIR_1 - TA_LIR) / dt - TR * TA_LIR_1

q_DIC_NN_1 = (DIC_NN_1 - DIC_NN) / dt - TR * DIC_NN_1
q_DIC_LIR_1 = (DIC_LIR_1 - DIC_LIR) / dt - TR * DIC_LIR_1

q_NO3_NN_1 = (NO3_NN_1 - NO3_NN) / dt - TR * NO3_NN_1
q_NO3_LIR_1 = (NO3_LIR_1 - NO3_LIR) / dt - TR * NO3_LIR_1

# make 3D
q_AT_NN_1_3D = np.full(ocnmask.shape, np.nan)
q_AT_NN_1_3D[ocnmask == 1] = np.reshape(q_AT_NN_1, (-1,), order='F') # rebuild 3D array

q_AT_LIR_1_3D = np.full(ocnmask.shape, np.nan)
q_AT_LIR_1_3D[ocnmask == 1] = np.reshape(q_AT_LIR_1, (-1,), order='F') # rebuild 3D array

q_DIC_NN_1_3D = np.full(ocnmask.shape, np.nan)
q_DIC_NN_1_3D[ocnmask == 1] = np.reshape(q_DIC_NN_1, (-1,), order='F') # rebuild 3D array

q_DIC_LIR_1_3D = np.full(ocnmask.shape, np.nan)
q_DIC_LIR_1_3D[ocnmask == 1] = np.reshape(q_DIC_LIR_1, (-1,), order='F') # rebuild 3D array

q_NO3_NN_1_3D = np.full(ocnmask.shape, np.nan)
q_NO3_NN_1_3D[ocnmask == 1] = np.reshape(q_NO3_NN_1, (-1,), order='F') # rebuild 3D array

q_NO3_LIR_1_3D = np.full(ocnmask.shape, np.nan)
q_NO3_LIR_1_3D[ocnmask == 1] = np.reshape(q_NO3_LIR_1, (-1,), order='F') # rebuild 3D array

# calculate soft tissue fluxes of AT, DIC
q_AT_NN_1_soft_3D = q_NO3_NN_1_3D * -1.36
q_AT_LIR_1_soft_3D = q_NO3_LIR_1_3D * -1.36
q_DIC_NN_1_soft_3D = q_NO3_NN_1_3D * 6.6
q_DIC_LIR_1_soft_3D = q_NO3_LIR_1_3D * 6.6

# calculate hard tissue fluxes of AT, DIC
q_AT_LIR_1_hard_3D = q_AT_LIR_1_3D - q_AT_LIR_1_soft_3D
q_AT_NN_1_hard_3D = q_AT_NN_1_3D - q_AT_NN_1_soft_3D
q_DIC_NN_1_hard_3D = 0.5*q_AT_NN_1_hard_3D
q_DIC_LIR_1_hard_3D = 0.5*q_AT_LIR_1_hard_3D

# calculate air-sea gas exchange of DIC
q_DIC_NN_1_air_sea_3D = q_DIC_NN_1_3D - q_DIC_NN_1_soft_3D - 0.5*q_AT_NN_1_hard_3D
q_DIC_LIR_1_air_sea_3D = q_DIC_LIR_1_3D - q_DIC_LIR_1_soft_3D - 0.5*q_AT_LIR_1_hard_3D

# to start, look at integrated fluxes of AT and DIC

# total fluxes
q_AT_NN_1_surf = np.nansum((q_AT_NN_1_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]),axis=0)
q_AT_NN_1_int = np.nansum((q_AT_NN_1_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]),axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_1_surf.T, -40, 40, 'RdBu', 'Integrated Surface AT_NN Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_1_int.T, -300, 300, 'RdBu', 'Integrated Interior AT_NN Flux (109 m to bottom, mol m-2 yr-1)')

q_AT_LIR_1_surf = np.nansum((q_AT_LIR_1_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]),axis=0)
q_AT_LIR_1_int = np.nansum((q_AT_LIR_1_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]),axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_1_surf.T, -50, 50, 'RdBu', 'Integrated Surface AT_LIR Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_1_int.T, -2000, 2000, 'RdBu', 'Integrated Interior AT_LIR Flux (109 m to bottom, mol m-2 yr-1)')

q_DIC_NN_1_surf = np.nansum((q_DIC_NN_1_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]),axis=0)
q_DIC_NN_1_int = np.nansum((q_DIC_NN_1_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]),axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_1_surf.T, -75, 75, 'RdBu', 'Integrated Surface DIC_NN Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_1_int.T, -1000, 1000, 'RdBu', 'Integrated Interior DIC_NN Flux (109 m to bottom, mol m-2 yr-1)')

q_DIC_LIR_1_surf = np.nansum((q_DIC_LIR_1_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]),axis=0)
q_DIC_LIR_1_int = np.nansum((q_DIC_LIR_1_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]),axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_1_surf.T, -100, 100, 'RdBu', 'Integrated Surface DIC_LIR Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_1_int.T, -4000, 4000, 'RdBu', 'Integrated Interior DIC_LIR Flux (109 m to bottom, mol m-2 yr-1)')

# soft tissue pump
q_AT_NN_1_soft_3D_surf = np.nansum((q_AT_NN_1_soft_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]), axis=0)
q_AT_NN_1_soft_3D_int = np.nansum((q_AT_NN_1_soft_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_1_soft_3D_surf.T, -15, 15, 'RdBu', 'Integrated Surface AT_NN Soft Tissue Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_1_soft_3D_int.T, -750, 750, 'RdBu', 'Integrated Interior AT_NN Soft Tissue Flux (109 m to bottom, mol m-2 yr-1)')

q_AT_LIR_1_soft_3D_surf = np.nansum((q_AT_LIR_1_soft_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]), axis=0)
q_AT_LIR_1_soft_3D_int = np.nansum((q_AT_LIR_1_soft_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_1_soft_3D_surf.T, -15, 15, 'RdBu', 'Integrated Surface AT_LIR Soft Tissue Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_1_soft_3D_int.T, -1000, 1000, 'RdBu', 'Integrated Interior AT_LIR Soft Tissue Flux (109 m to bottom, mol m-2 yr-1)')
#
q_DIC_NN_1_soft_3D_surf = np.nansum((q_DIC_NN_1_soft_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]), axis=0)
q_DIC_NN_1_soft_3D_int = np.nansum((q_DIC_NN_1_soft_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_1_soft_3D_surf.T, -200, 200, 'RdBu', 'Integrated Surface DIC_NN Soft Tissue Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_1_soft_3D_int.T, -3500, 3500, 'RdBu', 'Integrated Interior DIC_NN Soft Tissue Flux (109 m to bottom, mol m-2 yr-1)')

q_DIC_LIR_1_soft_3D_surf = np.nansum((q_DIC_LIR_1_soft_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]), axis=0)
q_DIC_LIR_1_soft_3D_int = np.nansum((q_DIC_LIR_1_soft_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_1_soft_3D_surf.T, -200, 200, 'RdBu', 'Integrated Surface DIC_LIR Soft Tissue Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_1_soft_3D_int.T, -4000, 4000, 'RdBu', 'Integrated Interior DIC_LIR Soft Tissue Flux (109 m to bottom, mol m-2 yr-1)')

# hard tissue pump
q_AT_NN_1_hard_3D_surf = np.nansum((q_AT_NN_1_hard_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]), axis=0)
q_AT_NN_1_hard_3D_int = np.nansum((q_AT_NN_1_hard_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_1_hard_3D_surf.T, -50, 50, 'RdBu', 'Integrated Surface AT_NN Hard Tissue Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_NN_1_hard_3D_int.T, -1000, 1000, 'RdBu', 'Integrated Interior AT_NN Hard Tissue Flux (109 m to bottom, mol m-2 yr-1)')

q_AT_LIR_1_hard_3D_surf = np.nansum((q_AT_LIR_1_hard_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]), axis=0)
q_AT_LIR_1_hard_3D_int = np.nansum((q_AT_LIR_1_hard_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_1_hard_3D_surf.T, -50, 50, 'RdBu', 'Integrated Surface AT_LIR Hard Tissue Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_AT_LIR_1_hard_3D_int.T, -2000, 2000, 'RdBu', 'Integrated Interior AT_LIR Hard Tissue Flux (109 m to bottom, mol m-2 yr-1)')
#
q_DIC_NN_1_hard_3D_surf = np.nansum((q_DIC_NN_1_hard_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]), axis=0)
q_DIC_NN_1_hard_3D_int = np.nansum((q_DIC_NN_1_hard_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_1_hard_3D_surf.T, -30, 30, 'RdBu', 'Integrated Surface DIC_NN Hard Tissue Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_1_hard_3D_int.T, -500, 500, 'RdBu', 'Integrated Interior DIC_NN Hard Tissue Flux (109 m to bottom, mol m-2 yr-1)')

q_DIC_LIR_1_hard_3D_surf = np.nansum((q_DIC_LIR_1_hard_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]), axis=0)
q_DIC_LIR_1_hard_3D_int = np.nansum((q_DIC_LIR_1_hard_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_1_hard_3D_surf.T, -30, 30, 'RdBu', 'Integrated Surface DIC_LIR Hard Tissue Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_1_hard_3D_int.T, -1500, 1500, 'RdBu', 'Integrated Interior DIC_LIR Soft Hard Flux (109 m to bottom, mol m-2 yr-1)')

# air-sea gas exchange
q_DIC_NN_1_air_sea_3D_surf = np.nansum((q_DIC_NN_1_air_sea_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]), axis=0)
q_DIC_NN_1_air_sea_3D_int = np.nansum((q_DIC_NN_1_air_sea_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_1_air_sea_3D_surf.T, -50, 50, 'RdBu', 'Integrated Surface DIC_NN Air-Sea Gas Ex Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_NN_1_air_sea_3D_int.T, -3000, 3000, 'RdBu', 'Integrated Interior DIC_NN Air-Sea Gas Ex Flux (109 m to bottom, mol m-2 yr-1)')

q_DIC_LIR_1_air_sea_3D_surf = np.nansum((q_DIC_LIR_1_air_sea_3D[0:9, :, :] * 10**-6 * model_rho[0:9, :, :] * model_thickness[0:9, :, :]), axis=0)
q_DIC_LIR_1_air_sea_3D_int = np.nansum((q_DIC_LIR_1_air_sea_3D[9:48, :, :] * 10**-6 * model_rho[9:48, :, :] * model_thickness[9:48, :, :]), axis=0)

p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_1_air_sea_3D_surf.T, -50, 50, 'RdBu', 'Integrated Surface DIC_LIR Air-Sea Gas Ex Flux (surface to 109 m, mol m-2 yr-1)')
p2.plot_surface2d(model_lon[0, :, 0], model_lat[0, 0, :], q_DIC_LIR_1_air_sea_3D_int.T, -3000, 3000, 'RdBu', 'Integrated Interior DIC_LIR Air-Sea Gas Ex Flux (109 m to bottom, mol m-2 yr-1)')

