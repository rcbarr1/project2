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
import PyCO2SYS as pyco2
import gsw

data_path = '/Users/Reese_1/Documents/Research Projects/project2/data/'
output_path = '/Users/Reese_1/Documents/Research Projects/project2/outputs/'
main_path = '/Users/Reese_1/Documents/Research Projects/project2'

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

#%% pull temperature and salinity data from model

ptemp = model_data.ptemp.to_numpy() # potential temperature [ºC]
sal = model_data.salt.to_numpy() # salinity [psu]

# convert potential temperature to temperature
pres = gsw.p_from_z(model_depth*-1, model_lat) # convert depth [m, positive up] to pressure [dbar] using latitude [-90º to +90º]
asal = gsw.SA_from_SP(sal, pres, model_lon, model_lat)  # convert practical salinity [unitless] to absolute salinity [g/kg] with pressure [dbar], latitude [-90º to +90º], and longitude [-360º to 360º]
ctemp = gsw.CT_from_pt(asal, ptemp) # convert potential temperature [ºC] to conservative temperature [ºC] using absolute salinity [g/kg]
temp = gsw.t_from_CT(asal, ctemp, pres) # convert conservative temperature [ºC[ to in-situ temperature [ºC] using absolute salinity [g/kg] and pressure [dbar] (GSW python toolbox does not have direct conversion)

#%% use pyESPERs with model temperature and salinity to estimate alkalinity

# flatten all arrays and turn into lists
longitude = model_lon[ocnmask == 1].flatten(order='F').tolist() # ºE
latitude = model_lat[ocnmask == 1].flatten(order='F').tolist() # ºN
depth = model_depth[ocnmask == 1].flatten(order='F').tolist() # [m, positive down]
salinity = sal[ocnmask == 1].flatten(order='F').tolist() # [unitless]
temperature = temp[ocnmask == 1].flatten(order='F').tolist() # [ºC]


# create dictionary for output coordinates and predictor measurements
output_coordinates = {}
predictor_measurements = {}

output_coordinates.update({'longitude' : longitude,
                          'latitude' : latitude,
                          'depth' : depth})

predictor_measurements.update({'salinity' : salinity,
                              'temperature' : temperature})



# call pyESPERs to calculate total alkalinity [µmol kg-1]

TA_NN, _= PyESPER.PyESPER_NN(['TA'], main_path, output_coordinates, predictor_measurements)
TA_LIR, _, _ = PyESPER.PyESPER_LIR(['TA'], main_path, output_coordinates, predictor_measurements)



#%%


















