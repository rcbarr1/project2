#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate data used in experiments (i.e. regrid GLODAP inputs).

Created on Fri Oct 17 17:22:01 2025

@author: Reese C. Barrett
"""
#%%
from src.utils import project2 as p2
import xarray as xr
import numpy as np

data_path = './data/'

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

#%% regrid GLODAP data
p2.regrid_glodap(data_path, 'TCO2', model_depth, model_lat, model_lon, ocnmask)
p2.regrid_glodap(data_path, 'TAlk', model_depth, model_lat, model_lon, ocnmask)
p2.regrid_glodap(data_path, 'pHtsinsitutp', model_depth, model_lat, model_lon, ocnmask)
p2.regrid_glodap(data_path, 'temperature', model_depth, model_lat, model_lon, ocnmask)
p2.regrid_glodap(data_path, 'salinity', model_depth, model_lat, model_lon, ocnmask)
p2.regrid_glodap(data_path, 'silicate', model_depth, model_lat, model_lon, ocnmask)
p2.regrid_glodap(data_path, 'PO4', model_depth, model_lat, model_lon, ocnmask)

#%% regrid NCEP/DOE reanalysis II data
p2.regrid_ncep_noaa(data_path, 'icec', model_lat, model_lon, ocnmask)
p2.regrid_ncep_noaa(data_path, 'wspd', model_lat, model_lon, ocnmask)
p2.regrid_ncep_noaa(data_path, 'sst', model_lat, model_lon, ocnmask)


#dates_2025 = 2025 * np.ones([output_coordinates.shape[0],1])

# upload regridded glodap data
T_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy')
S_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy')
DIC_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC.npy') # dissolved inorganic carbon [µmol kg-1]

# transpose to match requirements for TRACEv1
T_3D = T_3D.transpose([1, 2, 0])
S_3D = S_3D.transpose([1, 2, 0])
predictor_measurements = np.vstack([S_3D.flatten(order='F'), T_3D.flatten(order='F')]).T

# combine all into .csv file to export for use with TRACEv1 in MATLAB (on the edge of my seat for pyTRACE clearly)
#trace_data = np.hstack([output_coordinates, dates_2002, predictor_measurements])
#np.savetxt(data_path + 'TRACEv1/trace_inputs_2002.txt', trace_data, delimiter = ',')

# transpose temperature and salinity back
T_3D = T_3D.transpose([2, 0, 1])
S_3D = S_3D.transpose([2, 0, 1])

# load in TRACE data (GLODAP gridded represents year 2002, so must subtract off anthropogenic C in 2002 to get preindustrial levels)
Canth_2002 = p2.loadmat(data_path + 'TRACEv1/trace_outputs_2002.mat')
Canth_2002 = Canth_2002['trace_outputs_2002']
Canth_2002 = Canth_2002.reshape(len(model_lon), len(model_lat), len(model_depth), order='F')
Canth_2002 = Canth_2002.transpose([2, 0, 1])

# calculate preindustrial pH from GLODAP DIC minus Canth to get preindustrial DIC and GLODAP TA, assuming steady state
DIC_preind_3D = DIC_3D - Canth_2002
DIC_preind = p2.flatten(DIC_preind_3D, ocnmask)

# create "pressure" array by broadcasting depth array
pressure_3D = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[1], ocnmask.shape[2]))
pressure = pressure_3D[ocnmask == 1].flatten(order='F')

# calculate preindustrial pH from DIC in 2002 minus Canth in 2002 AND TA in 2002 (assuming steady state)

# pyCO2SYS v2
co2sys_preind = pyco2.sys(dic=DIC_preind, alkalinity=AT, salinity=S, temperature=T,
                pressure=pressure, total_silicate=Si, total_phosphate=P)

pH_preind = co2sys_preind['pH']

