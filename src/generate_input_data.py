#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate data used in experiments (i.e. regrid GLODAP inputs).

Created on Fri Oct 17 17:22:01 2025

@author: Reese C. Barrett
"""

from src.utils import project2 as p2
import xarray as xr

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
