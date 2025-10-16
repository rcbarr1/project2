#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regrid Rui's COBALT outputs for jdiss_cadet_arag_plus_btm,
jdiss_cadet_calc_plus_btm, jprod_cadet_arag, jprod_cadet_calc for use in model.
Then, try different parameterizations for ∆q terms based on this output.

Created on Mon Jun 16 16:15:15 2025

@author: Reese Barrett
"""

# NOTE: GOING TO NEED TO THINK VERY CAREFULLY ABOUT UNITS

import xarray as xr
import numpy as np
import project2 as p2
import time
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

cobalt_path = '/Volumes/LaCie/data/OM4p25_cobalt_v3/'
data_path = '/Users/Reese_1/Documents/Research Projects/project2/data/'
#output_path = '/Users/Reese_1/Documents/Research Projects/project2/outputs/'
output_path = '/Volumes/LaCie/outputs/'

# load ocim
TR = p2.loadmat(data_path + 'OCIM2_48L_base/OCIM2_48L_base_transport.mat')
TR = TR['TR']

# open up rest of data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].to_numpy()

model_depth = model_data['tz'].to_numpy()[:, 0, 0] # m below sea surface
model_lon = model_data['tlon'].to_numpy()[0, :, 0] # ºE
model_lat = model_data['tlat'].to_numpy()[0, 0, :] # ºN

#%% load cobalt
cobalt = xr.open_dataset(cobalt_path + '19580101.ocean_cobalt_fluxes_int.nc', decode_cf=False)

q_diss_arag_plus_btm = cobalt.jdiss_cadet_arag_plus_btm # [mol CACO3 m-2 s-1]
q_diss_calc_plus_btm = cobalt.jdiss_cadet_calc_plus_btm # [mol CACO3 m-2 s-1]
#q_diss_arag = cobalt.jdiss_cadet_arag # [mol CACO3 m-2 s-1]
q_diss_calc = cobalt.jdiss_cadet_calc # [mol CACO3 m-2 s-1]
q_prod_arag = cobalt.jprod_cadet_arag # [mol CACO3 m-2 s-1]
q_prod_calc = cobalt.jprod_cadet_calc # [mol CACO3 m-2 s-1]

data = [q_diss_calc, q_prod_arag, q_prod_calc]
#data = [q_diss_arag]


cobalt_vrbl = q_prod_calc

#%% call regridding & inpainting function
for d in data:
#d = data[0]
    print('now regridding ' + d.name)
    start = time.time()
    p2.regrid_cobalt(d, model_depth, model_lat, model_lon, ocnmask, output_path)
    end = time.time()
    print('regrid of ' + d.name + ' performed in ' + str(round(end - start,3)) + ' s')
    
#%% plot before and after longitude conversion
cobalt_var = q_prod_calc.copy()
# plot before conversion
cobalt_depth = cobalt_var['zl'].to_numpy() # m below sea surface
cobalt_lon = cobalt_var['xh'].to_numpy()     # ºE (originally -300 to +60, now 0 to 360)
cobalt_lat = cobalt_var['yh'].to_numpy()     # ºN (-80 to +90)

i = 74
t = 0

#test = cobalt_var.isel(zl=0)
test = cobalt_var
test['xh'] = (test['xh'] + 360) % 360 # convert
test = test.sortby('xh') # resort
test = test.where(test != 1e20)  
test = test.mean(dim='time', skipna=True)  

test_lon = test['xh'].values   # ºE (originally -300 to +60, now 0 to 360)
p2.plot_surface2d(test_lon, cobalt_lat, test[0, :, :], 1e-11, 1e-9, 'magma', 'calc fluxes')

#%% open and plot results

avg = np.load(output_path + 'jprod_cadet_calc_averaged.npy')
regr = np.load(output_path + 'jprod_cadet_calc_averaged_regridded.npy')
inp = np.load(output_path + 'jprod_cadet_calc_averaged_regridded_inpainted.npy')

#%% plots
# cobalt data
cobalt_depth = q_prod_calc['zl'].to_numpy() # m below sea surface
cobalt_lon = q_prod_calc['xh'].to_numpy()     # ºE (originally -300 to +60, now 0 to 360)
cobalt_lat = q_prod_calc['yh'].to_numpy()     # ºN (-80 to +90)
cobalt_lon_regr = (cobalt_lon + 360) % 360 # convert

#%% inpainted
i = 40
test2 = avg
#test2 = inp
#test2[test2<=0] = 1e-20
p2.plot_surface3d(model_lon, model_lat, test2, i, 1e-17, 1e-10, 'magma', 'calc fluxes', logscale=True, lon_lims=[0, 360])

#%% regridded
test1 = regr
#test1[test1<=0] = 1e-20
p2.plot_surface3d(model_lon, model_lat, test1, i, 1e-17, 1e-10, 'magma', 'calc fluxes', logscale=True, lon_lims=[0, 360])

#%% averaged across time
j = np.abs(cobalt_depth - model_depth[i]).argmin()
test0 = inp
#test0[test0<0] = 1e-20
p2.plot_surface3d(cobalt_lon, cobalt_lat, test0, j, 1e-17, 1e-10, 'magma', 'calc fluxes', logscale=True, lon_lims=[0, 360])

#%% plotting ocnmask to compare inpainting -> something is wrong, deep ocean layers do not look like deep ocean ocnmask
ocnmask_copy = ocnmask.copy()
ocnmask_copy[ocnmask_copy == 0] = np.NaN

p2.plot_surface3d(model_lon, model_lat, ocnmask_copy, i, 0, 2, 'magma', 'ocnmask')
