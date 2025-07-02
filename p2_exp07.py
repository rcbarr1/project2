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
from scipy.interpolate import RegularGridInterpolator
import project2 as p2
import time

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
q_diss_calc = cobalt.jdiss_cadet_calc # [mol CACO3 m-2 s-1]
q_prod_arag = cobalt.jprod_cadet_arag # [mol CACO3 m-2 s-1]
q_prod_calc = cobalt.jprod_cadet_calc # [mol CACO3 m-2 s-1]

data = [q_diss_calc, q_prod_arag, q_prod_calc]
#%%
def regrid_cobalt(cobalt_vrbl, model_depth, model_lat, model_lon, ocnmask, output_path):
    '''
    regrid COBALT data to model grid, inpaint nans, save as .npy file
    
    Parameters
    ----------
    cobalt_vrbl : variable from COBALT model to regrid
    model_depth : array of model depth levels
    model_lat : array pf model latitudes
    model_lon : array of model longitudes
    ocnmask : mask same shape as glodap_var where 1 marks an ocean cell and 0 marks land
    output_path : where data is stored
    
    '''
    cobalt_var = cobalt_vrbl.copy()
    var_name = cobalt_var.name

    # convert longitude to 0-360 from -300 to +60
    start_time = time.time()
    cobalt_var['xh'] = (cobalt_var['xh'] + 360) % 360 # convert
    cobalt_var = cobalt_var.sortby('xh') # resort
    end_time = time.time()
    print('\tlongitude converted to OCIM coordinates: ' + str(round(end_time - start_time,3)) + ' s')

    # replace 1e+20 values with np.NaN
    start_time = time.time()
    cobalt_var = cobalt_var.where(cobalt_var != 1e20)  
    end_time = time.time()
    print('\tNaN values replaced: ' + str(round(end_time - start_time,3)) + ' s')

    # average across time
    start_time = time.time()
    cobalt_var = cobalt_var.mean(dim='time', skipna=True)  
    end_time = time.time()
    print('\taveraged across time: ' + str(round(end_time - start_time,3)) + ' s')

    # pull out arrays of depth, latitude, and longitude from COBALT
    cobalt_depth = cobalt_var['zl'].to_numpy() # m below sea surface
    cobalt_lon = cobalt_var['xh'].to_numpy()     # ºE (originally -300 to +60, now 0 to 360)
    cobalt_lat = cobalt_var['yh'].to_numpy()     # ºN (-80 to +90)

    # pull out values from COBALT
    start_time = time.time()
    var = cobalt_var.values
    end_time = time.time()
    print('\tvalues extracted to numpy: ' + str(round(end_time - start_time,3)) + ' s')

    # switch order of COBALT dimensions (originally depth, lat, lon) to match
    # OCIM dimensions (depth, lon, lat)
    start_time = time.time()
    var = np.transpose(var, (0, 2, 1))

    np.save(output_path + var_name + '_averaged_1.npy', var)  
    end_time = time.time()
    print('\tvalues transposed, checkpoint array saved: ' + str(round(end_time - start_time,3)) + ' s')

    # create interpolator
    start_time = time.time()
    interp = RegularGridInterpolator((cobalt_depth, cobalt_lon, cobalt_lat), var, bounds_error=False, fill_value=None)

    # create meshgrid for OCIM grid
    depth, lon, lat = np.meshgrid(model_depth, model_lon, model_lat, indexing='ij')

    # reshape meshgrid points into a list of coordinates to interpolate to
    query_points = np.array([depth.ravel(), lon.ravel(), lat.ravel()]).T

    # perform interpolation (regrid COBALT data to match OCIM grid)
    var = interp(query_points)

    # transform results back to model grid shape
    var = var.reshape(depth.shape)
    
    np.save(output_path + var_name + '_averaged_regridded_1.npy', var)
    end_time = time.time()
    print('\tinterpolation performed, checkpoint array saved: ' + str(round(end_time - start_time,3)) + ' s')

    # inpaint nans
    start_time = time.time()
    var = p2.inpaint_nans3d(var, mask=ocnmask.astype(bool))
    end_time = time.time()
    print('\tNaNs inpainted: ' + str(round(end_time - start_time,3)) + ' s')

    # save regridded data
    np.save(output_path + var_name + '_averaged_regridded_inpainted_1.npy', var)
    print('\tfinal regridded array saved')
   
for d in data:
#d = data[0]
    print('now regridding ' + d.name)
    start = time.time()
    regrid_cobalt(d, model_depth, model_lat, model_lon, ocnmask, output_path)
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

## NOTE: I am trying to reproduce the weird stripiness within the function
# outside of the function. If I only do one depth layer, it doesn't happen.
# Need to run code overnight to do averaging across time while maintaining all
# three other dimensions to test. Otherwise I have no idea wtf is going on.
# Also not sure why the stripiness goes away after regridding–is it just a 
# colorbar issue?


#%% open and plot results

# before making copy of array in function
avg = np.load(output_path + 'jprod_cadet_calc_averaged.npy')
regr = np.load(output_path + 'jprod_cadet_calc_averaged_regridded.npy')
inp = np.load(output_path + 'jprod_cadet_calc_averaged_regridded_inpainted.npy')

# after editing function to include copy of array
avg_1 = np.load(output_path + 'jprod_cadet_calc_averaged_1.npy')
regr_1 = np.load(output_path + 'jprod_cadet_calc_averaged_regridded_1.npy')
inp_1 = np.load(output_path + 'jprod_cadet_calc_averaged_regridded_inpainted_1.npy')

#%% plots
# cobalt data
cobalt_depth = q_prod_calc['zl'].to_numpy() # m below sea surface
cobalt_lon = q_prod_calc['xh'].to_numpy()     # ºE (originally -300 to +60, now 0 to 360)
cobalt_lat = q_prod_calc['yh'].to_numpy()     # ºN (-80 to +90)
cobalt_lon_regr = (cobalt_lon + 360) % 360 # convert


#%% averaged across time

i = 74
test0 = avg
#test0[test0<0] = 1e-20
p2.plot_surface3d(cobalt_lon, cobalt_lat, test0, i, 1e-17, 1e-10, 'magma', 'calc fluxes', logscale=True, lon_lims=[-300, 60])


#%% regridded
i=0
test1 = regr_1
#test1[test1<=0] = 1e-20
p2.plot_surface3d(model_lon, model_lat, test1, i, 1e-17, 1e-10, 'magma', 'calc fluxes', logscale=True)

#%% inpainted
test2 = inp_1
#test2[test2<=0] = 1e-20
p2.plot_surface3d(model_lon, model_lat, test2, i, 1e-17, 1e-10, 'magma', 'calc fluxes', logscale=True)

#%% plotting ocnmask to compare inpainting -> something is wrong, deep ocean layers do not look like deep ocean ocnmask
ocnmask_copy = ocnmask.copy()
ocnmask_copy[ocnmask_copy == 0] = np.NaN

p2.plot_surface3d(model_lon, model_lat, ocnmask_copy, i, 0, 5, 'magma', 'ocnmask')










