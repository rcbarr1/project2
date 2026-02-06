"""
Functions supporting project2_main.py

Created on Tue Oct  8 11:15:51 2024

@author: Reese Barrett
"""

import scipy.io as spio
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import xarray as xr
import h5netcdf
import datetime as dt
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from tqdm import tqdm
import time
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import cartopy.crs as ccrs
import random
import PyCO2SYS as pyco2
import cftime
import matplotlib.animation as animation
import torch
import torch.nn as nn
import joblib
from pyTRACE import trace
from scipy.ndimage import uniform_filter

def loadmat(filename):
    '''
    stolen from: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)
    
def save_model_output(filename, time, lat, lon, depth, tracers, tracer_dims=[('time', 'lat', 'lon', 'depth')], tracer_names=None, tracer_units=None, global_attrs=None):
    '''
    Save model output to a NetCDF file.
    
    Parameters
    ----------
    filename (str): Name of the NetCDF file to create.
    time (list or array): Time values (e.g., in years).
    lat (list or array): Latitude values.
    lon (list or array): Longitude values.
    depth (list or array): Depth values.
    tracers (list of arrays): List of arrays for each variable of interest.
    tracer_dims (list of tuples): List of dimension tuples corresponding to each tracer (e.g., ('time', 'depth', 'lat', 'lon')).
    tracer_names (list of str, optional): Names of the tracers corresponding to each tracer set in `tracers`.
    tracer_units (list of str, optional): Units of the tracers corresponding to each tracer set in `tracers`.
    global_attrs (dict, optional): Additional global attributes for the NetCDF file.
    '''

    if tracer_names is None:
        tracer_names = [f"tracer_{i}" for i in range(len(tracers))]

    if tracer_units is None:
        tracer_units = ["" for _ in tracers]

    # Validate input lengths
    if len(tracers) != len(tracer_names):
        raise ValueError("The number of tracers must match the number of tracer names.")
    if len(tracers) != len(tracer_units):
        raise ValueError("The number of tracers must match the number of tracer units.")
    if len(tracers) != len(tracer_dims):
        raise ValueError("The number of tracers must match the number of tracer dimensions.")

    # Ensure all tracer_dims are tuples
    tracer_dims = [(dim,) if isinstance(dim, str) else dim for dim in tracer_dims]
    
    with h5netcdf.File('./outputs/' + filename, 'w', invalid_netcdf=True) as ncfile:
        # Create dimensions
        ncfile.dimensions['time'] = len(time)
        ncfile.dimensions['lat'] = len(lat)
        ncfile.dimensions['lon'] = len(lon)
        ncfile.dimensions['depth'] = len(depth)

        # Create coordinate variables
        nc_time = ncfile.create_variable('time', ('time',), dtype='f8')
        nc_lat = ncfile.create_variable('lat', ('lat',), dtype='f8')
        nc_lon = ncfile.create_variable('lon', ('lon',), dtype='f8')
        nc_depth = ncfile.create_variable('depth', ('depth',), dtype='f8')

        # Set units and descriptions for coordinate variables
        nc_time.attrs['units'] = 'years'
        nc_lat.attrs['units'] = 'degrees_north'
        nc_lon.attrs['units'] = 'degrees_east'
        nc_depth.attrs['units'] = 'meters'

        # Write coordinate data
        nc_time[:] = time
        nc_lat[:] = lat
        nc_lon[:] = lon
        nc_depth[:] = depth

        # Create and write tracer variables
        for tracer_name, tracer_data, tracer_dim, tracer_unit in zip(tracer_names, tracers, tracer_dims, tracer_units):
            tracer_var = ncfile.create_variable(tracer_name, tracer_dim, dtype='f8')
            tracer_var.attrs['description'] = f"{tracer_name} data"
            tracer_var.attrs['units'] = tracer_unit

            # Write tracer data
            tracer_var[...] = tracer_data

        # Add global attributes
        ncfile.attrs['history'] = 'Created ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ncfile.attrs['source'] = 'Python script'
        if global_attrs:
            for key, value in global_attrs.items():
                ncfile.attrs[key] = value

def flatten(e_3D, ocnmask):
    '''
    flattens array from 3D to 1D in Fortran ordering and simultaneously removes land points from flat array
    '''
    return e_3D.flatten(order='F')[ocnmask.flatten(order='F').astype(bool)]

def make_3D(e_flat, ocnmask):
    '''
    returns 1D array to 3D, adds np.NaN for land boxes as defined by ocnmask. Fortran ordering
    '''
    e_3D = np.full(ocnmask.shape, np.nan)
    flat_mask = ocnmask.flatten(order='F').astype(bool)
    
    e_3D_flat = e_3D.flatten(order='F')
   
    e_3D_flat[flat_mask] = e_flat
    e_3D = e_3D_flat.reshape(ocnmask.shape, order='F') 
    
    return e_3D

def smooth_tracer3D(e_flat, ocnmask):
    '''
    Smooths tracer distribution by averaging each cell with the cells surrounding it in all dimensions.

    :param e_flat: Flattened matrix representing tracer values.
    :param ocnmask: Mask representing land (0) and ocean (1) grid cells
    '''
    e_flat_3D = make_3D(e_flat,ocnmask)
    e_flat_3D = np.nan_to_num(e_flat_3D, nan=0.0) # make land cells == 0 for averaging

    # add pad for longitude wrapping
    e_flat_pad_3D = np.pad(e_flat_3D, ((0,0), (2,2), (0,0)), mode='wrap') 
    ocnmask_pad = np.pad(ocnmask, ((0,0), (2,2), (0,0)), mode='wrap') 
    
    # sum tracer across 3x3x3 neighborhood to get average of nearest cells
    e_sum_pad_3D = uniform_filter(e_flat_pad_3D * ocnmask_pad, size=(5,5,5), mode='constant', cval=0)

    # count ocean cells used in each neighborhood
    ocnmask_pad_count = uniform_filter(ocnmask_pad.astype(float), size=(5,5,5), mode='constant', cval=0)

    ocnmask_pad_count[ocnmask_pad_count == 0] = np.nan # avoid division by zero
    e_smooth_pad_3D = e_sum_pad_3D / ocnmask_pad_count # calculate average AT
    
    # remove pad and flatten
    e_flat_smooth_3D = e_smooth_pad_3D[:, 2:-2, :]
    e_flat_smooth = flatten(e_flat_smooth_3D, ocnmask)

    return e_flat_smooth

def get_depth_idx(ocnmask, depth_level):
    '''
    returns indicies in 3D array flattened using flatten() above that correspond to 
    values at a depth level represented by depth_idx
    '''
    surf_mask = np.zeros_like(ocnmask)
    surf_mask[:, :, 0] = 1

    ocn_surf_mask = ocnmask * surf_mask
    return np.argwhere(flatten(ocn_surf_mask, ocnmask)==1)


def find_MLD(model_lat, model_lon, ocnmask, MLD_da, latm, lonm, type_flag):
    """
    Reads in the Holte et al. monthly mixed layer climatology and interpolates
    it. Currently set up to interpolate the maximum or average monthly mixed
    layer depth, but it could be rewritten to allow minimum MLD. Interpolates
    to OCIM grid. Lots of interpolation in sea ice regions, but this doesn't
    matter a ton because we account for sea ice in air-sea gas exchange
    calculations.
    
    Keyword arguments:
        model_lat = latitudes of interest
        model_lon = longitudes of interest
        MLD_da = mixed layer depth from density algorithm from Holte et al.
                (should be mld_da_max or mld_da_mean)
        latm = from Holte et al.
        lonm = from Holte et al.
        type_flag = 0 for maximum monthly max MLD, 1 for mean monthly mean MLD
        
    Returns:
        interp_MLDs = maximum mixed layer depths at lons & lats
    """
    
    # extracting the maximum values along the first dimension
    # this is taking the maximum MLD across the monthly climatologies
    # --> whichever month had the largest MLD
    if type_flag == 0:
        MLDs = np.nanmax(MLD_da, axis=0)
    elif type_flag == 1:
        MLDs = np.nanmean(MLD_da, axis=0)
    else:
        print('ERROR: type_flag should be specified as 0 or 1')
    
    # transform lons to 0 to 360
    lonm[lonm<=0] += 360

    # reorder everything along the lon axis to have in ascending order
    lonm_1d = lonm[:,0]
    sort_idx = np.argsort(lonm_1d)

    lonm_1d = lonm_1d[sort_idx]
    lonm = lonm[sort_idx, :]
    MLDs = MLDs[sort_idx, :]

    # padding lonm and latm arrays
    lonm = np.vstack([lonm[-1, :] - 360, lonm, lonm[0, :] + 360])
    latm = np.vstack([latm[-1, :], latm, latm[0, :]])
    MLDs = np.vstack([MLDs[-1, :], MLDs, MLDs[0, :]])

    latm = np.hstack([latm[:, 0:1] - 1, latm, latm[:, -1:] + 1])
    lonm = np.hstack([lonm[:, 0:1], lonm, lonm[:, -1:]])
    MLDs = np.hstack([MLDs[:, 0:1], MLDs, MLDs[:, -1:]])

    # create interpolator
    interp = RegularGridInterpolator((latm[:,0], lonm[0,:]), MLDs, bounds_error=False, fill_value=None)

    # create meshgrid for OCIM grid
    lat, lon = np.meshgrid(model_lat, model_lon, indexing='ij')

    # reshape meshgrid points into a list of coordinates to interpolate to
    query_points = np.array([lat.ravel(), lon.ravel()]).T

    # perform interpolation (regrid WOA data to match OCIM grid)
    var = interp(query_points)

    # transform results back to model grid shape
    var = var.reshape(lon.shape)

    # inpaint nans
    interp_MLDs = inpaint_nans2d(var, mask=ocnmask[:,:,0].astype(bool))

    return interp_MLDs
    
def inpaint_nans3d(array_3d, iterations=10, mask=None):
        
    # copy input
    interpolated = array_3d.copy()
    
    # use a simple 6-connected stencil for neighbor averaging
    kernel = np.zeros((3, 3, 3))
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1  # neighbors in Z
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1  # neighbors in Y
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1  # neighbors in X
    
    # initialize NaNs with mean value
    nan_mask = np.isnan(interpolated)
    interpolated[nan_mask] = np.nanmean(interpolated)
    
    # optional land mask — persist NaNs in land areas
    if mask is not None:
        land_mask = ~mask.astype(bool)  # True where it's land (to be masked out)
    else:
        land_mask = np.zeros_like(array_3d, dtype=bool)
    
    for _ in range(iterations):
        # count valid neighbors
        valid = ~np.isnan(interpolated)
        neighbor_sum = convolve(np.nan_to_num(interpolated), kernel, mode='wrap')
        neighbor_count = convolve(valid.astype(float), kernel, mode='wrap')
    
        # avoid division by 0
        with np.errstate(invalid='ignore', divide='ignore'):
            new_vals = neighbor_sum / neighbor_count
    
        # only update former NaNs (and skip land if mask is given)
        update_mask = nan_mask & ~land_mask & (neighbor_count > 0)
        interpolated[update_mask] = new_vals[update_mask]
    
    # reapply land mask as NaN
    if mask is not None:
        interpolated[~mask.astype(bool)] = np.nan

    return interpolated

def inpaint_nans2d(array_2d, iterations=10, mask=None):
        
    # copy input
    interpolated = array_2d.copy()
    
    # use a simple 4-connected stencil for neighbor averaging
    kernel = np.zeros((3, 3))
    kernel[0, 1] = 1  # up
    kernel[2, 1] = 1  # down
    kernel[1, 0] = 1  # left
    kernel[1, 2] = 1  # right
    
    # initialize NaNs with mean value
    nan_mask = np.isnan(interpolated)
    interpolated[nan_mask] = np.nanmean(interpolated)
    
    # optional land mask — persist NaNs in land areas
    if mask is not None:
        land_mask = ~mask.astype(bool)  # true where it's land (to be masked out)
    else:
        land_mask = np.zeros_like(array_2d, dtype=bool)
    
    for _ in tqdm(range(iterations), desc="inpainting"):
        # count valid neighbors
        valid = ~np.isnan(interpolated)
        neighbor_sum = convolve(np.nan_to_num(interpolated), kernel, mode='wrap')
        neighbor_count = convolve(valid.astype(float), kernel, mode='wrap')
    
        # avoid division by 0
        with np.errstate(invalid='ignore', divide='ignore'):
            new_vals = neighbor_sum / neighbor_count
    
        # only update former NaNs (and skip land if mask is given)
        update_mask = nan_mask & ~land_mask & (neighbor_count > 0)
        interpolated[update_mask] = new_vals[update_mask]
    
    # reapply land mask as NaN
    if mask is not None:
        interpolated[~mask.astype(bool)] = np.nan

    return interpolated

def regrid_glodap(data_path, glodap_var, model_lat, model_lon, model_depth, ocnmask):
    '''
    regrid glodap data to model grid, inpaint nans, save as .npy file

    Parameters
    ----------
    data_path : path to folder which contains GLODAPv2.2016b.MappedProduct folder from https://glodap.info/index.php/mapped-data-product/
    glodap_var : variable to regrid as named in GLODAP (dimensions: depth, longitude, latitude)
    model_lat : array pf model latitudes
    model_lon : array of model longitudes
    model_depth : array of model depth levels
    ocnmask : mask same shape as glodap_var where 1 marks an ocean cell and 0 marks land

    '''
    print('begin regrid of ' + glodap_var)
    start_time = time.time()
    
    # load GLODAP data (https://glodap.info/index.php/mapped-data-product/)
    glodap_data = xr.open_dataset(data_path + 'GLODAPv2.2016b.MappedProduct/GLODAPv2.2016b.' + glodap_var + '.nc')

    # pull out arrays of depth, latitude, and longitude from GLODAP
    glodap_lat = glodap_data['lat'].to_numpy()     # ºN
    glodap_lon = glodap_data['lon'].to_numpy()     # ºE
    glodap_depth = glodap_data['Depth'].to_numpy() # m below sea surface

    # pull out values of DIC and TA from GLODAP
    var = glodap_data[glodap_var].transpose('lat', 'lon', 'depth_surface').copy().values # switch order of GLODAP dimensions to match OCIM dimensions
    
    # create interpolator
    interp = RegularGridInterpolator((glodap_lat, glodap_lon, glodap_depth), var, bounds_error=False, fill_value=None)

    # transform model_lon for anything < 20 (because GLODAP goes from 20ºE - 380ºE)
    model_lon[model_lon < 20] += 360

    # create meshgrid for OCIM grid
    lat, lon, depth = np.meshgrid(model_lat, model_lon, model_depth, indexing='ij')

    # reshape meshgrid points into a list of coordinates to interpolate to
    query_points = np.array([lat.ravel(), lon.ravel(), depth.ravel()]).T

    # perform interpolation (regrid GLODAP data to match OCIM grid)
    var = interp(query_points)

    # transform results back to model grid shape
    var = var.reshape(depth.shape)

    # inpaint nans
    var = inpaint_nans3d(var, mask=ocnmask.astype(bool))
    
    # transform model_lon and meshgrid back for anything > 360
    model_lon[model_lon > 360] -= 360
    
    # make sure no negative values in total silicate or total phosphate due to interpolation
    if glodap_var == 'PO4' or glodap_var == 'silicate':
        var[var < 0] = 0
    
    # save regridded data
    if glodap_var == 'TCO2':
        np.save(data_path + 'GLODAPv2.2016b.MappedProduct/DIC.npy', var)
    elif glodap_var == 'TAlk':
        np.save(data_path + 'GLODAPv2.2016b.MappedProduct/TA.npy', var)
    else:
        np.save(data_path + 'GLODAPv2.2016b.MappedProduct/' + glodap_var + '.npy', var)

    end_time = time.time()
    print('\tregrid complete in ' + str(round(end_time - start_time,3)) + ' s')


def regrid_woa(data_path, woa_var, model_lat, model_lon, model_depth, ocnmask):
    '''
    regrid woa data to model grid, inpaint nans

    Parameters
    ----------
    data_path : path to folder which contains WOA18 data from https://www.ncei.noaa.gov/access/world-ocean-atlas-2018/
    woa_var : variable to regrid (dimensions: depth, longitude, latitude)
    model_lat : array pf model latitudes
    model_lon : array of model longitudes
    ocnmask : mask same shape as woa_var wbere 1 marks an ocean cell and 0 marks land

    Returns
    -------
    woa_var : regridded to model grid
    '''
    
    # load WOA18 data
    if woa_var == 'S': # salinity [unitless]
        data = xr.open_dataset(data_path + 'WOA18/woa18_decav81B0_s00_01.nc', decode_times=False)
    elif woa_var == 'T': # temperature [ºC]
        data = xr.open_dataset(data_path + 'WOA18/woa18_decav81B0_t00_01.nc', decode_times=False)
    elif woa_var == 'Si': # silicate [µmol kg-1]
        data = xr.open_dataset(data_path + 'WOA18/woa18_all_i00_01.nc', decode_times=False)
    elif woa_var == 'P': # phosphate [µmol kg-1]
        data = xr.open_dataset(data_path + 'WOA18/woa18_all_p00_01.nc', decode_times=False)
    else:
        print("WOA data not found. Choose from woa_var = 'S', 'T', 'Si', 'P'")
        return
    
    print('begin regrid of ' + woa_var)
    start_time = time.time()

    # convert longitude to 0-360 from -180-180
    data['lon'] = (data['lon'] + 360) % 360 # convert
    data = data.sortby('lon') # resort
    
    # pull out arrays of depth, latitude, and longitude from WOA
    data_lat = data['lat'].to_numpy()     # ºN
    data_lon = data['lon'].to_numpy()     # ºE
    data_depth = data['depth'].to_numpy()     # m
    
    # pull out data variable from WOA (now that coordinates are correct)
    if woa_var == 'S': # salinity [unitless]
        var = data.s_an.isel(time=0).transpose('lat', 'lon', 'depth').values # transpose to match OCIM format (depth, lon, lat)
    elif woa_var == 'T': # temperature [ºC]
        var = data.t_an.isel(time=0).transpose('lat', 'lon', 'depth').values # transpose to match OCIM format (depth, lon, lat)
    elif woa_var == 'Si': # silicate [µmol kg-1]
        var = data.i_an.isel(time=0).transpose('lat', 'lon', 'depth').values # transpose to match OCIM format (depth, lon, lat)
    elif woa_var == 'P': # phosphate [µmol kg-1]
        var = data.p_an.isel(time=0).transpose('lat', 'lon', 'depth').values # transpose to match OCIM format (depth, lon, lat)

    # create interpolator
    interp = RegularGridInterpolator((data_lat, data_lon, data_depth), var, bounds_error=False, fill_value=None)

    # create meshgrid for OCIM grid
    lat, lon, depth = np.meshgrid(model_lat, model_lon, model_depth, indexing='ij')

    # reshape meshgrid points into a list of coordinates to interpolate to
    query_points = np.array([lat.ravel(), lon.ravel(), depth.ravel()]).T

    # perform interpolation (regrid WOA data to match OCIM grid)
    var = interp(query_points)

    # transform results back to model grid shape
    var = var.reshape(depth.shape)

    # inpaint nans
    var = inpaint_nans3d(var, mask=ocnmask.astype(bool))

    # save regridded data
    if woa_var == 'S':
        np.save(data_path + 'WOA18/S.npy', var)
    elif woa_var == 'T':
        np.save(data_path + 'WOA18/T.npy', var)
    elif woa_var == 'Si':
        np.save(data_path + 'WOA18/Si.npy', var)
    elif woa_var == 'P':
        np.save(data_path + 'WOA18/P.npy', var)
        
    end_time = time.time()
    print('\tregrid complete in ' + str(round(end_time - start_time,3)) + ' s')
    
    
def regrid_ncep_noaa(data_path, ncep_var, model_lat, model_lon, ocnmask):
    '''
    calculate annual average, regrid data to model grid, inpaint nans
    NCEP/DOE reanalysis II data and NOAA Extended Reconstruction SST V5

    Parameters
    ----------
    data_path : path to folder which contains NCEP data from https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html or NOAA data from https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html
    ncep_var : variable to regrid (dimensions: depth, longitude, latitude)
    model_lat : array pf model latitudes
    model_lon : array of model longitudes
    ocnmask : mask same shape as ncep_var where 1 marks an ocean cell and 0 marks land

    Returns
    -------
    ncep_var : regridded to model grid

    '''
    # load NCEP data (https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html) or NOAA data (https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html)
    if ncep_var == 'icec': # ice concentration
        data = xr.open_dataset(data_path + 'NCEP_DOE_Reanalysis_II/icec.sfc.mon.ltm.1991-2020.nc')
        var = data.icec.mean(dim='time', skipna=True).values # average across all months, pull out values from NCEP
    elif ncep_var == 'wspd': # u-wind [m/s]
        data = xr.open_dataset(data_path + 'NCEP_DOE_Reanalysis_II/wspd.10m.mon.mean.nc')
        var = data.wspd.isel(time=slice(552,924)).mean(dim='time', skipna=True).values # average across all months from 1994-01-01 to 2024-01-01, pull out values from NCEP
    elif ncep_var == 'sst': # sea surface temperature [ºC]
        data = xr.open_dataset(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst.mon.ltm.1991-2020.nc')
        var = data.sst.mean(dim='time', skipna=True).values # average across all months, pull out values from NCEP
    else:
        print('NCEP/NOAA data not found.')
        return
      
    print('begin regrid of ' + ncep_var)
    start_time = time.time()
    
    # pull out arrays of depth, latitude, and longitude from NCEP
    data_lat = data['lat'].to_numpy()     # ºN
    data_lon = data['lon'].to_numpy()     # ºE
   
    # create interpolator
    interp = RegularGridInterpolator((data_lat, data_lon), var, bounds_error=False, fill_value=None)

    # create meshgrid for OCIM grid
    lat, lon = np.meshgrid(model_lat, model_lon, indexing='ij')

    # reshape meshgrid points into a list of coordinates to interpolate to
    query_points = np.array([lat.ravel(), lon.ravel()]).T

    # perform interpolation (regrid GLODAP data to match OCIM grid)
    var = interp(query_points)

    # transform results back to model grid shape
    var = var.reshape(lon.shape)

    # inpaint nans
    var = inpaint_nans2d(var, mask=ocnmask[:, :, 0].astype(bool))

    # save regridded data
    if ncep_var == 'icec':
        np.save(data_path + 'NCEP_DOE_Reanalysis_II/icec.npy', var)
    elif ncep_var == 'wspd':
        np.save(data_path + 'NCEP_DOE_Reanalysis_II/wspd.npy', var)
    elif ncep_var == 'sst':
        np.save(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst.npy', var)
        
    end_time = time.time()
    print('\tregrid complete in ' + str(round(end_time - start_time,3)) + ' s')

def regrid_cobalt(cobalt_vrbl, model_lat, model_lon, model_depth, ocnmask, data_path):
    '''
    regrid COBALT data to model grid, inpaint nans, save as .npy file
    
    Parameters
    ----------
    cobalt_vrbl : variable from COBALT model to regrid
    model_lat : array of model latitudes
    model_lon : array of model longitudes
    model_depth : array of model depth levels
    ocnmask : mask same shape as glodap_var where 1 marks an ocean cell and 0 marks land
    data_path : where data is stored
    
    '''
    # set up
    cobalt_var = cobalt_vrbl.copy()
    var_name = cobalt_var.name
    print('begin regrid of ' + var_name)

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

    # convert longitude to 0 to 360 from -300 to +60 
    start_time = time.time()
    cobalt_var['xh'] = (cobalt_var['xh'] + 360) % 360 # convert
    cobalt_var = cobalt_var.sortby('xh') # resort
    end_time = time.time()
    print('\tlongitude converted to OCIM coordinates: ' + str(round(end_time - start_time,3)) + ' s')

    # pull out arrays of depth, latitude, and longitude from COBALT
    cobalt_lat = cobalt_var['yh'].to_numpy()     # ºN (-80 to +90)
    cobalt_lon = cobalt_var['xh'].to_numpy()     # ºE (originally -300 to +60, now 0 to 360)
    cobalt_depth = cobalt_var['zl'].to_numpy() # m below sea surface

    # pull out values from COBALT
    start_time = time.time()
    var = cobalt_var.transpose('yh', 'xh', 'zl').values # switch order of COBALT dimensions to match OCIM
    end_time = time.time()
    print('\tvalues extracted to numpy: ' + str(round(end_time - start_time,3)) + ' s')

    # create interpolator
    start_time = time.time()
    interp = RegularGridInterpolator((cobalt_lat, cobalt_lon, cobalt_depth), var, method='linear', bounds_error=False, fill_value=None)

    # create meshgrid for OCIM grid
    lat, lon, depth = np.meshgrid(model_lat, model_lon, model_depth, indexing='ij')

    # reshape meshgrid points into a list of coordinates to interpolate to
    query_points = np.array([lat.ravel(), lon.ravel(), depth.ravel()]).T

    # perform interpolation (regrid COBALT data to match OCIM grid)
    var_interped = interp(query_points)

    # transform results back to model grid shape
    var_interped = var_interped.reshape(depth.shape)

    #np.save(output_path + var_name + '_averaged_regridded.npy', var_interped)
    end_time = time.time()
    print('\tinterpolation performed: ' + str(round(end_time - start_time,3)) + ' s')
    
    # inpaint nans
    start_time = time.time()
    var_inpainted = inpaint_nans3d(var_interped, mask=ocnmask.astype(bool))
    end_time = time.time()
    print('\tNaNs inpainted: ' + str(round(end_time - start_time,3)) + ' s')

    # save regridded data
    #np.save(output_path + var_name + '_averaged_regridded_inpainted.npy', var_inpainted)
    np.save(data_path + 'COBALT_regridded/' + var_name + '.npy', var_inpainted)
    print('\tfinal regridded array saved')

def interp_TRACE(data_path, time, scenario, model_lat, model_lon, model_depth, ocnmask):
    '''
    Given a time and an emissions scenario, interpolate TRACE gridded product
    (downloaded from https://zenodo.org/records/15003059) to return a 3-D array
    of Canth matching OCIM grid.

    If time is before 2020, historical/none scenario is used. If time is
    after 2020, one of the following emissions scenarios must be specified:
    ssp119, ssp126, ssp245, ssp370, ssp370_lowNTCF, ssp434, ssp460, ssp534_OS	

    Parameters
    ----------
    data_path = path to folder which contains TRACE gridded product data
    time = time, in decimal years, of interest
    scenario = emissions scenario of interest ('historical', 'ssp119', 'ssp126',
               'ssp245', 'ssp370', 'ssp370_lowNTCF', 'ssp434', 'ssp460', 'ssp534_OS')
    model_depth = array of model depth levels
    model_lon = array of model longitudes
    model_lat = array of model latitudes
    ocnmask = mask with shape len(model_depth) x len(model_lon) x len(model_lat) where
              1 marks an ocean cell and 0 marks land
    
    Returns
    ----------
    canth = array with same shape as ocnmask that has values of canth at each grid cell
            interpolated from TRACE gridded product given a time of interest and an emissions scenario

    '''
    # pull correct data from gridded dataset
    scenarios = {'none' : 1, 'ssp119' : 2, 'ssp126' : 3, 'ssp245' : 4, 'ssp370' : 5,
                 'ssp370_lowNTCF' : 6,'ssp434' : 7,'ssp460' : 8,'ssp534_OS' : 9}

    if scenario not in list(scenarios.keys()):
        raise ValueError(f"Invalid value: {scenario!r}. Must be one of: {', '.join(list(scenarios.keys()))}")

    if time > 2500:
        print('Warning: TRACE data only is available to 2500. Interpolating past this point is not advised.')

    if time >= 2020:
        if scenario == 'none':
            raise ValueError("'none' scenario chosen, but time > 2020 selected.")
        trace_data = xr.open_dataset(data_path + 'TRACE_gridded/CanthFromTRACECO2Pathway' + str(scenarios[scenario]) + '.nc', decode_times=False)    
    else:
        trace_data = xr.open_dataset(data_path + 'TRACE_gridded/CanthFromTRACECO2Pathway1.nc', decode_times=False)    

    # interpolate to time of interest
    trace_data = trace_data.interp(time=time)

    # pull out arrays of depth, latitude, and longitude from TRACE gridded product
    trace_lat = trace_data['lat'].to_numpy()     # ºN
    trace_lon = trace_data['lon'].to_numpy()     # ºE
    trace_depth = trace_data['depth'].to_numpy() # m below sea surface

    # pull out values of DIC and TA from TRACE, switch order of TRACE dimensions to match OCIM dimensions
    canth = trace_data['canth'].transpose('lat', 'lon', 'depth').copy().values
    
    # create interpolator
    interp = RegularGridInterpolator((trace_lat, trace_lon, trace_depth), canth, bounds_error=False, fill_value=None)

    # transform model_lon for anything < 20 (because TRACE goes from 20ºE - 380ºE)
    model_lon[model_lon < 20] += 360

    # create meshgrid for OCIM grid
    lat, lon, depth = np.meshgrid(model_lat, model_lon, model_depth, indexing='ij')

    # reshape meshgrid points into a list of coordinates to interpolate to
    query_points = np.array([lat.ravel(), lon.ravel(), depth.ravel()]).T
    
    # perform interpolation (regrid TRACE data to match OCIM grid)
    canth = interp(query_points)

    # transform results back to model grid shape
    canth = canth.reshape(depth.shape)

    # inpaint nans
    canth = inpaint_nans3d(canth, mask=ocnmask.astype(bool))

    # transform model_lon and meshgrid back for anything > 360
    model_lon[model_lon > 360] -= 360
    
    return canth

def schmidt(gas, temperature):
    '''
    Calculate the Schmidt number (Sc) for a given gas in seawater based on Wanninkhof (2014).
    
    Parameters
    ----------
    gas : the gas of interest (e.g., 'O2', 'CO2', 'N2', 'Ar').
    temperature : seawater temperature in degrees Celsius (°C).
    
    Returns
    -------
    Sc : Schmidt number (Sc)
    '''
    # Schmidt number coefficients from Wanninkhof (2014) for seawater from -2ºC to 40ºC
    sc_coeffs = {
        'O2':  [1920.4, -135.6, 5.2122, -0.10939, 0.00093777],  # oxygen
        'CO2': [2116.8, -136.25, 4.7353, -0.092307, 0.0007555], # carbon dioxide
        'N2':  [2403.8, -162.75, 6.2557, -0.13129, 0.0011255],  # nitrogen
        'Ar':  [2078.1, -146.74, 5.6403, -0.11838, 0.0010148]   # argon
    }

    if gas not in sc_coeffs:
        raise ValueError(f"Gas '{gas}' not supported. Choose from {list(sc_coeffs.keys())}")

    a, b, c, d, e = sc_coeffs[gas]
    
    # Compute Schmidt number
    Sc = a + (b * temperature) + (c * temperature**2) + (d * temperature**3) + (e * temperature**4)
    
    return Sc

def calculate_AT_to_add(pH_preind, DIC, AT, T, S, pressure, Si, P, AT_mask=None, low=0, high=200, tol=1e-6, maxiter=50):
    '''
    Calculate the amount of alkalinity to add to the surface ocean in order to
    reach preindustrial pH in the surface. This alkalinity will be between
    "low" and "high", which right now is 0 and 200 µmol kg-1

    Parameters
    ----------
    pH : Array of floats.
        Preindustrial pH values at each ocean grid cell [unitless]
    DIC : Array of floats.
        Present-day dissolved inorganic carbon values at each ocean grid cell
        [µmol kg-1]
    AT : Array of floats.
        Present-day alkalinity values at each ocean grid cell [µmol kg-1]
    T : Array of floats.
        Present-day temperature values at each ocean grid cell [ºC]
    S : Array of floats.
        Present-day salinity values at each ocean grid cell [unitless]
    pressure : Array of floats.
        Present-day pressure values at each ocean grid cell [dbar]
    Si : Array of floats.
        Present-day total silicate values at each ocean grid cell [µmol kg-1]
    P : Array of floats.
        Present-day total phosphate values at each ocean grid cell [µmol kg-1].
    AT_mask : Mask same shape as pH/DIC/AT, etc. where 1 marks an cell that
        recieves AT and 0 marks cells that do not get AT (i.e. scenario in
        which AT is added to surface only)
    low : Float.
        Initial lower bound for output (AT_to_offset)
    high : Float.
        Initial upper bound for output (AT_to_offset)
    tol : Float.
        Tolerance for convergence [µmol kg-1]
    maxiter: Int.
        Maximum numer of iterations

    Returns
    -------
    AT_to_offset: Array of floats.
        Amount of AT to apply at each present-day ocean grid cell in order to
        reach preindustrial pH in the surface ocean [µmol kg-1]
    '''
    # check which grid cells have pH < pH_preind
    co2sys_init = pyco2.sys(dic=DIC, alkalinity=AT, salinity=S,
                            temperature=T, pressure=pressure, 
                            total_silicate=Si, total_phosphate=P)
    pH_init = co2sys_init['pH']
    
    # mask for grid cells that will have AT added
    gets_AT = pH_init < pH_preind
    
    # combine with AT_mask if provided
    if AT_mask is not None:
        gets_AT = gets_AT & (AT_mask == 1)

    # initialize arrays representing low and high guesses
    AT_to_offset = np.zeros_like(AT, dtype=float) # to record amount of AT to add
    low_arr = np.full(np.count_nonzero(gets_AT), low, dtype=float)
    high_arr = np.full(np.count_nonzero(gets_AT), high, dtype=float)

    # extract only cells that need AT for faster processing
    DIC_gets_AT = DIC[gets_AT]
    AT_gets_AT = AT[gets_AT]
    T_gets_AT = T[gets_AT]
    S_gets_AT = S[gets_AT]
    pressure_gets_AT = pressure[gets_AT]
    Si_gets_AT = Si[gets_AT]
    P_gets_AT = P[gets_AT]
    pH_preind_gets_AT = pH_preind[gets_AT]
    
    # iterate through solve
    for it in range(maxiter):
        mid_arr = 0.5 * (low_arr + high_arr)

        # compute pH for midpoints
        co2sys_mid = pyco2.sys(dic=DIC_gets_AT, alkalinity=AT_gets_AT+mid_arr,
                           salinity=S_gets_AT, temperature=T_gets_AT,
                           pressure=pressure_gets_AT,
                           total_silicate=Si_gets_AT,
                           total_phosphate=P_gets_AT)

        f_mid = co2sys_mid['pH'] - pH_preind_gets_AT

        # evaluate sign at low bound
        co2sys_low = pyco2.sys(dic=DIC_gets_AT, alkalinity=AT_gets_AT+low_arr,
                               salinity=S_gets_AT, temperature=T_gets_AT,
                               pressure=pressure_gets_AT,
                               total_silicate=Si_gets_AT,
                               total_phosphate=P_gets_AT)

        f_low = co2sys_low['pH'] - pH_preind_gets_AT

        # update brackets
        same_sign = (f_mid * f_low) > 0
        low_arr[same_sign] = mid_arr[same_sign]
        high_arr[~same_sign] = mid_arr[~same_sign]

        # check convergence
        if np.all((high_arr - low_arr) < tol):
            break
    # assign results back to full grid
    AT_to_offset[gets_AT] = 0.5 * (low_arr + high_arr)
    
    return AT_to_offset

def calculate_canth(scenario, year, T_3D, S_3D, ocnmask, model_lat, model_lon, model_depth):
    """
    calculates anthropogenic carbon stored in the ocean relative to a preindustrial
    (280 ppm) baseline using pyTRACE.

    Parameters
    ----------
    scenario : string of the following choices: 'none', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp460_NTCF', 'ssp434', 'ssp460', 'ssp534_OS'
    year : year to calculate anthropogenic carbon at
    T_3D : array of temperatures [ºC] of shape len(model_depth), len(model_lon), len(model_lon)
    S_3D : array of salinities of shape len(model_depth), len(model_lon), len(model_lon)
    ocnmask : mask of shape len(model_depth), len(model_lon), len(model_lon) where 1 marks an ocean cell and 0 marks land
    model_lat : array pf model latitudes
    model_lon : array of model longitudes
    model_depth : array of model depth levels
    
    Returns
    ----------
    Canth_3D : anthropogenic carbon [µmol kg-1] at each combination of depth/lat/lon provided.
    """

    # truncate year to nearest whole number

    if year > 2015 and scenario == 'none':
        raise ValueError('error: future year chosen (after 2015) but no emissions scenario selected')

    # set up emissions scenario
    # note: none = no perturbation to atmospheric co2, NOT trace historical scenario. error is raised
    # if projections forward are attempted with none. GLODAP 2002 data is used as baseline and no
    # change to Revelle factors due to emissions are included. Other scenarioees follow Meinshausen et al. (2020)
    scenario_dict = {'none' : 2, 'ssp119': 2, 'ssp126' : 3, 'ssp245' : 4, 'ssp370' : 5,
                     'ssp360_NTCF' : 6, 'ssp434' : 7, 'ssp460' : 8, 'ssp534_OS' : 9}

    # transpose to match requirements for PyTRACE (lon, lat, depth)
    T_3D_T = T_3D.transpose([1, 0, 2])
    S_3D_T = S_3D.transpose([1, 0, 2])
    ocnmask_T = ocnmask.transpose([1, 0, 2])

    predictor_measurements = np.vstack([S_3D_T.flatten(order='F'), T_3D_T.flatten(order='F')]).T
    predictor_measurements = predictor_measurements[ocnmask_T.flatten(order='F').astype(bool)]
    
    # create list of longitudes (ºE), latitudes (ºN), and depths (m) in TRACE format
    # this order is required for TRACE
    lon, lat, depth = np.meshgrid(model_lon, model_lat, model_depth, indexing='ij')
    ocim_coordinates = np.array([lon.ravel(order='F'), lat.ravel(order='F'), depth.ravel(order='F'), ]).T # reshape meshgrid points into a list of coordinates to interpolate to
    ocim_coordinates = ocim_coordinates[ocnmask_T.flatten(order='F').astype(bool)]

    dates = year * np.ones([ocim_coordinates.shape[0],1])

    trace_output = trace(output_coordinates=ocim_coordinates,
                         dates=dates[:,0],
                         predictor_measurements=predictor_measurements,
                         predictor_types=[1, 2],
                         atm_co2_trajectory=scenario_dict[scenario],
                         verbose_tf=False)
    
    # right now, pyTRACE is estimating some preformed P and Si as <0, which is
    # resulting in NaN Canth. for now, am fixing this by setting <0 values =0.
    # also pulling preformed AT and scale factors to make second pyTRACE
    # calculation faster
    preformed_p = trace_output.preformed_p.values
    preformed_p[preformed_p < 0] = 0
    preformed_ta = trace_output.preformed_ta.values
    preformed_si = trace_output.preformed_si.values
    preformed_si[preformed_si < 0] = 0
    scale_factors = trace_output.scale_factors.values

    trace_output = trace(output_coordinates=ocim_coordinates,
                         dates=dates[:,0],
                         predictor_measurements=predictor_measurements,
                         predictor_types=[1, 2],
                         atm_co2_trajectory=scenario_dict[scenario],
                         preformed_p=preformed_p,
                         preformed_ta=preformed_ta,
                         preformed_si=preformed_si,
                         scale_factors=scale_factors,
                         verbose_tf=False)
    
    Canth_3D = make_3D(trace_output.canth.values, ocnmask_T).transpose([1, 0, 2])

    return Canth_3D

def get_CO2_scenario(scenario, times):
    """
    Finds atmospheric CO2 concentrations over the historical record and a
    selected shared socioeconomic pathway (SSP). Returns time in units of years
    and emissions in units of mol CO2 (mol air-1). Scenarios available are
    'none', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp370_NTCF', 'ssp434',
    'ssp460', and 'ssp534_OS'. Data is from pyTRACE CO2TrajectoriesAdjusted.txt,
    which is originally from:
    https://greenhousegases.science.unimelb.edu.au/#!/ghg?mode=downloads

    note: historical data is different from SSPs by < 1 ppm even before SSPs
    diverge starting in 2016 
    
    Parameters
    ----------
    scenario: String
        name of historical or future emissions scenario of interest
        ('none', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp370_NTCF',
        'ssp434', 'ssp460', 'ssp534_OS')
    times: 1-D array of floats
        array of times (in decimal years CE) of interest
        
    Returns
    ----------
    atmospheric_CO2: 1-D array of floats
        cumulative amount of emissions in µmol CO2 (mol air)-1 (aka ppm) since year 0 CE
    """

    scenarios = {'none' : 1, 'ssp119' : 2, 'ssp126' : 3, 'ssp245' : 4, 'ssp370' : 5,
                'ssp370_lowNTCF' : 6,'ssp434' : 7,'ssp460' : 8,'ssp534_OS' : 9}

    if scenario not in list(scenarios.keys()):
        raise ValueError(f"Invalid value: {scenario!r}. Must be one of: {', '.join(list(scenarios.keys()))}")

    # open file, pull out scenario of interest

    data_file = './src/utils/pyTRACE/pyTRACE/data/CO2TrajectoriesAdjusted.txt'
    data = np.loadtxt(data_file)

    CO2_data_years = data[:,0]
    CO2_data = data[:,scenarios[scenario]]
    print(len(CO2_data_years))
    print(len(CO2_data))

    if scenario != 'none':
        # interpolate for times of interest
        atmospheric_CO2 = np.interp(times, CO2_data_years, CO2_data)

    else:
        if times[0] >= 2020:
            raise ValueError("'none' scenario chosen, but time > 2020 selected.")
        
        # return constant atmospheric CO2 based on start year & historical scenario
        atmospheric_CO2 = np.interp(times[0], CO2_data_years, CO2_data) * np.ones_like(times)

    return atmospheric_CO2

def plot_surface2d(lats, lons, variable, vmin, vmax, cmap, title):
    
    # mask out zero values 
    variable_masked = np.ma.masked_where(variable == 0, variable)

    # create colormap copy, set masked to black
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color='black')
    
    # main plot
    fig = plt.figure(figsize=(10,7))
    #fig = plt.figure(figsize=(8,4))
    ax = fig.gca()
    levels = np.linspace(vmin-0.1, vmax, 100)
    cntr = plt.contourf(lons, lats, variable_masked, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    c = plt.colorbar(cntr, ax=ax)
    c.set_ticks(np.round(np.linspace(vmin, vmax, 10),2))
    #c.set_label('mol DIC m$^{-2}$ yr$^{-1}$', fontsize=12)

    # overlay black for land
    #zero_mask = (variable == 0).astype(float)
    #ax.contourf(lons, lats, zero_mask.T, levels=[0.5, 1.5], colors='black', alpha=1.0)
    
    plt.xlabel('longitude (ºE)')
    plt.ylabel('latitude (ºN)')
    #plt.xlabel('Longitude (ºE)')
    #plt.ylabel('Latitude (ºN)')
    plt.title(title)
    plt.xlim([0, 360]), plt.ylim([-90,90])


def plot_surface3d(lats, lons, variable, depth_level, vmin, vmax, cmap, title, logscale=None, lon_lims=None):
    fig = plt.figure(figsize=(10,7), dpi=200)
    ax = fig.gca()
    
    if logscale:
        #levels = np.logspace(vmin, vmax, 100)
        cntr = plt.contourf(lons, lats, variable[:, :, depth_level], norm=LogNorm(), cmap=cmap, vmin=vmin, vmax=vmax) # log scale
    else:
        levels = np.linspace(vmin-1e-7, vmax, 100)
        cntr = plt.contourf(lons, lats, variable[:, :, depth_level], levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    
    c = plt.colorbar(cntr, ax=ax)
    c.set_ticks(np.round(np.linspace(vmin, vmax, 10),2))
    plt.xlabel('longitude (ºE)')
    plt.ylabel('latitude (ºN)')
    plt.title(title)
    
    if lon_lims==None:
        plt.xlim([0, 360]), plt.ylim([-90,90])
    else:
        plt.ylim([-90,90])
        plt.xlim(lon_lims)
        
    return fig

def plot_longitude3d(lats, depths, variable, longitude, vmin, vmax, cmap, title):
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca()
    levels = np.linspace(vmin-1e-7, vmax, 100)
    cntr = plt.contourf(lats, depths, variable[:, longitude, :].T, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    c = plt.colorbar(cntr, ax=ax)
    c.set_ticks(np.round(np.linspace(vmin, vmax, 10),2))
    ax.invert_yaxis()
    plt.xlabel('latitude (ºN)')
    plt.ylabel('depth (m)')
    plt.title(title)
    plt.xlim([-90, 90]), plt.ylim([depths.max(), 0])
    
def build_lme_masks(shp_path, ocnmask, lats, lons):
    lmes = gpd.read_file(shp_path)
    if lmes.crs != "EPSG:4326":
        lmes = lmes.to_crs(epsg=4326)

    # Convert lons for spatial check
    lons_for_test = ((lons + 180) % 360) - 180
    lon_grid, lat_grid = np.meshgrid(lons_for_test, lats)

    points = [Point(lon, lat) for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]
    points_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

    lme_id_grid = np.zeros(lon_grid.shape, dtype=int).T
    lme_id_to_name = {}
    lme_masks = {}

    for idx, row in lmes.iterrows():
        lme_id = int(row["LME_NUMBER"])
        name = row["LME_NAME"]
        mask_flat = points_gdf.within(row.geometry)
        mask = mask_flat.to_numpy().reshape(lat_grid.shape).T
        
        mask = np.logical_and(mask, ocnmask[:, :, 0].astype(bool))
        
        # manually mask out point (177,76) from LME 22 (because it overlaps with LME 60)
        if lme_id == 22:
            mask[177,76] = False
            
        # manually mask in holes in regions created with grid conversion
        if lme_id == 3:
            mask[119,63] = True
        elif lme_id == 9:
            mask[152:154,69] = True
        elif lme_id == 11:
            mask[131,53] = True
        elif lme_id == 12:
            mask[138,56] = True
        elif lme_id == 13:
            mask[141,39] = True
            mask[144,36] = True
            mask[144,31] = True
        elif lme_id == 17:
            mask[152,48] = True
        elif lme_id == 18:
            mask[146,80] = True
            mask[148,79] = True
            mask[153,81] = True
            mask[154,77] = True
        elif lme_id == 20:
            mask[7,84] = True
            mask[9,85] = True
            mask[30,86] = True
        elif lme_id == 25:
            mask[175,65] = True
            mask[176,67] = True
        elif lme_id == 26:
            mask[176,64] = True
            mask[177:180,63:65] = True
            mask[0:7,63:65] = True
            mask[5,62] = True
        elif lme_id == 27:
            mask[173,59] = True
        elif lme_id == 28:
            mask[0,48] = True
        elif lme_id == 32:
            mask[25,51] = True
        elif lme_id == 34:
            mask[47,53] = True
        elif lme_id == 36:
            mask[52,50] = True
        elif lme_id == 37:
            mask[60,48] = True
        elif lme_id == 39:
            mask[65,39] = True
        elif lme_id == 43:
            mask[69,27] = True
        elif lme_id == 45:
            mask[57,34] = True
        elif lme_id == 54:
            mask[89,81] = True
            mask[95,77] = True
            mask[101,81] = True
        elif lme_id == 58:
            mask[47,85:87] = True
            mask[45,86] = True
            mask[32,80] = True
            mask[33,81] = True
        elif lme_id == 59:
            mask[169,78] = True
        elif lme_id == 61:
            mask[0,9] = True
            mask[1:15,10] = True
            mask[16,11] = True
            mask[17,10] = True
            mask[20:25,11] = True
            mask[25:29,12] = True
            mask[29,11] = True
            mask[34,10] = True
            mask[38,10] = True
            mask[41,11] = True
            mask[42:46,12] = True
            mask[51:72,12] = True
            mask[72:79,11] = True
            mask[81:84,10] = True
            mask[84,8] = True
            mask[81,6] = True
            mask[102,6] = True
            mask[104:107,7] = True
            mask[110:129,8] = True
            mask[168,8] = True
            mask[170:176,9] = True
            mask[176:179,10] = True
        elif lme_id == 66:
            mask[123:125,83] = True
            mask[127,85] = True
            mask[133:137,86] = True
            mask[142,85] = True
            mask[144,84] = True
            mask[144,87] = True
            mask[157,87] = True
            
        if np.any(mask):
            lme_id_grid[mask] = lme_id
            lme_masks[lme_id] = mask
            lme_id_to_name[lme_id] = name

    return lme_masks, lme_id_to_name
    
def plot_lmes(lme_masks, ocnmask, lats, lons):
    # convert lons to -180 to 180 for plotting
    lons_shifted = np.where(lons > 180, lons - 360, lons)
    
    # create an array to hold lme ids
    id_grid = np.full((len(lons), len(lats)), np.nan)
    centers = [] # center of each LME

    # assign each mask a numeric id
    for idx, (lme_id, mask) in enumerate(lme_masks.items(), start=1):
        id_grid[mask] = int(lme_id)
        
        # calculate geographic center for label
        if np.any(mask):
            lat_center = np.mean(lats[np.any(mask, axis=0)])
            lon_center = np.mean(lons[np.any(mask, axis=1)])
            if lon_center > 180:
                lon_center -= 360
                
            centers.append((lon_center, lat_center, int(lme_id)))

    # set up plot
    fig = plt.figure(figsize=(14, 8), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_global()
    
    # plot land with ocnmask
    mesh = ax.pcolormesh(
        lons_shifted, lats, ocnmask[0, :, :].T,
        transform=ccrs.PlateCarree(),
        cmap='Greys_r', shading='nearest'
    )

    # create discrete colormap
    hsv_colors = [(i / len(lme_masks), 0.75, 0.85) for i in range(len(lme_masks)+4)]
    rgb_colors = [mcolors.hsv_to_rgb(c) for c in hsv_colors]
    random.Random(48).shuffle(rgb_colors)
    cmap = mcolors.ListedColormap(rgb_colors)
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(0.5, len(lme_masks) + 4 + 1.5, 1),
        ncolors=len(lme_masks)+4
    )

    # plot each lme
    mesh = ax.pcolormesh(
        lons_shifted, lats, id_grid.T,
        transform=ccrs.PlateCarree(),
        cmap=cmap, alpha=0.8, norm=norm, shading='nearest'
    )
    
    for lon_c, lat_c, idx in centers:
        if idx == 2:
            ax.text(lon_c, lat_c+2, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 3:
            ax.text(lon_c-1, lat_c-2, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 11:
            ax.text(lon_c+1, lat_c+2, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 19:
            ax.text(-6, lat_c+5, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 21:
            ax.text(3, lat_c-2, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 22:
            ax.text(3, lat_c, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 26:
            ax.text(18, lat_c, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 27:
            ax.text(lon_c-5, lat_c, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 28:
            ax.text(353, lat_c, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 29:
            ax.text(lon_c-4, lat_c, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 30:
            ax.text(lon_c+5, lat_c, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 39:
            ax.text(lon_c, lat_c+3, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 46:
            ax.text(lon_c-4, lat_c+2, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 49:
            ax.text(lon_c+5, lat_c, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 51:
            ax.text(lon_c+2, lat_c-2, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 61:
            ax.text(310, lat_c-3, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 64:
            ax.text(174, lat_c+1, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 65:
            ax.text(lon_c+6, lat_c-4, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        elif idx == 66:
            ax.text(lon_c-18, lat_c-1, str(idx), transform=ccrs.PlateCarree(), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        else:
            ax.text(lon_c, lat_c, str(idx),
                    transform=ccrs.PlateCarree(),
                    fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
            
    plt.title("Large Marine Ecosystems (62 out of 66 can be represented on OCIM grid)")
    #plt.xlim([-190, 190])
    plt.show()

def make_surf_animation(variable, colorbar_label, model_lat, model_lon, t, nt, vmin, vmax, cmap, filename):
    fig, ax = plt.subplots(figsize=(10,7))
    
    # first frame of animation
    cntr = ax.contourf(model_lon, model_lat,
                       variable.isel(time=0).values[:,:,0],
                       levels=np.linspace(vmin, vmax, 100),
                       cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cntr, ax=ax,label=colorbar_label)
    ax.set_xlabel('Longitude (ºE)')
    ax.set_ylabel('Latitude (ºN)')
    title = ax.set_title('t = ' + str(np.round(t[0],3)) + ' yr')

    # update function: this updates each frame with the new "axis", which is the subsequent contour plot
    def update_frame(idx):
        ax.clear()
        ax.contourf(model_lon, model_lat,
                    variable.isel(time=idx).values[:,:,0],
                    levels=np.linspace(vmin, vmax, 100),
                    cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Longitude (ºE)')
        ax.set_ylabel('Latitude (ºN)')
        ax.set_title('t = ' + str(np.round(t[idx],3)) + ' yr')
        return []
    
    # make and save animation
    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=100, blit=False)
    writer = animation.writers['ffmpeg'](fps=10)
    ani.save(filename, writer=writer, dpi=200)
    
def make_surf_animation_pH(pH, colorbar_label, model_lon, model_lat, t, nt, ocnmask, vmin, vmax, cmap, filename):
    fig, ax = plt.subplots(figsize=(10,7))
    
    # first frame of animation
    pH_3D = make_3D(pH[0], ocnmask)
    
    cntr = ax.contourf(model_lon, model_lat,
                       pH_3D[:,:,0],
                       levels=np.linspace(vmin, vmax, 100),
                       cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cntr, ax=ax,label=colorbar_label)
    ax.set_xlabel('Longitude (ºE)')
    ax.set_ylabel('Latitude (ºN)')
    title = ax.set_title('t = ' + str(np.round(t[0],3)) + ' yr')

    # update function: this updates each frame with the new "axis", which is the subsequent contour plot
    def update_frame(idx):
        ax.clear()
        pH_3D = make_3D(pH[idx], ocnmask)
        ax.contourf(model_lon, model_lat,
                    pH_3D[:,:,0],
                    levels=np.linspace(vmin, vmax, 100),
                    cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Longitude (ºE)')
        ax.set_ylabel('Latitude (ºN)')
        ax.set_title('t = ' + str(np.round(t[idx],3)) + ' yr')
        return []
    
    # make and save animation
    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=100, blit=False)
    writer = animation.writers['ffmpeg'](fps=10)
    ani.save(filename, writer=writer, dpi=200)
    
def make_section_animation(variable, colorbar_label, model_depth, model_lat, t, nt, vmin, vmax, cmap, filename):
    fig, ax = plt.subplots(figsize=(10,7))
    
    # first frame of animation
    cntr = ax.contourf(model_lat, model_depth,
                       variable.isel(time=0).values[:,90,:].T,
                       levels=np.linspace(vmin, vmax, 100),
                       cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cntr, ax=ax,label=colorbar_label)
    ax.invert_yaxis()
    ax.set_xlabel('Latitude (ºN)')
    ax.set_ylabel('Depth (m)')
    title = ax.set_title('t = ' + str(np.round(t[0],3)) + ' yr at 181ºE')

    
    # update function: this updates each frame with the new "axis", which is the subsequent contour plot
    def update_frame(idx):
        ax.clear()
        ax.contourf(model_lat, model_depth,
                    variable.isel(time=idx).values[:,90,:].T,
                    levels=np.linspace(vmin, vmax, 100),
                    cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_xlabel('Latitude (ºN)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('t = ' + str(np.round(t[idx],3)) + ' yr at 181 ºE')
        return []
    
    # make and save animation
    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=100, blit=False)
    writer = animation.writers['ffmpeg'](fps=10)
    ani.save(filename, writer=writer, dpi=200)
    
def make_section_animation_pH(pH, colorbar_label, model_depth, model_lat, t, nt, ocnmask, vmin, vmax, cmap, filename):
    fig, ax = plt.subplots(figsize=(10,7))
    
    # first frame of animation
    pH_3D = make_3D(pH[0], ocnmask)
    cntr = ax.contourf(model_lat, model_depth,
                       pH_3D[:,90,:].T,
                       levels=np.linspace(vmin, vmax, 100),
                       cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cntr, ax=ax,label=colorbar_label)
    ax.invert_yaxis()
    ax.set_xlabel('Latitude (ºN)')
    ax.set_ylabel('Depth (m)')
    title = ax.set_title('t = ' + str(np.round(t[0],3)) + ' yr at 181ºE')

    
    # update function: this updates each frame with the new "axis", which is the subsequent contour plot
    def update_frame(idx):
        pH_3D = make_3D(pH[idx], ocnmask)
        ax.clear()
        ax.contourf(model_lat, model_depth,
                    pH_3D[:,90,:].T,
                    levels=np.linspace(vmin, vmax, 100),
                    cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_xlabel('Latitude (ºN)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('t = ' + str(np.round(t[idx],3)) + ' yr at 181 ºE')
        return []
    
    # make and save animation
    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=100, blit=False)
    writer = animation.writers['ffmpeg'](fps=10)
    ani.save(filename, writer=writer, dpi=200)