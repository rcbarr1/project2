#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions supporting project2_main.py

Created on Tue Oct  8 11:15:51 2024

@author: Reese Barrett
"""

import scipy.io as spio
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import xarray as xr
import h5netcdf
import datetime as dt

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
    
def save_model_output(filename, depth, lon, lat, time, tracers, tracer_names=None, tracer_units=None, global_attrs=None):
    '''
    Save model output to a NetCDF file.
    
    Parameters
    ----------
    filename (str): Name of the NetCDF file to create.
    depth (list or array): Depth values.
    lon (list or array): Longitude values.
    lat (list or array): Latitude values.
    time (list or array): Time values (e.g., in years).
    tracers (list of 4D arrays): List of 4D arrays (time x depth x lon x lat) for each tracer.
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

    with h5netcdf.File(filename, 'w', invalid_netcdf=True) as ncfile:
        # Create dimensions
        ncfile.dimensions['time'] = len(time)
        ncfile.dimensions['depth'] = len(depth)
        ncfile.dimensions['lon'] = len(lon)
        ncfile.dimensions['lat'] = len(lat)

        # Create coordinate variables
        nc_time = ncfile.create_variable('time', ('time',), dtype='f8')
        nc_depth = ncfile.create_variable('depth', ('depth',), dtype='f8')
        nc_lon = ncfile.create_variable('lon', ('lon',), dtype='f8')
        nc_lat = ncfile.create_variable('lat', ('lat',), dtype='f8')

        # Set units and descriptions for coordinate variables
        nc_time.attrs['units'] = 'years'
        nc_depth.attrs['units'] = 'meters'
        nc_lon.attrs['units'] = 'degrees_east'
        nc_lat.attrs['units'] = 'degrees_north'

        # Write coordinate data
        nc_time[:] = time
        nc_depth[:] = depth
        nc_lon[:] = lon
        nc_lat[:] = lat

        # Create and write tracer variables
        for tracer_name, tracer_data, tracer_unit in zip(tracer_names, tracers, tracer_units):
            tracer_var = ncfile.create_variable(tracer_name, ('time', 'depth', 'lon', 'lat'), dtype='f8')
            tracer_var.attrs['description'] = f"{tracer_name} data"
            tracer_var.attrs['units'] = tracer_unit

            # Write 4D tracer data
            tracer_var[:, :, :, :] = tracer_data

        # Add global attributes
        ncfile.attrs['history'] = 'Created ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ncfile.attrs['source'] = 'Python script'
        if global_attrs:
            for key, value in global_attrs.items():
                ncfile.attrs[key] = value

def inpaint_nans3d(array_3d, mask=None, iterations=100):
    '''
    adapted from https://stackoverflow.com/questions/73206073/interpolation-of-missing-values-in-3d-data-array-in-python
    to incorporate nan mask so inpainting doesn't happen over land 
    '''
    # dimensions of input
    size = array_3d.shape

    # get index of nan in corrupted data
    nan_mask = np.isnan(array_3d)
    
    # If a mask is provided, combine it with the NaN mask
    if mask is not None:
        nan_mask = nan_mask & mask
        
    nanIndex = nan_mask.nonzero()

    interpolatedData = array_3d.copy()

    # make an initial guess for the interpolated data using the mean of the non NaN values
    interpolatedData[nanIndex] = np.nanmean(array_3d)

    def sign(index, max_index):
        # pass additional max_index to this func as this is now a variable
        if index == 0:
            return [1, 0]
        elif index == max_index - 1:
            return [-1, 0]
        else:
            return [-1, 1]

    # calculate the sign for each dimension separately
    nanIndexX, nanIndexY, nanIndexZ = nanIndex[0], nanIndex[1], nanIndex[2]
    signsX = np.array([sign(x, size[0]) for x in nanIndexX])
    signsY = np.array([sign(y, size[1]) for y in nanIndexY])
    signsZ = np.array([sign(z, size[2]) for z in nanIndexZ])
            
    for _ in range(iterations):
        for i in range(len(nanIndexX)):
            x, y, z = nanIndexX[i], nanIndexY[i], nanIndexZ[i]
            dx, dy, dz = signsX[i], signsY[i], signsZ[i]

            neighbors = []
            if dx[0] != 0:  # can average with the previous x neighbor
                neighbors.append(interpolatedData[np.clip(x + dx[0], 0, size[0] - 1), y, z])
            if dx[1] != 0:  # can average with the next x neighbor
                neighbors.append(interpolatedData[np.clip(x + dx[1], 0, size[0] - 1), y, z])

            if dy[0] != 0:
                neighbors.append(interpolatedData[x, np.clip(y + dy[0], 0, size[1] - 1), z])
            if dy[1] != 0:
                neighbors.append(interpolatedData[x, np.clip(y + dy[1], 0, size[1] - 1), z])

            if dz[0] != 0:
                neighbors.append(interpolatedData[x, y, np.clip(z + dz[0], 0, size[2] - 1)])
            if dz[1] != 0:
                neighbors.append(interpolatedData[x, y, np.clip(z + dz[1], 0, size[2] - 1)])

            # average the neighbors to interpolate the NaN value
            interpolatedData[x, y, z] = np.nanmean(neighbors)
    
    return interpolatedData

def inpaint_nans2d(array_2d, mask=None, iterations=100):
    '''
    adapted from https://stackoverflow.com/questions/73206073/interpolation-of-missing-values-in-3d-data-array-in-python
    to incorporate nan mask so inpainting doesn't happen over land 
    '''
    # dimensions of input
    size = array_2d.shape

    # get index of nan in corrupted data
    nan_mask = np.isnan(array_2d)
    
    # If a mask is provided, combine it with the NaN mask
    if mask is not None:
        nan_mask = nan_mask & mask
        
    nanIndex = nan_mask.nonzero()

    interpolatedData = array_2d.copy()

    # make an initial guess for the interpolated data using the mean of the non NaN values
    interpolatedData[nanIndex] = np.nanmean(array_2d)

    def sign(index, max_index):
        # pass additional max_index to this func as this is now a variable
        if index == 0:
            return [1, 0]
        elif index == max_index - 1:
            return [-1, 0]
        else:
            return [-1, 1]

    # calculate the sign for each dimension separately
    nanIndexX, nanIndexY = nanIndex[0], nanIndex[1]
    signsX = np.array([sign(x, size[0]) for x in nanIndexX])
    signsY = np.array([sign(y, size[1]) for y in nanIndexY])
            
    for _ in range(iterations):
        for i in range(len(nanIndexX)):
            x, y = nanIndexX[i], nanIndexY[i]
            dx, dy = signsX[i], signsY[i]

            neighbors = []
            if dx[0] != 0:  # can average with the previous x neighbor
                neighbors.append(interpolatedData[np.clip(x + dx[0], 0, size[0] - 1), y])
            if dx[1] != 0:  # can average with the next x neighbor
                neighbors.append(interpolatedData[np.clip(x + dx[1], 0, size[0] - 1), y])

            if dy[0] != 0:
                neighbors.append(interpolatedData[x, np.clip(y + dy[0], 0, size[1] - 1)])
            if dy[1] != 0:
                neighbors.append(interpolatedData[x, np.clip(y + dy[1], 0, size[1] - 1)])

            # average the neighbors to interpolate the NaN value
            interpolatedData[x, y] = np.nanmean(neighbors)
    
    return interpolatedData

def regrid_glodap(data_path, glodap_var, model_depth, model_lat, model_lon, ocnmask):
    '''
    regrid glodap data to model grid, inpaint nans, save as .npy file

    Parameters
    ----------
    data_path : path to folder which contains GLODAPv2.2016b.MappedProduct folder from https://glodap.info/index.php/mapped-data-product/
    glodap_var : variable to regrid as named in GLODAP (dimensions: depth, longitude, latitude)
    model_depth : array of model depth levels
    model_lat : array pf model latitudes
    model_lon : array of model longitudes
    ocnmask : mask same shape as glodap_var when 1 marks an ocean cell and 0 marks land

    '''
    
    # load GLODAP data (https://glodap.info/index.php/mapped-data-product/)
    glodap_data = xr.open_dataset(data_path + 'GLODAPv2.2016b.MappedProduct/GLODAPv2.2016b.' + glodap_var + '.nc')

    # pull out arrays of depth, latitude, and longitude from GLODAP
    glodap_depth = glodap_data['Depth'].to_numpy() # m below sea surface
    glodap_lon = glodap_data['lon'].to_numpy()     # ºE
    glodap_lat = glodap_data['lat'].to_numpy()     # ºN

    # pull out values of DIC and TA from GLODAP
    var = glodap_data[glodap_var].values

    # switch order of GLODAP dimensions to match OCIM dimensions
    var = np.transpose(var, (0, 2, 1))
    
    # create interpolator
    interp = RegularGridInterpolator((glodap_depth, glodap_lon, glodap_lat), var, bounds_error=False, fill_value=None)

    # transform model_lon for anything < 20 (because GLODAP goes from 20ºE - 380ºE)
    model_lon[model_lon < 20] += 360

    # create meshgrid for OCIM grid
    depth, lon, lat = np.meshgrid(model_depth, model_lon, model_lat, indexing='ij')

    # reshape meshgrid points into a list of coordinates to interpolate to
    query_points = np.array([depth.ravel(), lon.ravel(), lat.ravel()]).T

    # perform interpolation (regrid GLODAP data to match OCIM grid)
    var = interp(query_points)

    # transform results back to model grid shape
    var = var.reshape(depth.shape)

    # inpaint nans
    var = inpaint_nans3d(var, mask=ocnmask.astype(bool))
    
    # transform model_lon and meshgrid back for anything > 360
    model_lon[model_lon > 360] -= 360
    

    # save regridded data
    if glodap_var == 'TCO2':
        np.save(data_path + 'GLODAPv2.2016b.MappedProduct/DIC_AO.npy', var)
    elif glodap_var == 'TAlk':
        np.save(data_path + 'GLODAPv2.2016b.MappedProduct/TA_AO.npy', var)
    else:
        np.save(data_path + 'GLODAPv2.2016b.MappedProduct/' + glodap_var + '_AO.npy', var)


def regrid_woa(woa_var, woa_lat, woa_lon, model_lat, model_lon, ocnmask):
    '''
    regrid woa data to model grid, inpaint nans

    Parameters
    ----------
    woa_var : variable to regrid (dimensions: depth, longitude, latitude)
    woa_lat : array of glodap latitudes
    woa_lon : array of glodap longitudes
    model_lat : array pf model latitudes
    model_lon : array of model longitudes
    ocnmask : mask same shape as woa_var when 1 marks an ocean cell and 0 marks land

    Returns
    -------
    woa_var : regridded to model grid

    '''
    # create interpolator
    interp = RegularGridInterpolator((woa_lon, woa_lat), woa_var, bounds_error=False, fill_value=None)

    # create meshgrid for OCIM grid
    lon, lat = np.meshgrid(model_lon, model_lat, indexing='ij')

    # reshape meshgrid points into a list of coordinates to interpolate to
    query_points = np.array([lon.ravel(), lat.ravel()]).T

    # perform interpolation (regrid GLODAP data to match OCIM grid)
    woa_var = interp(query_points)

    # transform results back to model grid shape
    woa_var = woa_var.reshape(lon.shape)

    # inpaint nans
    woa_var = inpaint_nans2d(woa_var, mask=ocnmask.astype(bool))

    return woa_var
    
def plot_surface2d(lons, lats, variable, depth_level, vmin, vmax, cmap, title):
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca()
    levels = np.linspace(vmin-0.1, vmax, 100)
    cntr = plt.contourf(lons, lats, variable, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    c = plt.colorbar(cntr, ax=ax)
    c.set_ticks(np.round(np.linspace(vmin, vmax, 10),2))
    plt.xlabel('longitude (ºE)')
    plt.ylabel('latitude (ºN)')
    plt.title(title)
    plt.xlim([0, 380]), plt.ylim([-90,90])

def plot_surface3d(lons, lats, variable, depth_level, vmin, vmax, cmap, title):
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca()
    levels = np.linspace(vmin-0.1, vmax, 100)
    cntr = plt.contourf(lons, lats, variable[depth_level, :, :].T, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    c = plt.colorbar(cntr, ax=ax)
    c.set_ticks(np.round(np.linspace(vmin, vmax, 10),2))
    plt.xlabel('longitude (ºE)')
    plt.ylabel('latitude (ºN)')
    plt.title(title)
    plt.xlim([0, 360]), plt.ylim([-90,90])

def plot_longitude3d(lats, depths, variable, longitude, vmin, vmax, cmap, title):
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca()
    levels = np.linspace(vmin-0.1, vmax, 100)
    cntr = plt.contourf(lats, depths, variable[:, longitude, :], levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    c = plt.colorbar(cntr, ax=ax)
    c.set_ticks(np.round(np.linspace(vmin, vmax, 10),2))
    ax.invert_yaxis()
    plt.xlabel('latitude (ºN)')
    plt.ylabel('depth (m)')
    plt.title(title)
    plt.xlim([-90, 90]), plt.ylim([5500, 0])
