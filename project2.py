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
    
def inpaint_nans(array_3d, mask=None, iterations=100):
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

def regrid_glodap(glodap_var, glodap_depth, glodap_lat, glodap_lon, model_depth, model_lat, model_lon, ocnmask):
    '''
    regrid glodap data to model grid, inpaint nans

    Parameters
    ----------
    glodap_var : variable to regrid (dimensions: depth, longitude, latitude)
    glodap_depth : array of glodap depth levels
    glodap_lat : array of glodap latitudes
    glodap_lon : array of glodap longitudes
    model_depth : array of model depth levels
    model_lat : array pf model latitudes
    model_lon : array of model longitudes
    ocnmask : mask same shape as glodap_var when 1 marks an ocean cell and 0 marks land

    Returns
    -------
    glodap_var : regridded to model grid

    '''
    # create interpolator
    interp = RegularGridInterpolator((glodap_depth, glodap_lon, glodap_lat), glodap_var, bounds_error=False, fill_value=None)

    # transform model_lon for anything < 20 (because GLODAP goes from 20ºE - 380ºE)
    model_lon[model_lon < 20] += 360

    # create meshgrid for OCIM grid
    depth, lon, lat = np.meshgrid(model_depth, model_lon, model_lat, indexing='ij')

    # reshape meshgrid points into a list of coordinates to interpolate to
    query_points = np.array([depth.ravel(), lon.ravel(), lat.ravel()]).T

    # perform interpolation (regrid GLODAP data to match OCIM grid)
    glodap_var = interp(query_points)

    # transform results back to model grid shape
    glodap_var = glodap_var.reshape(depth.shape)

    # inpaint nans
    glodap_var = inpaint_nans(glodap_var, mask=ocnmask.astype(bool))
    
    # transform model_lon and meshgrid back for anything > 360
    model_lon[model_lon > 360] -= 360

    return glodap_var
    
def plot_surface(lons, lats, variable, depth_level, vmin, vmax, cmap, title):
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca()
    cntr = plt.contourf(lons, lats, variable[depth_level, :, :].T, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    c = plt.colorbar(cntr, ax=ax)
    plt.xlabel('longitude (ºE)')
    plt.ylabel('latitude (ºN)')
    plt.title(title)
    plt.xlim([0, 380]), plt.ylim([-90,90])

def plot_longitude(lats, depths, variable, longitude, vmin, vmax, cmap, title):
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca()
    cntr = plt.contourf(lats, depths, variable[:, longitude, :], levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    c = plt.colorbar(cntr, ax=ax)
    ax.invert_yaxis()
    plt.xlabel('longitude (ºE)')
    plt.ylabel('depth (m)')
    plt.title(title)
    plt.xlim([-90, 90]), plt.ylim([5500, 0])