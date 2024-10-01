#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:46:18 2024

@author: Reese Barrett

Functions for translating Khatiwala calc_steady_state_PaTh.m example code from
MATLAB to python to teach myself how to work with this model
https://www.ldeo.columbia.edu/%7Espk/Research/TMM/tmm.html
"""

import numpy as np
import scipy.io as spio

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

def gcm2matrix(TR, Ibox, ixb, iyb, izb, **kwargs):
    '''
    average GCM (general circulation model, aka transport matrix?) onto box grid
    
    TR(x, y, z) or TR(x, y)
    '''
    
    # get optional keyword arguments
    alpha = kwargs.get('alpha', None)
    kd = kwargs.get('kd', None)
    
    nb = len(Ibox)
    C = np.zeros(nb)
    
    if kd != None:
        TR[kd] = 0 # not sure this is translated correctly because I don't have an example here to test with, just a heads up to future Reese

    if TR.ndim == 2: # TR(x, y)
        for j in range(nb):
            i = Ibox[j]
            C[j] = TR[ixb[i], iyb[i]];
        
    else: # TR(x, y, z)
        for j in range(nb):
            i = Ibox[j]
            C[j] = TR[ixb[i], iyb[i], izb[i]];
            
    return C

def calc_reversible_scavenging(C, Cbulk, K, ws, dz):
    '''
    flux is positive downward
    discretize using upward scheme
    C = Cp + Cd     total tracer concentration
    Cp              tracer concentration in particulate phase
    Cd              tracer concentration in dissolved phase
    Cp = (K * Cbulk/(K * Cbulk + 1)) * C
    
    q = d(ws Cp)/dz, ws > 0 downward
    '''
    
    nz = len(C)
    Cp = (K * Cbulk/(K * Cbulk + 1)) * C

    F = np.zeros(nz+1)

    F[1:nz+1] = ws * Cp
    q = (F[0:nz] - F[1:nz+1]) / dz
    
    return q
    
def grid_boxes3d(C, I, boxFile, gridFile, interpType='rearrange', keepEmptyLayers=0):
    
    '''
    function to rearrange boxes onto regular grid (no interpolation)
    
    rearranges vector C onto grid xg, yg, zg based on nominal box positions
    I is index vector referenced to all boxes, so C = C_all_boxes(I)
    
    e.g., if C is a tracer at interior points (index vector Ii), use:
        Cg, xg, yg, zg = grid_boxes3d(C, Ii, boxFile, gridFile)
        
    to do 2-d gridding, suppose C is tracer at interior points, but we only
    want to grid onto a single depth layer:
        I = np.argwhere(Zboxnom == zLayer)
        I1 = np.arhwere(Zboxnom(Ii) == zLayer)
        Cg, xg, yg, zg = grid_boxes3d(C(I1), I, boxFile, gridFile)
    
    if interpType = 'rearrange' (default), then the script simply 'rearranges'
    the elements of C onto the GCM grid. in this case, Cg will be the same size
    as the GCM grid (in x and y, in z it will still extract only required
    layers based on 'I'). *this is generally not the case, since GCM boxes with
    land will not be represented in the matrix model
    '''

    if len([C, I, boxFile, gridFile]) < 4:
        raise ValueError('ERROR: must pass 4 arguments')

    box = loadmat(boxFile)
    Xboxnom = box['Xboxnom']
    Yboxnom = box['Yboxnom']
    Zboxnom = box['Zboxnom']
    nb = box['nb']
    if interpType == 'rearrange':
        ixBox = box['ixBox']
        iyBox = box['iyBox']
        izBox = box['izBox']
    else:
        ixb = box['ixb']
        iyb = box['iyb']
        izb = box['izb']
        
    grid = loadmat(gridFile)
    x = grid['x']
    y = grid['y']
    z = grid['z']
    nx = grid['nx']
    ny = grid['ny']
    nz = grid['nz']
    
    if I.size() == 0:
        I = np.array(range(0, nb))
        
    if (interpType != 'linear') | (interpType != 'nearest') | (interpType != 'rearrange'):
        raise ValueError("ERROR: interpType must equal 'linear', 'nearest', or 'rearrange'")
    
    if interpType == 'rearrange':
        xg = x
        yg = y
        Zb = Zboxnom[I]
        zg = np.unique(Zb)
        Cg = np.full([nx, ny, nz, nt], np.NaN)
        
        if np.ndim(C) == 1:
            nt = 1
            
        else:
            nt = np.shape(C)[1]
        
        
        #tmp = np.full([nx, ny, nz], np.NaN)
        #idx = 
        for it in range(nt):
            Cg[:, :, :, it] = C[:, it] 
        
        
            
            
            
            







