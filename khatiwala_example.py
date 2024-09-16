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
    
















