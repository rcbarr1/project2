#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:53:54 2024

@author: Reese Barrett

Functions for translating DeVries OCIM1 example code from MATLAB to python to teach myself
how to work with this model
https://tdevries.eri.ucsb.edu/models-and-data-products/

"""

import scipy.io as spio
import scipy as sp
import numpy as np


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

def eq_wmfrac(TR, REG, M3d, grid):
    '''
    solve dF/dt = A*F subject to F prescribed at surface
    
    - partition F and A into surface and interior grid points such that:
        
        d [F_s] = [A_ss A_si][F_s]
        d [F_i]   [A_is A_ii][F_i]
        
    - if F_s is prescribed then d/dt F_i = A_ii*F_i + A_is*F_s
    - note that A_is*F_s is the effective source from having prescribed the surface sst
    
    to compute A_is*F_s, we use the sbc=0 transport operator and compute A*F where
    
        F = [F_s]  so that A*F = [A_ss A_si][F_s] = [A_ss * F_s]
            [0  ]                [A_is A_ii][0  ]   [A_is * F_s]
            
    thus, the interior grid points of A*F give the needed quantity
    '''

    # partition the transport operator
    n = np.size(TR,0)
    ny, nx, nz = np.shape(M3d)
    
    iocn = np.argwhere(M3d.flatten(order='F')) # 1-d indexes of whole ocean (cells = 1), need to do row-major order for this to work for some reason? otherwise 3rd dimension gets messed up when calculating iint
    isurf = np.argwhere(M3d[:, :, 0].flatten(order='F')) # 1-d indexes of surface ocean (cells = 1)
    iint = np.setdiff1d(iocn, isurf) # 1-d indexes of interior ocean (points in whole ocean that are not in surface ocean)
    
    ns = len(isurf)
    Aii = TR[ns::, ns::]
    Ais = TR[ns::, 0:ns]
    
    # solve for the interior distribution given surface boundary conditions
    REG_flat = REG.flatten(order='F')
    f = sp.sparse.linalg.spsolve(-Aii, (Ais * REG_flat[isurf]))
    
    # make 3d array assigning results of time step
    F = np.empty(np.shape(M3d))
    F[:] = np.nan
    F_flat = F.flatten(order='F')
    F_flat[isurf] = REG_flat[isurf]
    F_flat[iint] = f;
    F = np.reshape(F_flat, np.shape(F), order='F')
    
    return F
    
    
    
    


















