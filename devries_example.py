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
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_array
from scipy.sparse import csc_matrix
from scipy.sparse import identity


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
    ny, nx, nz = np.shape(M3d) # number of x, y, z points
    
    iocn = np.argwhere(M3d.flatten(order='F')) # 1-d indexes of whole ocean (cells = 1), need to do row-major order for this to work for some reason? otherwise 3rd dimension gets messed up when calculating iint
    isurf = np.argwhere(M3d[:, :, 0].flatten(order='F')) # 1-d indexes of surface ocean (cells = 1)
    iint = np.setdiff1d(iocn, isurf) # 1-d indexes of interior ocean (points in whole ocean that are not in surface ocean)
    
    ns = len(isurf)
    Aii = TR[ns::, ns::]
    Ais = TR[ns::, 0:ns]
    
    # solve for the interior distribution given surface boundary conditions
    REG_flat = REG.flatten(order='F')
    f = spsolve(-Aii, (Ais * REG_flat[isurf]))
    
    # make 3d array assigning results of time step
    F = np.empty(np.shape(M3d))
    F[:] = np.nan
    F_flat = F.flatten(order='F')
    F_flat[isurf] = REG_flat[isurf]
    F_flat[iint] = f;
    F = np.reshape(F_flat, np.shape(F), order='F')
    
    return F
    
def d0(r):
    '''
    downloaded scripts do not include documentation for this function but it
    turns an array (r) into a sparse matrix (A) that has the values of r along
    its diagonal
    '''
    m = len(r)
    K = identity(m) # create sparse identity matrix
    idx = np.argwhere(K)
    A = csc_array((r,(idx[:,0], idx[:,1])),shape=(m,m))
    
    return A
    
def eqage(TR, grid, M3d):
    '''
    compute the first and centered second moments of the last passage time
    distribution (LP1 & LP2) and of the first passage time distribution (FP1 &
    FP2)
    
    d(age)/dt - TR * age = 1
    age = 0 at sea surface
    
    d  [age(isurf)]       [TR(isurf, isurf)  TR(isurf, iint)][age(isurf)]       [1(isurf)]
    -  [          ]   -   [                                 ][          ]   =   [        ]
    dt [age(iint) ]       [TR(iint, isurf)   TR(iint, iint) ][age(iint) ]       [1(iint) ]
    
    subject to age(isurf) = 0
    
    at steady state, d/dt --> 0
    
    TR(iint, iint) * age(iint) = 1(iint) --> age(iint) = TR(iint, iint) \ 1(iint)
    
    note: transport operator (TR) has units [yr^-1]
    
    '''
    
    ny, nx, nz = np.shape(M3d) # number of x, y, z points
    
    # land & sea masks
    iocn = np.squeeze(np.argwhere(M3d.flatten(order='F')))
    iland = np.setdiff1d(range(0,len(M3d.flatten(order='F'))), iocn)
    
    # find the surface points
    Q = 0*M3d
    Q[:,:,0] = 1
    Q = Q.flatten(order='F') # interior points
    iint = np.squeeze(np.argwhere(Q[iocn]==0))
    
    # get relevant components of transport operator
    A = TR[iint,:]
    A = A[:,iint]
    
    # calculate volume of each grid cell
    vol = grid['DXT3d'] * grid['DYT3d'] * grid['DZT3d']
    w = d0(vol.flatten(order='f')[iocn[iint]])
    
    # solve for mean last passage time (ideal age, aka time water parcel last had contact with atmosphere)
    lp1 = np.zeros((ny, nx, nz), dtype=np.cfloat) # "last passage 1"
    lp1_flat = lp1.flatten(order='f')
    
    rhs = np.ones(len(iint))
    
    lp1_flat[iocn[iint]] = -1 * spsolve(A, rhs, use_umfpack=True)
    lp1_flat[iland] = np.NaN + np.NaN*1j
    lp1 = np.reshape(lp1_flat,np.shape(lp1),order='F')
    
    # centered second moment of the last passage time distribution
    lp2 = np.zeros((ny, nx, nz), dtype=np.cfloat) # "last passage 2"
    lp2_flat = lp2.flatten(order='f')
    lp2_flat[iocn[iint]] = -2 * spsolve(A, lp1_flat[iocn[iint]], use_umfpack=True) # use results from previous time step here
    lp2_flat[iland] = np.NaN + np.NaN*1j
    lp2 = np.reshape(lp2_flat,np.shape(lp2),order='F')
    
    lp2 = np.sqrt(lp2 - lp1**2) # you compare the second moment to the sqare of the first moment? https://en.wikipedia.org/wiki/Second_moment_method

    # mean first passage time (average time it will take for a water parcel starting in this state to return to surface I think?)
    fp1 = np.zeros((ny, nx, nz), dtype=np.cfloat) # "first passage 1"
    fp1_flat = fp1.flatten(order='f')
    rhs = w @ np.ones((len(iint),1)) # need @ to do dot product instead of element-wise
    rhs = csc_array(rhs).astype(np.cfloat)
    fp1_flat[iocn[iint]] = spsolve(-1*csc_matrix(w).astype(np.cfloat), spsolve(A.transpose(), rhs, use_umfpack=True), use_umfpack=True)
    fp1_flat[iland] = np.NaN + np.NaN*1j
    fp1 = np.reshape(fp1_flat,np.shape(fp1),order='F')
    
    # centered second mument of the first passage time distribution
    fp2 = np.zeros((ny, nx, nz), dtype=np.cfloat)
    fp2_flat = fp2.flatten(order='f')
    fp2_flat[iocn[iint]] = -2 * spsolve(A, fp1_flat[iocn[iint]], use_umfpack=True) # use results from previous time step here
    fp2_flat[iland] = np.NaN + np.NaN*1j
    fp2 = np.reshape(fp2_flat,np.shape(fp2),order='F')
    
    fp2 = np.sqrt(fp2 - fp1**2) # you compare the second moment to the sqare of the first moment? https://en.wikipedia.org/wiki/Second_moment_method

    return lp1, fp1, lp2, fp2











