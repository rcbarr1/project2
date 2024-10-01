#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:45:58 2024

@author: Reese Barrett

Translating Khatiwala calc_steady_state_PaTh.m example code from MATLAB to
python to teach myself how to work with this model
https://www.ldeo.columbia.edu/%7Espk/Research/TMM/tmm.html
"""

import khatiwala_example as kw_p2
import numpy as np
import itertools
from scipy.sparse import csc_array
from scipy.sparse import lil_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

base_path = '/Users/Reese_1/Documents/Research Projects/project2/khatiwala/MIT_Matrix_Global_2.8deg/'

# constants
w = 1000                # settling velocity [m yr^-1]
rho = 1024000           # mean water density [g m ^-3]
z0 = 200                # mixed layer thickness
zpen = 2000             # penetration depth
eps = -0.858            # decay factor
ws = w / (86400 * 365)  # settling velocity

BetaPa = 2.33e-3 / (86400 * 365)        # Pa-231 source from U decay [dpm m^-3 s^-1]
BetaTh = 2.52e-2 / (86400 * 365)        # Th-230 source from U decay [dpm m^-3 s^-1]
lambdaDecayPa = 2.13e-5 / (86400 * 365) # [s^-1]
lambdaDecayTh = 9.22e-6 / (86400 * 365) # [s^-1]

# scavenging coefficients
Kref = 1e7
K_Th_car = Kref
K_Th_opal = Kref/20
K_Th_pom = Kref
K_Th_dust = 0

K_Pa_car = Kref/40
K_Pa_opal = Kref/6
K_Pa_pom = Kref
K_Pa_dust = 0

# settings - come back and understand what these do
bigMat = 0
rearrangeProfiles = 1
periodicMatrix = 0
dt = 1200 # make sure this matches gcm
advScheme = 'dst3'

# load in grid specs
grid = kw_p2.loadmat(base_path + 'grid.mat')
nx = grid['nx']
ny = grid['ny']
nz = grid['nz']
dz = grid['dz']
x = grid['x']
y = grid['y']
z = grid['z']

# load in matrix 5 geometry
matrix5 = kw_p2.loadmat(base_path + 'Matrix5/Data/boxes.mat')
Xboxnom = matrix5['Xboxnom'] # nominal values of x boxes (longitude)
Yboxnom = matrix5['Yboxnom'] # nominal values of y boxes (latitude)
Zboxnom = matrix5['Zboxnom'] # nominal values of z boxes
ixBox = matrix5['ixBox'] # index associated with each x box
ixBox = ixBox - 1 # subtract 1 to shift indexing from matlab convention to python convention
iyBox = matrix5['iyBox'] # index associated with each y box 
iyBox = iyBox - 1 # subtract 1 to shift indexing from matlab convention to python convention
izBox = matrix5['izBox'] # index associated with each z box 
izBox = izBox - 1 # subtract 1 to shift indexing from matlab convention to python convention
nb = matrix5['nb'] # total number of boxes
volb = matrix5['volb'] # volume of each box

Ib = np.argwhere(izBox == 0) # surface depth level
nbb = len(Ib) # number of surface depth boxes

# load in transport matricies
# remember: c^n+1 = A_i^n * (A_e^n * c^n + q^n), where c is a tracer and q is sources/sinks of that tracer (aka the biogeochemical model)
A = kw_p2.loadmat(base_path + 'Matrix5/Scripts/matrix_nocorrection_annualmean.mat')
Aexpms = A['Aexpms'] # transport matrix (A) for "explicit-in-time" component of advection-diffusion
Aimpms = A['Aimpms'] # transport matrix (A) for implicit transport

# compute indexing to rearrange all objects by profile
Ip = []
for i in range(0, nbb):
   ibs = Ib[i]
   depth_idx = np.argwhere((Xboxnom == Xboxnom[ibs]) & (Yboxnom == Yboxnom[ibs])) # get list of all indicies of boxes at a given x and y (indicies of different depth levels at that location)
   Ip.append(depth_idx) # add array of indicies to list
   zp = np.sort(Zboxnom[Ip[i]], axis=0) # find and sort depth levels 
   izp = np.argsort(Zboxnom[Ip[i]], axis=0) # find and sort indicies of depth levels
   Ip[i] = Ip[i][izp].reshape(len(Ip[i][izp])) # add sorted indicies of depth levels to list of lists
Ir = np.array(list(itertools.chain.from_iterable(Ip))) # flatten into one list

# figure out what this does ("average gcm onto box grid?")
# I think it converts "dz" or the difference between depth levels into the grid we're using for this calculation?
dzb = kw_p2.gcm2matrix(dz,list(range(nb)),ixBox,iyBox,izBox)

# save these for later
Ip_prev = Ip
Ir_prev = Ir

# if this flag is turned on, reorder profiles by Ir indexing (I think all depth levels at a specific lat/lon are next to each other)
if rearrangeProfiles:
    Xboxnom = Xboxnom[Ir] 
    Yboxnom = Yboxnom[Ir]
    Zboxnom = Zboxnom[Ir]
    ixBox=ixBox[Ir]
    iyBox=iyBox[Ir]
    izBox=izBox[Ir]
    volb=volb[Ir]
    dzb=dzb[Ir]
    
    # if it is not a periodic matrix, reorder transport matricies accordingly ()
    if ~periodicMatrix:
        Aexpms=Aexpms[Ir,:][:,Ir]
        Aimpms=Aimpms[Ir,:][:,Ir]
        
    Ib = np.argwhere(izBox == 0)
    Irr = np.argsort(Ir, axis=0) # takes a rearranged vector back to its original arrangement
    
    # repeat with rearranged profiles
    Ip = []
    for i in range(nbb):
        ibs = Ib[i]
        depth_idx = np.argwhere((Xboxnom == Xboxnom[ibs]) & (Yboxnom == Yboxnom[ibs])) # get list of all indicies of boxes at a given x and y (indicies of different depth levels at that location)
        Ip.append(depth_idx) # add array of indicies to list
        zp = np.sort(Zboxnom[Ip[i]], axis=0) # find and sort depth levels 
        izp = np.argsort(Zboxnom[Ip[i]], axis=0) # find and sort indicies of depth levels
        Ip[i] = Ip[i][izp].reshape(len(Ip[i][izp])) # add sorted indicies of depth levels to list of lists
    Ir = np.array(list(itertools.chain.from_iterable(Ip))) # flatten into one list

# particle fields

# Uncomment one of these blocks:
    
###### 1. Spatially constant flux
# Ct = np.ones(nbb) * 28.2526 # flux from surface, g m^-2 yr^-1
# Ot = np.ones(nbb) * 2.3574 # flux from surface, g m^-2 yr^-1
# Pt = np.ones(nbb) * 56.8978 # flux from surface, g m^-2 yr^-1
# Dt = np.ones(nbb) * 2.0960 # flux from surface, g m^-2 yr^-1
######

###### 2. Spatially varying flux
opal = kw_p2.loadmat(base_path + 'PaTh_Matrix5_1/opal.mat')
Ot = opal['PrSi']

car = kw_p2.loadmat(base_path + 'PaTh_Matrix5_1/car.mat')
Ct = car['PrCO3']

pom = kw_p2.loadmat(base_path + 'PaTh_Matrix5_1/pom.mat')
Pt = pom['PrPOM']

dust = kw_p2.loadmat(base_path + 'PaTh_Matrix5_1/dust.mat')
Dt = dust['Prdust']
######

Cz=np.zeros(nb)
Oz=np.zeros(nb)
Pz=np.zeros(nb)
Duz=np.zeros(nb)

# loop over all surface boxes
for i in range(nbb):
    Iploc = Ip[i] # global indicies of local profile
    nzloc = len(Iploc) # number of grid points in local profile
    Imlloc = np.argwhere(Zboxnom[Iploc] < z0) # local indices of mixed layer for local profile
    
    # CaCO3
    C = Ct[i] / (w * rho) # concentration normalized to water density
    Cloc = C * np.exp((-1 * Zboxnom[Iploc] + z0) / zpen) # decay with respect to depth
    Cloc[Imlloc] = C # mixed layer
    Cz[Iploc] = Cloc

    # opal
    O = Ot[i] / (w * rho) # concentration normalized to water density
    #Oloc = O * exp((-1 * Zboxnom + z0) / zpen) # decay with respect to depth
    Oloc = O * np.ones(nzloc) # no decay with respect to depth for now, similar to conditions at poles
    Oloc[Imlloc] = O # mixed layer
    Oz[Iploc] = Oloc
    
    # POM
    P = Pt[i] / (w * rho) # concentration normalized to water density
    Ploc = P * np.power(Zboxnom[Iploc] / z0, eps) # decay with respect to depth
    Ploc[Imlloc] = P # mixed layer
    Pz[Iploc] = Ploc
    
    # dust
    D = Dt[i] / (w * rho) # concentration normalized to water density
    Duz[Iploc] = D * np.ones(nzloc) # no decay with respect to depth for now
    
# calculate bulk concentration of particles (unitless, Cbulk = Pz + Cz + Duz + Oz)
KxCbulkTh = (K_Th_pom * Pz) + (K_Th_car * Cz) + (K_Th_dust * Duz) + (K_Th_opal * Oz)
KxCbulkPa = (K_Pa_pom * Pz) + (K_Pa_car * Cz) + (K_Pa_dust * Duz) + (K_Pa_opal * Oz)

# compute matrix Q such that the reversible scavanging source term is q = Q*C,
# where C is the vector of tracer concentrations
# Q will be a block diagonal matrix
# we compute Q for the entire domain here (including boundary points)

# Th-230
KxCbulk = KxCbulkTh
Q_Th = lil_matrix(np.zeros((nb, nb)))
for i in range(nbb): # loop over all surface boxes
    Iploc = Ip[i] # global indices of local profile
    nzloc = len(Iploc) # number of grid points in local profile
    
    # construct Qloc a column at a time by probing using unit vectors
    Qloc = np.zeros((nzloc, nzloc)) # local block
    e = np.zeros(nzloc)
    for iz in range(nzloc):
        e[:] = 0
        e[iz] = 1
        Qloc[:,iz] = kw_p2.calc_reversible_scavenging(e, KxCbulk[Iploc], 1, ws, dzb[Iploc]);
    
    Q_Th[Ip[i][0]:(Ip[i][-1] + 1), Ip[i][0]:(Ip[i][-1] + 1)] = Qloc # insert into global Q (this is the stupidest indexing ever wtf python)
    Q_Th = Q_Th.tocsc()
    
# Pa-231
KxCbulk = KxCbulkPa
Q_Pa = lil_matrix(np.zeros((nb, nb)))
for i in range(nbb): # loop over all surface boxes
    Iploc = Ip[i] # global indices of local profile
    nzloc = len(Iploc) # number of grid points in local profile
    
    # construct Qloc a column at a time by probing using unit vectors
    Qloc = np.zeros((nzloc, nzloc)) # local block
    e = np.zeros(nzloc)
    for iz in range(nzloc):
        e[:] = 0
        e[iz] = 1
        Qloc[:,iz] = kw_p2.calc_reversible_scavenging(e, KxCbulk[Iploc], 1, ws, dzb[Iploc]);
    
    Q_Pa[Ip[i][0]:(Ip[i][-1] + 1), Ip[i][0]:(Ip[i][-1] + 1)] = Qloc # insert into global Q (this is the stupidest indexing ever wtf python)
    Q_Pa = Q_Pa.tocsc()

# make discrete
Q_Th = dt * Q_Th # discrete in time
Q_Pa = dt * Q_Pa # discrete in time

I = identity(nb)
Aexpms = dt * Aexpms
Aexpms = I + Aexpms

DdecayTh = lambdaDecayTh * I * dt
DdecayPa = lambdaDecayPa * I * dt

# create matrix of length nb that is = Beta * time step
BetaTh = BetaTh * dt * np.ones(nb)
BetaPa = BetaPa * dt * np.ones(nb)

# construct right-hand side and steady state operator
# steady state equation is:
# c = Ai * [Ae * c - Dd * c + Beta + Q * c], or
# [Ai * (Ae - Dd + Q) - I] * c = -Ai * Beta

b = -Aimpms * BetaPa
M = Aimpms * (Aexpms - DdecayPa + Q_Pa) - I
Pa = spsolve(M, b) # solve for steady state

b = -Aimpms * BetaTh
M = Aimpms * (Aexpms - DdecayTh + Q_Th) - I
Th = spsolve(M, b)

# convert total to dissolved concentration
Thd = (1 / (1 + KxCbulkTh)) * Th
Thp = Th - Thd
#Thdg = grid_boxes3d(Thd[Irr], np.array(range(nb)), base_path + 'Matrix5/Data/boxes.mat', base_path + 'grid.mat')









