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
periodicMatrx = 0
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
Xboxnom = matrix5['Xboxnom']
Yboxnom = matrix5['Yboxnom']
Zboxnom = matrix5['Zboxnom']
ixBox = matrix5['ixBox']
iyBox = matrix5['iyBox']
izBox = matrix5['izBox']
nb = matrix5['nb']
volb = matrix5['volb']

Ib = np.argwhere(izBox == 1)
nbb = len(Ib)

# load in transport matricies
# remember: c^n+1 = A_i^n * (A_e^n * c^n + q^n), where c is a tracer and q is sources/sinks of that tracer (aka the biogeochemical model)
A = kw_p2.loadmat(base_path + 'Matrix5/Scripts/matrix_nocorrection_annualmean.mat')
Aexpms = A['Aexpms'] # transport matrix (A) for "explicit-in-time" component of advection-diffusion
Aimpms = A['Aimpms'] # transport matrix (A) for implicit transport

# compute indexing to rearrange all objects by profile
#for i in range(0, nbb):
#   ibs = Ib[i]
#   Ip[i] = do a list of lists here? 




