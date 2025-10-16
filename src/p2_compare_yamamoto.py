#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 13:39:59 2025

COMPARE_YAMAMOTO: created to compare outputs from Kana's MATLAB file with
results from my work to try to isolate what is causing discrepancies in the
answers
- step 1: compare my "A" matrix with Kana's "M" matrix

Note: Using OCIM1 grid because it matches Kana's file structure

Naming convention for saving model runs (see .txt file for explanation of experiments)
    exp##__YYYY-MM-DD-a.nc (where expXX corresponds to the python file used to
    run the experiment; a, b, c etc. represent model runs from the same day)

@author: Reese C. Barrett
"""

import project2 as p2
import xarray as xr
import numpy as np
import PyCO2SYS as pyco2
from scipy import sparse
#from tqdm import tqdm
from scipy.sparse.linalg import spilu, LinearOperator, lgmres
from time import time
import matplotlib.pyplot as plt

data_path = '/Users/Reese_1/Documents/Research Projects/project2/data/'
output_path = '/Users/Reese_1/Documents/Research Projects/project2/outputs/'

#%% load transport matrix (OCIM1)
model_data = p2.loadmat('/Users/Reese_1/Documents/Research Projects/project2/examples/CTL.mat')

TR = model_data['output']['TR']
ocnmask = np.asfortranarray(model_data['output']['M3d'])
ocnmask = np.transpose(ocnmask, axes=(2, 1, 0))

model_depth = np.array(model_data['output']['grid']['zt']) # m below sea surface defined at tracer grid point
model_lon = np.array(model_data['output']['grid']['xt']) # ºE defined at tracer grid point
model_lat = np.array(model_data['output']['grid']['yt']) # ºN defined at tracer grid point
model_vols = np.asfortranarray(model_data['output']['grid']['DXT3d']) * np.asfortranarray(model_data['output']['grid']['DYT3d']) * np.asfortranarray(model_data['output']['grid']['DZT3d']) # m^3
model_vols = np.transpose(model_vols, axes=(2, 1, 0))

# seawater density for volume to mass [kg m-3]
rho = 1025 

#%% depth of first model layer (need bottom of grid cell, not middle) [m]
z1 = np.array(model_data['output']['grid']['zw'])[1]

# to help with conversions
sec_per_year = 60 * 60 * 24 * 365.25 # seconds in a year

# number of surface grid cells
nsurf = np.sum(ocnmask[0, :, :])

#%% set up air-sea gas exchange (Wanninkhof 2014)

# upload (or regrid) woa18 data for use in CO2 system calculations

# regrid WOA18 data
#p2.regrid_woa(data_path, 'S', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_woa(data_path, 'T', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_woa(data_path, 'Si', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_woa(data_path, 'P', model_depth, model_lat, model_lon, ocnmask)

# upload regridded WOA18 data
S_3D = np.load(data_path + 'WOA18/S_AO_OCIM1.npy')   # salinity [unitless]
T_3D = np.load(data_path + 'WOA18/T_AO_OCIM1.npy')   # temperature [ºC]
Si_3D = np.load(data_path + 'WOA18/Si_AO_OCIM1.npy') # silicate [µmol kg-1]
P_3D = np.load(data_path + 'WOA18/P_AO_OCIM1.npy')   # phosphate [µmol kg-1]

# flatten data
S = p2.flatten(S_3D, ocnmask)
T = p2.flatten(T_3D, ocnmask)
Si = p2.flatten(Si_3D, ocnmask)
P = p2.flatten(P_3D, ocnmask)

#p2.plot_surface3d(model_lon, model_lat, S_3D, 0, 25, 38, 'magma', 'WOA salinity distribution')
#p2.plot_surface3d(model_lon, model_lat, T_3D, 0, -10, 35, 'magma', 'WOA temp distribution')
#p2.plot_surface3d(model_lon, model_lat, Si_3D, 0, 0, 30, 'magma', 'WOA silicate distribution')
#p2.plot_surface3d(model_lon, model_lat, P_3D, 0, 0, 2.5, 'magma', 'WOA phosphate distribution')

# regrid NCEP/DOE reanalysis II data
#p2.regrid_ncep_noaa(data_path, 'icec', model_lat, model_lon, ocnmask)
#p2.regrid_ncep_noaa(data_path, 'uwnd', model_lat, model_lon, ocnmask)
#p2.regrid_ncep_noaa(data_path, 'sst', model_lat, model_lon, ocnmask)

# upload regridded NCEP/DOE reanalysis II data
f_ice_2D = np.load(data_path + 'NCEP_DOE_Reanalysis_II/icec_AO_OCIM1.npy') # annual mean ice fraction from 0 to 1 in each grid cell
uwnd_2D = np.load(data_path + 'NCEP_DOE_Reanalysis_II/uwnd_AO_OCIM1.npy') # annual mean of forecast of U-wind at 10 m [m/s]
sst_2D = np.load(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst_AO_OCIM1.npy') # annual mean sea surface temperature [ºC]

# calculate Schmidt number using Wanninkhof 2014 parameterization
vec_schmidt = np.vectorize(p2.schmidt)
Sc_2D = vec_schmidt('CO2', sst_2D)

# solve for k (gas transfer velocity) for each ocean cell
a = 0.251 # from Wanninkhof 2014
k_2D = a * uwnd_2D**2 * (Sc_2D/660)**-0.5 # [cm/h] from Yamamoto et al., 2024, adapted from Wanninkhof 2014

#p2.plot_surface2d(model_lon, model_lat, k.T, 0, 20, 'magma', 'Gas transfer velocity (k, cm/hr)')

k_2D *= (24*365.25/100) # [m/yr] convert units

#p2.plot_surface2d(model_lon, model_lat, uwnd_3D.T, -15, 15, 'seismic', 'U-wind at 10 m (m/s)')
#p2.plot_surface2d(model_lon, model_lat, sst_3D.T, -2, 40, 'magma', 'sst (ºC)')

# set up linearized CO2 system (Nowicki et al., 2024)

# upload (or regrid) glodap data for use as initial conditions for marine carbonate system 

# regrid GLODAP data
#p2.regrid_glodap(data_path, 'TCO2', model_depth, model_lat, model_lon, ocnmask)
#p2.regrid_glodap(data_path, 'TAlk', model_depth, model_lat, model_lon, ocnmask)

# upload regridded GLODAP data
DIC_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC_AO_OCIM1.npy') # dissolved inorganic carbon [µmol kg-1]
AT_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA_AO_OCIM1.npy')   # total alkalinity [µmol kg-1]

# flatten data
DIC = p2.flatten(DIC_3D, ocnmask)
AT = p2.flatten(AT_3D, ocnmask)

# create "pressure" array by broadcasting depth array
pressure_3D = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[1], ocnmask.shape[2]))
pressure = pressure_3D[ocnmask == 1].flatten(order='F')

# use CO2SYS with GLODAP and WOA data to solve for carbonate system at each grid cell
# do this for only ocean grid cells
# this is PyCO2SYSv2
co2sys = pyco2.sys(dic=DIC, alkalinity=AT, salinity=S, temperature=T,
                   pressure=pressure, total_silicate=Si, total_phosphate=P)

# extract key results arrays
pCO2 = co2sys['pCO2'] # pCO2 [µatm]
aqueous_CO2 = co2sys['CO2'] # aqueous CO2 [µmol kg-1]
R_C = co2sys['revelle_factor'] # revelle factor w.r.t. DIC [unitless]

# calculate revelle factor w.r.t. AT [unitless]
# must calculate manually, R_AT defined as (dpCO2/pCO2) / (dAT/AT)
co2sys_000001 = pyco2.sys(dic=DIC, alkalinity=AT+0.000001, salinity=S,
                       temperature=T, pressure=pressure, total_silicate=Si,
                       total_phosphate=P)

pCO2_000001 = co2sys_000001['pCO2']
R_A = ((pCO2_000001 - pCO2)/pCO2) / (0.000001/AT)

# calculate Nowicki et al. parameters
Ma = 1.8e26 # number of micromoles of air in atmosphere [µmol air]
beta_C = DIC/aqueous_CO2 # [unitless]
beta_A = AT/aqueous_CO2 # [unitless]
K0 = aqueous_CO2/pCO2*rho # [µmol CO2 m-3 (µatm CO2)-1], in derivation this is defined in per volume units so used density to get there
Patm = 1e6 # atmospheric pressure [µatm]
V = p2.flatten(model_vols, ocnmask) # volume of model cells

# add layers of "np.NaN" for all subsurface layers in k, f_ice, then flatten
k_3D = np.full(ocnmask.shape, np.nan)
k_3D[0, :, :] = k_2D
k = p2.flatten(k_3D, ocnmask)

f_ice_3D = np.full(ocnmask.shape, np.nan)
f_ice_3D[0, :, :] = f_ice_2D
f_ice = p2.flatten(f_ice_3D, ocnmask)

gammax = k * V * (1 - f_ice) / Ma / z1
gammaC = -1 * k * (1 - f_ice) / z1

#%% set up time stepping

# more complicated time stepping
# set up time domain
dt1 = 1/360 # 1 day
dt2 = 1/12 # 1 month
dt3 = 1 # 1 year
dt4 = 10 # 10 years
dt5 = 100 # 100 years

# Kana's time stepping
t1 = np.arange(0, 90/360, dt1) # use a 1 day time step for the first 90 days
t2 = np.arange(90/360, 5, dt2) # use a 1 month time step until the 5th year
t3 = np.arange(5, 100, dt3) # use a 1 year time step until the 100th year
t4 = np.arange(100, 500, dt4) # use a 10 year time step until the 500th year
t5 = np.arange(500, 1000+dt5, dt5) # use a 100 year time step until the 1000th year

# shortened time stepping to test solvers#
#t1 = np.arange(0, 30/360, dt1) # use a 1 day time step for the first 30 days
#t2 = np.arange(30/360, 1, dt2) # use a 1 month time step until the 1st year
#t3 = np.arange(1, 3, dt3) # use a 1 year time step until the 3rd year
#t4 = np.arange(3, 23, dt4) # use a 10 year time step until the 23rd year
#t5 = np.arange(23, 123+dt5, dt5) # use a 100 year time step until the 1000th year

t = np.concatenate((t1, t2, t3, t4, t5))

#%% construct "A" matrix
# add CDR perturbation - NONE FOR THIS COMPARISON
# Add surface ocean perturbation of -1 µmol kg-1 yr-1 in DIC, no change in AT
# Goal: compare results with Yamamoto et al., 2024 supplemental figures

# ∆q_CDR,AT (change in alkalinity due to CDR addition) - final units: [µmol AT kg-1 yr-1]
del_q_CDR_AT_3D = np.full(ocnmask.shape, np.nan)
del_q_CDR_AT_3D[ocnmask == 1] = 0
del_q_CDR_AT = p2.flatten(del_q_CDR_AT_3D, ocnmask)

# ∆q_CDR,DIC (change in DIC due to CDR addition) - final units: [µmol DIC kg-1 yr-1]
del_q_CDR_DIC_3D = np.full(ocnmask.shape, np.nan)
del_q_CDR_DIC_3D[ocnmask == 1] = 0
del_q_CDR_DIC = p2.flatten(del_q_CDR_DIC_3D, ocnmask)

# construct matricies
# matrix form:
#  dc/dt = A * c + q
#  c = variable(s) of interest
#  A = transport matrix (TR) plus any processes with dependence on c 
#    = source/sink vector (processes not dependent on c)
    
# UNITS NOTE: all xCO2 units are mol CO2 (mol air)-1 all AT units are µmol AT kg-1, all DIC units are µmol DIC kg-1
# see comment at top for more info

# m = # ocean grid cells
# nt = # time steps

m = TR.shape[0]
nt = len(t)

# c = [ ∆xCO2 ] --> 1 * nt
#     [ ∆DIC  ] --> m * nt
#     [ ∆AT   ] --> m * nt

c = np.zeros((1 + 2*m, nt))

# q = [ 0                                                                           ] --> 1 * nt, q[0]
#     [ ∆q_CDR,DIC + ∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc      ] --> m * nt, q[1:(m+1)]
#     [ ∆q_CDR,AT + 2 * (∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc) ] --> m * nt, q[(m+1):(2*m+1)]

# which translates to...
# q = [ 0                                      ] --> 1 * nt, q[0]
#     [ ∆q_CDR,DIC + ∆q_diss,DIC - ∆q_prod,DIC ] --> m * nt, q[1:(m+1)]
#     [ ∆q_CDR,AT + ∆q_diss,AT - ∆q_prod,AT    ] --> m * nt, q[(m+1):(2*m+1)]

q = np.zeros((1 + 2*m, nt))

# add in source/sink vectors for ∆AT, only add perturbation for time step 0

# for ∆DIC, add perturbation for first 30 days
q[1:(m+1),0:30] = np.tile(del_q_CDR_DIC[:, np.newaxis], (1,30))

# for ∆AT, no change

# dimensions
# A = [1 x 1][1 x m][1 x m] --> total size 2m + 1 x 2m + 1
#     [m x 1][m x m][m x m]
#     [m x 1][m x m][m x m]

# what acts on what
# A = [THIS BOX * ∆xCO2][THIS BOX * ∆DIC][THIS BOX * ∆AT] --> to calculate new ∆xCO2 (still need q)
#     [THIS BOX * ∆xCO2][THIS BOX * ∆DIC][THIS BOX * ∆AT] --> to calculate new ∆DIC (still need q)
#     [THIS BOX * ∆xCO2][THIS BOX * ∆DIC][THIS BOX * ∆AT] --> to calculate new ∆AT (still need q)

# math in each box (note: air-sea gas exchange terms only operate in surface boxes, they are set as main diagonal of identity matrix)
# A = [-gammax * K0 * Patm      ][gammax * rho * R_DIC / beta_DIC][gammax * rho * R_AT / beta_AT]
#     [-gammaC * K0 * Patm / rho][TR + gammaC * R_DIC / beta_DIC ][gammaC * R_AT / beta_AT      ]
#     [0                        ][0                              ][TR                           ]

# notation for setup
# A = [A00][A01][A02]
#     [A10][A11][A12]
#     [A20][A21][A22]

# to solve for ∆xCO2
A00 = -1 * Patm * np.nansum(gammax * K0) # using nansum because all subsurface boxes are NaN, we only want surface
A01 = np.nan_to_num(gammax * rho * R_C / beta_C) # nan_to_num sets all NaN = 0 (subsurface boxes, no air-sea gas exchange)
A02 = np.nan_to_num(gammax * rho * R_A / beta_A)

# combine into A0 row
A0_ = np.full(1 + 2*m, np.nan)
A0_[0] = A00
A0_[1:(m+1)] = A01
A0_[(m+1):(2*m+1)] = A02

#del A00, A01, A02

# to solve for ∆DIC
A10 = np.nan_to_num(-1 * gammaC * K0 * Patm / rho) # is csc the most efficient format? come back to this
A11 = TR + sparse.diags(np.nan_to_num(gammaC * R_C / beta_C), format='csc')
A12 = sparse.diags(np.nan_to_num(gammaC * R_A / beta_A))

A1_ = sparse.hstack((sparse.csc_matrix(np.expand_dims(A10,axis=1)), A11, A12))

#del A10, A11, A12

# to solve for ∆AT
A20 = np.zeros(m)
A21 = 0 * TR
A22 = TR

A2_ = sparse.hstack((sparse.csc_matrix(np.expand_dims(A20,axis=1)), A21, A22))

#del A20, A21, A22

# build into one mega-array!!
A = sparse.vstack((sparse.csc_matrix(np.expand_dims(A0_,axis=0)), A1_, A2_))

del A0_, A1_, A2_

#del TR

#%% load in matlab output for comparison
yamamoto_data = p2.loadmat('/Users/Reese_1/Documents/Research Projects/project2/examples/run_DAC_OAE/M.mat')
M = yamamoto_data['M']
Atau = yamamoto_data['Atau']
Qatm = yamamoto_data['Qatm']
Satm = yamamoto_data['Satm']
Vatm = yamamoto_data['Vatm']

#%% compare!
# Atau should equal A11
print((Atau != A11).nnz == 0)

# Qatm should equal A10
print(np.array_equal(np.asarray(Qatm.toarray(), order="F").ravel(order="F"), A10, equal_nan=True))

# Satm should equal A00
print(np.equal(Satm, A00))

# Vatm should equal A01
print(np.array_equal(np.asarray(Vatm.toarray(), order="F").ravel(order="F"), A01, equal_nan=True))



















