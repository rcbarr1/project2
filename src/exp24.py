
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 2026

EXP24: Attempting to replicate map of air-sea gas exchange efficiency from Zhou et al., 2025
https://www.nature.com/articles/s41558-024-02179-9
- experimental setup: adding AT at rate of 10 mol m−2 yr−1 for 1 month, then let equilibrate
  (impulse response function), one ocean grid cell at a time for whole ocean surface

Governing equations (based on my own derivation + COBALT governing equations)
1. d(xCO2)/dt = ∆q_sea-air,xCO2 --> [µatm CO2 (µatm air)-1 yr-1] or [µmol CO2 (µmol air)-1 yr-1]
2. d(∆DIC)/dt = TR * ∆DIC + ∆q_air-sea,DIC + ∆q_CDR,DIC --> [µmol DIC (kg seawater)-1 yr-1]
3. d(∆AT)/dt = TR * ∆AT + ∆q_CDR,AT --> [µmol AT (kg seawater)-1 yr-1]

Air-sea gas exchange fluxes have to be multiplied by "c" vector because they
rely on ∆c's, which means they are incorporated with the transport matrix into
vector "A"

∆q_sea-air,xCO2 = k * V * (1 - f_ice) / Ma / z1 * (rho * R_DIC * del_DIC / beta_DIC + rho * R_AT * del_AT / beta_AT - K0 * Patm * del_xCO2)
∆q_air-sea,DIC = - k * (1 - f_ice) / z1 * (R_DIC * del_DIC / beta_DIC + R_AT * del_AT / beta_AT - K0 * Patm / rho * del_xCO2)

simplify with parameter "gamma"
gammax = k * V * (1 - f_ice) / Ma / z1
gammaC = - k * (1 - fice) / z1

∆q_sea-air,xCO2 = gammax * (rho * R_DIC * del_DIC / beta_DIC + rho * R_AT * del_AT / beta_AT - K0 * Patm * del_xCO2)
∆q_air-sea,DIC = gammaC * (R_DIC * del_DIC / beta_DIC + R_AT * del_AT / beta_AT - K0 * Patm / rho * del_xCO2)

Note about transport matrix set-up
- This was designed in matlab, which uses "fortran-style" aka column major ordering
- This means that "c" and "q" vectors must be constructed in this order
- This is complicated by the fact that the land boxes are excluded from the transport matrix
- The length of "c" and "q" vectors, as well as the length and width of the
  transport operator, are equal to the total number of ocean boxes in the model

Naming convention for saving model runs (see .txt file for explanation of experiments)
    exp##__YYYY-MM-DD-a.nc (where expXX corresponds to the python file used to
    run the experiment; a, b, c etc. represent model runs from the same day)

@author: Reese C. Barrett
"""
#%%
from src.utils import project2 as p2
import xarray as xr
from datetime import datetime
from netCDF4 import Dataset
import numpy as np
import PyCO2SYS as pyco2
from scipy import sparse
from tqdm import tqdm
from petsc4py import PETSc
from time import time
import matplotlib.pyplot as plt
import argparse
import jax
import gc
from pyTRACE import trace

# load model architecture
data_path = './data/'
output_path = './outputs/'

# load transport matrix (OCIM2-48L, from Holzer et al., 2021)
# transport matrix is referred to as "A" vector in John et al., 2020 (AWESOME OCIM)
TR = p2.loadmat(data_path + 'OCIM2_48L_base/OCIM2_48L_base_transport.mat')
TR = TR['TR']

# open up rest of data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].to_numpy()

model_depth = model_data['tz'].to_numpy()[:, 0, 0] # m below sea surface
model_lon = model_data['tlon'].to_numpy()[0, :, 0] # ºE
model_lat = model_data['tlat'].to_numpy()[0, 0, :] # ºN
model_vols = model_data['vol'].to_numpy() # m^3

# some other important numbers
grid_cell_depth = model_data['wz'].to_numpy() # depth of model layers (need bottom of grid cell, not middle) [m]
z1 = grid_cell_depth[1, 0, 0] # depth of first model layer [m]
rho = 1025 # seawater density for volume to mass [kg m-3]
surf_idx = p2.get_depth_idx(ocnmask,0) # indicies of surface grid cells in 3D array flattened by p2.flatten()

#%%
# rules for saving files
t_per_file = 2000 # number of time steps 

def set_experiment_parameters(test=False):
    # SET EXPERIMENTAL VARIABLES
    # - length of time of experiment/time stepping
    # - depth of addition
    # - location of addition
    # - year to start simulation
    # - emissions scenarios
    # - experiment names
    # in this experiment, amount of addition is set as the maximum amount of AT
    # that can be added to a grid cell before exceeding preindustrial pH, so it is
    # not treated as a variable

    # TIME
    dt0 = 1/8640 # 1 hour
    dt1 = 1/360 # 1 day
    dt2 = 1/12 # 1 month
    dt3 = 1 # 1 year

    # just year time steps
    exp0_t = np.arange(0,16,dt3)
    
    # experiment with dt = 1/12 (1 month) time steps
    exp1_t = np.arange(0,16,dt2)

    # experiment with dt = 1/360 (1 day) time steps
    exp2_t = np.arange(0,16,dt1) 

    # another with dt = 1/8640 (1 hour) time steps
    exp3_t = np.arange(0,16,dt0) 

    # daily timestep for first 90 days, a monthly until year 5,  annual until year 15
    t0 = np.arange(0, 0.25, dt1) # use a 1 day time step for the first 90 days
    t1 = np.arange(0.25, 5, dt2) # use a 1 month time step until the 5th year
    t2 = np.arange(5, 16, dt3) # use a 1 year time step through the 15th year
    exp4_t = np.concatenate((t0, t1, t2))

    #exp_ts = [exp0_t, exp1_t, exp2_t, exp3_t, exp4_t]
    exp_ts = [exp0_t]
    exp_t_names = ['6deg']

    # DEPTHS OF ADDITION

    # to do addition in first (or first two, or first three, etc.) model layer(s)
    #q_AT_depths = ocnmask.copy()
    #q_AT_depths[1::, :, :] = 0 # all ocean grid cells in surface layer (~10 m) are 1, rest 0

    # to do all ocean lat/lons individually
    #ocn_idxs = np.argwhere(q_AT_depths == 1) # find the indices where mask == 1
    #grid_cell_idxs = np.arange(len(q_AT_depths[q_AT_depths == 1]))

    # to do chunks in surface of ~5x5 model cells at at time
    chunks = []
    lat_chunk = 3
    lon_chunk = 3

    for lat0 in range(0, len(model_lat), lat_chunk):
        for lon0 in range(0, len(model_lon), lon_chunk):

            # handle edge of model grid
            lat1 = min(lat0 + lat_chunk, len(model_lat))
            lon1 = min(lon0 + lon_chunk, len(model_lon))

            block = ocnmask[0, lon0:lon1, lat0:lat1]
            
            # filter out blocks with no ocean
            if not np.any(block):
                continue

            lon_idx, lat_idx = np.where(block)
            lon_idx += lon0
            lat_idx += lat0

            depth_idx = np.zeros_like(lon_idx)

            q_AT_location = (depth_idx, lon_idx, lat_idx)
            chunks.append(q_AT_location)

    # to constrain lat/lon of addition to LME(s)
    # get masks for each large marine ecosystem (LME)
    #lme_masks, lme_id_to_name = p2.build_lme_masks(data_path + 'LMES/LMEs66.shp', ocnmask, model_lat, model_lon)
    #p2.plot_lmes(lme_masks, ocnmask, model_lat, model_lon) # note: only 62 of 66 can be represented on OCIM grid
    #lme_idx = [22,52] # subset of LMEs
    #lme_idx = list(lme_masks.keys()) # all LMES
    #q_AT_latlons = sum(lme_masks[idx] for idx in lme_idx)

    # EMISSIONS SCENARIOS
    # no emissions scenario
    #q_emissions = np.zeros(nt)

    # with starting year
    start_year = 2002 # year to start simulation
    start_CDR = 2002 # year to start CDR deployment

    # with emissions scenario
    scenarios = ['none'] 

    # set up experiments to run 
    experiments = []

    # test experiment
    if test:
        for exp_t in [np.arange(0,6,1)]: # 5 years, dt = 1 year
            #for ocn_idx, grid_cell_idxs in zip([ocn_idxs[0]], [grid_cell_idxs[0]]):
            for chunk in [chunks[2]]:
                for scenario in ['none']:
                    experiments.append({'exp_t': exp_t,
                                        'q_AT_location': chunk,
                                        'scenario': scenario,
                                        'start_year': 2002,
                                        'start_CDR' : 2002,
                                        'tag': 'TEST'})
    # real experiments
    else:
        for exp_t, exp_t_name in zip(exp_ts, exp_t_names):
            #for ocn_idx, grid_cell_idx in zip(ocn_idxs, grid_cell_idxs):
            for grid_cell_idx, chunk in enumerate(chunks):
                for scenario in scenarios:
                        experiments.append({'exp_t': exp_t,
                                            'q_AT_location': chunk,
                                            'scenario': scenario,
                                            'start_year' : start_year,
                                            'start_CDR' : start_CDR,
                                            'tag': datetime.now().strftime("%Y-%m-%d") + '_' + exp_t_name + '_' + str(grid_cell_idx)})
    return experiments

def run_experiment(experiment):
    experiment_name = 'exp24_' + experiment['tag']
    print('\nnow running experiment ' + experiment_name + '\n')

    # pull experimental parameters out of dictionary
    t = experiment['exp_t'] # time steps (starting from zero) [yr]
    nt = len(t) # total number of time steps
    dt = np.diff(t, prepend=np.nan) # difference between each time step [yr]
    q_AT_location = experiment['q_AT_location']
    q_AT_locations_mask = np.zeros(ocnmask.shape)
    q_AT_locations_mask[q_AT_location] = 1
    start_year = experiment['start_year']
    start_CDR = experiment['start_CDR']
    scenario = experiment['scenario']

    # upload regridded GLODAP data
    T_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy') # temperature [ºC]
    S_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy') # salinity [unitless]
    DIC_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC.npy')   # dissolved inorganic carbon [µmol kg-1]
    AT_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA.npy')   # total alkalinity [µmol kg-1]
    Si_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/silicate.npy') # silicate [µmol kg-1]
    P_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/PO4.npy') # phosphate [µmol kg-1]

    AT = p2.flatten(AT_3D, ocnmask)
    T = p2.flatten(T_3D, ocnmask)
    S = p2.flatten(S_3D, ocnmask)
    Si = p2.flatten(Si_3D, ocnmask)
    P = p2.flatten(P_3D, ocnmask)

    # create "pressure" array by broadcasting depth array
    pressure_3D = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[1], ocnmask.shape[2]))
    pressure = pressure_3D[ocnmask == 1].flatten(order='F')

    # calculate preindustrial DIC using TRACE
    Canth_2002_3D = p2.interp_TRACE(data_path, 2002, 'none', model_depth, model_lon, model_lat, ocnmask)

    # calculate preindustrial pH from GLODAP DIC minus Canth to get preindustrial DIC and GLODAP TA, assuming steady state
    DIC_preind_3D = DIC_3D - Canth_2002_3D
    DIC_preind = p2.flatten(DIC_preind_3D, ocnmask)
    # pyCO2SYS v2
    co2sys_preind = pyco2.sys(dic=DIC_preind, alkalinity=AT, salinity=S, temperature=T,
                    pressure=pressure, total_silicate=Si, total_phosphate=P)

    # calculate anthropogenic carbon at starting year with TRACE
    Canth_3D = p2.interp_TRACE(data_path, start_year, scenario, model_depth, model_lon, model_lat, ocnmask)
    Canth = p2.flatten(Canth_3D, ocnmask)

    # set up air-sea gas exchange (Wanninkhof, 2014)

    # upload regridded NCEP/DOE reanalysis II data
    f_ice_2D = np.load(data_path + 'NCEP_DOE_Reanalysis_II/icec.npy') # annual mean ice fraction from 0 to 1 in each grid cell
    wspd_2D = np.load(data_path + 'NCEP_DOE_Reanalysis_II/wspd.npy') # annual mean of forecast of wind speed at 10 m [m/s]
    sst_2D = np.load(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst.npy') # annual mean sea surface temperature [ºC]

    # calculate Schmidt number using Wanninkhof 2014 parameterization
    vec_schmidt = np.vectorize(p2.schmidt)
    Sc_2D = vec_schmidt('CO2', sst_2D)

    # solve for k (gas transfer velocity) for each ocean cell
    a = 0.251 # from Wanninkhof 2014
    k_2D = a * wspd_2D**2 * (Sc_2D/660)**-0.5 # [cm/h] from Yamamoto et al., 2024, adapted from Wanninkhof 2014

    k_2D *= (24*365.25/100) # [m/yr] convert units

    # set up linearized CO2 system (Nowicki et al., 2024)

    # upload (or regrid) glodap data for use as initial conditions for marine carbonate system 

    # calculate Nowicki et al. parameters
    Ma = 1.8e26 # number of micromoles of air in atmosphere [µmol air]

    Patm = 1e6 # atmospheric pressure [µatm]
    V = p2.flatten(model_vols, ocnmask) # volume of first layer of model [m^3]

    # add layers of "np.NaN" for all subsurface layers in k, f_ice, then flatten
    k_3D = np.full(ocnmask.shape, np.nan)
    k_3D[0, :, :] = k_2D
    k = p2.flatten(k_3D, ocnmask)

    f_ice_3D = np.full(ocnmask.shape, np.nan)
    f_ice_3D[0, :, :] = f_ice_2D
    f_ice = p2.flatten(f_ice_3D, ocnmask)

    gammax = k * V * (1 - f_ice) / Ma / z1
    gammaC = -1 * k * (1 - f_ice) / z1
    
    # set up file saving rules (multiple files to avoid running out of working memory)
    ds = None
    file_number = -1
    
    # construct matrix C
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
    
    # c = [ ∆xCO2 ] --> 1 * nt
    #     [ ∆DIC  ] --> m * nt
    #     [ ∆AT   ] --> m * nt
    
    c = np.zeros((1 + 2*m, 2)) # c[:,0] = c at previous time step, c[:,1] = c at current time step
    
    # construct initial q vector (it is going to change every iteration)
    # q = [ 0                                                                           ] --> 1 * nt, q[0]
    #     [ ∆q_CDR,DIC + ∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc      ] --> m * nt, q[1:(m+1)]
    #     [ ∆q_CDR,AT + 2 * (∆q_diss,arag + ∆q_diss,calc - ∆q_prod,arag - ∆q_prod,calc) ] --> m * nt, q[(m+1):(2*m+1)]
    
    # which translates to...
    # q = [ 0                                      ] --> 1 * nt, q[0]
    #     [ ∆q_CDR,DIC + ∆q_diss,DIC - ∆q_prod,DIC ] --> m * nt, q[1:(m+1)]
    #     [ ∆q_CDR,AT + ∆q_diss,AT - ∆q_prod,AT    ] --> m * nt, q[(m+1):(2*m+1)]
    
    q = np.zeros((1 + 2*m))

    # time stepping simulation forward
    
    # set initial baseline to evaluate if co2sys recalculation is needed
    AT_current = AT.copy()
    DIC_current = DIC_preind + Canth
  
    # not calculating delAT/delDIC/delxCO2 at time = 0 (this time step is initial conditions only)
    for idx in tqdm(range(1,nt)):
        
        # open new file if previous file is full (or one hasn't been opened yet)
        if (idx-1) % t_per_file == 0:
            
            # close previous file if open
            if ds is not None: ds.close()
            
            file_number += 1
            fname = experiment_name + f'_{file_number:03d}.nc'
            print('starting new file: ' + fname)
            fname = output_path + fname

            ds = Dataset(fname, "w", format="NETCDF4")
        
            ds.createDimension('time', None)
            ds.createDimension('depth', len(model_depth))
            ds.createDimension('lon', len(model_lon))
            ds.createDimension('lat', len(model_lat))
            
            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = 'year'
            
            depth_var = ds.createVariable('depth', 'f4', ('depth',))
            depth_var[:] = model_depth
            depth_var.units = 'meters'
            
            lon_var = ds.createVariable('lon', 'f4', ('lon',))
            lon_var[:] = model_lon
            lon_var.units = 'degrees_east'
            
            lat_var = ds.createVariable('lat', 'f4', ('lat',))
            lat_var[:] = model_lat
            lat_var.units = 'degrees_north'
        
            # create 4D variables
            tracers = {}
            tracers_4D = {'delDIC', 'DIC_added', 'delAT', 'AT_added'}
            for tracer in tracers_4D:
                tracers[tracer] = ds.createVariable(
                    tracer,
                    'f4',
                    ('time', 'depth', 'lon', 'lat',),
                    zlib=True,
                    complevel=4,
                    chunksizes=(1, len(model_depth), len(model_lon), len(model_lat)),)
            
            # create 1D variables
            tracers_1D = {'delxCO2', 'xCO2_added'}
            for tracer in tracers_1D:
                tracers[tracer] = ds.createVariable(
                    tracer,
                    'f4',
                    ('time'),
                    zlib=True,
                    complevel=4,
                    chunksizes=(1,),)
                
            # reset time index in the file
            idx_file = 0
            
        # if first iteration, add baseline state to file (should all be zeros)
        if idx == 1:
            
            ds.variables['time'][idx_file] = t[idx] + 2015
            tracers['delxCO2'][idx_file] = c[0, 0].astype('float32')
            tracers['delDIC'][idx_file, :, :, :] = p2.make_3D(c[1:(m+1), 0],ocnmask).astype('float32')
            tracers['delAT'][idx_file, :, :, :] = p2.make_3D(c[(m+1):(2*m+1), 0],ocnmask).astype('float32')
            tracers['xCO2_added'][idx_file] = q[0].astype('float32')
            tracers['DIC_added'][idx_file, :, :, :] = p2.make_3D(q[1:(m+1)], ocnmask).astype('float32')
            tracers['AT_added'][idx_file, :, :, :] = p2.make_3D(q[(m+1):(2*m+1)], ocnmask).astype('float32')
            idx_file += 1
        
        # update c vector to move results from idx-1 to "previous timestep" slot
        c[:,0] = c[:,1]
        c[:,1] *= 0
        
        # reset q vector
        q *= 0

        # recalculate anthropogenic carbon if the year has incremented since the last whole number
        # do not recalculate if no scenario selected --> anthropogenic CO2 will remain at level
        # at specified start year
        if scenario != 'none':
            Canth_3D = p2.interp_TRACE(data_path, t[idx] + start_year, scenario, model_depth, model_lon, model_lat, ocnmask)
            Canth = p2.flatten(Canth_3D, ocnmask)
            # print('new total Canth = ' + str(np.nansum(Canth)))

        # AT and DIC are equal to initial AT and DIC + whatever the change
        # in AT and DIC seen in previous time step are + whatever the 
        # anthropogenic carbon in this year is
        AT_current = AT + c[(m+1):(2*m+1), 0]
        DIC_current = DIC_preind + c[1:(m+1), 0] + Canth

        # print('new current avg DIC (surf, unweighted)= ' + str(np.nanmean(DIC_current[surf_idx])))
        # print('new current avg DIC (unweighted) = ' + str(np.nanmean(DIC_current)))

        # calculate carbonate system
        # use CO2SYS with GLODAP data to solve for carbonate system at each grid cell
        # do this for only surface ocean grid cells
        # this is PyCO2SYSv2
        co2sys_current = pyco2.sys(dic=DIC_current, alkalinity=AT_current,
                                salinity=S, temperature=T, pressure=pressure,
                                total_silicate=Si, total_phosphate=P)
    
        # extract key results arrays
        pCO2 = co2sys_current['pCO2'] # pCO2 [µatm]
        aqueous_CO2 = co2sys_current['CO2'] # aqueous CO2 [µmol kg-1]
        R_C = co2sys_current['revelle_factor'] # revelle factor w.r.t. DIC [unitless]

        # print('new RC = ' + str(np.nanmean(R_C[surf_idx])))

        # calculate revelle factor w.r.t. AT [unitless]
        # must calculate manually, R_AT defined as (dpCO2/pCO2) / (dAT/AT)
        # to speed up, only calculating this in surface
        co2sys_000001 = pyco2.sys(dic=DIC_current[surf_idx], alkalinity=AT_current[surf_idx]+0.000001, salinity=S[surf_idx],
                                temperature=T[surf_idx], pressure=pressure[surf_idx], total_silicate=Si[surf_idx],
                                total_phosphate=P[surf_idx])
    
        pCO2_000001 = co2sys_000001['pCO2']
        R_A_surf = ((pCO2_000001 - pCO2[surf_idx])/pCO2[surf_idx]) / (0.000001/AT[surf_idx])
        R_A = np.full(R_C.shape, np.nan)
        R_A[surf_idx] = R_A_surf
        
        # print('new RA = ' + str(np.nanmean(R_A[surf_idx])))
        
        # calculate rest of Nowicki et al. parameters
        beta_C = DIC_current/aqueous_CO2 # [unitless]
        beta_A = AT/aqueous_CO2 # [unitless]
        K0 = aqueous_CO2/pCO2*rho # [µmol CO2 m-3 (µatm CO2)-1], in derivation this is defined in per volume units so used density to get there
        
        # calculate "A" matrix
    
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
    
        del A00, A01, A02
    
        # to solve for ∆DIC
        A10 = np.nan_to_num(-1 * gammaC * K0 * Patm / rho) 
        A11 = TR + sparse.diags(np.nan_to_num(gammaC * R_C / beta_C), format='csr')
        A12 = sparse.diags(np.nan_to_num(gammaC * R_A / beta_A))
    
        A1_ = sparse.hstack((sparse.csr_matrix(np.expand_dims(A10,axis=1)), A11, A12))
    
        del A10, A11, A12
    
        # to solve for ∆AT
        A20 = np.zeros(m)
        A21 = 0 * TR
        A22 = TR
    
        A2_ = sparse.hstack((sparse.csr_matrix(np.expand_dims(A20,axis=1)), A21, A22))
    
        del A20, A21, A22
    
        # build into one mega-array!!
        A = sparse.vstack((sparse.csr_matrix(np.expand_dims(A0_,axis=0)), A1_, A2_))
    
        del A0_, A1_, A2_
            
        # calculate left hand side according to Euler backward method
        LHS = sparse.eye(A.shape[0], format="csr") - dt[idx] * A
    
        # for now, assuming NaOH (no change in DIC)
        # add 10 mol m-2 yr of AT to surface ocean for one month only
        # must convert to µmol kg-1 yr-1
        if dt[idx] == 1 and t[idx] <= 1:
            del_q_CDR_AT = p2.flatten(q_AT_locations_mask, ocnmask) * (10 * 1e6 / z1 / rho) / 12 # divide by 12 if year-long time step to only add one month's worth
            q[(m+1):(2*m+1)] = del_q_CDR_AT # add in source/sink vectors for ∆AT to q vector
        elif dt[idx] > 0.5/12 and dt[idx] < 1.5/12 and t[idx] <= 31/360:            
            del_q_CDR_AT = p2.flatten(q_AT_locations_mask, ocnmask) * (10 * 1e6 / z1 / rho) 
            q[(m+1):(2*m+1)] = del_q_CDR_AT # add in source/sink vectors for ∆AT to q vector
        elif dt[idx] > 0.5/360 and dt[idx] < 1.5/360 and t[idx] <= 30.5/360:
            del_q_CDR_AT = p2.flatten(q_AT_locations_mask, ocnmask) * (10 * 1e6 / z1 / rho) # add in source/sink vectors for ∆AT to q vector
            q[(m+1):(2*m+1)] = del_q_CDR_AT
        
        # calculate right hand side according to Euler backward method
        RHS = c[:,0] + np.squeeze(dt[idx] * q)
        
        # convert matricies from scipy sparse to PETSc to parallelize
        LHS_petsc = PETSc.Mat().createAIJ(size=LHS.shape,
                                        csr=(LHS.indptr, LHS.indices, LHS.data))
        RHS_petsc = PETSc.Vec().createWithArray(RHS)

        # set up PETSc solver
        ksp = PETSc.KSP().create()
        ksp.setOperators(LHS_petsc)
        ksp.setType('lgmres')
        ksp.setGMRESRestart(30)  # restart after 30 iterations

        # set up preconditioner
        ksp.getPC().setType('bjacobi')  # block Jacobi with ILU on each block

        # set convergence tolerances
        ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=1000)

        # set up output array (PETSc vector object)
        c_petsc = LHS_petsc.createVecRight()
        c_petsc.setArray(c[:,0])
        ksp.setInitialGuessNonzero(True) # tell solver to use initial guess

        # solve system (perform time stepping)
        ksp.solve(RHS_petsc, c_petsc)
        c[:,1] = c_petsc.array.copy()

        # check for convergence 
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(
                f'solver failed to converge '
                f'reason code: {ksp.getConvergedReason()}, '
                f'iterations: {ksp.getIterationNumber()}, '
                f'residual: {ksp.getResidualNorm():.2e} '
            )

        # fix tracer drift (transport matrix is not perfectly conservative)
        # calculate amount erroneously added between time steps, subtract off mean of this for each box
        # weights = p2.flatten(model_vols, ocnmask)
        
        # sum_DIC0 = np.sum(c[1:(m+1), 0] * rho * weights) # amount of DIC at previous time step [µmol]
        # sum_DIC1 = np.sum(c[1:(m+1), 1] * rho * weights) # amount of DIC at current time step [µmol]
        # flux_q_DIC = np.sum(dt[idx] * q[1:(m+1)] * rho * weights) # amount of DIC added on purpose [µmol]
        # #flux_A_DIC = np.sum(-1 * c[0, 1] * Ma) # amount of DIC added via air-sea gas exchange [µmol]
        # drift_DIC = (sum_DIC1 - sum_DIC0 - flux_q_DIC) / np.sum(rho * weights)
        # c[1:(m+1), 1] -= drift_DIC
        
        # sum_AT0 = np.sum(c[(m+1):(2*m+1), 0] * rho * weights) # amount of AT at previous time step [µmol]
        # sum_AT1 = np.sum(c[(m+1):(2*m+1), 1] * rho * weights) # amount of AT at current time step [µmol]
        # flux_AT = np.sum(dt[idx] * q[(m+1):(2*m+1)] * rho * weights) # amount of AT added on purpose [µmol]
        # drift_AT = (sum_AT1 - sum_AT0 - flux_AT) / np.sum(rho * weights)
        # c[(m+1):(2*m+1), 1] -= drift_AT
        
        # partition "c" into xCO2, DIC, and AT
        c_delxCO2 = c[0, 1]
        c_delDIC  = c[1:(m+1), 1]
        c_delAT   = c[(m+1):(2*m+1), 1]

        # partition "q" into xCO2, DIC, and AT
        # convert from flux (amount yr-1) to amount by multiplying by dt [yr]
        q_delxCO2 = q[0] * dt[idx]
        q_delDIC  = q[1:(m+1)] * dt[idx]
        q_delAT   = q[(m+1):(2*m+1)] * dt[idx]
        
        # convert delxCO2 units from unitless [µatm CO2 / µatm air] or [µmol CO2 / µmol air] to ppm
        c_delxCO2 *= 1e6
        q_delxCO2 *= 1e6
        
        # rebuild 3D concentrations from 1D array used for solving matrix equation
        c_delDIC_3D = np.full(ocnmask.shape, np.nan)
        c_delAT_3D = np.full(ocnmask.shape, np.nan)
        q_delDIC_3D = np.full(ocnmask.shape, np.nan)
        q_delAT_3D = np.full(ocnmask.shape, np.nan)
        
        c_delDIC_3D = p2.make_3D(c_delDIC, ocnmask)
        c_delAT_3D = p2.make_3D(c_delAT, ocnmask)
        q_delDIC_3D = p2.make_3D(q_delDIC, ocnmask)
        q_delAT_3D = p2.make_3D(q_delAT, ocnmask)
            
        # write data to xarray
        ds.variables['time'][idx_file] = t[idx] + 2015
        tracers['delxCO2'][idx_file] = c_delxCO2.astype('float32')
        tracers['delDIC'][idx_file, :, :, :] = c_delDIC_3D.astype('float32')
        tracers['delAT'][idx_file, :, :, :] = c_delAT_3D.astype('float32')
        tracers['xCO2_added'][idx_file] = q_delxCO2.astype('float32')
        tracers['DIC_added'][idx_file, :, :, :] = q_delDIC_3D.astype('float32')
        tracers['AT_added'][idx_file, :, :, :] = q_delAT_3D.astype('float32')

        # delete pyco2sys objects to avoid running out of memory
        if 'co2sys' in globals(): del co2sys
        if 'co2sys_preind' in globals(): del co2sys_preind
        if 'co2sys_000001' in globals(): del co2sys_000001
        if 'co2sys_current' in globals(): del co2sys_current
        if 'co2sys_desired' in globals(): del co2sys_desired
        gc.collect()
        jax.clear_caches()

        # sync data to output file every 20 time steps (in case of crash)
        if idx % 20 == 0:
            ds.sync()
        
        # increment within-file index
        idx_file += 1
    
    if ds is not None: ds.close()

def main():
    # parse function call (from command line)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-id', type=int,
                       help='experiment ID (0 to N-1)')
    parser.add_argument('--list', action='store_true',
                       help='list all experiments and exit')
    parser.add_argument('--test', action='store_true',
                       help='use quick test experiments instead of full set')
    args = parser.parse_args()
    
    # get all experiment configurations
    test = False
    if args.test:
        test = True
    experiments = set_experiment_parameters(test)
    
    # handle --list option
    if args.list:
        print(f"total experiments: {len(experiments)}")
        for i, experiment in enumerate(experiments):
            print(f"  {i}: exp21_{experiment['tag']}")
        return
    
    # validate exp_id
    if not test:
        if args.exp_id < 0 or args.exp_id >= len(experiments):
            print(f"ERROR: exp-id must be between 0 and {len(experiments)-1}")
            print(f"use --list to see all experiments")
            return
    
    # run the specified experiment
    if not test: experiment = experiments[args.exp_id]
    else: experiment = experiments[0]
    run_experiment(experiment)
    
# run main function
if __name__ == '__main__':
    main()
# %%
