
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:37 2025

EXP21: Attempting maximum alkalinity calculation with parallel sparse matrix solve and running experiments in parallel (optimizing for HPC)
- Import anthropogenic carbon prediction at each grid cell using TRACEv1 to be initial ∆DIC conditions
- From ∆DIC and gridded GLODAP pH and DIC, calculate how much ∆pH or ∆pCO2 to get back to preindustrial surface conditions
- With this, calculate ∆AT required to offset
- Add this ∆AT as a surface perturbation (globally for now, but eventually do this for each large marine ecosystem)
- Also, add the relevant ∆xCO2 for the time step given the emissions scenario of interest?
- Repeat for each LMES & maybe a few combinations of LMES?

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

#%% diagnostics
print("numpy config:")
np.show_config()
print("PETSc info:", PETSc.Sys.getVersion())

#%% load model architecture
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
ns = int(np.nansum(ocnmask[0,:,:])) # number of surface grid cells
rho = 1025 # seawater density for volume to mass [kg m-3]

# rules for saving files
t_per_file = 2000 # number of time steps 

def set_experiment_parameters(test=False):
    # SET EXPERIMENTAL VARIABLES: WEEKEND RUN
    # - length of time of experiment/time stepping
    # - depth of addition
    # - location of addition
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
    exp0_t = np.arange(0,10,dt3) 

    # experiment with dt = 1/12 (1 month) time steps
    exp1_t = np.arange(0,10/12,dt2) 
    #exp1_t = np.arange(0,400,dt2)

    # experiment with dt = 1/360 (1 day) time steps
    exp2_t = np.arange(0,10/360,dt1) 

    # another with dt = 1/8640 (1 hour) time steps
    exp3_t = np.arange(0,10/8640,dt0) 

    exp_ts = [exp0_t, exp1_t, exp2_t, exp3_t]
    exp_t_names = ['t0', 't1', 't2', 't3'] 

    # DEPTHS OF ADDITION

    # to do addition in mixed layer...
    # pull mixed layer depth at each lat/lon from OCIM model data, then create mask
    # of ocean cells that are at or below the mixed layer depth
    mld = model_data.mld.values # [m]
    # create 3D mask where for each grid cell, mask is set to 1 if the depth in the
    # grid cell depths array is less than the mixed layer depth for that column
    # note: this does miss cells where the MLD is close but does not reach the
    # depth of the next grid cell below (i.e. MLD = 40 m, grid cell depths are at
    # 30 m and 42 m, see lon_idx, lat_idx = 20, 30). I am intentionally leaving
    # this for now to ensure what enters the ocean stays mostly within the mixed
    # layer, but the code could be changed to a different method if needed.exp2_t

    mldmask = (grid_cell_depth < mld[None, :, :]).astype(int)
    q_AT_depths = [mldmask]

    # to do addition in first (or first two, or first three, etc.) model layer(s)
    #q_AT_depths = ocnmask.copy()
    #q_AT_depths[1::, :, :] = 0 # all ocean grid cells in surface layer (~10 m) are 1, rest 0
    #q_AT_depths[2::, :, :] = 0 # all ocean grid cells in top 2 surface layers (~30 m) are 1, rest 0
    #q_AT_depths[3::, :, :] = 0 # all ocean grid cells in top 3 surface layers (~50 m) are 1, rest 0

    # to do all lat/lons
    q_AT_latlons = [ocnmask[0,:,:].copy()]

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

    # with emissions scenario
    scenarios = ['none'] 

    # set up experiments to run 
    experiments = []

    # test experiment
    if test:
        for exp_t in [np.arange(0,6,1)]: # 5 years, dt = 1 year
            for q_AT_depth in q_AT_depths:
                for q_AT_latlon in q_AT_latlons:
                    for scenario in ['none']:
                            experiments.append({'exp_t': exp_t,
                                                'q_AT_locations_mask': q_AT_depth * q_AT_latlon, # combine depth and lat/lon masks into one
                                                'scenario': scenario,
                                                'threshold': 0.00,
                                                'tag': 'TEST'})
    # real experiments
    else:
        for exp_t, exp_t_name in zip(exp_ts, exp_t_names):
            for q_AT_depth in q_AT_depths:
                for q_AT_latlon in q_AT_latlons:
                    for scenario in scenarios:
                            experiments.append({'exp_t': exp_t,
                                                'q_AT_locations_mask': q_AT_depth * q_AT_latlon, # combine depth and lat/lon masks into one
                                                'scenario': scenario,
                                                'threshold': 0.00,
                                                'tag': datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + exp_t_name + '_' + scenario})
    return experiments

def run_experiment(experiment):
    experiment_name = 'exp21_' + experiment['tag']
    print('\nnow running experiment ' + experiment_name + '\n')

    # pull experimental parameters out of dictionary
    t = experiment['exp_t'] # time steps (starting from zero) [yr]
    nt = len(t) # total number of time steps
    dt = np.diff(t, prepend=np.nan) # difference between each time step [yr]
    q_AT_locations_mask = experiment['q_AT_locations_mask']
    scenario = experiment['scenario']
    threshold = experiment['threshold']

    # getting initial ∆DIC conditions from TRACEv1
    # note, doing set up with Fortran ordering for consistency

    # create list of longitudes (ºE), latitudes (ºN), and depths (m) in TRACE format
    # this order is required for TRACE
    lon, lat, depth = np.meshgrid(model_lon, model_lat, model_depth, indexing='ij')

    # reshape meshgrid points into a list of coordinates to interpolate to
    output_coordinates = np.array([lon.ravel(order='F'), lat.ravel(order='F'), depth.ravel(order='F'), ]).T

    # create required input of dates
    # first simulation year will be 2015 (I think), so do then 
    dates_2015 = 2015 * np.ones([output_coordinates.shape[0],1])
    #dates_2025 = 2025 * np.ones([output_coordinates.shape[0],1])

    # upload regridded glodap data
    T_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy')
    S_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy')
    #S_3D[S_3D < 30] = 30 # ensure no very low S values (might be causing co2sys to lose it?)

    # transpose to match requirements for TRACEv1
    T_3D = T_3D.transpose([1, 2, 0])
    S_3D = S_3D.transpose([1, 2, 0])
    predictor_measurements = np.vstack([S_3D.flatten(order='F'), T_3D.flatten(order='F')]).T

    # combine all into .csv file to export for use with TRACEv1 in MATLAB (on the edge of my seat for pyTRACE clearly)
    #trace_data = np.hstack([output_coordinates, dates_2015, predictor_measurements])
    #np.savetxt(data_path + 'TRACEv1/trace_inputs_2015.txt', trace_data, delimiter = ',')

    # transpose temperature and salinity back
    T_3D = T_3D.transpose([2, 0, 1])
    S_3D = S_3D.transpose([2, 0, 1])

    # load in TRACE data
    Canth_2015 = p2.loadmat(data_path + 'TRACEv1/trace_outputs_2015.mat')
    Canth_2015 = Canth_2015['trace_outputs_2015']
    Canth_2015 = Canth_2015.reshape(len(model_lon), len(model_lat), len(model_depth), order='F')
    Canth_2015 = Canth_2015.transpose([2, 0, 1])

    #p2.plot_surface3d(model_lon, model_lat, Canth_2015, 0, -1, 82, 'viridis', 'anthropogenic carbon')

    # calculate preindustrial pH from GLODAP DIC minus Canth to get preindustrial DIC and GLODAP TA, assuming steady state

    # upload regridded GLODAP data
    DIC_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC.npy') # dissolved inorganic carbon [µmol kg-1]
    AT_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA.npy')   # total alkalinity [µmol kg-1]
    pH_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/pHtsinsitutp.npy') # pH on total scale at in situ temperature and pressure 
    Si_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/silicate.npy') # silicate [µmol kg-1]
    P_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/PO4.npy') # phosphate [µmol kg-1]

    DIC = p2.flatten(DIC_3D, ocnmask)
    AT = p2.flatten(AT_3D, ocnmask)
    pH = p2.flatten(pH_3D, ocnmask)
    T = p2.flatten(T_3D, ocnmask)
    S = p2.flatten(S_3D, ocnmask)
    Si = p2.flatten(Si_3D, ocnmask)
    P = p2.flatten(P_3D, ocnmask)

    # calculate preindustrial pH by subtracting anthropogenic carbon
    DIC_preind_3D = DIC_3D - Canth_2015
    DIC_preind = p2.flatten(DIC_preind_3D, ocnmask)

    # create "pressure" array by broadcasting depth array
    pressure_3D = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[1], ocnmask.shape[2]))
    pressure = pressure_3D[ocnmask == 1].flatten(order='F')

    # calculate preindustrial pH from DIC in 2015 minus Canth in 2015 AND TA in 2015 (assuming steady state)

    # is it okay to use modern-day temperatures for this?? probably not, but not
    # sure if there's a TRACE for this and trying to stick with data-based, not
    # model-based
    # pyCO2SYS v2
    co2sys_preind = pyco2.sys(dic=DIC_preind, alkalinity=AT, salinity=S, temperature=T,
                    pressure=pressure, total_silicate=Si, total_phosphate=P)

    pH_preind = co2sys_preind['pH']
    avg_pH_preind = np.nanmean(pH_preind)

    pH_preind_3D = p2.make_3D(pH_preind, ocnmask)

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
    nfiles = nt // t_per_file + (nt % t_per_file > 0) # number of files for this simulation
    ds = None
    file_number = -1
    
    # set up emissions scenario
    
    # get annual emissions
    atmospheric_xCO2 = np.zeros(nt)

    if scenario != 'none':
        atmospheric_xCO2_time, atmospheric_xCO2_annual = p2.get_emissions_scenario(data_path, scenario) 

        # interpolate atmospheric CO2 to match time stepping of simulation
        atmospheric_xCO2 = np.interp(t + 2015, atmospheric_xCO2_time, atmospheric_xCO2_annual)

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
    AT_at_last_calc = AT.copy()
    DIC_at_last_calc = DIC.copy()
    AT_current = AT.copy()
    DIC_current = DIC.copy()
  
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
        
        # AT and DIC are equal to initial AT and DIC + whatever the change
        # in AT and DIC seen in previous time step are
        AT_current = AT + c[(m+1):(2*m+1), 0]
        DIC_current = DIC + c[1:(m+1), 0]
        
        # recalculate carbonate system every time >5% of grid cells see change
        # in AT or DIC >5% since last recalculation
        frac_AT = np.mean(np.abs(AT_current - AT_at_last_calc) > threshold * np.abs(AT_at_last_calc)) # calculate fraction of grid cells with change in AT above 10%
        frac_DIC = np.mean(np.abs(DIC_current - DIC_at_last_calc) > threshold * np.abs(DIC_at_last_calc)) # calculate fraction of grid cells with change in DIC above 10%
 
        # (re)calculate carbonate system if it has not yet been calculated or
        # needs to be recalculated
        if idx == 1 or frac_AT > threshold or frac_DIC > threshold: 
            AT_at_last_calc = AT_current.copy()
            DIC_at_last_calc = DIC_current.copy()
            # use CO2SYS with GLODAP data to solve for carbonate system at each grid cell
            # do this for only surface ocean grid cells
            # this is PyCO2SYSv2
            co2sys = pyco2.sys(dic=DIC_current, alkalinity=AT_current,
                               salinity=S, temperature=T, pressure=pressure,
                               total_silicate=Si, total_phosphate=P)
        
            # extract key results arrays
            pCO2 = co2sys['pCO2'] # pCO2 [µatm]
            aqueous_CO2 = co2sys['CO2'] # aqueous CO2 [µmol kg-1]
            R_C = co2sys['revelle_factor'] # revelle factor w.r.t. DIC [unitless]
        
            # calculate revelle factor w.r.t. AT [unitless]
            # must calculate manually, R_AT defined as (dpCO2/pCO2) / (dAT/AT)
            # to speed up, only calculating this in surface
            co2sys_000001 = pyco2.sys(dic=DIC_current[0:ns], alkalinity=AT_current[0:ns]+0.000001, salinity=S[0:ns],
                                   temperature=T[0:ns], pressure=pressure[0:ns], total_silicate=Si[0:ns],
                                   total_phosphate=P[0:ns])
        
            pCO2_000001 = co2sys_000001['pCO2']
            R_A_surf = ((pCO2_000001 - pCO2[0:ns])/pCO2[0:ns]) / (0.000001/AT[0:ns])
            R_A = np.full(R_C.shape, np.nan)
            R_A[0:ns] = R_A_surf
         
            # calculate rest of Nowicki et al. parameters
            beta_C = DIC/aqueous_CO2 # [unitless]
            beta_A = AT/aqueous_CO2 # [unitless]
            K0 = aqueous_CO2/pCO2*rho # [µmol CO2 m-3 (µatm CO2)-1], in derivation this is defined in per volume units so used density to get there
            
            print('\ncarbonate system recalculated (t = ' + str(t[idx]) + ')')
        
        # must (re)calculate A matrix if 1. it has not yet been calculated
        # 2. the carbonate system needs to be recalculated or 3. the time
        # step interval (dt) has changed
        if idx == 1 or frac_AT > threshold or frac_DIC > threshold or np.round(dt[idx],10) != np.round(dt[idx-1],10):
            
            if frac_AT > threshold and threshold > 0: print('frac_AT > ' + str(threshold))
            if frac_DIC > threshold and threshold > 0: print('frac_DIC > ' + str(threshold))
            if dt[idx] != dt[idx-1]: print('dt[' + str(idx) + '] != dt[' + str(idx-1) + ']')
        
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

            # diagnostics
            t0 = time()

            '''

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
            t1 = time() # diagnostics
            '''            
            # set up with PETSc (to reduce time required to convert from scipy)
            LHS = PETSc.Mat().createAIJ([2*m+1, 2*m+1])
            LHS.setUp()

            # set up A0 row (to solve for xCO2)
            A00 = 1 - dt[idx] * (-1 * Patm * np.nansum(gammax * K0)) # using nansum because all subsurface boxes are NaN, we only want surface
            A01 = dt[idx] * (np.nan_to_num(gammax * rho * R_C / beta_C)) # nan_to_num sets all NaN = 0 (subsurface boxes, no air-sea gas exchange)
            A02 = dt[idx] * (np.nan_to_num(gammax * rho * R_A / beta_A))
            
            LHS.setValue(0, 0, A00)
            j_idx = 0
            for j in range(1,m+1):
                if A01[j_idx] != 0:
                    LHS.setValue(0,j,A01[j_idx])
                j_idx += 1
            j_idx = 0
            for j in range(m+1,2*m+1):
                if A02[j_idx] != 0:
                    LHS.setValue(0,j,-1 * A02[j_idx])
                j_idx += 1
            
            # set up A1 row (to solve for ∆DIC)
            A10 = dt[idx] * (np.nan_to_num(-1 * gammaC * K0 * Patm / rho))
            A11 = sparse.eye_array(m, format="csr") - dt[idx] * (TR + sparse.diags(np.nan_to_num(gammaC * R_C / beta_C), format='csr'))
            A12 = dt[idx] * (sparse.diags(np.nan_to_num(gammaC * R_A / beta_A), format='csr'))
            
            # for A10
            i_idx = 0
            for i in range(1,m+1):
                if A10[i_idx] != 0:
                    LHS.setValue(i,0,A10[i_idx]) 
                i_idx += 1
                
            # for A11
            for i in range(m):
                row = 1 + i
                start, end = A11.indptr[i], A11.indptr[i+1]
                cols = 1 + A11.indices[start:end]
                vals = A11.data[start:end]
                LHS.setValues(row, cols, vals)

            # for A12
            for i in range(m):
                row = 1 + i
                start, end = A12.indptr[i], A12.indptr[i+1]
                cols = (m+1) + A12.indices[start:end]
                vals = A12.data[start:end]
                LHS.setValues(row, cols, vals)

           # set up A2 row (to solve for ∆AT)
            A20 = dt[idx] * np.zeros(m)
            A21 = dt[idx] * (0 * TR)
            A22 = sparse.eye_array(m, format="csr") - dt[idx] * (TR)

            # for A20
            i_idx = 0 
            for i in range(m+1,2*m+1):
                if A20[i_idx] != 0:
                    LHS.setValue(i,0,A20[i_idx]) 
                i_idx += 1

            # for A21
            for i in range(m):
                row = (m+1) + i
                start, end = A21.indptr[i], A21.indptr[i+1]
                cols = 1 + A21.indices[start:end]
                vals = A21.data[start:end]
                LHS.setValues(row, cols, vals)

            # for A22
            for i in range(m):
                row = (m+1) + i
                start, end = A22.indptr[i], A22.indptr[i+1]
                cols = (m+1) + A22.indices[start:end]
                vals = A22.data[start:end]
                LHS.setValues(row, cols, vals)
            
            LHS.assemblyBegin()
            LHS.assemblyEnd()
            
            del A00, A01, A02, A10, A11, A12, A20, A21, A22

            t1 = time() # diagnostics
        
        # for now, assuming NaOH (no change in DIC)
        
        # calculate AT required to return to preindustrial pH
        # using DIC at previous time step (initial DIC + modeled change in DIC) and preindustrial pH
        # apply mask (q_AT_locations_mask) at this step to choose which grid cells AT is added to
        DIC_new = DIC + c[1:(m+1), 0]
        AT_new = AT + c[(m+1):(2*m+1), 0]
        # diagnostics
        AT_to_offset = p2.calculate_AT_to_add(pH_preind, DIC_new, AT_new, T, S, pressure, Si, P, AT_mask=p2.flatten(q_AT_locations_mask,ocnmask), low=0, high=200, tol=1e-6, maxiter=50)
        t2 = time()

        # make sure there are no negative values
        if len(AT_to_offset[AT_to_offset<0]) != 0:
            print('error: AT offset is negative')
            break

        # from this offset, calculate rate at which AT must be applied
        # by solving discretized equation for q(t)
        # OCIM manual eqn. 40: (I - dt * TR) * c_t = c_(t-dt) + dt * q(t)
        # solve for q(t): q(t) = [(I - dt * TR) * c_t - c_(t-dt)] / dt
        # --> define c_t as modern DIC + whatever AT required to get to
        #     preind pH
        # --> define c_(t-1) as preindustrial DIC + preindustrial AT (which
        #     we are saying is the same as GLODAP AT)
        # set CDR perturbation equal to this RATE in mixed layer only
        # ∆q_CDR,AT (change in alkalinity due to CDR addition) - final units: [µmol AT kg-1 yr-1]
        #del_q_CDR_AT = (((sparse.eye(TR.shape[0], format="csr") - dt[idx] * TR) * (AT + AT_to_offset) - AT)) / dt[idx]
        #del_q_CDR_AT *= p2.flatten(q_AT_locations_mask, ocnmask) # apply in mixed layer only
        
        # this doesn't work... try q(t) = AT_to_offset / dt [µmol AT kg-1 yr-1]??
        del_q_CDR_AT = AT_to_offset / dt[idx]        
 
        # add in source/sink vectors for ∆AT to q vector
        q[(m+1):(2*m+1)] = del_q_CDR_AT
        
        # calculate right hand side according to Euler backward method
        RHS = c[:,0] + np.squeeze(dt[idx] * q)

        # diagnostics

        # convert matricies from scipy sparse to PETSc to parallelize
        #LHS = PETSc.Mat().createAIJ(size=LHS.shape,
        #                                csr=(LHS.indptr, LHS.indices, LHS.data))
        RHS_petsc = PETSc.Vec().createWithArray(RHS)

        # set up PETSc solver
        ksp = PETSc.KSP().create()
        ksp.setOperators(LHS)
        ksp.setType('lgmres')
        ksp.setGMRESRestart(30)  # restart after 30 iterations

        # set up preconditioner
        ksp.getPC().setType('bjacobi')  # block Jacobi with ILU on each block

        # set convergence tolerances
        ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=1000)

        # set up output array (PETSc vector object)
        c_petsc = LHS.createVecRight()
        c_petsc.setArray(c[:,0])
        ksp.setInitialGuessNonzero(True) # tell solver to use initial guess

        # diagnostics
        t3 = time()

        # solve system (perform time stepping)
        ksp.solve(RHS_petsc, c_petsc)
        c[:,1] = c_petsc.array.copy()

        # diagnostics
        t4 = time()
        print("KSP iters:", ksp.getIterationNumber(), "reason:", ksp.getConvergedReason()) # diagnostics
        print("assemble petsc: ", np.round(t1-t0,5), "solve for AT to add: ", np.round(t2-t1, 5), "set up solver: ", np.round(t3-t2, 5), "solve: ", np.round(t4-t3, 5))

        # check for convergence 
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(
                f"Solver failed to converge! "
                f"Reason code: {ksp.getConvergedReason()}, "
                f"Iterations: {ksp.getIterationNumber()}, "
                f"Residual: {ksp.getResidualNorm():.2e}"
            )

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
        
        c_delDIC_3D[ocnmask == 1] = np.reshape(c_delDIC, (-1,), order='F')
        c_delAT_3D[ocnmask == 1] = np.reshape(c_delAT, (-1,), order='F')
        q_delDIC_3D[ocnmask == 1] = np.reshape(q_delDIC, (-1,), order='F')
        q_delAT_3D[ocnmask == 1] = np.reshape(q_delAT, (-1,), order='F')
            
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
        if 'co2sys_preind' in globals(): del co2sys
        if 'co2sys_000001' in globals(): del co2sys
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