'''
Make higher resolution gridded TRACE product to speed up Canth calculations with better accuracy
than published TRACE product
https://zenodo.org/records/15692788


Reese C. Barrett
March 26, 2026
'''

#%%
import numpy as np
import os
from src.utils import project2 as p2
import xarray as xr
from pyTRACE import trace
from tqdm import tqdm
import matplotlib.pyplot as plt

# load model architecture
data_path = './data/'

# load transport matrix (OCIM2-48L, from Holzer et al., 2021)
# transport matrix is referred to as "A" vector in John et al., 2020 (AWESOME OCIM)
TR = p2.loadmat(data_path + 'OCIM2_48L_base/OCIM2_48L_base_transport.mat')
TR = TR['TR']

# open up rest of data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()

model_lat = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()    # ºN
model_lon = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()     # ºE
model_depth = model_data['tz'].isel(longitude=0, latitude=0).to_numpy() # m below sea surface
model_vols = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy() # m^3

T_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy') # temperature [ºC]
S_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy') # salinity [unitless]

rho = 1025 # seawater density [kg m-3]

#%% upload old TRACE gridded product for comparison

trace_gridded = xr.open_dataset(data_path + 'TRACE_gridded/CanthFromTRACECO2Pathway2.nc',
                                decode_times=False)

# these datasets have years 1750, 1800, 1850, 1900, 1950, 1980, 1994.5, 2000, 2002, 2007, 2010
# 2014, 2020, 2030, 2050, 2100, 2200, 2300, 2400, 2500

#%% make TRACE gridded product on OCIM grid every year from 2000 to 2100 by calling TRACE
scenario_dict = {'none' : 1, 'ssp119': 2, 'ssp126' : 3, 'ssp245' : 4, 'ssp370' : 5,
                 'ssp360_NTCF' : 6, 'ssp434' : 7, 'ssp460' : 8, 'ssp534_OS' : 9, 'REMIND' : 10}

# transpose to match requirements for PyTRACE (lon, lat, depth)
T_3D_T = T_3D.transpose([1, 0, 2])
S_3D_T = S_3D.transpose([1, 0, 2])
ocnmask_T = ocnmask.transpose([1, 0, 2])

predictor_measurements = np.vstack([S_3D_T.flatten(order='F'), T_3D_T.flatten(order='F')]).T
predictor_measurements = predictor_measurements[ocnmask_T.flatten(order='F').astype(bool)]

# create list of longitudes (ºE), latitudes (ºN), and depths (m) in TRACE format
# this order is required for TRACE
lon, lat, depth = np.meshgrid(model_lon, model_lat, model_depth, indexing='ij')
ocim_coordinates = np.array([lon.ravel(order='F'), lat.ravel(order='F'), depth.ravel(order='F'), ]).T # reshape meshgrid points into a list of coordinates to interpolate to
ocim_coordinates = ocim_coordinates[ocnmask_T.flatten(order='F').astype(bool)]

# loop over all scenarios
output_path = './data/TRACE_gridded/'
years = np.arange(2000, 2101)  # 2000 to 2100 inclusive

for scenario_name, scenario_idx in scenario_dict.items():
    print(f'Processing scenario: {scenario_name}')
    
    # initialize list to store canth arrays for all years
    canth_time_series = []
    
    # loop over all years
    for year in tqdm(years):
        dates = year * np.ones([ocim_coordinates.shape[0], 1])
        
        # first TRACE call
        trace_output = trace(output_coordinates=ocim_coordinates,
                            dates=dates[:,0],
                            predictor_measurements=predictor_measurements,
                            predictor_types=[1, 2],
                            atm_co2_trajectory=scenario_idx,
                            verbose_tf=False)
        
        # right now, pyTRACE is estimating some preformed P and Si as <0, which is
        # resulting in NaN Canth. for now, am fixing this by setting <0 values =0.
        # also pulling preformed AT and scale factors to make second pyTRACE
        # calculation faster
        preformed_p = trace_output.preformed_p.values
        preformed_p[preformed_p < 0] = 0
        preformed_ta = trace_output.preformed_ta.values
        preformed_si = trace_output.preformed_si.values
        preformed_si[preformed_si < 0] = 0
        scale_factors = trace_output.scale_factors.values
        
        # second TRACE call with preformed values
        trace_output = trace(output_coordinates=ocim_coordinates,
                            dates=dates[:,0],
                            predictor_measurements=predictor_measurements,
                            predictor_types=[1, 2],
                            atm_co2_trajectory=scenario_idx,
                            preformed_p=preformed_p,
                            preformed_ta=preformed_ta,
                            preformed_si=preformed_si,
                            scale_factors=scale_factors,
                            verbose_tf=False)
        
        # reshape back to 3D and store
        canth_3D = p2.make_3D(trace_output.canth.values, ocnmask_T).transpose([1, 0, 2])
        canth_time_series.append(canth_3D)
    
    # convert time series list to xarray dataset
    canth_xr = xr.DataArray(
        np.array(canth_time_series),
        dims=['time', 'lat', 'lon', 'depth'],
        coords={
            'time': years,
            'lat': model_lat,
            'lon': model_lon,
            'depth': model_depth
        },
        name='Canth'
    )
    
    # create dataset and save
    ds_output = canth_xr.to_dataset()
    output_filename = output_path + f'OCIM_CanthFromTRACECO2Pathway{scenario_idx}.nc'
    ds_output.to_netcdf(output_filename)
    print(f'Saved: {output_filename}')

# %% test trace data produced
scenario_dict = {'none' : 1, 'ssp119': 2, 'ssp126' : 3, 'ssp245' : 4, 'ssp370' : 5,
                 'ssp360_NTCF' : 6, 'ssp434' : 7, 'ssp460' : 8, 'ssp534_OS' : 9, 'REMIND' : 10}

# plot total Canth over time
fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

for scenario_name, scenario_idx in scenario_dict.items():
    
    trace_gridded_new = xr.open_dataset(data_path + f'TRACE_gridded/OCIM_CanthFromTRACECO2Pathway{scenario_idx}.nc',
                                        decode_times=False)

    model_vols_ds = xr.DataArray(model_vols, dims=["lat", "lon", "depth"],
                                 coords={"lat": trace_gridded_new.lat,
                                         "lon": trace_gridded_new.lon,
                                         "depth": trace_gridded_new.depth})

    Canth = trace_gridded_new * model_vols_ds * rho / 1000000

    ax.plot(trace_gridded_new.time.values, Canth.sum(dim=["lat", "lon", "depth"]).Canth.values, label=scenario_name)

ax.set_xlabel('Year')
ax.set_ylabel('Total Canth (mol C)')
ax.legend()

#%% plot how surface Canth changes over time
scenario_name = 'ssp119'
scenario_idx = scenario_dict[scenario_name]

trace_gridded_new = xr.open_dataset(data_path + f'TRACE_gridded/OCIM_CanthFromTRACECO2Pathway{scenario_idx}.nc',
                                    decode_times=False)

for time in trace_gridded_new.time:
    p2.plot_surface3d(model_lat, model_lon, trace_gridded_new.Canth.sel(time=time), 0, 0, 100, 'viridis',
                      f'Canth at t = {time.values}')

# %%
