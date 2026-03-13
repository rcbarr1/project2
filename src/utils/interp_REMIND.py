'''
Created on Wed Mar 11 2026

Interpolating REMIND data from Maria to a smooth yearly curve for use in LCA simulations

@author: Reese C. Barrett
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

#%% data
co2_trajs = np.genfromtxt('./src/utils/pyTRACE/pyTRACE/data/CO2Trajectories_OLD.txt', 
                            delimiter='\t', 
                            dtype=None,
                            skip_header=0,
                            missing_values=None, 
                            filling_values=np.NaN,
                            encoding=None,
                            )
co2_trajs = np.vstack(co2_trajs.tolist())

#%% remove data with NaN in co2 column
co2_trajs_no_nans = co2_trajs[~np.isnan(co2_trajs).any(axis=1)]

years = co2_trajs_no_nans[:, 0] 
remind_co2 = co2_trajs_no_nans[:, 10]

#%% perform interpolation
interp = PchipInterpolator(years, remind_co2, extrapolate=True)

years_new = np.arange(0, 2500)
remind_co2_interp = interp(years_new)

# set constant co2 level after 2100
remind_co2_interp[2100::] = remind_co2_interp[2100]

#%% save unadjusted CO2 values to CO2Trajectories.txt
co2_trajs_interp = co2_trajs.copy()
co2_trajs_interp[:,10] = remind_co2_interp

np.savetxt('./src/utils/pyTRACE/pyTRACE/data/CO2Trajectories.txt', co2_trajs_interp, delimiter='\t')
#%% adjust CO2 values via eqn. 5 from Carter et al. (2025)
# https://essd.copernicus.org/articles/17/3073/2025/essd-17-3073-2025.html
# this accounts for the fact there is incomplete air-sea gas exchange and a lag in ocean mixing time relative to the atmosphere
# pCO2, oce, year = xCO2, atm, year - 0.144 * (xCO2, atm, year - xCO2, atm, year-65)
# start making adjustment after year 65 following previous CO2Trajectories.txt & CO2TrajectoriesAdjusted.txt

remind_co2_interp_adj = np.zeros_like(remind_co2_interp)
remind_co2_interp_adj[0:65] = remind_co2_interp[0:65] # keep values <65 unadjusted
remind_co2_interp_adj[65::] = remind_co2_interp[65:] - 0.144 * (remind_co2_interp[65:] - remind_co2_interp[:-65])

# pull in rest of adjusted data
co2_trajs_adj = np.genfromtxt('./src/utils/pyTRACE/pyTRACE/data/CO2TrajectoriesAdjusted_OLD.txt', 
                            delimiter='\t', 
                            dtype=None,
                            skip_header=0,
                            missing_values=None, 
                            filling_values=np.NaN,
                            encoding=None,
                            )

co2_trajs_adj = np.vstack(co2_trajs_adj.tolist())
co2_trajs_adj = np.hstack((co2_trajs_adj, np.expand_dims(remind_co2_interp_adj, axis=1)))

#%% save to CO2 adjusted trajectories file
np.savetxt('./src/utils/pyTRACE/pyTRACE/data/CO2TrajectoriesAdjusted.txt', co2_trajs_adj, delimiter='\t')

# %% plot results
fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

ax.scatter(years, remind_co2, marker='o', facecolors='none', edgecolors='steelblue', label='Preindustrial CO$_{2}$ and REMIND data')
ax.plot(years_new, remind_co2_interp, label='Interpolated REMIND data')
ax.plot(years_new, remind_co2_interp_adj, label='REMIND data adjusted for TRACE')

ax.set_xlim([1900, 2150])
ax.set_ylim([200, 550])
ax.set_xlabel('Year')
ax.set_ylabel('Atmospheric CO$_{2}$ Concentration (ppm)')
ax.legend()

#%% make sure everything looks good by plotting all co2 trajectories & adjusted trajectories
# trajectories
co2_trajs_all = np.genfromtxt('./src/utils/pyTRACE/pyTRACE/data/CO2Trajectories.txt',
                              delimiter='\t', 
                              dtype=None,
                              skip_header=0,
                              missing_values=None, 
                              filling_values=np.NaN,
                              encoding=None
                              )

co2_trajs_all = np.vstack(co2_trajs_all.tolist())
years = co2_trajs_all[:,0]
labels = ['Historical/Linear', 'SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 
          'SSP3-7.0_lowNTCF', 'SSP4-3.4', 'SSP4-6.0', 'SSP5-3.4_over', 'REMIND']

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

for i, label in enumerate(labels):
    ax.plot(years, co2_trajs_all[:,i+1], label=label)

ax.set_xlim([1900, 2150])
#ax.set_ylim([200, 550])
ax.set_xlabel('Year')
ax.set_ylabel('Atmospheric CO$_{2}$ Concentration (ppm)')
ax.legend()

# adjusted trajectories
co2_trajs_adj_all = np.genfromtxt('./src/utils/pyTRACE/pyTRACE/data/CO2TrajectoriesAdjusted.txt',
                              delimiter='\t', 
                              dtype=None,
                              skip_header=0,
                              missing_values=None, 
                              filling_values=np.NaN,
                              encoding=None
                              )

co2_trajs_adj_all = np.vstack(co2_trajs_adj_all.tolist())
years = co2_trajs_adj_all[:,0]
labels = ['Historical/Linear', 'SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 
          'SSP3-7.0_lowNTCF', 'SSP4-3.4', 'SSP4-6.0', 'SSP5-3.4_over', 'REMIND']

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

for i, label in enumerate(labels):
    ax.plot(years, co2_trajs_adj_all[:,i+1], label=label)

ax.set_xlim([1900, 2150])
#ax.set_ylim([200, 550])
ax.set_xlabel('Year')
ax.set_ylabel('Atmospheric CO$_{2}$ Concentration (ppm)')
ax.legend()

# %%
