#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A small diversion: testing if I can assume that changes to pCO2 induced by changes in DIC and TA are additive

i.e. is my equation pCO2_CDR = pCO2_CDR,DIC + pCO2_CDR,TA

Created on Mon Apr 21 16:20:50 2025

@author: Reese Barrett
"""

import PyCO2SYS as pyco2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# create test scenarios -> assume 1 µmol/kg CaCO3 added,
# so DIC increases by 1 µmol/kg and TA increases by 2 µmol/kg
# scenario = [starting DIC, DIC perturbation, starting TA, TA perturbation] (all µmol/kg)
scenarios = [[2000, 0.1, 2200, 0.1],
             [2000, 1, 2200, 2],
             [2000, 2, 2200, 4],
             [2000, 5, 2200, 10],
             [2000, 25, 2200, 50],
             [1800, 1, 2000, 2],
             [2200, 1, 2400, 1],]

scenarios = []
num_scenarios = 100
for i in np.logspace(0.0001, 2, num=num_scenarios):
    scenario = [2000, i, 2200, i*2]
    scenarios.append(scenario)
    
#%% define functions to calculate each way

# using pyCO2SYS: calculate change in pCO2 from t = 0 to t = 1
def calculate_pyco2sys(DIC0, del_DIC, TA0, del_TA):
    results0 = pyco2.sys(par1=DIC0, par2=TA0, par1_type=2, par2_type=1)
    results1 = pyco2.sys(par1=DIC0+del_DIC, par2=TA0+del_TA, par1_type=2, par2_type=1)

    pCO20 = results0['pCO2']
    pCO21 = results1['pCO2']
    
    return pCO21 - pCO20


# using my linearization:
# 1. calculate Revelle factor (R_DIC)
# 2. calculate TA equivalent of Revelle factor (R_TA)
# 3. apply ∆pCO2 = R_DIC * (pCO20/DIC0) * ∆DIC + R_TA * (pCO20/TA0) * ∆TA

def calculate_linear(DIC0, del_DIC, TA0, del_TA):
    # calculate Revelle factor
    results0 = pyco2.sys(par1=DIC0, par2=TA0, par1_type=2, par2_type=1)
    R_DIC = results0['revelle_factor']
    pCO20 = results0['pCO2']

    
    # calculate TA equivalent of Revelle factor
    results000001 = pyco2.sys(par1=DIC0, par2=TA0+0.000001, par1_type=2, par2_type=1)
    pCO2000001 = results000001['pCO2']
    R_TA = ((pCO2000001 - pCO20)/pCO20) / (0.000001/TA0) # checked and this gives the same answer for DIC as revelle factor to 7 sig figs so should work for TA
    

    del_pCO2 = R_DIC * (pCO20/DIC0) * del_DIC + R_TA * (pCO20/TA0) * del_TA
    
    return del_pCO2

#%% calculate results
print('')
py_pCO2 = np.zeros(num_scenarios)
linear_pCO2 = np.zeros(num_scenarios)
for i, scenario in enumerate(scenarios):
    py_pCO2[i] = calculate_pyco2sys(scenario[0], scenario[1], scenario[2], scenario[3])
    linear_pCO2[i] = calculate_linear(scenario[0], scenario[1], scenario[2], scenario[3])

    #print('DIC = ' + str(scenario[0]) + ', ∆DIC = ' + str(scenario[1]) + ', TA = ' + str(scenario[2]) + ', ∆TA = ' + str(scenario[3]) + ' (all µmol/kg)')
    #print('with pyCO2SYS: ∆pCO2 = ' + str(np.round(py_pCO2,5)) + ' µatm')
    #print('with additive assumption: ∆pCO2 = ' + str(np.round(linear_pCO2,5)) + ' µatm\n')

#%% plot results
rcParams['font.family'] = 'Avenir'

amt_CaCO3 = [scenario[1] for scenario in scenarios] # extract amount of CaCO3 added in each scenario [µmol kg-1]

fig = plt.figure(figsize=(4.5,2))
ax = fig.gca()

ax.semilogx(np.array(amt_CaCO3), py_pCO2, label='Calculated with PyCO$_{2}$SYS',c='#1649b3')
ax.semilogx(np.array(amt_CaCO3), linear_pCO2, label='Calculated with linearization',c='#DA9497')

ax.set_xlabel('Amount of CaCO$_{3}$ Added (µmol kg$^{-1}$)')
ax.set_ylabel('Change in Seawater pCO$_{2}$')
plt.suptitle('Simulation of Ocean Alkalinity Enhancement', y=1.09)
plt.title('($A_\mathrm{T}$ = 2200 µmol kg$^{-1}$, DIC = 2000 µmol kg$^{-1}$)',fontsize=10)
plt.legend()









