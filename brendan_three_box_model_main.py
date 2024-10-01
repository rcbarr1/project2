# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:54:08 2024

@author: Reese Barrett

Translation of Brendan's GIGTempAllSensTAForced.m to python to teach myself 
what is happening with this model

Three-box model of the impact of simulated sensitivity of carbonate mineral
formation rates on CO2 and temperature in glacial interglacial cycles

Assumptions:
    1. The Earth is initially at steady state in all respects
    2. The input of TA through rivers equals 1 µmol kg-1 yr-1 into the surface
       ocean. This is derived from the 33 Tmol yr-1 TA input into the ocean
       through rivers (Cai et al., 2008) and the assumption of about 3e6 Tmol
       TA in the ocean (2200 µmol kg-1 average ocean TA with a 1.4e21 kg
       ocean) for a ~100,000 year TA residence time (~2200 µmol kg-1 / 1 µmol
       kg-1 yr-1 * 51 [kg ocean/kg surface ocean]). Due to assumption 1,
       initial burial is set to equal the riverine input.
                   REESE NOTE: did we not decide 51 kg ocean/kg surface ocean is incorrect? this would also affect SizRun variable!
    3. There are no external sources or sinks of borate, and riverine inputs
       and CaCO3 cycling are the only processes affecting marine alkalinity.
                   REESE NOTE: we would need to include biological pump processes if we wanted to include effects of remineralization in the model (still part of CaCO3 cycling though technically)
    4. The surface ocean and atmosphere increase temperature by 3ºC per
       doubling of CO2 from 280 µatm
    5. The deep ocean temperature is fixed by Stommel's Mixed Layer Demon. This
       assumption isn't necessary, but it is simple and I'm [Brendan is] trying
       to avoid including feedbacks that are not directly related to
       interactions between CO2, temperature, and CaCO3 burial.
                   REESE NOTE: god science was so weird back in the day
"""
import brendan_three_box_model as b3bm
import numpy as np
import PyCO2SYS as pyco2
import time
import matplotlib.pyplot as plt

# need to load in 'OAAnom' here once Brendan sends it to me (ocean acidification anomalies?)
# file doens't load exactly correctly, so lines 43-45 are fixing that to match matlab
OAAnom = b3bm.loadmat('/Users/Reese_1/Documents/Research Projects/project2/brendan/OAAnom.mat')
OAAnom = OAAnom['OAAnom']
OAAnom = np.stack((np.arange(800000, 0, -1000), OAAnom)).T
OAAnom = np.concatenate((OAAnom, np.array([0, 0], ndmin=2)), axis=0)
OAAnom[798, 1] = 0
OAAnom[799, 1] = 0

# set up all 16 experiments!
# four forcing types:
    # 1. TempForced: 300 kyr simulation with a temperature change of -1ºC after 10 ka that becomes a temperature perturbation of +1ºC afer 200 ka (for a net swing of 2ºC in the 200,000th year) 
    # 2. CO2Forced: 300 kyr simulation with a CO2 removal of ~0.082 PgC yr-1 (i.e. ~0.30 PgCO2, or exactly enough to increase the surface ocean by 0.5 µmol DIC kg-1 yr-1) from year 5,000 to year 10,000 of the simulation (effectively resulting in a ~1ºC temperature decrease) and a CO2 removal of 0.16 µmol kg-1 yr-1 from year 200,000 to 201,000 (for a ~2 ºC increase, or net increase of ~1ºC)
    # 3. OmegaForced: (control run??) an 800 kyr simulation using the anomaly from the mean Omega in the Omega timeseries approximated from the EPICA temperature and pCO2 records
    # 4. TAForced: 400 kyr simulation that doubles the TA from rivers for 5000 years starting in year 5000 and that then removes twice the anomalously-added TA from years 200,000 to 205,000. This initial perturbation is intended to represent an attempt at ocean alkalinity enhancement. The reversal in year 200,000 is only plausible if rivers become acidic and is included only to assess hysterisis.
    # REESE NOTE: is there a better way to normalize to comparem magnitudes of the effects?

# use for loop to iterate through different subgroups of 16 experiments
#for experiment in range(12,16):
for experiment in range(13,14):
    if experiment == 1:     # temperature forced with all sensitivities
        TempForced = 1      # flag setting whether the experiment is forced by a tempearture perturbation (i.e. orbital parameters change)
        CO2Forced = 0       # flag setting whether the experiment is forced by a CO2 perturbation
        OmegaForced = 0     # flag setting if experiment is based on omega data (control run)
        TAForced = 0        # flag setting whether to do any geoengineering
        HTPSense = 1        # flag setting if the response of the surface production of CaCO3 and export is sensitive to carbonate mineral saturation
        BurialSense = 1     # flag setting the response of the burial of CaCO3 to carbonate mineral saturation
        SizRun = 150000     # the number of timesteps for the run, with each timestep equalling the length of time required to flush 1/10th of the surface ocean into the deep ocean. If the deep ocean has a residence time of 1000 years and a volume of 50 times the surface ocean, then a timestep is about 2 years.
        filename = 'GIGOutputV2TempAllSens'
    elif experiment == 2:   # temperature forced with no sensitivites
        TempForced = 1
        CO2Forced = 0
        OmegaForced = 0
        TAForced = 0
        HTPSense = 0
        BurialSense = 0
        SizRun = 150000
        filename = 'GIGOutputV2TempNoSens'
    elif experiment == 3:   # temperature forced with only gradient
        TempForced = 1
        CO2Forced = 0
        OmegaForced = 0
        TAForced = 0
        HTPSense = 1
        BurialSense = 0
        SizRun = 150000
        filename = 'GIGOutputV2TempNoBurialSens'
    elif experiment == 4:   # temperature forced with burial but no gradient
        TempForced = 1
        CO2Forced = 0
        OmegaForced = 0
        TAForced = 0
        HTPSense = 0
        BurialSense = 1
        SizRun = 150000
        filename = 'GIGOutputV2TempNoHTPSens'
    elif experiment == 5:   # CO2 forced with all sensitivities
        TempForced = 0
        CO2Forced = 1
        OmegaForced = 0
        TAForced = 0
        HTPSense = 1
        BurialSense = 1
        SizRun = 150000
        filename = 'GIGOutputV2CO2AllSens'
    elif experiment == 6:   # CO2 forced with no sensitivities
        TempForced = 0
        CO2Forced = 1
        OmegaForced = 0
        TAForced = 0
        HTPSense = 0
        BurialSense = 0
        SizRun = 150000
        filename = 'GIGOutputV2CO2NoSens'
    elif experiment == 7:   # CO2 forced with only gradient
        TempForced = 0
        CO2Forced = 1
        OmegaForced = 0
        TAForced = 0
        HTPSense = 1
        BurialSense = 0
        SizRun = 150000
        filename = 'GIGOutputV2CO2NoBurialSens'
    elif experiment == 8:   # CO2 forced with burial but no gradient
        TempForced = 0
        CO2Forced = 1
        OmegaForced = 0
        TAForced = 0
        HTPSense = 0
        BurialSense = 1
        SizRun = 150000
        filename = 'GIGOutputV2CO2NoHTPSens'
    elif experiment == 9:   # Omega forced with all sensitivities
        TempForced = 0
        CO2Forced = 0
        OmegaForced = 1
        TAForced = 0
        HTPSense = 1
        BurialSense = 1
        SizRun = 150000
        filename = 'GIGOutputV2OmegaAllSens'
    elif experiment == 10:  # Omega forced with no sensitivities
        TempForced = 0
        CO2Forced = 0
        OmegaForced = 1
        TAForced = 0
        HTPSense = 0
        BurialSense = 0
        SizRun = 150000
        filename = 'GIGOutputV2OmegaNoSens'
    elif experiment == 11:  # Omega forced with only gradient
        TempForced = 0
        CO2Forced = 0
        OmegaForced = 1
        TAForced = 0
        HTPSense = 1
        BurialSense = 0
        SizRun = 150000
        filename = 'GIGOutputV2OmegaNoBurialSens'
    elif experiment == 12:  # Omega forced with burial but no gradient
        TempForced = 0
        CO2Forced = 0
        OmegaForced = 1
        TAForced = 0
        HTPSense = 0
        BurialSense = 1
        SizRun = 150000
        filename = 'GIGOutputV2OmegaNoHTPSens'
    elif experiment == 13:  # TA forced with all sensitivities
        TempForced = 0
        CO2Forced = 0
        OmegaForced = 0
        TAForced = 1
        HTPSense = 1
        BurialSense = 1
        SizRun = 1000
        filename = 'GIGOutputV2TAAllSens'
    elif experiment == 14:  # TA forced with no sensitivities
        TempForced = 0
        CO2Forced = 0
        OmegaForced = 0
        TAForced = 1
        HTPSense = 0
        BurialSense = 0
        SizRun = 150000
        filename = 'GIGOutputV2TANoSens'
    elif experiment == 15:  # TA forced with only gradient
        TempForced = 0
        CO2Forced = 0
        OmegaForced = 0
        TAForced = 1
        HTPSense = 1
        BurialSense = 0
        SizRun = 150000
        filename = 'GIGOutputV2TANoBurialSens'
    elif experiment == 16:  # TA forced with burial but no gradient
         TempForced = 0
         CO2Forced = 0
         OmegaForced = 0
         TAForced = 1
         HTPSense = 0
         BurialSense = 1
         SizRun = 150000
         filename = 'GIGOutputV2TANoHTPSens'
         
    # preallocate variables for efficiency
    deepBC = np.full((SizRun), np.nan) # deep basic inorganic carbon, half carbonate alkalinity
    surfBC = np.full((SizRun), np.nan) # surface basic inorganic carbon, half carbonate alkalinity
    
    deepAC = np.full((SizRun), np.nan) # deep acidic inorganic carbon, remainder of DIC
    surfAC = np.full((SizRun), np.nan) # surface acidic inorganic carbon, remainder of DIC
    
    temp_pert = np.full((SizRun), np.nan) # magintude of the temperature perturbation induced by unconsidered feedbacks (i.e. orbital forcing)
    
    omega_fraction = np.full((SizRun), np.nan) # current aragonate saturation minus 1, divided by the initial aragonate saturation minus 1
    
    htp = np.full((SizRun), np.nan) # hard tissue pump, equalling the µmol of basic carbon (CaCO3) removed from the surface ocean
    
    burial = np.full((SizRun), np.nan) # the burial of CaCO3 lost from the surface ocean. the residual between htp and burial is added to the deep ocean as dissolved basic carbon
    
    ocean_BC_tot = np.full((SizRun), np.nan) # the total quantity of dissovled CaCO3 in the ocean
    ocean_DIC_tot = np.full((SizRun), np.nan) # the total quantity of DIC in the ocean
    added_carbon = np.full((SizRun), np.nan) # sum of carbon added to total earth carbon system
    total_CO2 = np.full((SizRun), np.nan) # total amount of carbon in earth system
    
    pCO2 = np.full((SizRun), np.nan) # atmospheric pCO2 (ppm)
    
    surf_temp = np.full((SizRun), np.nan) # surface ocean temperature (ºC)
    
    # set initial conditions
    n = 0
    overturn_fraction = 0.1 # with this, each time step is about 2 years
    river_in = 1 # input of 1 µmol kg-1 TA into the surface ocean per year (see assumptions above for derivation)
    burial[n] = river_in #initially at steady state, so inputs = outputs
    deep2surf_mass_ratio = 50 # ratio of the deep ocean mass divided by the surface ocean mass
    surfBC[n] = 1150 # initial quantity of basic carbon in the surface ocean, equivalent to a TA of 2300 (this may be an overestimate when considering borate alkalinity, but this model is only sensitive to the surface-to-deep TA gradient, and deep TA is similarly overestimated)
    deepBC[n] = 1250 # initial quantity of basic carbon in the deep ocean, equivalent to a TA of 2500
    pCO2[n] = 280 # initial atmospheric pCO2
    surf_temp[n] = 13 # initial surface ocean temperature
    omega_fraction[n] = 1 # scaling factof for htp and burial referenced to model's steady (initial) state and expressed as a ratio (hence it is initially 1)
    htp[n] = (deepBC[n] - surfBC[n]) * overturn_fraction * river_in # setting hard tissue pump to be the value required to offset deep-to-surface TA transport through overturning
    
    # use CO2SYS to calculate DIC at the surface TA, pCO2, and temperature
    # - many unimportant assumptions are hidden in this calculation
    # - this is "impatient" to run faster with a higher pH convergence tolerance 
    #   that is still much more precise than needed for this model
    out_surf = pyco2.sys(surfBC[n] * 2, pCO2[n], 1, 4, 35, surf_temp[n], 0, surf_temp[n], 0) # surface DIC
    surfDIC = out_surf['dic']
    surfAC[n] = surfDIC - surfBC[n]
    surfOmega = out_surf['saturation_aragonite_out'] - 1 # degree of supersaturation
   
    out_deep = pyco2.sys(surfBC[n] * 2, pCO2[n], 1, 4, 35, 4, 0, 4, 0) # deep DIC, also assuming pressure = 0 here though
    deepDIC = out_deep['dic']
    deepAC[n] = deepDIC - deepBC[n]
    
    # sum of total initial marine carbon
    ocean_DIC_tot[n] = surfAC[n] + surfBC[n] + deep2surf_mass_ratio*(deepAC[n] + deepBC[n])
        
    # sum of total initial earth system carbon
    # - the final term assumes the atmosphere initially holds 1/40th the carbon that the ocean does
    total_CO2[n] = ocean_DIC_tot[n] - np.nansum(burial) + river_in + pCO2[n]/280*ocean_DIC_tot[n]/40
    added_carbon[n] = 0
    
    # begin iterations
    for n in range(n+1,SizRun):
        # initializing the temperature, alkalinity, and CO2 perturbations, overwritten later in temperature/alkalinity/CO2 experiments
        temp_pert[n] = 0
        TA_pert = 0
        added_carbon[n] = added_carbon[n-1]
        
        # set temperature perturbation, if necessary
        if TempForced == 1:
            if n >= 5000: temp_pert[n] = -1
            if n >= 100000: temp_pert[n] = 1
            
        # set CO2 perturbation, if necessary
        if CO2Forced == 1:
            if n >= 5000 and n <= 10000: added_carbon[n] = added_carbon[n-1] - 0.5
            if n >= 100000 and n <= 105000: added_carbon[n] = added_carbon[n-1] + 1
        
        # set TA perturbation, if necessary
        if TAForced == 1:
            if n >= 5000 and n <= 10000: TA_pert = 1
            if n >= 100000 and n <= 105000: TA_pert = -2
            
        # recalculating the temperature bawed on current perturbations and
        # final pCO2 of atmosphere during last time step
        surf_temp[n] = surf_temp[0] + temp_pert[n] + 3 * np.log2(pCO2[n-1]/280) # should probably look up where this relationship comes from, kind of cool
        
        # using CO2SYS to calculate the carbonate system properties in the surface ocean
        # - several assumptions, including constant salinity of 35
        out_surf = pyco2.sys(surfBC[n-1] * 2, pCO2[n-1], 1, 4, 35, surf_temp[n], 0, surf_temp[n], 0)
        
        # repeating for cooled surface waters entering deep ocean
        # - assuming constant 4 ºC deep temperature (and standard pressure, why?)
        out_new_deep = pyco2.sys(surfBC[n-1] * 2, pCO2[n-1], 1, 4, 35, 4, 0, 4, 0)
        
        # calculate new aragonite supersaturation
        omega_fraction[n] = ((out_surf['saturation_aragonite_out'] - 1) / surfOmega) # cubic term can be added to test higher sensitivity/different parameterizations 
        if OmegaForced == 1:
            omega_anom = np.interp(n*2, OAAnom[:,0][::-1], OAAnom[:,1][::-1])
            omega_fraction[n] = ((out_surf['saturation_aragonite_out'] - 1 + omega_anom) / surfOmega) # same as above

        # flag for hard tissue pump sensitivity to the carbonate mineral supersaturation
        if HTPSense == 1:
            htp[n] = htp[0] * omega_fraction[n] # sensitivity case: scaling the hard tissue pump by supersaturation changes
        else:
            htp[n] = htp[0] # no sensitivity case, fixed hard tissue pump strength
        
        # flag for burial sensitivity to the carbonate mineral supersaturation
        if BurialSense == 1:
            burial[n] = river_in * omega_fraction[n] # sensitivity case: scaling burial by supersaturation changes
        else:
            burial[n] = river_in # no sensitivity case, fixed hard tissue pump strength
        
        # surface basic carbon persists from the previous timestep, minus what is lost to the htp and deep water formation and plus what is gained from rivers, deep water entrainment, and geoengineering
        surfBC[n] = surfBC[n-1] - htp[n] + river_in + TA_pert + overturn_fraction*(deepBC[n-1] - surfBC[n-1])
        
        # surface acidic carbon is set by gas exchange calculated previously, this is expressing that AC = DIC - BC
        surfAC[n] = out_surf['dic'] - out_surf['alkalinity']/2
        
        # the deep basic carbon persists, but is modified by gains from the
        # hard tissue pump (gains that are attenuated by burial of a fraction
        # of the HTP) and gains and losses from surface-to-deep seawater exchange
        deepBC[n] = deepBC[n-1] + (htp[n] - burial[n])/deep2surf_mass_ratio - overturn_fraction*(deepBC[n-1] - surfBC[n-1])/deep2surf_mass_ratio
        
        # deep acidic carbon persists, but is modulated by losses to and gains
        # from the surface. the surface acidic carbon term is replaced by the
        # difference between the surface DIC at 4 ºC and the surface basic
        # carbon (which is unaffected by temperature or air-sea equilibration)
        # because freshly formed deep water is assumed to have time to
        # equilibrate with the atmosphere before subducting
        deepAC[n] = deepAC[n-1] - overturn_fraction*(deepAC[n-1] - out_new_deep['dic'] + surfBC[n])/deep2surf_mass_ratio
        
        # summing total amount of basic carbon in ocean (diagnostic)
        ocean_BC_tot[n] = surfBC[n] + deepBC[n]*deep2surf_mass_ratio
        
        # summing total amount of carbon in the ocean
        ocean_DIC_tot[n] = ocean_BC_tot[n] + surfAC[n] + deepAC[n]*deep2surf_mass_ratio
        
        # summing total amount of carbon in the system
        # = initial carbon (scaled by 41/40 to account for initial atmospheric carbon)
        # + riverine basinc carbon contribution
        # + added co2 contribution
        # - burial of carbonate minerals
        total_CO2[n] = ocean_DIC_tot[0]*41/40 - np.nansum(burial) + TA_pert + river_in*(n+1) + added_carbon[n]
        
        # atmospheric CO2 calculated by subtracting the total oceanic carbon from the total carbon present
        # pCO2 calculated by scaling the initial pCO2 by the ratio of the atmospheric CO2 at this timestep to atmospheric CO2 initially
        pCO2[n] = pCO2[0] * (total_CO2[n] - ocean_DIC_tot[n])/(total_CO2[0] - ocean_DIC_tot[0])
        
# diagnostic plotting and printing

# Calculate x-axis values for the first two plots
x_vals = np.arange(0, n+1) * 1000 * overturn_fraction / deep2surf_mass_ratio

# Create figure
plt.figure(1)
plt.clf()

# Create first plot with two y-axes (PCO2 and SurfTemp)
fig, ax1 = plt.subplots()

# First plot (PCO2 vs x_vals)
ax1.plot(x_vals, pCO2, 'b-')
ax1.set_xlabel('Years')
ax1.set_ylabel('pCO2', color='b')

# Create second y-axis
ax2 = ax1.twinx()
ax2.plot(x_vals, surf_temp, 'r-')
ax2.set_ylabel('surface temperature', color='r')

# Overlay the TempPert plot
#plt.plot(x_vals, temp_pert, 'g-')

# Show the plot
plt.show()

# Record and print elapsed time
start_time = time.time()
toc = time.time() - start_time
print(f'Time elapsed: {toc:.2f} seconds')



















