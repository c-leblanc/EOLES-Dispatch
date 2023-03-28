"""
EOLES-Dispatch Model by Clément Leblanc, January 2022
***
Based the EOLES Model developped by Behrang Shirizadeh, Quentin Perrier and Philippe Quirion (May 2021)
and adapted in Python by Nilam De Oliveira-Gill, June 2021

NB: Set the directory containing this script as working directory before running
"""


"""IMPORTS

Import modules and libraries needed for the programm 
"""
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import time
import itertools
import os
import gc
import subprocess

#%% Functions' Def
### Definition of functions used to launch runs of the model


def set_model():
    ### INITIALISATION OF THE MODEL
    model = pyo.ConcreteModel()
    
    #Dual Variable, used to get the marginal value of an equation.
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    ### INPUTS
    ## TIME-SERIES - SUPPLY / DEMAND
        # Demand profile in each our in GW
    demand = pd.read_csv("inputs/demand.csv", header=None, names=['a','h','demand']).set_index(['a','h']).squeeze(axis=1)
        # Non-market dependant production in GW
    nmd = pd.read_csv("inputs/nmd.csv", header=None, names=['a','h','nmd']).set_index(['a','h']).squeeze(axis=1)
        # Prices in neighboring (non-modeled) countries in EUR/MWh (exogenous)
    exoPrices = pd.read_csv("inputs/exoPrices.csv", header=None, names=['exo_a','h','exoPrice']).set_index(['exo_a','h']).squeeze(axis=1)
        # Production profiles of VRE
    load_factor = pd.read_csv("inputs/vre_profiles.csv", header=None, names=['a','vre','h','load_factor']).set_index(['a','vre','h']).squeeze(axis=1)
        # Monthly lake inflows in TWh
    lake_inflows = pd.read_csv("inputs/lake_inflows.csv", header=None, names=['a','month','lake_inflows']).set_index(['a','month']).squeeze(axis=1)    
    
    
    ## INSTALLED SYSTEM
        # Capacities of the technologies in GW
    capa = pd.read_csv("inputs/capa.csv", header=None, names=['a','tec','capa']).set_index(['a','tec']).squeeze(axis=1)
        # Storage inflow capacity in GW (max amount stored in an hour)
    capa_in = pd.read_csv("inputs/capa_in.csv",header=None, names=['a','sto','capa']).set_index(['a','sto']).squeeze(axis=1)
        # Maximum volume of energy can be stored in TWh
    stockMax = pd.read_csv("inputs/stockMax.csv",header=None, names=['a','sto','capa']).set_index(['a','sto']).squeeze(axis=1)
        # EAF (equivalent availability factor) - maximum yearly average CF
    eaf = pd.read_csv("inputs/yEAF.csv",header=None, names=['a','thr','capa']).set_index(['a','thr']).squeeze(axis=1)
        # Maximum availability factor (hourly constraint)
    maxaf = pd.read_csv("inputs/maxAF.csv",header=None, names=['a','thr','capa']).set_index(['a','thr']).squeeze(axis=1)
        # Weekly max AF for nuclear (based on observed historic data)
    nucMaxAF = pd.read_csv("inputs/nucMaxAF.csv",header=None, names=['a','week','nucMaxAF']).set_index(['a','week']).squeeze(axis=1)
        # Monthly max production for hydro (replace capacity based on observed historic data)
    hMaxOut = pd.read_csv("inputs/hMaxOut.csv",header=None, names=['a','month','hMaxOut']).set_index(['a','month']).squeeze(axis=1)
        # Monthly max storing for hydro (replace capacity based on observed historic data)
    hMaxIn = pd.read_csv("inputs/hMaxIn.csv",header=None, names=['a','month','hMaxIn']).set_index(['a','month']).squeeze(axis=1)
        # Interconnexion maximum capacity in GW (import to a1 from a2)
    links = pd.read_csv("inputs/links.csv",header=None, names=['a1','a2','links']).set_index(['a1','a2']).squeeze(axis=1)
        # Maximum import capacity from an exogenous neighboring country in GW (import to a from exo_a)
    exo_IM = pd.read_csv("inputs/exo_IM.csv",header=None, names=['a','exo_a','exo_IM']).set_index(['a','exo_a']).squeeze(axis=1)
        # Maximum export capacity to an exogenous neighboring country in GW (export to exo_a from a)
    exo_EX = pd.read_csv("inputs/exo_EX.csv",header=None, names=['a','exo_a','exo_EX']).set_index(['a','exo_a']).squeeze(axis=1)    
        # Additional FRR requirement for variable renewable energies because of forecast errors
    rsv_req = pd.read_csv("inputs/rsv_req.csv",header=None, names=['vre','rsv_req']).set_index('vre').squeeze(axis=1)    
    ## VARIABLE COSTS & OTHER TECH's CHARACTERISTICS        
    # Fuel price computation (monthly and area adjusted)
        # Fuel fuel used by each technology
    thr_fuel = pd.read_csv("inputs/thr_fuel.csv",header=None, names=['thr','fuel']).set_index('thr')
        # Fuel price for the technology in EUR/netGJ
    fuel_price = pd.read_csv("inputs/fuel_price.csv", header=None, names=['thr','price']).set_index('thr')
        # Fuel price monthly variation (index normalized to average = 1)
    fuel_timeFactor = pd.read_csv("inputs/fuel_timeFactor.csv", header=None, names=['fuel','month','timeFactor']).set_index('fuel')
        # Fuel price variation depending on the area  (index normalized to average = 1)
    fuel_areaFactor = pd.read_csv("inputs/fuel_areaFactor.csv", header=None, names=['fuel','area','areaFactor']).set_index('fuel')    
        # => Fuel price adjusted for time and location
    fuel_price_adj = fuel_price.join(thr_fuel, how='outer').join(fuel_areaFactor, how='outer', on='fuel').join(fuel_timeFactor, how='outer', on='fuel')
    fuel_price_adj.index.name ='thr'
    fuel_price_adj.set_index(['area','month'],append=True,inplace=True)
    fuel_price_adj['fuel_price_adj'] =  fuel_price_adj.price*fuel_price_adj.timeFactor + fuel_price_adj.areaFactor
    fuel_price_adj = fuel_price_adj.drop(['price','fuel','timeFactor','areaFactor'],axis=1).dropna()
    del thr_fuel, fuel_price, fuel_timeFactor, fuel_areaFactor    
    # Other VOM parameters
    GJ_MWh = 3.6
        # Standard NCV Efficiency of the technology
    efficiency = pd.read_csv("inputs/efficiency.csv", header=None, names=['thr','efficiency']).set_index('thr')
        # Standard NCV Efficiency of the technology when running at 50% capacity factor
    eff50 = pd.read_csv("inputs/eff50.csv", header=None, names=['thr','eff50']).set_index('thr')
        # CO2 emmisions associated with fuel consumption in kg/netGJ
    co2_factor = pd.read_csv("inputs/co2_factor.csv", header=None, names=['thr','co2_factor']).set_index('thr')
        # Cost associated to co2 emmisions (taxes, ETS payments) in EUR/t
    co2_price = pd.read_csv("inputs/co2_price.csv", header=None, names=['thr','co2_price']).set_index('thr')
        # Non fuel cost of production in EUR/MWh
    nonFuel_vOM = pd.read_csv("inputs/nonFuel_vOM.csv", header=None, names=['thr','nonFuel_vOM']).set_index('thr')
        # Start-up fixed cost in EUR/MW
    su_fixedCost = pd.read_csv("inputs/su_fixedCost.csv", header=None, names=['thr','su_fixedCost']).set_index('thr')
        # Start-up fuel consumption in netGJ/MW
    su_fuelCons = pd.read_csv("inputs/su_fuelCons.csv", header=None, names=['thr','su_fuelCons']).set_index('thr')
        # Ramp-up fuel consumption in netGJ/MW
    ramp_fuelCons = pd.read_csv("inputs/ramp_fuelCons.csv", header=None, names=['thr','ramp_fuelCons']).set_index('thr')    
        # => Variable operation and maintenance costs in EUR/MWh (ie kEUR/GWh)
    vOM = fuel_price_adj.join(efficiency,how='left').join(co2_factor,how='left').join(co2_price,how='left').join(nonFuel_vOM,how='left')
    vOM['vOM'] = (1/vOM.efficiency)*GJ_MWh*(vOM.fuel_price_adj + vOM.co2_factor*vOM.co2_price/1000) + vOM.nonFuel_vOM
    vOM = vOM[['vOM']].squeeze(axis=1)    
        # => Variable OM costs applied to GENERATION in EUR/MWh (ie kEUR/GWh)
    genOM = fuel_price_adj.join(efficiency,how='left').join(eff50,how='left').join(co2_factor,how='left').join(co2_price,how='left').join(nonFuel_vOM,how='left')
    genOM['genOM'] = (2/genOM.efficiency - 1/genOM.eff50)*GJ_MWh*(genOM.fuel_price_adj + genOM.co2_factor*genOM.co2_price/1000)
    genOM = genOM[['genOM']].squeeze(axis=1)    
        # => Variable OM costs applied to CAPACITY ON in EUR/MW (ie kEUR/GW)
    onOM = fuel_price_adj.join(efficiency,how='left').join(eff50,how='left').join(co2_factor,how='left').join(co2_price,how='left').join(nonFuel_vOM,how='left')
    onOM['onOM'] = (1/onOM.eff50 - 1/onOM.efficiency)*GJ_MWh*(onOM.fuel_price_adj + onOM.co2_factor*onOM.co2_price/1000) + onOM.nonFuel_vOM
    onOM = onOM[['onOM']].squeeze(axis=1)    
        # => Start-up cost of the technology in EUR/MW (or kEUR/GW)
    su_cost = fuel_price_adj.join(su_fuelCons,how='left').join(co2_factor,how='left').join(co2_price,how='left').join(su_fixedCost,how='left')
    su_cost['su_cost'] = su_cost.su_fuelCons*(su_cost.fuel_price_adj+su_cost.co2_factor*su_cost.co2_price/1000) + su_cost.su_fixedCost
    su_cost = su_cost[['su_cost']].squeeze(axis=1)    
        # => Ramping cost of the technology in EUR/MW (or kEUR/GW)
    ramp_cost = fuel_price_adj.join(ramp_fuelCons,how='left').join(co2_factor,how='left').join(co2_price,how='left')
    ramp_cost['ramp_cost'] = ramp_cost.ramp_fuelCons*(ramp_cost.fuel_price_adj+ramp_cost.co2_factor*ramp_cost.co2_price/1000)
    ramp_cost = ramp_cost[['ramp_cost']].squeeze(axis=1)    
        # => Emissions related to steady generation in kgCO2/MWh (ie tCO2/GWh)
    genCarb = efficiency.join(eff50,how='left').join(co2_factor,how='left')
    genCarb['genCarb'] = (2/genCarb.efficiency - 1/genCarb.eff50)*GJ_MWh*genCarb.co2_factor
    genCarb = genCarb[['genCarb']].squeeze(axis=1)
        # => Emissions related to steady generation in kgCO2/MWh (ie tCO2/GWh)
    onCarb = efficiency.join(eff50,how='left').join(co2_factor,how='left')
    onCarb['onCarb'] = (1/onCarb.eff50 - 1/onCarb.efficiency)*GJ_MWh*onCarb.co2_factor
    onCarb = onCarb[['onCarb']].squeeze(axis=1)
        # => Emissions related to start-ups in kgCO2/MWh (ie tCO2/GW)
    suCarb = su_fuelCons.join(co2_factor,how='left')
    suCarb['suCarb'] = suCarb.su_fuelCons*suCarb.co2_factor
    suCarb = suCarb[['suCarb']].squeeze(axis=1)
        # => Emissions related to ramp-ups in kgCO2/MWh (ie tCO2/GW)
    rampCarb = ramp_fuelCons.join(co2_factor,how='left')
    rampCarb['rampCarb'] = rampCarb.ramp_fuelCons*rampCarb.co2_factor
    rampCarb = rampCarb[['rampCarb']].squeeze(axis=1)    
    del fuel_price_adj, GJ_MWh, efficiency, co2_factor, co2_price, nonFuel_vOM, su_fixedCost, su_fuelCons, ramp_fuelCons
    # Other technologies parameters
        # Minimum stable generation (%)
    minSG = pd.read_csv("inputs/minSG.csv", header=None, names=['thr','minSG']).set_index('thr').squeeze(axis=1)
        # Minimum time off (hours)
    minTimeOFF = pd.read_csv("inputs/minTimeOFF.csv", header=None, names=['thr','minTimeOFF']).set_index('thr').squeeze(axis=1)
        # Minimum time on (hours)
    minTimeON = pd.read_csv("inputs/minTimeON.csv", header=None, names=['thr','minTimeON']).set_index('thr').squeeze(axis=1)
    
        # Variable cost of storage (EUR/MWh)
    str_vOM = pd.read_csv("inputs/str_vOM.csv", header=None, names=['str','str_vOM']).set_index('str').squeeze(axis=1)
    
    ## SET HOUR BY MONTHS & WEEKS
    ##
    ## Set up matching between hours index and months/weeks indices    
        # months-hours matching
    months_hours = pd.read_csv("inputs/hour_month.csv", header=None, names=['hour','month']).set_index('month').squeeze(axis=1)
    # Call all hours from january 2019 with the command <months_hours['201901']>
    # e.g. <sum(1 for h in months_hours['201901'])> => number of hours in January 2019
        # weeks-hours matching
    weeks_hours = pd.read_csv("inputs/hour_week.csv", squeeze=True, header=None, names=['hour','week']).set_index('week').squeeze(axis=1)
    # Same as for months    
        # hours-months matching
    hours_months = pd.read_csv("inputs/hour_month.csv", header=None, names=['hour','month']).set_index('hour').squeeze(axis=1)
    # Call all hours from january 2019 with the command <months_hours['201901']>
    # e.g. <sum(1 for h in months_hours['201901'])> => number of hours in January 2019
        # hours-weeks matching
    hours_weeks = pd.read_csv("inputs/hour_week.csv", squeeze=True, header=None, names=['hour','week']).set_index('hour').squeeze(axis=1)
    # Same as for months
    
    ## SETS
    ##
    ## Definition of set as an object of the model
        ## AREA & TIME SETS
    #Set of countries/areas (extracted from a dedicated file in inputs)
    model.a = pyo.Set(initialize = pd.read_csv("inputs/areas.csv", squeeze=True, header=None).array, ordered=False)
    #Set of exogenous countries/areas (extracted from a dedicated file in inputs)
    model.exo_a = pyo.Set(initialize = pd.read_csv("inputs/exo_areas.csv", squeeze=True, header=None).array, ordered=False)
    #Set of hours (extracted from a dedicated file in inputs)
    model.h = pyo.Set(initialize = pd.read_csv("inputs/hours.csv", squeeze=True, header=None).array)
    #Set of weeks (extracted from a dedicated file in inputs)
    model.week = pyo.Set(initialize = pd.read_csv("inputs/weeks.csv", squeeze=True, header=None).array)
    #Set of months (extracted from a dedicated file in inputs)
    model.month = pyo.Set(initialize = pd.read_csv("inputs/months.csv", squeeze=True, header=None).array)
        ## TECHNOLOGIES SETS
    #All technologies (extracted from a dedicated file in inputs)
    model.tec = pyo.Set(initialize = pd.read_csv("inputs/tec.csv", squeeze=True, header=None).array, ordered=False)
    #Variable technologies (extracted from a dedicated file in inputs)
    model.vre = pyo.Set(initialize = pd.read_csv("inputs/vre.csv", squeeze=True, header=None).array, ordered=False)
    #Thermal/Dispatchable technologies (extracted from a dedicated file in inputs)
    model.thr = pyo.Set(initialize = pd.read_csv("inputs/thr.csv", squeeze=True, header=None).array, ordered=False)
    #Storage technologies (extracted from a dedicated file in inputs)
    model.sto = pyo.Set(initialize = pd.read_csv("inputs/str.csv", squeeze=True, header=None).array, ordered=False)
    #Technologies used for upward FRR (extracted from a dedicated file in inputs)
    model.frr = pyo.Set(initialize = pd.read_csv("inputs/frr.csv", squeeze=True, header=None).array, ordered=False)
    #Technologies NOT used for upward FRR (extracted from a dedicated file in inputs)
    model.no_frr = pyo.Set(initialize = pd.read_csv("inputs/no_frr.csv", squeeze=True, header=None).array, ordered=False)
    
    ## SCALAR PARAMETERS
    load_uncertainty = 0.01     # Uncertainty coefficient for hourly demand
    delta = 0.1                 # Load variation factor 
    voll = 15000                # Value of lost load (virtual cost of unserved demand)
    eta_in = pd.Series([0.95,0.9,0.59], index=['lake_phs','battery','methanation']) # Charging efficiency of storage technologies
    eta_out = pd.Series([0.9,0.95,0.45], index=['lake_phs','battery','methanation']) # Discharging efficiency of storage technologies
    trloss = 0.02               # Transportation loss applied to power trade 
    
    ## VARIABLES
    ## Definition of variable as an object of the model
        # Hourly energy generation in GWh/h
    model.gene = pyo.Var(((a, tec, h) for a in model.a for tec in model.tec for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # Thermal capacity on and available for generation in GW
    model.on = pyo.Var(((a, thr, h) for a in model.a for thr in model.thr for h in model.h), within=pyo.NonNegativeReals,initialize=0)
    for a in model.a:
        for thr in model.thr:
            for h in model.h: model.on[a,thr,h].value = capa[a,thr]*maxaf[a,thr]
        # Hourly increase in thermal capacity on and available for generation in GW (i.e. hourly start-ups)
    model.startup = pyo.Var(((a, thr, h) for a in model.a for thr in model.thr for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # Hourly decrease in thermal capacity on and available for generation in GW (i.e. hourly turn-offs)
    model.turnoff = pyo.Var(((a, thr, h) for a in model.a for thr in model.thr for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # Hourly (positive) increase in thermal generation in GW (i.e. hourly ramp-up)
    model.ramp_up = pyo.Var(((a, thr, h) for a in model.a for thr in model.thr for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # Hourly electricity input of battery storage GW
    model.storage = pyo.Var(((a, sto, h) for a in model.a for sto in model.sto for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # Energy stored in each storage technology in GWh = Stage of charge
    model.stored = pyo.Var(((a, sto, h) for a in model.a for sto in model.sto for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # Required upward frequency restoration reserve in GW    
    model.rsv = pyo.Var(((a, tec, h) for a in model.a for tec in model.tec for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # Hourly loss of load in GW    
    model.hll = pyo.Var(((a, h) for a in model.a for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # Hourly imports from another country in GW
    model.im = pyo.Var(((a1, a2, h) for a1 in model.a for a2 in model.a for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # Hourly exports to another country in GW
    model.ex = pyo.Var(((a1, a2, h) for a1 in model.a for a2 in model.a for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # Hourly imports from another non-modeled country in GW
    model.exo_im = pyo.Var(((a, exo_a, h) for a in model.a for exo_a in model.exo_a for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # Hourly exports to another non-modeled country in GW
    model.exo_ex = pyo.Var(((a, exo_a, h) for a in model.a for exo_a in model.exo_a for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # hourly dispatch cost definition (in kEUR)
    model.hcost = pyo.Var(((a, h) for a in model.a for h in model.h), within=pyo.NonNegativeReals,initialize=0)
        # hourly CO2 emissions (in tCO2)
    model.hcarb = pyo.Var(((a, h) for a in model.a for h in model.h), within=pyo.NonNegativeReals,initialize=0)
    
    ### CONSTRAINTS RULE DEFINITION
    # Set up a function which will return the equation of the constraint.
        # NON DISPATCHABLE GENERATION CONSTRAINTS
    def gene_vre_constraint_rule(model, a, h, vre):
        """Get constraint on variables renewable profiles generation."""
        return model.gene[a, vre, h] <= capa[a,vre]*load_factor[a,vre,h]
    def gene_nmd_constraint_rule(model, a, h):
        """Get constraint on non market dependant generation."""
        return model.gene[a, 'nmd', h] == nmd[a,h]
        # THERMAL GENERATION CONSTRAINTS
    def on_capa_constraint_rule(model, a, thr, h):
        """Get constraint on maximum thermal capacity on at a given hour."""
        return model.on[a,thr,h] <= capa[a,thr]*maxaf[a,thr]
    def gene_on_hmax_constraint_rule(model, a, thr, h):
        """Get constraint on maximum thermal generation given the capacity on at a given hour."""
        return model.gene[a,thr,h] + model.rsv[a,thr,h] <= model.on[a,thr,h]
    def gene_on_hmin_constraint_rule(model, a, thr, h):
        """Get constraint on minimum thermal generation given the capacity on at a given hour."""
        return model.on[a,thr,h]*minSG[thr] <= model.gene[a,thr,h]
    def yearly_maxON_constraint_rule(model, a, thr):
        """Get constraint on maximum thermal capacity on over the whole period on average."""
        return sum(model.on[a,thr,h] for h in model.h)/len(model.h) <= capa[a,thr]*eaf[a,thr]
    def nuc_maxON_constraint_rule(model, a, h):
        """Get constraint on maximum nuclear capacity on at a given hour."""
        return model.on[a,'nuclear',h] <= capa[a,'nuclear']*nucMaxAF[a,hours_weeks[h]]
    def on_off_constraint_rule(model, a, thr, h):
        """Get constraint on dynamics of starting up and turning off capacity."""
        h_next = h+1 if h<model.h.last() else model.h.first()
        return model.on[a,thr,h_next] == model.on[a,thr,h] + model.startup[a,thr,h] - model.turnoff[a,thr,h]
    def cons_startup_constraint_rule(model, a, thr, h):
        """Get constraint on capacity available for start up."""
        if h-minTimeOFF[thr] >= model.h.first(): recently_off = range(h-minTimeOFF[thr],h)
        if h-minTimeOFF[thr] < model.h.first(): recently_off = itertools.chain(range(model.h.first(),h),range(model.h.last()-minTimeOFF[thr]+(h-model.h.first())+1,model.h.last()+1))
        return model.startup[a,thr,h] <= capa[a,thr]*maxaf[a,thr] - model.on[a,thr,h] - sum(model.turnoff[a,thr,h_bis] for h_bis in recently_off)
    def cons_turnoff_constraint_rule(model, a, thr, h):
        """Get constraint on capacity available for turning off."""
        if h-minTimeON[thr] >= model.h.first(): recently_on = range(h-minTimeON[thr],h)
        if h-minTimeON[thr] < model.h.first(): recently_on = itertools.chain(range(model.h.first(),h),range(model.h.last()-minTimeON[thr]+(h-model.h.first())+1,model.h.last()+1))
        return model.turnoff[a,thr,h] <= model.on[a,thr,h] - sum(model.startup[a,thr,h_bis] for h_bis in recently_on)
    def ramping_up_constraint_rule(model, a, thr, h):
        """Get constraint on generation up variation from one hour to the next."""
        h_next = h+1 if h<model.h.last() else model.h.first()
        return model.ramp_up[a,thr,h_next] >= model.gene[a,thr,h_next] - model.gene[a,thr,h]    
        # STORAGE CONSTRAINTS
    def stored_cap_constraint_rule(model, a, sto, h):
        #Get constraint on maximum energy that is stored in storage units."""
        return model.stored[a,sto,h] <= stockMax[a,sto]*1000
    def stor_in_constraint_rule(model, a, sto, h):
        #Get constraint on the capacity with hourly charging relationship of storage."""
        if sto=='lake_phs':
            return model.storage[a,sto,h] <= capa_in[a,sto]*hMaxIn[a,hours_months[h]]
        else:
            return model.storage[a,sto,h] <= capa_in[a,sto]
    def stor_out_constraint_rule(model, a, sto, h):
        #Get constraint on the capacity with hourly charging relationship of storage."""
        if sto=='lake_phs':
            return model.gene[a,sto,h] + model.rsv[a,sto,h] <= capa[a,sto]*hMaxOut[a,hours_months[h]]
        else:
            return model.gene[a,sto,h] + model.rsv[a,sto,h] <= capa[a,sto]
    def storing_constraint_rule(model, a, sto, h):
        #Get constraint on the definition of stored energy in the storage options."""
        h_next = h+1 if h<model.h.last() else model.h.first()
        if sto=='lake_phs':
            return model.stored[a,sto,h_next] == model.stored[a,sto,h] + model.storage[a,sto,h]*eta_in[sto] - model.gene[a,sto,h]/eta_out[sto] + (lake_inflows[a,hours_months[h]]*1000/len(months_hours[hours_months[h]]))/eta_out[sto]
        else :
            return model.stored[a,sto,h_next] == model.stored[a,sto,h] + model.storage[a,sto,h]*eta_in[sto] - model.gene[a,sto,h]/eta_out[sto]
    def lake_res_constraint_rule(model, a, month):
        #Get constraint on water for lake reservoirs."""
        return sum(model.gene[a,'lake_phs',h] - model.storage[a,'lake_phs',h]*eta_in['lake_phs']*eta_out['lake_phs'] for h in months_hours[month]) == lake_inflows[a,month]*1000
        # RESERVE CONSTRAINTS
    def reserves_constraint_rule(model, a, h):
        #Get constraint on FRR requirement."""
        return sum(model.rsv[a,frr,h] for frr in model.frr) == sum(rsv_req[vre]*capa[a,vre] for vre in model.vre) + demand[a,h]*load_uncertainty*(1+delta)
    def no_FRR_contrib_constraint_rule(model, a, no_frr, h):
        #Get constraint on technologies not able to contribute to FRR. (Avoid hidden curtailment if constrained)"""
        return model.rsv[a,no_frr,h] == 0
        # BALANCE CONSTRAINTS
    def trade_bal_constraint_rule(model, a1, a2, h):
        #Get constraint on equilibrium in trade between countries."""
        return model.im[a1,a2,h] == model.ex[a2,a1,h]*(1-trloss)
    def icIM_constraint_rule(model, a1, a2, h):
        #Get constraint on capacity limitation of interconnexions for imports."""
        return model.im[a1,a2,h] <= links[a1,a2]
    def exoIM_constraint_rule(model, a, exo_a, h):
        #Get constraint on capacity limitation on imports from non-modeled countries."""
        return model.exo_im[a,exo_a,h] <= exo_IM[a,exo_a]
    def exoEX_constraint_rule(model, a, exo_a, h):
        #Get constraint on capacity limitation on exports to non-modeled countries."""
        return model.exo_ex[a,exo_a,h] <= exo_EX[a,exo_a]
    def adequacy_constraint_rule(model, a, h):
        #Get constraint on supply and demand equality."""
        return sum(model.gene[a,tec,h] for tec in model.tec) + sum(model.im[a,trader,h] for trader in model.a) + sum(model.exo_im[a,exo_a,h] for exo_a in model.exo_a) == \
            demand[a,h] + sum(model.ex[a,trader,h] for trader in model.a) + sum(model.exo_ex[a,exo_a,h] for exo_a in model.exo_a) + sum(model.storage[a,sto,h] for sto in model.sto) - model.hll[a,h]
    def hcost_definition(model, a, h):
        #Define hourly cost for every country."""
        return model.hcost[a,h] == \
            sum(model.gene[a,thr,h]*genOM[thr,a,hours_months[h]] for thr in model.thr) + \
            sum(model.on[a,thr,h]*onOM[thr,a,hours_months[h]] for thr in model.thr) + \
            sum(model.startup[a,thr,h]*su_cost[thr,a,hours_months[h]] for thr in model.thr) + \
            sum(model.ramp_up[a,thr,h]*ramp_cost[thr,a,hours_months[h]] for thr in model.thr) + \
            sum(model.gene[a,sto,h]*str_vOM[sto] for sto in model.sto) + \
            sum(((model.exo_im[a,exo_a,h]-model.exo_ex[a,exo_a,h])/(1-trloss))*exoPrices[exo_a,h] for exo_a in model.exo_a) + \
            model.hll[a,h]*voll
    def hcarb_definition(model, a, h):
        #Define hourly cost for every country."""
        return model.hcarb[a,h] == \
            sum(model.gene[a,thr,h]*genCarb[thr] for thr in model.thr) + \
            sum(model.on[a,thr,h]*onCarb[thr] for thr in model.thr) + \
            sum(model.startup[a,thr,h]*suCarb[thr] for thr in model.thr) + \
            sum(model.ramp_up[a,thr,h]*rampCarb[thr] for thr in model.thr)
    def objective_rule(model):
        #Get constraint for the final objective function (total cost in billion EUR)."""
        return sum(model.hcost[a,h] for a in model.a for h in model.h)/1000000
                   
    ### CONSTRAINTS CREATION & SOLVE STATEMENT
    # Create the constraint as an object of the model with the function declared earlier as a rule.
    model.gene_vre_constraint = pyo.Constraint(model.a, model.h, model.vre, rule=gene_vre_constraint_rule)
    model.gene_nmd_constraint = pyo.Constraint(model.a, model.h, rule=gene_nmd_constraint_rule)
    model.on_capa_constraint = pyo.Constraint(model.a, model.thr, model.h, rule=on_capa_constraint_rule)
    model.gene_on_hmax_constraint = pyo.Constraint(model.a, model.thr, model.h, rule=gene_on_hmax_constraint_rule)
    model.gene_on_hmin_constraint = pyo.Constraint(model.a, model.thr, model.h, rule=gene_on_hmin_constraint_rule)
    model.yearly_maxON_constraint = pyo.Constraint(model.a, model.thr, rule=yearly_maxON_constraint_rule)
    model.nuc_maxON_constraint = pyo.Constraint(model.a, model.h, rule=nuc_maxON_constraint_rule)
    model.on_off_constraint = pyo.Constraint(model.a, model.thr, model.h, rule=on_off_constraint_rule)
    model.cons_startup_constraint = pyo.Constraint(model.a, model.thr, model.h, rule=cons_startup_constraint_rule)
    model.cons_turnoff_constraint = pyo.Constraint(model.a, model.thr, model.h, rule=cons_turnoff_constraint_rule)
    model.ramping_up_constraint = pyo.Constraint(model.a, model.thr, model.h, rule=ramping_up_constraint_rule)
    model.stored_cap_constraint = pyo.Constraint(model.a, model.sto, model.h, rule=stored_cap_constraint_rule)
    model.stor_in_constraint = pyo.Constraint(model.a, model.sto, model.h, rule=stor_in_constraint_rule)
    model.stor_out_constraint = pyo.Constraint(model.a, model.sto, model.h, rule=stor_out_constraint_rule)
    model.storing_constraint = pyo.Constraint(model.a, model.sto, model.h, rule=storing_constraint_rule)
    model.lake_res_constraint = pyo.Constraint(model.a, model.month, rule=lake_res_constraint_rule)
    model.reserves_constraint = pyo.Constraint(model.a, model.h, rule=reserves_constraint_rule)
    model.no_FRR_contrib_constraint = pyo.Constraint(model.a, model.no_frr, model.h, rule=no_FRR_contrib_constraint_rule)
    model.trade_bal_constraint = pyo.Constraint(model.a, model.a, model.h, rule=trade_bal_constraint_rule)
    model.icIM_constraint = pyo.Constraint(model.a, model.a, model.h, rule=icIM_constraint_rule)
    model.exoIM_constraint = pyo.Constraint(model.a, model.exo_a, model.h, rule=exoIM_constraint_rule)
    model.exoEX_constraint = pyo.Constraint(model.a, model.exo_a, model.h, rule=exoEX_constraint_rule)
    model.adequacy_constraint = pyo.Constraint(model.a, model.h, rule=adequacy_constraint_rule)
    model.hcost_constraint = pyo.Constraint(model.a, model.h, rule=hcost_definition)
    model.hcarb_constraint = pyo.Constraint(model.a, model.h, rule=hcarb_definition)
    #Creation of the objective -> Cost
    model.objective = pyo.Objective(rule=objective_rule)
    
    return(model)

def save_model(model, outputs):
    production = pd.DataFrame(index = range(len(model.a*model.h)),columns=['area','hour','nmd','pv','river','nuclear','lake_phs','wind','coal','gas','oil','battery','phs_in','battery_in','net_imports','net_exo_imports'])
    production.area = np.repeat(list(model.a._values),len(model.h), axis=0)
    production.hour = list(model.h._values)*len(model.a)
    production.nmd = pyo.value(model.gene[:,'nmd',:])
    production.pv = pyo.value(model.gene[:,'pv',:])
    production.river = pyo.value(model.gene[:,'river',:])
    production.nuclear = pyo.value(model.gene[:,'nuclear',:])
    production.lake_phs = pyo.value(model.gene[:,'lake_phs',:])
    production.wind = (np.array(pyo.value(model.gene[:,'onshore',:])) + np.array(pyo.value(model.gene[:,'offshore',:]))).tolist()
    production.coal = (np.array(pyo.value(model.gene[:,'coal_SA',:])) + np.array(pyo.value(model.gene[:,'coal_1G',:])) + np.array(pyo.value(model.gene[:,'lignite',:]))).tolist()
    production.gas = (np.array(pyo.value(model.gene[:,'gas_ccgt1G',:])) + np.array(pyo.value(model.gene[:,'gas_ccgt2G',:])) + np.array(pyo.value(model.gene[:,'gas_ccgtSA',:])) + np.array(pyo.value(model.gene[:,'gas_ocgtSA',:]))).tolist()
    production.oil = pyo.value(model.gene[:,'oil_light',:])
    production.battery = pyo.value(model.gene[:,'battery',:])
    production.phs_in = (-np.array(pyo.value(model.storage[:,'lake_phs',:]))).tolist()
    production.battery_in = (-np.array(pyo.value(model.storage[:,'battery',:]))).tolist()
    production.net_imports = (sum(np.array(pyo.value(model.im[:,trader,:])) - np.array(pyo.value(model.ex[:,trader,:])) for trader in model.a) + \
        sum(np.array(pyo.value(model.exo_im[:,trader,:])) - np.array(pyo.value(model.exo_ex[:,trader,:])) for trader in model.exo_a)).tolist()
    production = production.set_index(['area','hour'])
    demand = pd.read_csv("inputs/demand.csv", header=None, names=['area','hour','demand']).set_index(['area','hour']).squeeze(axis=1)
    production = production.join(demand)
    capa_on = pd.DataFrame(index = range(len(model.a*model.h)),columns=['area','hour','nuclear','coal_SA','coal_1G','coal_LI','gas_ccgt1G','gas_ccgt2G','gas_ccgtSA','gas_ocgtSA','oil'])
    capa_on.area = np.repeat(list(model.a._values),len(model.h), axis=0)
    capa_on.hour = list(model.h._values)*len(model.a)
    capa_on.nuclear = pyo.value(model.on[:,'nuclear',:])
    capa_on.coal_SA = pyo.value(model.on[:,'coal_SA',:])
    capa_on.coal_1G = pyo.value(model.on[:,'coal_1G',:])
    capa_on.coal_LI = pyo.value(model.on[:,'lignite',:])
    capa_on.gas_ccgt1G = pyo.value(model.on[:,'gas_ccgt1G',:])
    capa_on.gas_ccgt2G = pyo.value(model.on[:,'gas_ccgt2G',:])
    capa_on.gas_ccgtSA = pyo.value(model.on[:,'gas_ccgtSA',:])
    capa_on.gas_ocgtSA = pyo.value(model.on[:,'gas_ocgtSA',:])
    capa_on.oil = pyo.value(model.on[:,'oil_light',:])
    capa_on = capa_on.set_index(['area','hour'])
    
    prices = pd.DataFrame(index = range(len(model.h)), columns=['hour']+list(model.a._values))
    i = 0
    for hour in model.h:
        prices.hour[i] = hour
        for a in model.a:
            prices[a][i] = 1000000 * model.dual[model.adequacy_constraint[a,hour]]
        i+=1
    prices = prices.set_index('hour')
    
    # Net imports to France
    trloss = 0.02  
    FRtrade = pd.DataFrame(index = range(len(model.h)), columns=['hour'] + list(model.a._values))
    FRtrade.hour = list(model.h._values)
    for a in model.a:
        FRtrade[a] = (np.array(pyo.value(model.im['FR',a,:]))*(1-trloss) - np.array(pyo.value(model.ex['FR',a,:]))).tolist()
    FRtrade = FRtrade.set_index('hour')
    
    if not os.path.exists(outputs): os.makedirs(outputs)
    production.to_csv(outputs+"/production.csv",index=True)
    capa_on.to_csv(outputs+"/capa_on.csv",index=True)
    prices.to_csv(outputs+"/prices.csv",index=True)
    FRtrade.to_csv(outputs+"/FRtrade.csv",index=True)

def run_model(scenario, year, outputs):
    print("SCENARIO =",scenario)
    print("YEAR =",year)
    start_time = time.localtime()
    print("Started at ",time.asctime(start_time))
    print("-- Load the input data")
    subprocess.run("Rscript --vanilla format_inputs_EXEC.R "+scenario+" "+year, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    print("-- Initializing the model")
    model = set_model()
    print("-- Solving the model")
    opt = SolverFactory('cbc.exe')
    opt.solve(model)
    print("-- Saving the outputs")
    save_model(model,outputs)
    print("EXECUTION TIME: ",time.strftime("%H:%M:%S",time.gmtime(time.time() - time.mktime(start_time))))
    print("-- Cleaning...")
    del model, opt
    gc.collect()


#%% RUNS
### Runs to be launched

#Launch as many simulations as needed with the run_model() function, specifying the following parameters...
## scenario : Name of the .xlsx in which specifications of the scenario to simulate are gathered
## year     : Which year (among 2016-2019) should be considered for weather related and other time-varying inputs (hourly power demand, wind and solar production, ...)
## outputs  : Name of the (existing) directory in which outputs will be saved
run_model(scenario="Scenario_BASELINE", year="2019", outputs="outputs/BASELINE_2019")

