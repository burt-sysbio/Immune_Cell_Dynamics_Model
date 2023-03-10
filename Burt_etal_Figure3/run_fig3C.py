#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:14:25 2020
note that the ODE solver will throw some warnings, this is expected due to 
the unbounded cell growth in the naive model.
@author: burt
"""

from utils import Simulation, SimList
import models as model
from params_fig2 import d
import numpy as np

# =============================================================================
# make time course
# =============================================================================
# for carrying capacacity I need a long long timecourse
# because the peak comes so late
time_long = [(0,1500)]
time_null = [(0,20)]

# for timer and IL2 timecourse needs to be a bit longer
# for the cells to return to 0 at the end because they have a high peak
time_il2 = [(0,400)]
time_timer = [(0,100)]

def vir_model_const(t):
    return 0

sim1 = Simulation(name="Null", mode=model.null_model, parameters=d, start_times=time_null, vir_model= vir_model_const)
sim2 = Simulation(name="Carrying Capacity", mode=model.carry_prolif, parameters=d, start_times=time_long, vir_model= vir_model_const)
sim3 = Simulation(name="IL2", mode=model.il2_prolif, parameters=d, start_times=time_il2, vir_model= vir_model_const)
sim4 = Simulation(name="Timer", mode=model.timer_prolif, parameters=d, start_times=time_timer, vir_model= vir_model_const)

sim_il2 = [sim1, sim2, sim3, sim4]

# =============================================================================
# make parameter scan
# =============================================================================
exp2 = SimList(sim_il2)
pnames = ["beta_p"]

n = 30
arr = np.geomspace(0.7,70, n)
arr_smooth = np.linspace(1.5,3,30)
arr = np.sort(np.concatenate((arr,arr_smooth)))

pscan = exp2.pscan(pnames = pnames, arr = arr, n = len(arr), normtype= "first")

### store output
pscan.to_csv("Burt_etal_Figure2/output_fig2C.csv", index = False)
