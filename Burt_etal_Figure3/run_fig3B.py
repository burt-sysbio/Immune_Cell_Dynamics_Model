#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:08:29 2020

@author: burt
code to generate output and plot figure 2
note that the ODE solver will throw some warnings, this is expected due to 
the unbounded cell growth in the naive model.
"""
#%%
from utils import Simulation, SimList, make_sim_list, change_param
import models as model
import numpy as np
import matplotlib.pyplot as plt
from params_fig2 import d

d = dict(d)
d["deg_myc"] = 0.39

# =============================================================================
# make time course
# =============================================================================
start_times = [(0,50)]

def vir_model_const(t):
    return 0

sim1 = Simulation(name = "Null", mode = model.null_model, parameters = d, start_times = start_times,vir_model = vir_model_const)
sim2 = Simulation(name = "Carrying Capacity", mode = model.carry_prolif, parameters = d, start_times = start_times,vir_model = vir_model_const)
sim3 = Simulation(name = "IL2", mode = model.il2_prolif, parameters = d, start_times = start_times,vir_model = vir_model_const)
sim4 = Simulation(name = "Timer", mode = model.timer_prolif, parameters = d, start_times = start_times,vir_model = vir_model_const)

#
sims = [sim1, sim2, sim3, sim4]

res = 30
name1 = "beta_p"
arr = np.geomspace(1,35,res)
arr = np.linspace(0,28,res)
arr_name = "divisions per day"
cmap = "Greys"

simlist2 = [make_sim_list(sim, n = res) for sim in sims]
simlist3 = [change_param(simlist, name1, arr) for simlist in simlist2]
# make simlist3 flat
flat_list = [item for sublist in simlist3 for item in sublist]

exp = SimList(flat_list)
#%%
g, data = exp.plot_timecourses(arr, arr_name,
                               log = False,
                               cmap = cmap,
                               log_scale = True,
                               norm_arr = 7)
g.set(ylim = (1,1e6), xlim = (0,start_times[0][1]),
      ylabel = "cells", xlabel = "time (d)")

# add additional lineplot
df_norm = []
titles = ["Null", "Carrying Capacity", "IL2", "Timer"]
for ax, sim, title in zip(g.axes.flat, sims, titles):
    sim.compute_cellstates()
    df = sim.cells.loc[sim.cells["species"] == "CD4_all"]
    ax.plot(df["time"], df["value"], color = "tab:red", zorder = 10000000)
    # line for carrying capacity
    ax.axhline(y = d["n_crit"], linewidth = 2., ls = "--", color = "k")
    ax.set_xticks([0,10,20,30,40,50])
    ax.set_title(title)
    df_norm.append(df)

plt.show()

# %%
