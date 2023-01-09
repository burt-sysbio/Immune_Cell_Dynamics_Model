#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:38:50 2020

@author: burt
one simple diff x--> y with differentiation
keep death rate constant and vary feedback for prolif within a
range so that it does not top the death rate
"""

import numpy as np
from utils import pscan2d, Sim

savedir ="../../../../"

d1 = {
    "alpha_naive": 10,
    "beta": 10,
    "div_naive": 0,
    "div_eff": 1,
    "alpha_eff1": 10,
    "alpha_eff2": 10,
    "beta_p": 0.001,
    "d_naive": 0,
    "d_eff": 0,
    "EC50_myc": 0.5,
    "deg_myc": 0.1,
    "r_cyto1" : 1,
    "r_cyto2" : 1,
    "uptake_cyto1" : 1,
    "uptake_cyto2" : 1,
    "fb_eff1" : 1e4,
    "fb_eff2" : 1e4,
    "fb_EC50" : 0.1,
    "p1" : 0.498,
    "block_fb_start" : 0,
    "block_dur" : 0,
    "use_fb_regular" : False,
}

d2 = dict(d1)
d2["alpha_naive"] = 1
d2["beta"] = 1

# set up simulations
time = np.arange(0,8,0.01)
sim1 = Sim("delay", d1, time)
sim2 = Sim("nodelay", d2, time)

# plot heatmap
prange1 = (0,2)
pname1 = "block_fb_start"

pname2 = "block_dur"
prange2 = (0.01,1)

res = 100

timepoints = [3]
cellnames = ["eff1", "eff2"]
cellnames_maxima = cellnames

titles = ["Delay", "NoDelay"]
simlist = [sim1, sim2]
#
for sim, title in zip(simlist, titles):
    sim.reset()

    df = pscan2d(sim, prange1, prange2, pname1, pname2, res,
                 timepoints = [3],
                 cellnames = cellnames,
                 cellnames_maxima = cellnames)
    df.to_csv(savedir + "output/fig1/cyto_blockade_heatmap_" + title + ".csv", index=False)
