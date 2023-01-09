#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:38:50 2020

@author: burt
one simple diff x--> y with differentiation
keep death rate constant and vary feedback for prolif within a
range so that it does not top the death rate
"""

# %%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils import Sim
import matplotlib

d1 = {
    "alpha_naive": 10,
    "beta": 10,
    "div_naive": 0,
    "div_eff": 1,
    "alpha_eff1": 10,
    "alpha_eff2": 10,
    "beta_p": 1,
    "d_naive": 0,
    "d_eff": 1,
    "EC50_myc": 0.5,
    "deg_myc": 0.0000001,
    #"r_cyto1" : 1,
    #"r_cyto2" : 1,
    #"uptake_cyto1" : 1,
    #"uptake_cyto2" : 1,
    "fb_eff1" : 1e1,
    #"fb_eff2" : 1e4,
    #"fb_EC50" : 0.1,
    "p1" : 0.6,
    "block_fb_start" : 100,
    "block_dur" : 1,
    "use_fb_regular" : True,
}

d2 = dict(d1)
d2["alpha_naive"] = 1
d2["beta"] = 1

# set up simulations
time = np.arange(0,8,0.05)
sim1 = Sim("RTM", d1, time)
sim2 = Sim("SSM", d2, time)

def vary_fb(sim, arr, drop_eff1 = True, drop_naive = True):
    """
    vary the delay for a given perturbation setting
    """


    cntrl = []
    # first just change alpha without perturbation as control
    for x in arr:
        sim.params["fb_eff1"] = x

        cells, _ = sim.run_sim()
        if drop_eff1:
            cells = cells.loc[cells.cell != "eff1"]
        if drop_naive:
            cells = cells.loc[cells.cell != "naive"]

        cells["value"] = cells["value"] * 100
        cells["param_value"] = x

        sim.reset()
        cntrl.append(cells)

    df_cntrl = pd.concat(cntrl)

    return df_cntrl

# %%
arr = np.geomspace(1,100,2)
out = vary_fb(sim1, arr, drop_eff1 = False)
out2 = vary_fb(sim2, arr, drop_eff1 = False)

df = pd.concat([out,out2]).reset_index()


g = sns.relplot(data= df, x = "time", y = "value", style = "param_value", hue = "cell", col = "name", kind = "line",
                height = 1.75, legend = False, aspect =1, palette = ["tab:blue", "tab:red"])

g.set_titles("{col_name}")
g.set(xlabel = "time (d)", ylabel = "Cells", ylim = [0,None], xlim = [0,4],
      xticks = [0,1,2,3,4])
sns.despine(top = False, right = False)
plt.show()

# %%
arr = np.geomspace(1,100,31)
out = vary_fb(sim1, arr)
out2 = vary_fb(sim2, arr)

df = pd.concat([out,out2]).reset_index()

cmap = "rocket_r"
sm = matplotlib.colors.LogNorm(vmin = arr[0], vmax = arr[-1])
g = sns.relplot(data= df, x = "time", y = "value", hue = "param_value", col = "name", kind = "line",
                hue_norm = sm, palette = cmap, height = 1.6, legend = False, aspect =0.8)

g.set_titles("{col_name}")
g.set(xlabel = "time (d)", ylabel = "ThX cells", ylim = [0,None], xlim = [0,4],
      xticks = [0,1,2,3,4])
sns.despine(top = False, right = False)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=sm)
plt.colorbar(sm, ax= g.axes)
plt.show()

# %%
arr = np.geomspace(0.01,100,31)
out = vary_fb(sim1, arr)
out2 = vary_fb(sim2, arr)
df = pd.concat([out,out2]).reset_index()

out = df.groupby(["name", "param_value"])["value"].max().reset_index()
out["val_norm"] = out.groupby(["name"])["value"].transform(lambda x: np.log2(x/x.median()))

g = sns.relplot(data = out, x = "param_value", y = "val_norm", kind = "line", hue = "name", height =1.6,
                palette = ["k", "purple"], lw = 1.5, legend = False, aspect = 1.15)
g.set(xscale = "log",
      xlim = [0.01,100],
      xlabel = "feedback fold-change",
      ylabel = "effect size",
      yticks = [0,0.25,0.5],
      ylim = [-0.05,0.5])
sns.despine(top = False, right = False)
plt.show()
