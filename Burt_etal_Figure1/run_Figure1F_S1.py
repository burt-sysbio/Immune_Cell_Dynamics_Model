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
import pandas as pd
from utils import draw_new_params, Sim

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
    "fb_eff1" : 1e4,
    "p1" : 0.5,
    "block_fb_start" : 100,
    "block_dur" : 1,
    "use_fb_regular" : False,
}

d2 = dict(d1)
d2["alpha_naive"] = 1
d2["beta"] = 1

# set up simulations
time = np.arange(0,8,0.01)
sim1 = Sim("delay", d1, time)
sim2 = Sim("nodelay", d2, time)

# plot timecourse with default params
df11, df12 = sim1.run_sim()
df21, df22 = sim2.run_sim()
cell_df = pd.concat([df11, df21])
mol_df = pd.concat([df12, df22])

cell_df = cell_df.loc[cell_df.cell != "naive"]
cell_df.loc[:, "value"] = cell_df.value * 100
cell_df["t0"] = 0
cell_df.to_csv("Burt_etal_Figure1/output/cyto_blockade_timecourse_control.csv", index = False)

# plot timecourse for optimal intervention times
def get_cell_df(block_start, block_dur, sim1, sim2):
    # run delay and no delay simulation with given perturbation time and combine
    sim1.params["block_fb_start"] = block_start
    sim2.params["block_fb_start"] = block_start
    sim1.params["block_dur"] = block_dur
    sim2.params["block_dur"] = block_dur

    df11, df12 = sim1.run_sim()
    df21, df22 = sim2.run_sim()
    cells = pd.concat([df11, df21])
    cells["t0"] = block_start

    sim1.reset()
    sim2.reset()
    return cells

def proc_onset(df_perturb, df_cntrl, grouping = "t0"):
    """
    take two data frames and generate total difference and maxximal difference vs contrl
    """

    # merge data frames
    assert (grouping=="t0") | (grouping=="SD") | (grouping == "ID")
    if grouping == "t0":
        # if onset is varied, t0 is only in the perturb data frame
        out_combined = pd.merge(df_perturb, df_cntrl, how = "left", on = ["time", "cell", "name"])
    else:
        #
        out_combined = pd.merge(df_perturb, df_cntrl, how = "left", on = ["time", "cell", "name", grouping])
    out_combined = out_combined.loc[out_combined["cell"] == "eff2"].copy()

    # compute difference perturb no perturb
    out_combined["val_FC"] = np.abs((out_combined["value_x"] - out_combined["value_y"]))

    # groupby and check maximal difference and compute integral
    out_grouped = out_combined.groupby(["name", grouping, "cell"])
    out_max = out_grouped["val_FC"].max().reset_index()
    out_sum = out_grouped.apply(lambda x: (np.trapz(x["value_x"], x["time"])) - (np.trapz(x["value_y"], x["time"])))
    out_sum = out_sum.reset_index()
    out = pd.merge(out_max, out_sum, how = "inner", on = ["name", grouping, "cell"])
    out.columns = ["name", grouping, "cell", "val_FC", "val_INT"]
    return out

def vary_onset(sim1, sim2, perturb_start_arr, block_dur = 0.5):
    """
    for RTM and SSM models, vary perturbation onset
    """
    cntrl1, _ = sim1.run_sim()
    cntrl2, _ = sim2.run_sim()

    df_cntrl = pd.concat([cntrl1,cntrl2])

    mylist = []
    for val in perturb_start_arr:
        cells = get_cell_df(val, block_dur, sim1, sim2)
        mylist.append(cells)

    sim1.reset()
    sim2.reset()

    # combine with cntrl df and compute max difference
    df_perturb = pd.concat(mylist)
    out = proc_onset(df_perturb, df_cntrl)
    return out


def vary_delay(sim, block_start, block_dur):
    """
    vary the delay for a given perturbation setting
    """
    arr = np.arange(1,11,1)
    simlist = []

    cntrl_list = []
    # first just change alpha without perturbation as control
    for x in arr:
        sim.params["alpha_naive"] = x
        sim.params["beta"] = x
        cells, _ = sim.run_sim()

        cells["SD"] = x
        sim.reset()
        cntrl_list.append(cells)

    for x in arr:
        sim.params["alpha_naive"] = x
        sim.params["beta"] = x
        sim.params["block_fb_start"] = block_start
        sim.params["block_dur"] = block_dur
        cells, _ = sim.run_sim()

        # currently uses chain length instead of SD
        cells["SD"] = x
        sim.reset()

        simlist.append(cells)

    df_perturb = pd.concat(simlist)
    df_cntrl = pd.concat(cntrl_list)

    out = proc_onset(df_perturb, df_cntrl, grouping = "SD")
    out["t0"] = str(block_start)
    sim.reset()

    return out


def mcarlo_blockade(sim1, sim2,
                    block_start, block_dur,
                    CV = 0.1, res = 50,
                    pnames = ["p1"]):
    """
    run blockade for multiple times with parameters drawn from lognrom dist
    based on CV
    """
    simlist = []
    simlist_cntrl = []
    for i in range(res):
        # draw new parameters but only once, then adjust also for other simulation
        draw_new_params(sim1, param_names = pnames, heterogeneity= CV, use_CV=True)


        # adjust parameters in other simulation
        for p in pnames:
            if p != "beta":
                sim2.params[p] = sim1.params[p]
            else:
                # default beta is 10 but sim2.beta is 1
                sim2.params[p] = 0.1*sim1.params[p]
        cells = get_cell_df(block_start, block_dur, sim1, sim2)
        # kickout t0 column and add later, not needed here
        cells.drop(columns=["t0"], inplace=True)

        # get the result without perturbation to compare endstates with this precise simulation
        cells_baseline = get_cell_df(0,0,sim1,sim2)
        cells_baseline.drop(columns=["t0"], inplace=True)

        cells["ID"] = i
        cells_baseline["ID"] = i

        simlist.append(cells)
        simlist_cntrl.append(cells_baseline)

        sim1.reset()
        sim2.reset()

    df_perturb = pd.concat(simlist)
    df_cntrl = pd.concat(simlist_cntrl)

    out = proc_onset(df_perturb, df_cntrl, grouping= "ID")
    out["t0"] = block_start
    return out

def proc_delay(delay_sim : list):
    """
    for given delay scan, get endstate with comparison to default simulation
    """
    df_list = [df.groupby(["cell", "name", "t0", "SD"]).tail(1) for df in delay_sim]
    df_list = [df.loc[df["cell"] == "eff2"] for df in df_list]
    df = pd.concat(df_list)
    # here I can really divide by 0.5 because the parameters for default should be 50%
    df.loc[:, "val_FC"] = np.log2(df["value"] / 0.5)
    df["t0"] = df["t0"].astype("str")
    return df

# process data for early intermediate and late timepoints
vals = [0.1, 1.5, 0.7]
block_dur = 0.5

perturb_start_arr = np.linspace(0,3,30)
out = vary_onset(sim1, sim2, perturb_start_arr, block_dur = 0.5)
out.to_csv("Burt_etal_Figure1/output/cyto_blockade_perturbation_start.csv", index = False)

#
# ##### mcarlo simulation
mcarlo_sim = [mcarlo_blockade(sim1, sim2, v, block_dur, res = 50) for v in vals]
mcarlo_endstates = pd.concat(mcarlo_sim)
mcarlo_endstates.to_csv("Burt_etal_Figure1/output/cyto_blockade_mcarlo.csv", index = False)
#
delay_sim = [vary_delay(sim1, v, block_dur) for v in vals]
delay_endstate = pd.concat(delay_sim).reset_index(drop=True)
delay_endstate.to_csv("Burt_etal_Figure1/output/cyto_blockade_delay.csv", index = False)
# #
# #
# # plot timecourse for given perturbations, delay and no delay
cells = [get_cell_df(v, block_dur, sim1, sim2) for v in vals]
cells = pd.concat(cells)
cells = cells.loc[cells.cell != "naive"]
cells.loc[:, "value"] = cells.value * 100
cells["t0"] = cells["t0"].astype("str")
cells.to_csv("Burt_etal_Figure1/output/cyto_blockade_timecourse.csv", index = False)
