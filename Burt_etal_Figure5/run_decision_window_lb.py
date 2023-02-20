import copy

import numpy as np
from lmfit import minimize, fit_report
import pandas as pd
import sys
import pickle
import seaborn as sns
sys.path.append("../../")

from models.late_branching import late_branching
from utils_data_model import vir_model_expdecay
from analysis.late_branching.model_params import myparams as myparams

# run decision window
res = 50
fb_stren = 100
xlim = 20
ylim = 20
t_start_arr = np.linspace(0,xlim,res)
t_dur_arr = np.linspace(0,ylim,res)

sname = "../../output/decision_window/hm_lb_"

# get model
time = np.arange(0,150,0.5)

names = ["arm_gamma", "chr_gamma"]

fit_name = "latebranch_fit_local"
sim1 = late_branching("arm_gamma", time, myparams, vir_model_expdecay)
sim1.load_fit(fit_name)
sim1.set_params({"vir_load" : 1, "vir_decay" : 0.26, "fb_decwindow" : fb_stren})

sim2 = late_branching("chr_gamma", time, myparams, vir_model_expdecay)
sim2.load_fit(fit_name)
sim2.set_params({"vir_load" : 5, "vir_decay" : 0.01, "fb_decwindow" : fb_stren})

simlist = [sim1, sim2]
for sim in simlist:
    # run analysis
    df = sim.run_decision_window(t_start_arr, t_dur_arr)
    # store
    sname1 = sname + sim.name + "_res" + str(res) + "_fb_" + str(fb_stren) + "_xlim" + str(xlim) + "_ylim" + str(ylim)+ ".csv.gz"
    df.to_csv(sname1, index = False, compression = "gzip")


