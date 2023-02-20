import pandas as pd
import sys
import numpy as np

sys.path.append("../../")

from models.late_branching import late_branching
from utils_data_model import vir_model_expdecay
from analysis.late_branching.model_params import myparams
from analysis.late_branching.model_params import fit_names_local as fit_names
from analysis.late_branching.model_params import fit_lb as best_fit

import copy
# get models
time = np.arange(0, 80, 0.1)

sim = late_branching("Arm", time, myparams, vir_model_expdecay)
# load fit results
sim.load_fit(best_fit)
sim2 = copy.deepcopy(sim)
sim2.name = "Cl13"
sim2.set_params({"vir_load": 5, "vir_decay": 0.01})

cells, mols = sim.compute_cellstates()
cells2, mols2 = sim2.compute_cellstates()

#fit_latebranch = "latebranch_fit_local"
data_params = ["beta_eff", "beta_naive", ("death_eff", "deg_chr_th1", "deg_chr_tfh"), "death_mem"]

fit_params_late = ["p1", "prop_ag", "p1_ag", "p2_IL10", "fb_IL10", "fbfc_ag_chr", "r_mem", "r_chr","r_IL10_chr", "deg_IL10",
                   "initial_cells"]
#fit_params_early = ["p1", "p1_ag", "p2_IL10", "fb_IL10", "fbfc_ag_chr", "r_mem", "rate_chr1","r_IL10_chr", "deg_IL10"]

#fit_params_late = ["p1", "fb_IL10", "prop_ag", "r_mem", "r_chr", "initial_cells"]
#fit_params_early = ["p1", "fb_IL10", "prop_ag", "r_mem", "rate_chr1", "initial_cells"]


#load fit and simulate
#load fit results

scan_types = ["mcarlo", "param_scans"]
scan_types = ["param_scans"]
# for each parameter and each infection, do monte carlo simulation and parameter scan
res = 30
res2 = 3
CV = 0.1

fit_names = [best_fit]
for fit_name in fit_names:

    if "early" in fit_name:
        fit_params = fit_params_early
    elif "late" in fit_name:
        fit_params = fit_params_late
    elif fit_name == best_fit:
        fit_params_late

    pnames = [*data_params, *fit_params]
    for pname in pnames:
        for scan_type in scan_types:

            if scan_type == "mcarlo":
                myarr1 = sim.get_lognorm_array(pname, res=res, CV=CV)
                myarr2 = sim2.get_lognorm_array(pname, res=res, CV=CV)

            else:
                myarr1 = sim.get_pscan_array(pname, res = res2, myrange = "10perc")
                myarr2 = sim2.get_pscan_array(pname, res=res2, myrange = "10perc")

            sim.param_scan2(pname=pname,
                            arr=myarr1,
                            infection="Arm",
                            use_fit=fit_name,
                            scan_type=scan_type,
                            timepoints=np.arange(1,60))

            sim2.param_scan2(pname=pname,
                            arr=myarr2,
                            infection="Cl13",
                            use_fit=fit_name,
                            scan_type=scan_type,
                            timepoints=np.arange(1,60))
#
