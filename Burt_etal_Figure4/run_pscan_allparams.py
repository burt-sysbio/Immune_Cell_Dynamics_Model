import pandas as pd
import sys
import numpy as np

sys.path.append("../../")

from models.late_branching import late_branching
from utils_data_model import vir_model_expdecay
from analysis.late_branching.model_params import myparams
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


#fit_latebranch = "latebranch_fit_local"
data_params = []

pnames = ["beta_eff", "beta_naive", ("death_eff", "deg_chr_th1", "deg_chr_tfh"), "death_mem",
          "p1", "p3", "prop_ag", "p1_ag", "p2_IL10", "fb_IL10", "fbfc_ag_chr", "r_mem", "r_chr","r_IL10_chr", "deg_IL10",
        "initial_cells", "deg_myc", "r_IL2_naive", "r_IL2_eff", "deg_IL10"]


# for each parameter and each infection, do monte carlo simulation and parameter scan


def run_pscan(sim, pnames, fit, infection, scan_type = "param_scans", timepoints = np.arange(1,60)):
    assert (scan_type == "param_scans") | (scan_type == "mcarlo")
    assert (infection == "Arm") | (infection == "Cl13")

    if scan_type == "param_scans":
        res = 3
    else:
        res = 30
    for pname in pnames:
        myarr = sim.get_pscan_array(pname, res = res, myrange = "10perc")
        sim.param_scan2(pname=pname,
                        arr=myarr,
                        infection=infection,
                        use_fit=fit,
                        scan_type=scan_type,
                        timepoints= timepoints)

run_pscan(sim, pnames, best_fit, "Arm")
run_pscan(sim2, pnames, best_fit, "Cl13")

# for pname in pnames:
#
#     myarr1 = sim.get_pscan_array(pname, res = res2, myrange = "10perc")
#     myarr2 = sim2.get_pscan_array(pname, res=res2, myrange = "10perc")
#
#     sim.param_scan2(pname=pname,
#                     arr=myarr1,
#                     infection="Arm",
#                     use_fit=best_fit,
#                     scan_type=scan_type,
#                     timepoints=np.arange(1,60))
#
#     sim2.param_scan2(pname=pname,
#                     arr=myarr2,
#                     infection="Cl13",
#                     use_fit=best_fit,
#                     scan_type=scan_type,
#                     timepoints=np.arange(1,60))
#
