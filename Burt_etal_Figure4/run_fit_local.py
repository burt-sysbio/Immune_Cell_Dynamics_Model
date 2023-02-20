import numpy as np
from lmfit import minimize, fit_report, Parameters
import pandas as pd
import seaborn as sns
from utils_data_model import dataplot
from models.late_branching import late_branching
from utils_data_model import vir_model_expdecay
from utils_data_model import get_residuals_combined_fullmodel
from analysis.late_branching.model_params import myparams, myparams_SSM
import pickle
import sys
sys.path.append("../../")
sns.set(context = "talk", style = "ticks")

path_data = "../../data/differentiation_kinetics/"
data_arm = pd.read_csv(path_data + "data_arm.csv")
data_cl13 = pd.read_csv(path_data + "data_cl13.csv")


params1 = Parameters()
params1.add("prop_ag", value=0.15, min=0.1, max=0.45)
params1.add("r_chr", value=0.005, vary = True, min = 0.001, max = 0.01)
params1.add("r_mem", value=0.005, vary = True, min = 0.001, max = 0.01)
params1.add("initial_cells", value = 3000, min = 2000, max = 4000)
params1.add("p1", value=0.6, min=0.5, max=0.8)
params1.add("p2", expr = "1-p1")
params1.add("fb_IL10", value = 0.2, min = 0.1, max = 0.4)
# I previously did keep params below fixed at certain value
#params1.add("p1_ag", value= 0.5, min=0, max=100.0)
#params1.add("p2_IL10", value = 10, min = 0, max = 100.0)
#params1.add("r_IL10_chr", value=100, min = 0, max = 10000)
#params1.add("fbfc_ag_chr", value = 500, min = 1, max = 10000)

use_SSM = False
if use_SSM:
    params = myparams_SSM
    fname = "_SSM"
    params1.add("initial_cells", value = 2000, min = 2000, max = 4000)
    params1.add("prop_ag", value=0.15, min=0.1, max=0.45)
    params1.add("r_chr", value=0.005, vary = True, min = 0.001, max = 0.01)
    params1.add("r_mem", value=0.005, vary = True, min = 0.001, max = 0.01)
    params1.add("p1", value=0.6, min=0.5, max=0.8)
    params1.add("p2", expr = "1-p1")
    params1.add("fb_IL10", value = 0.2, min = 0.1, max = 0.4)
else:
    params = myparams
    fname = ""
    params1.add("initial_cells", value = 3000, min = 2000, max = 4000)
    params1.add("prop_ag", value=0.15, min=0.1, max=0.45)
    params1.add("r_chr", value=0.005, vary = True, min = 0.001, max = 0.01)
    params1.add("r_mem", value=0.005, vary = True, min = 0.001, max = 0.01)
    params1.add("p1", value=0.6, min=0.5, max=0.8)
    params1.add("p2", expr = "1-p1")
    params1.add("fb_IL10", value = 0.2, min = 0.1, max = 0.4)

time = np.arange(0,70,0.1)
sim = late_branching("Arm", time, params, vir_model_expdecay)

sim.compute_cellstates()
dataplot(sim, data_arm, data_cl13)

fit_cells = ["Th1_all", "Tfh_all"]

fit_reports = []
#methods = ["leastsq", "least_squares"]
methods = ["leastsq"]
for method in methods:
    out = minimize(get_residuals_combined_fullmodel, params1,
                   args = (sim, data_arm, data_cl13), method= method)
    fit_reports.append(out)

    print(fit_report(out))

chisqr_arr = np.asarray([x.chisqr for x in fit_reports])
idx = np.argmin(chisqr_arr)

best_fit = fit_reports[idx]

fit_name = "latebranch_fit_local_20221020" + fname
sim.store_fit(best_fit, fit_name)

sim.load_fit(fit_name)
dataplot(sim, data_arm, data_cl13)

# also store fit report

# store full fit object
fit_dir = "../../output/fit_results/"
with open(fit_dir + fit_name + '_fit_report.p', 'wb') as fit_object:
    pickle.dump(out, fit_object, protocol=pickle.HIGHEST_PROTOCOL)

