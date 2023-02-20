import numpy as np
from lmfit import minimize, fit_report, Parameters
import pandas as pd
import seaborn as sns
import lmfit
import sys
sys.path.append("../../")
import lmfit
from utils_data_model import dataplot
from models.late_branching import late_branching
from utils_data_model import vir_model_expdecay
from utils_data_model import get_residuals_combined_fullmodel
from analysis.late_branching.model_params import myparams as myparams
from analysis.late_branching.model_params import fit_names_local, fit_names_global
import pickle
from analysis.late_branching.fit_params import params1, params2, params3, params4


sns.set(context = "talk", style = "ticks")

path_data = "../../data/differentiation_kinetics/"
data_arm = pd.read_csv(path_data + "data_arm.csv")
data_cl13 = pd.read_csv(path_data + "data_cl13.csv")


time = np.arange(0,70,0.1)
sim = late_branching("Arm", time, myparams, vir_model_expdecay)
sim.load_fit("latebranch_fit_local")
sim.compute_cellstates()
dataplot(sim, data_arm, data_cl13)

#methods = ["leastsq", "least_squares"]

run_local = False
if run_local:
    method = "leastsq"
    fit_names = fit_names_local
else:
    method = "bashinhopping"
    fit_names = fit_names_global

param_list = [params1, params2, params3, params4]

# focus on late branching constrained vs unconstrained
param_list = [param_list[0], param_list[2]]
fit_names = [fit_names[0], fit_names[2]]
print("running fits for late branching topology")

sim_list = [late_branching("Arm", time, myparams, vir_model_expdecay) for _ in param_list]

ci_list = []
fit_result_list = []

for sim, fitparams, fit_name in zip(sim_list, param_list, fit_names):
    print(len(fitparams))

    mini = lmfit.Minimizer(userfcn= get_residuals_combined_fullmodel, params = fitparams,
                           fcn_args= (sim, data_arm, data_cl13))
    out = mini.minimize(method = method)
    print(fit_report(out))
    #print("computing confidence intervals after fit...")
    #ci = lmfit.conf_interval(mini, out)
    #ci_list.append(ci)
    fit_result_list.append(out)
    #out = minimize(get_residuals_combined_fullmodel, fitparams,
    #               args = (sim, data_arm, data_cl13), method= method)


    fit_dir = "../../output/fit_results/"

    # store full fit object
    with open(fit_dir + fit_name + '_fit_report.p', 'wb') as fit_object:
         pickle.dump(out, fit_object, protocol=pickle.HIGHEST_PROTOCOL)

    # store fit report full
    sim.store_fit(out, fit_name)

    sim.load_fit(fit_name)
    dataplot(sim, data_arm, data_cl13)

