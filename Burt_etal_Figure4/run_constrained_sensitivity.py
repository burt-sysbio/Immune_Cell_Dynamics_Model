"""
take fit reports for constrained and unconstrained fit and run sensitivity analysis with SD taken from these fit reports
"""
import pickle
import numpy
import pandas as pd
import seaborn as sns
import pandas
import sys
sys.path.append("../../")
from analysis.late_branching.model_params import fit_names_local as fit_names
from analysis.late_branching.model_params import myparams, fit_params_lb, data_params_lb, myparams_SSM
from models.late_branching import late_branching
from utils_data_model import vir_model_expdecay
import numpy as np
import copy
# load fit reports
fit_dir = "../../output/fit_results/"

fit_reports = []
fit_names = [fit_names[0],
             fit_names[2],
             "lb_fit_SSM_fitparams_latechr",
             "lb_fit_SSM_allparams_latechr"]

for fit_name in fit_names:
    with open(fit_dir + fit_name + '_fit_report.p', 'rb') as fp:
        fit_result = pickle.load(fp)
    fit_reports.append(fit_result)

# create data frame with fit reports and SD vals
df_list = []

model_names = ["constrained",
               "unconstrained",
               "constrained",
               "unconstrained"]

for fit_report, name in zip(fit_reports, model_names):

    mydict = fit_report.params
    params = mydict.keys()
    fit_report_errors = [mydict[p].stderr for p in params]
    fit_report_values = [mydict[p].value for p in params]
    mydf = pd.DataFrame({"param_value" : fit_report_values, "param_std" : fit_report_errors, "param_name" : params})
    mydf["name"] = name

    df_list.append(mydf)


time = np.arange(0, 80, 0.1)

sim_list = [late_branching("Arm", time, myparams, vir_model_expdecay),
            late_branching("Arm", time, myparams, vir_model_expdecay),
            late_branching("Arm", time, myparams_SSM, vir_model_expdecay),
            late_branching("Arm", time, myparams_SSM, vir_model_expdecay)]

param_list = [fit_params_lb,
              fit_params_lb + data_params_lb,
              fit_params_lb,
              fit_params_lb + data_params_lb]

sname_list = ["constrained",
              "unconstrained",
              "constrained_SSM",
              "unconstrained_SSM"]

res = 100

def get_SD(df, p):
    """
    get standard deviation for a specific parameter p or parameter tuple (p,) for a specific fit report (df)
    """
    # edge case if tuple is provided
    if type(p) == tuple:
        p = p[0]
    SD = df.loc[df.param_name == p, "param_std"]
    # check if series is not empty

    assert len(SD) != 0
    SD = SD.values[0]
    return SD


assert len(sim_list) == len(fit_names) == len(df_list) == len(param_list) == len(sname_list)

for sim, fit_name, df_fit, param_names, sname in zip(sim_list, fit_names, df_list, param_list, sname_list):
    sim.load_fit(fit_name)
    sim2 = copy.deepcopy(sim)
    sim2.name = "Cl13"

    sim2.load_fit(fit_name)

    sim.set_params({"vir_load": 1, "vir_decay": 0.26})
    sim2.set_params({"vir_load": 5, "vir_decay": 0.01})

    # use 1 sigma confidence interval?
    CV = [get_SD(df_fit, param_name) for param_name in param_names]
    # CV = 0.05
    # CV = None # for multivariate case, do not need CV
    #
    sampling = "random"
    sim.global_sensitivtity(param_names=param_names, fname = fit_name, infection= "Arm", CV = CV,
                             res = res, sampling = sampling)
    sim.reset_params()
    #
    sim2.global_sensitivtity(param_names=param_names, fname = fit_name, infection= "Cl13", CV = CV,
                              res = res, sampling = sampling)
    sim2.reset_params()


