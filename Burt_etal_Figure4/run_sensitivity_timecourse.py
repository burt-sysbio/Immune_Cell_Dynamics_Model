import pandas as pd
import sys
import numpy as np
sys.path.append("../../")
from models.direct_branching import realistic_il2_direct
from models.late_branching import late_branching
from utils_data_model import vir_model_const, vir_model_expdecay
from analysis.direct_branching.model_params import myparams as myparams_db
from analysis.late_branching.model_params import myparams as myparams_lb
from analysis.late_branching.model_params import fit_lb, fit_params_lb, data_params_lb
from analysis.direct_branching.model_params import fit_fahey, fit_deboer, fit_params_db, data_params_db
import copy

def myfun(sim, params, CV=0.05, res = 50):
    """
    run sim multiple times with lognorm params and append cells
    """
    mylist= []
    for i in range(res):
        sim.set_params_lognorm(params, CV)
        sim.compute_cellstates()

        out = sim.cells

        out["ID"] = i
        # also add relative cell numbers
        out2 = sim.get_relative_cells()
        out2.rename(columns={"value": "value_rel"}, inplace=True)
        out = pd.merge(out, out2, how="inner", on=["time", "species"])

        sim.reset_params()
        mylist.append(out)

    df = pd.concat(mylist)
    return df


def myfun2(sim1, sim2, params, CV, res, fit_name, sname):
    """
    for acute and chronic infection (sim1, sim2) (make sure vir load is set)
    run timecourse and store output combined
    """
    cells_acute = myfun(sim1, params=params, CV=CV, res=res)
    cells_chronic = myfun(sim2, params=params, CV=CV, res=res)
    cells_all = pd.concat([cells_acute, cells_chronic])

    savedir = "../../output/global_sensitivity/"
    filename = savedir + "timecourse_global_" + sname + "_" + fit_name + "_CV" + str(CV) + ".csv"
    cells_all.to_csv(filename)

    sim1.reset_params()
    sim2.reset_params()


def prep_simulations(mymodel):
    time = np.arange(0, 80, 0.1)

    if mymodel == "db":
        sim1 = realistic_il2_direct("Arm", time, myparams_db, vir_model_const)
        sim2 = realistic_il2_direct("Cl13", time, myparams_db, vir_model_const)

        sim1.load_fit(fit_deboer)
        sim1.load_fit(fit_fahey)
        sim2.load_fit(fit_deboer)
        sim2.load_fit(fit_fahey)
        sim1.set_params({"vir_load": 0})
        sim2.set_params({"vir_load": 1})

        all_params = [*data_params_db, *fit_params_db]

        fit_name = fit_fahey

    elif mymodel == "lb":
        sim1 = late_branching("Arm", time, myparams_lb, vir_model_expdecay)
        sim2 = late_branching("Cl13", time, myparams_lb, vir_model_expdecay)
        sim1.load_fit(fit_lb)
        sim2.load_fit(fit_lb)

        sim1.set_params({"vir_load" : 1, "vir_decay" : 0.26})
        sim2.set_params({"vir_load": 5, "vir_decay": 0.01})

        all_params = [*data_params_lb, *fit_params_lb]

        fit_name = fit_lb

    return sim1, sim2, all_params, fit_name


#### the output depends strongly on whether I use fitparams or not
sname = "all_params"
CV = 0.05
res = 50

for mymodel in ["lb"]:
    sim1, sim2, all_params, fit_name = prep_simulations(mymodel)

    # note that fit_name does not update fit params
    myfun2(sim1, sim2, params = all_params, CV = CV, res = res, fit_name = fit_name, sname = sname)
