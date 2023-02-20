"""
plot acute and chronc
"""
import matplotlib.colors
import numpy as np
from lmfit import minimize, fit_report
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../../")

from models.late_branching import late_branching
from utils_data_model import vir_model_expdecay, run_infection, get_probs
from analysis.late_branching.model_params import myparams as myparams
import matplotlib
import copy
plt.style.use("../paper_theme_python.mplstyle")
sns.set_palette("deep")

path_data = "../../data/differentiation_kinetics/"
data = pd.read_csv(path_data + "fahey_data.csv")
data.rename(columns={"cell": "species", "value": "value_data", "name" : "Infection"}, inplace=True)
data["eps"] = data["value_data"] * 0.1
data = data.astype({"time": "float64"})

time = np.arange(0,70,0.1)

sim1 = late_branching("Prolif. with antigen", time, myparams, vir_model_expdecay)
sim1.load_fit("latebranch_fit_local")
cells, mols = run_infection(sim1)

palette = ["k", "grey"]
xlabel = "time post infection (d)"


cells2 = cells.loc[cells["species"].isin(["Naive", "Precursor", "Th1", "Tfh", "Th1_c", "Tfh_c", "Th1_mem",
                                          "Tfh_mem", "CD4_All"])]
g = sns.relplot(data = cells2, x = "time", y = "value", col = "species",
                hue = "Infection", kind = "line", height = 1.8, col_wrap = 5,
                palette = palette, aspect = 0.8)
g.set(yscale = "log", ylim = [1, 1e8], xlabel = xlabel, ylabel = "cells")
g.set_titles("{col_name}")
plt.show()

cells2.to_csv("../../output/timecourse_heatmaps/timecourse_bestfit_cells.csv", index = False)

g.savefig("../../figures/timecourses/timecourse_cells_bestfit.svg")
g.savefig("../../figures/timecourses/timecourse_cells_bestfit.pdf")


mols3 = mols.loc[mols["species"].isin(["IL2", "Antigen", "IL10"]), :].copy()
g = sns.relplot(data = mols3, x = "time", y = "value", hue = "Infection", col = "species", kind = "line",
                facet_kws = {"sharey" : False}, height = 1.8, palette = palette)
g.set(xlabel = xlabel, ylabel = "conc.")
g.set_titles("{col_name}")
plt.show()

g.savefig("../../figures/timecourses/timecourse_mols_bestfit.svg")
g.savefig("../../figures/timecourses/timecourse_mols_bestfit.pdf")

mols3.to_csv("../../output/timecourse_heatmaps/timecourse_bestfit_mols.csv", index = False)
