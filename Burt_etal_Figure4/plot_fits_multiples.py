import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils_data_model import dataplot
from models.late_branching import late_branching
from utils_data_model import vir_model_expdecay
from utils_data_model import run_infection
from analysis.late_branching.model_params import myparams as myparams
from analysis.late_branching.model_params import fit_names_global as fit_names
import sys
sys.path.append("../../")
plt.style.use("../paper_theme_python.mplstyle")
palette = ["steelblue", "indianred"]

# load data
path_data = "../../data/differentiation_kinetics/"
data = pd.read_csv(path_data + "fahey_data.csv")
data.rename(columns={"value": "value_data", "cell": "species"}, inplace=True)

time = np.arange(0,70,0.1)
sim = late_branching("Arm", time, myparams, vir_model_expdecay)
# set this as basic behavior
sim.load_fit("latebranch_fit_local")

# load every fit and plot
for fit_name in fit_names:
    sim.load_fit(fit_name)

    cells, mols = run_infection(sim)

    xlabel_time = "time post infection (d)"

    df2 = cells.loc[cells["species"].isin(["Th1_all", "Tfh_all"])]

    # for df2 (plot with actual cell numbers, merge data from fahey
    g1 = sns.relplot(data = df2, x="time", y= "value",
                    col="Infection", hue="species",
                    palette = palette,
                    kind = "line", legend = False,
                    ci = "sd", aspect = 1.0, height = 2)
    for ax, n in zip(g1.axes.flatten(), ["Arm", "Cl13"]):

        # prc data
        data2 = data.loc[data["name"] == n,:]
        x1 = data2.loc[data2["species"] == "Th1_all"]
        x2 = data2.loc[data2["species"] == "Tfh_all"]
        sns.scatterplot(data=data2, x="time", y="value_data", hue="species", ax=ax, legend=False, palette=[palette[1], palette[0]])
        ax.errorbar(x1["time"], x1["value_data"], yerr=x1["eps"], ecolor=palette[0], fmt="none")
        ax.errorbar(x2["time"], x2["value_data"], yerr=x2["eps"], ecolor=palette[1], fmt="none")


    g1.set(yscale="log", ylim=[1e4, 1e7])

    axes = g1.axes.flat
    axes[0].set_title("Acute")
    axes[1].set_title("Chronic")

    sns.despine(top = False, right = False)
    g1.set(xticks = [0,20,40,60], xlim = [0,70], yticks = [1e4, 1e5,1e6,1e7], xlabel = xlabel_time, ylabel = "cells")
    plt.show()

    g1.savefig("../../figures/fit_results/fit_result_" + fit_name + ".svg")
    g1.savefig("../../figures/fit_results/fit_result_" + fit_name + ".pdf")
