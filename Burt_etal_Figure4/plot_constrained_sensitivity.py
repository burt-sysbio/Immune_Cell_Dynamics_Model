import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from analysis.late_branching.model_params import fit_lb
from analysis.direct_branching.model_params import fit_fahey

import seaborn as sns
import matplotlib.pyplot as plt
from utils_data_model import dataplot
from models.late_branching import late_branching
from utils_data_model import vir_model_expdecay
from utils_data_model import run_infection
from analysis.late_branching.model_params import myparams as myparams
from matplotlib import colors

import sys
sys.path.append("../../")

plt.style.use("../../../paper_theme_python.mplstyle")

PROPS = {
    'boxprops':{'facecolor':'white', 'edgecolor':'k'},
    'medianprops':{'color':'k'},
    'whiskerprops':{'color':'k'},
    'capprops':{'color':'k'}
}

time = np.arange(0,70,0.1)
sim = late_branching("Arm", time, myparams, vir_model_expdecay)

sim.load_fit("latebranch_fit_local")
cells, mols = run_infection(sim)

from utils_data_model import proc_pscan

savedir = "../../figures/global_sensitivity/"
mydir = "../../output/global_sensitivity/uniform/"
files = os.listdir(mydir)

print("make sure only correct files are loaded by filters set below")
#CV = "0.1"
#files = [f for f in files if "mcarlo" in f and "strain" in f and CV in f]
df_list = [pd.read_csv(mydir + f) for f in files]

# annotate the data with infection type
for f, df in zip(files, df_list):
    if "Cl13" in f:
        df["Infection"] = "Cl13"
    else:
        df["Infection"] = "Arm"

    if "all_params" in f:
        df["scan_type"] = "all_params"
    elif "fitparams" in f:
        df["scan_type"] = "fit_params"
    else:
        df["scan_type"] = "data_params"

    if "SSM" in f:
        df["model_type"] = "SSM"
    else:
        df["model_type"] = "RTM"
# df = load_pscan_data(pname, fit_name)

df_all = pd.concat(df_list)

# split data into subsets for plotting
mydf = df_all.loc[df_all["readout"].str.contains("cellnumber")].copy()

mydf2 = mydf.groupby(["time", "Infection", "species", "scan_type", "model_type"])["value"].agg(["mean", "std"]).reset_index()


def plot_filled_plot(mydf2, celltype, color, xlabel = "time post infection (d)", ylabel = "cells", sigma = 1):

    edgecolor = colors.to_rgba("k", alpha = 1)

    mydf2["upper"] = mydf2["mean"] + sigma*mydf2["std"]
    mydf2["lower"] = mydf2["mean"] - sigma*mydf2["std"]

    # split RTM and SSSM models
    mydf1 = mydf2.loc[mydf2["model_type"] == "SSM"].copy()
    mydf2 = mydf2.loc[mydf2["model_type"] == "RTM"].copy()

    alpha = 0.2
    alpha2 = 0.5
    alpha3 = 1
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (1.95, 0.9))

    # plot outer tube (unconstrained RTM)
    df = mydf2.loc[mydf2.scan_type == "data_params"]
    mydf3 = df.loc[(df.Infection == "Arm") & (df.species == celltype)]
    mydf4 = df.loc[(df.Infection == "Cl13") & (df.species == celltype)]
    ax1.fill_between(mydf3["time"], mydf3["lower"], mydf3["upper"], facecolor = colors.to_rgba(color, alpha))
    ax2.fill_between(mydf4["time"], mydf4["lower"], mydf4["upper"], facecolor = colors.to_rgba(color, alpha))


    # plot middle tube (constrained to fit params SSM)
    df = mydf1.loc[mydf1.scan_type == "fit_params"]
    mydf3 = df.loc[(df.Infection == "Arm") & (df.species == celltype)]
    mydf4 = df.loc[(df.Infection == "Cl13") & (df.species == celltype)]
    ax1.fill_between(mydf3["time"], mydf3["lower"], mydf3["upper"], facecolor = colors.to_rgba(color, alpha2))
    ax2.fill_between(mydf4["time"], mydf4["lower"], mydf4["upper"], facecolor = colors.to_rgba(color, alpha2))


    # plot inner tube (constrained to fit params RTM)
    df = mydf2.loc[mydf2.scan_type == "fit_params"]
    mydf3 = df.loc[(df.Infection == "Arm") & (df.species == celltype)]
    mydf4 = df.loc[(df.Infection == "Cl13") & (df.species == celltype)]
    ax1.fill_between(mydf3["time"], mydf3["lower"], mydf3["upper"], facecolor = colors.to_rgba(color, alpha3), edgecolor = edgecolor)
    ax2.fill_between(mydf4["time"], mydf4["lower"], mydf4["upper"], facecolor = colors.to_rgba(color, alpha3), edgecolor = edgecolor)

    for ax in (ax1,ax2):
        ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim([0,65])
        ax.set_xticks([0,30,60])
        ax.set_yticks([1e5,1e7,1e9])
        ax.set_ylim([1e4, 1e10])

    ax2.set_yticklabels([])
    plt.tight_layout()
    plt.show()

    fig.savefig("../../figures/confidence_intervals/timecourse_shaded_" + celltype + "_sigma" + str(sigma) + ".svg", transparent = True)
    fig.savefig("../../figures/confidence_intervals/timecourse_shaded_" + celltype + "_sigma" + str(sigma) + ".pdf", transparent = True)

# plot with 1sigma and 2sigma conf interval
#plot_filled_plot(mydf2, "Th1_all", "steelblue", sigma = 1)
#plot_filled_plot(mydf2, "Tfh_all", "indianred", sigma = 1)

plot_filled_plot(mydf2, "Th1_all", "steelblue", sigma = 2)
plot_filled_plot(mydf2, "Tfh_all", "indianred", sigma = 2)


#################### plotting
#########################################################################################################
#########################################################################################################
PROPS = {
    'boxprops':{'facecolor':'white', 'edgecolor':'0.1'},
    'medianprops':{'color':'0.1'},
    'whiskerprops':{'color':'0.1'},
    'capprops':{'color':'0.1'}
}

# rename best fit simulation
cells.rename(columns = {"value" : "value_bestfit"}, inplace=True)

mydf = mydf.loc[mydf["model_type"] == "RTM"].copy()
out = pd.merge(mydf, cells, how = "left", on = ["time", "species", "Infection"])

out["curvediff"] = (out["value"] - out["value_bestfit"]) / out["value_bestfit"]


mypal = sns.color_palette("dark")
mypal2 = [mypal[-2], mypal[-3]]

out2 = out.groupby(["species", "run_ID", "Infection", "scan_type"])["curvediff"].agg(["sum", "std", "mean"]).reset_index()

yaxis = "sum"
g = sns.catplot(data = out2, col = "Infection", y = yaxis, x = "scan_type", kind = "box", **PROPS,
                showfliers = False, palette = mypal2, height = 1.6, sharey = False, aspect = 0.6)

g.map_dataframe(sns.stripplot, x="scan_type", y=yaxis, palette= mypal2,
                dodge = False, s = 1.5, alpha = 0.9, rasterized = True)

g.set_xticklabels(rotation = 90, labels = ["free fit", "preannotated"])

g.axes[0,0].set_ylim([-300,3e3])
g.axes[0,1].set_ylim([-2e4,2e5])

g.set_titles("{col_name}")
g.set(ylabel = "curve difference (norm.)", xlabel = "")
g.axes[0,1].set_ylabel("")
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

g.savefig("../../figures/confidence_intervals/sum_curve_diff_constrained.svg", dpi = 300)
g.savefig("../../figures/confidence_intervals/sum_curve_diff_constrained.pdf", dpi = 300)

