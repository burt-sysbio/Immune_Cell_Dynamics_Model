import pandas as pd
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from analysis.direct_branching.model_params import fit_fahey as fit_db
from analysis.late_branching.model_params import fit_lb

############# plot settings
plt.style.use("../../../paper_theme_python.mplstyle")
#sns.set(context = "poster", style = "ticks")
palette = ["steelblue", "indianred"]


################################ load data
sys.path.append("../../")
readdir = "../../output/global_sensitivity/timecourses/"
savedir = "../../figures/global_sensitivity/"

path_data = "../../data/differentiation_kinetics/"
data = pd.read_csv(path_data + "fahey_data.csv")
data.rename(columns={"value": "value_data", "cell": "species"}, inplace=True)

# load simulation data
CV = 0.05
sname = "all_params"

########################################### choose model
# use fit_db or fit_lb here
fit_name = fit_lb
filename = readdir + "timecourse_global_" + sname + "_" + fit_name + "_CV" + str(CV) + ".csv"
df = pd.read_csv(filename)


df2 = df.loc[df["species"].isin(["Th1_all", "Tfh_all"])]

# for df2 (plot with actual cell numbers, merge data from fahey

g1 = sns.relplot(data = df2, x="time", y= "value",
                col="name", hue="species",
                palette = palette,
                kind = "line", legend = False,
                ci = "sd", aspect = 0.8, height = 1.6)

for ax, n in zip(g1.axes.flatten(), ["Arm", "Cl13"]):

    # prc data
    data2 = data.loc[data["name"] == n,:]
    x1 = data2.loc[data2["species"] == "Th1_all"]
    x2 = data2.loc[data2["species"] == "Tfh_all"]
    sns.scatterplot(data=data2, x="time", y="value_data", hue="species", ax=ax, legend=False, palette=[palette[1], palette[0]])
    ax.errorbar(x1["time"], x1["value_data"], yerr=x1["eps"], ecolor=palette[0], fmt="none", capsize = 3, elinewidth = 0.5)
    ax.errorbar(x2["time"], x2["value_data"], yerr=x2["eps"], ecolor=palette[1], fmt="none", capsize = 3, elinewidth = 0.5)


g1.set(yscale="log", ylim=[1e4, 1e7])

axes = g1.axes.flat
axes[0].set_title("Acute")
axes[1].set_title("Chronic")

sns.despine(top = False, right = False)
xlabel_time = "time post infection (d)"

g1.set(xticks = [0,20,40,60], xlim = [0,70], yticks = [1e4, 1e5,1e6,1e7], xlabel = xlabel_time, ylabel = "cells")
plt.show()

g1.savefig("../../figures/global_sensitivity/global_sensitivity_timecourse.svg")
g1.savefig("../../figures/global_sensitivity/global_sensitivity_timecourse.pdf")

# same plot but with individual runs
# g1 = sns.relplot(data = df2, x="time", y= "value",
#                 col="name", row="species", hue = "ID",
#                 kind = "line", legend = False,
#                 aspect = 1.0)

# for ax, n in zip(g1.axes.flatten(), ["Arm", "Cl13"]):
#
#     # prc data
#     data2 = data.loc[data["name"] == n,:]
#     x1 = data2.loc[data2["species"] == "Th1_all"]
#     x2 = data2.loc[data2["species"] == "Tfh_all"]
#     sns.scatterplot(data=data2, x="time", y="value_data", hue="species", ax=ax, legend=False, palette=[palette[1], palette[0]])
#     ax.errorbar(x1["time"], x1["value_data"], yerr=x1["eps"], ecolor=palette[0], fmt="none", capsize=10)
#     ax.errorbar(x2["time"], x2["value_data"], yerr=x2["eps"], ecolor=palette[1], fmt="none", capsize=10)


# g1.set(yscale="log", ylim=[1e4, 1e7])
#
# axes = g1.axes.flat
# axes[0].set_title("Acute")
# axes[1].set_title("Chronic")
#
# sns.despine(top = False, right = False)
# xlabel_time = "time post infection (d)"
#
# g1.set(xticks = [0,20,40,60], xlim = [0,70], yticks = [1e4, 1e5,1e6,1e7], xlabel = xlabel_time, ylabel = "cells")
# plt.show()
