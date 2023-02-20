import numpy as np
from lmfit import minimize, fit_report
import pandas as pd
import sys
import pickle
import seaborn as sns
sys.path.append("../../")
from models.direct_branching import realistic_il2_direct
from models.late_branching import late_branching
import copy
from utils_data_model import vir_model_const, vir_model_expdecay
from analysis.late_branching.model_params import myparams
import matplotlib.pyplot as plt
palette = ["steelblue", "indianred"]

plt.style.use("../paper_theme_python.mplstyle")

# get model
fit_name = "latebranch_fit_local"
time = np.arange(0,200,0.5)
xlabel = "time (d)"
ylabel = "Tfh cells"

n = 50
models = ["d" + str(x) for x in range(n)]

models = ["cntrl", *models]


param_list = [copy.copy(myparams) for _ in models]
sim_list = [late_branching(n, time, params, vir_model_expdecay) for n, params in zip(models, param_list)]

# set feedback (for cntrl is does not matter bc I do not set the time point of feedback)
for sim in sim_list:
    sim.load_fit(fit_name)
    sim.set_params({"vir_load" : 1, "vir_decay" : 0.26, "fb_decwindow" : 1000})

# copy simlist and add chronic parameters
sim_list2 = [copy.deepcopy(sim) for sim in sim_list]

for sim in sim_list2:
    sim.set_params({"vir_load" : 5, "vir_decay" : 0.01, "fb_decwindow" : 1000})

sim_list[0].set_params({"fb_decwindow": 1})
sim_list2[0].set_params({"fb_decwindow": 1})


t_cntrl = (300,301)

start_times = np.linspace(0,50,n)
dur = 5

myspans = [(i, i+dur) for i in start_times]

myspans2 = [t_cntrl, *myspans]

# optimal time RTM
for sim in [sim_list, sim_list2]:
    for s, t in zip(sim, myspans2):
        s.params["t_start"] = t[0]
        s.params["t_end"] = t[1]

# for perturbation, set feedback strength

mylist = []
mylist2 = []

for sim, sim2, model in zip(sim_list, sim_list2, models):
    cells, mols = sim.compute_cellstates()
    cells2, mols2 = sim2.compute_cellstates()

    mylist.append(cells)
    mylist2.append(cells2)

df_all = pd.concat(mylist[1:len(mylist)])
df_lb = pd.concat(mylist2[1:len(mylist)])

df_cntrl = mylist[0]
df_cntrl_lb = mylist2[0]


names = ["Tfh_all"]

df_cntrl = df_cntrl.loc[df_cntrl["species"].isin(names)]
df_cntrl_lb = df_cntrl_lb.loc[df_cntrl_lb["species"].isin(names)]

# plot both Th1 and Tfh for one topt
df_all = df_all.loc[df_all["species"].isin(names)]
df_all = df_all.reset_index(drop = True)

df_lb = df_lb.loc[df_lb["species"].isin(names)]
df_lb = df_lb.reset_index(drop = True)


df_all["regulation"] = "Acute"
df_lb["regulation"] = "Chronic"


# plot both, direct and not direct?
df_dlb = pd.concat([df_all, df_lb])
df_dlb.reset_index(inplace=True, drop = True)

# g = sns.relplot(data = df_dlb, x ="time", y = "value",
#                 col = "name", row = "regulation",
#                 kind = "line", aspect = 0.9, height = 1.)
#
# axes = g.axes.flatten()
#
# for ax, val in zip(g.axes[0,:], myspans):
#     ax.axvspan(val[0], val[1], color = "grey", alpha = 0.3)
#     sns.lineplot(data = df_cntrl, x = "time", y = "value", ax = ax, color = "k")
#
# for ax, val in zip(g.axes[1,:], myspans):
#     ax.axvspan(val[0], val[1], color = "grey", alpha = 0.3)
#     sns.lineplot(data = df_cntrl_lb, x = "time", y = "value", ax = ax, color = "k")
#
# g.set(yscale = "log", ylim = [1e2, None], ylabel = "cells",
#       xlabel = xlabel, xticks = [0,100,200])
#
# g.set_titles("")
# plt.tight_layout()
# sns.despine(top = False, right = False)
# plt.show()
#
#g.savefig("../../figures/decision_window/timecourse_timed_perturbation_latebranch_longterm.svg")
#g.savefig("../../figures/decision_window/timecourse_timed_perturbation_latebranch_longterm.pdf")

df_cntrl["regulation"] = "Acute"
df_cntrl_lb["regulation"] = "Chronic"
df_cntrl = pd.concat([df_cntrl, df_cntrl_lb]).reset_index(drop = True)
df_cntrl = df_cntrl[["regulation", "time", "value"]]
df_cntrl.rename(columns = {"value" : "value_cntrl"}, inplace = True)

test = pd.merge(df_dlb, df_cntrl, how = "left", on = ["time", "regulation"])

test["diff"] = np.abs(test["value"] - test["value_cntrl"])

out1 = test.groupby(["regulation", "name"])["diff"].agg(["max", "sum"]).reset_index()
out1 = out1.melt(id_vars= ["regulation", "name"], var_name= "readout", value_name= "diff")

out2 = test.loc[test["time"] == test["time"].max(), ["regulation", "name", "diff"]]
out2["readout"] = "enddiff"

out3 = pd.concat([out1,out2])

out3["name"] = out3["name"].str.slice(1).values.astype(float)

out3 = out3.loc[out3.readout == "max"]
out3["val_norm"] = out3.groupby(["readout"])["diff"].transform(lambda x: (x-x.min()) / (x.max()-x.min()))
g = sns.relplot(data = out3, x = "name", hue = "regulation", y = "val_norm", kind = "line",
                height = 1.5, aspect = 1.15, palette = ["0.1", "0.5"], lw = 1.5)
g.set(xlabel = "perturbation start (d)", yticks = [0,0.5,1], xticks = [0,10,20,30,40], xlim = [0,40], ylim = [0,None], ylabel = "Response Max",
      )
sns.despine(top = False, right=False)
#g.axes[0,0].set_ylim([0,5e5])
#g.axes[0,1].set_ylim([0,2e3])

plt.show()

g.savefig("../../figures/decision_window/lineplot_decision_window_relapse.svg")
g.savefig("../../figures/decision_window/lineplot_decision_window_relapse.pdf")
