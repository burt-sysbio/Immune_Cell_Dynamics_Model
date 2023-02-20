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
time = np.arange(0,80,0.5)
xlabel = "time post infection (d)"
ylabel = "Tfh cells"
models = ["cntrl", "early", "int.", "late"]

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

sim_list[0].set_params({"fb_decwindow" : 1})
sim_list2[0].set_params({"fb_decwindow" : 1})

dur = 7
t_cntrl = (100,101)
t1 = (0,dur)
t2 = (dur, 2*dur)
t3 = (14, 21)
myspans = [t_cntrl, t1,t2,t3]
myspans2 = [t1,t2,t3]
# optimal time RTM
for sim in [sim_list, sim_list2]:
    for s, t in zip(sim, myspans):
        s.params["t_start"] = t[0]
        s.params["t_end"] = t[1]


# for perturbation, set feedback strength

mylist = []
mylist2 = []

for sim, sim2, model in zip(sim_list, sim_list2, models):
    cells, mols = sim.compute_cellstates()
    cells2, mols2 = sim2.compute_cellstates()

    cells_rel = sim.get_readouts(timepoints = [15.0,30.0], cell_names = ["Th1_all", "Tfh_all"])
    cells_rel["model"] = model

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

g = sns.relplot(data = df_dlb, x ="time", y = "value",
                row = "name", col = "regulation",
                kind = "line", aspect = 1., height = 1.5)

axes = g.axes.flatten()

for ax, val, t in zip(g.axes[:,0], myspans2, models):
    ax.axvspan(val[0], val[1], color = "grey", alpha = 0.3)
    sns.lineplot(data = df_cntrl, x = "time", y = "value", ax = ax, color = "k")

for ax, val in zip(g.axes[:,1], myspans2):
    ax.axvspan(val[0], val[1], color = "grey", alpha = 0.3)
    sns.lineplot(data = df_cntrl_lb, x = "time", y = "value", ax = ax, color = "k")

g.set(yscale = "log", ylim = [1e4, None], xlim = [0,30], ylabel = "cells",
      xlabel = xlabel, xticks = [0,7,14,21,28])

g.set_titles("")
plt.tight_layout()
sns.despine(top = False, right = False)
plt.show()

g.savefig("../../figures/decision_window/timecourse_timed_perturbation_latebranch.svg")
g.savefig("../../figures/decision_window/timecourse_timed_perturbation_latebranch.pdf")


df_cntrl = df_cntrl[["time", "species", "value"]].rename(columns = {"value" : "value2"})
df_cntrl["regulation"] = "Acute"

df_cntrl_lb = df_cntrl_lb[["time", "species", "value"]].rename(columns = {"value" : "value2"})
df_cntrl_lb["regulation"] = "Chronic"

df_cntrl = pd.concat([df_cntrl, df_cntrl_lb]).reset_index(drop = True)

out = pd.merge(df_dlb, df_cntrl, how = "left", on = ["time", "species", "regulation"])
out["val_norm"] = np.abs(np.log2(out.value / out.value2))


out = out.loc[out.time == 21.0]

g = sns.catplot(data = out, x = "name", y = "val_norm", hue = "regulation", kind = "bar", height = 1.8,
                palette = ["0.1", "grey"])
g.set_xticklabels(rotation = 90)
g.set(ylabel = "FC Tfh (vs cntrl)", xlabel = "")
sns.despine(top = False, right = False)
plt.show()

g.savefig("../../figures/decision_window/barplot_timed_perturbation_latebranch.svg")
g.savefig("../../figures/decision_window/barplot_timed_perturbation_latebranch.pdf")
