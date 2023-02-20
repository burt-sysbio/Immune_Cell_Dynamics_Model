import numpy as np
from lmfit import minimize, fit_report
import pandas as pd
import sys
import pickle
import seaborn as sns

sns.set(context = "poster", style ="ticks")
sys.path.append("../../")

from models.direct_branching import realistic_il2_direct
from models.late_branching import late_branching

from utils_data_model import vir_model_const, vir_model_expdecay
from analysis.direct_branching.model_params import myparams as myparams_db
from analysis.late_branching.model_params import myparams as myparams_lb
import matplotlib.pyplot as plt
palette = ["steelblue", "indianred"]

# get model
time = np.arange(0,60,0.5)
xlabel = "time (d)"
ylabel = "Tfh cells"
models = ["cntrl", "short, early", "short, intermediate", "long, early", "long, late"]

param_list = [myparams_db, myparams_db, myparams_db, myparams_db, myparams_db]
param_list2 = [myparams_lb, myparams_lb, myparams_lb, myparams_lb, myparams_lb]


sim_list = [realistic_il2_direct(n, time, params, vir_model_const) for n, params in zip(models, param_list)]
sim_list2 = [late_branching(n, time, params, vir_model_expdecay) for n, params in zip(models, param_list2)]

for sim in sim_list2:
    sim.load_fit("latebranch_fit_local")

dur = 1
t1 = 0.5
t2 = t1+dur
t3 = 1.1
t4 = t3+dur
t5 = 1
t6 = 5
t7 = 5
t8 = 9

# optimal time RTM
sim_list[1].params["t_start"] = t1
sim_list[1].params["t_end"] = t2
sim_list[2].params["t_start"] = t3
sim_list[2].params["t_end"] = t4
sim_list[3].params["t_start"] = t5
sim_list[3].params["t_end"] = t6
sim_list[4].params["t_start"] = t7
sim_list[4].params["t_end"] = t8

sim_list2[1].params["t_start"] = t1
sim_list2[1].params["t_end"] = t2
sim_list2[2].params["t_start"] = t3
sim_list2[2].params["t_end"] = t4
sim_list2[3].params["t_start"] = t5
sim_list2[3].params["t_end"] = t6
sim_list2[4].params["t_start"] = t7
sim_list2[4].params["t_end"] = t8


fb_stren = 10

use_chr = False

for sim, sim2, model in zip(sim_list, sim_list2, models):
    # first set de boer params
    fit_name = "fit_deboer2003_realistic_il2_direct"
    sim.load_fit(fit_name)

    # load fahey fit on top of de boer fit
    fit_name = "fit_fahey_realistic_il2_direct"
    sim.load_fit(fit_name)
    sim.params["fb_decwindow"] = 10.0

    sim2.params["fb_decwindow"] = 100.0
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

df_all = pd.concat(mylist[1:5])

df_lb = pd.concat(mylist2[1:5])

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


df_all["regulation"] = "direct"
df_lb["regulation"] = "prolif prec"


# plot both, direct and not direct?
df_dlb = pd.concat([df_all, df_lb])
df_dlb.reset_index(inplace=True, drop = True)

g = sns.relplot(data = df_all, x ="time", y = "value",
                col = "name", hue = "regulation",
                kind = "line", aspect = 0.9)

axes = g.axes.flatten()

myspans = [(t1,t2), (t3,t4), (t5,t6), (t7,t8)]

for ax, val in zip(axes, myspans):
    ax.axvspan(val[0], val[1], color = "grey", alpha = 0.3)
    sns.lineplot(data = df_cntrl, x = "time", y = "value", ax = ax, color = "k")
    #sns.lineplot(data= df_cntrl_lb, x="time", y="value", ax=ax, color="grey")

g.set(yscale = "log", ylim = [5e4, None], xlim = [0, 20], ylabel = "cells",
      xlabel = xlabel, xticks = [0,5,10,15,20])

g.set_titles("{col_name}")
sns.despine(top = False, right = False)
plt.show()

g.savefig("../../figures/decision_window/timecourse_short1.svg")


df_cntrl = df_cntrl[["time", "species", "value"]].rename(columns = {"value" : "value2"})
df_cntrl["regulation"] = "direct"

df_cntrl_lb = df_cntrl_lb[["time", "species", "value"]].rename(columns = {"value" : "value2"})
df_cntrl_lb["regulation"] = "prolif prec"

df_cntrl = pd.concat([df_cntrl, df_cntrl_lb]).reset_index(drop = True)

out = pd.merge(df_dlb, df_cntrl, how = "left", on = ["time", "species", "regulation"])
out["val_norm"] = np.log2(out.value / out.value2)


g = sns.relplot(data = out, x ="time", y = "val_norm",
                col = "name", hue = "regulation",
                kind = "line",
                palette = palette, aspect = 0.9)

axes = g.axes.flatten()

for ax, val in zip(axes, myspans):
    ax.axvspan(val[0], val[1], color = "grey", alpha = 0.3)

g.set(xlim = [0, 20], ylabel = "cells",
      xlabel = xlabel, xticks = [0,5,10,15,20])

g.set_titles("{col_name}")
sns.despine(top = False, right = False)
plt.show()

g.savefig("../../figures/decision_window/decision_window_timecourse2.svg")
