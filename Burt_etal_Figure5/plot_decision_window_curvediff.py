import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils_data_model import vir_model_expdecay
from analysis.late_branching.model_params import myparams
from models.late_branching import late_branching
from utils_data_model import run_infection
# make wide

plt.style.use("../paper_theme_python.mplstyle")

def proc_df(df, readout):
    """
    make df wide for heatmap
    """
    df = df.loc[df["readout"] == readout].copy()
    df = df[["t_start", "t_dur", "value"]]
    x = df.pivot_table(index="t_dur", columns="t_start", values="value").values

    return x

def plot_single_heatmap(df1, df2, r, t_start_arr, t_dur_arr, cmap = "rocket_r", xticks = [1], yticks = [1]):

    vals1 = proc_df(df1, r)
    vals2 = proc_df(df2, r)

    vals1 = vals1.T
    vals2 = vals2.T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.1, 1.8))
    vmin = 0
    vmax = 1.
    ylabel = "t (d)"
    xlabel = "perturbation window (d)"

    for ax, vals in zip([ax1, ax2], [vals1, vals2]):
        out = ax.pcolormesh(t_dur_arr, t_start_arr, vals, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
                            rasterized = True)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_xticks(xticks)
        #ax.set_yticks(yticks)
    cbar = fig.colorbar(out, ax=[ax1,ax2])
    cbar.set_label("curvediff")
    ax2.set_yticklabels([])
    ax2.set_ylabel("")
    ax1.set_title("Acute")
    ax2.set_title("Chronic")
    plt.tight_layout()
    plt.show()
    return fig

def plot_multi_heatmap(df, readouts, t_start_arr, t_dur_arr, cmap = "rocket_r", xticks = [0,25,50], yticks = [0,25,50]):

    df1 = df.loc[df["Infection"] == "Arm"]
    df2 = df.loc[df["Infection"] == "Cl13"]

    df_arm = []
    df_cl13 = []

    for r in readouts:
        vals1 = proc_df(df1, r)
        vals2 = proc_df(df2, r)

        vals1 = vals1.T
        vals2 = vals2.T
        df_arm.append(vals1)
        df_cl13.append(vals2)

    fig, axes = plt.subplots(1, 4, figsize=(5.6, 1.0))
    vmin = 0
    vmax = 1.0
    ylabel = "t (d)"
    xlabel = "perturbation window (d)"

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    for v, ax in zip(df_arm, [ax1,ax2]):
        out = ax.pcolormesh(t_dur_arr, t_start_arr, v, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)

    for v, ax in zip(df_cl13, [ax3,ax4]):
        out = ax.pcolormesh(t_dur_arr, t_start_arr, v, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)

    cbar = fig.colorbar(out, ax=[ax1,ax2, ax3, ax4])
    cbar.set_label("Curve diff. norm.")

    xticks = [0,5,10,15,20]
    xlim = [0,20]

    for ax in [ax1,ax2,ax3,ax4]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_xticks(xticks)
        ax.set_yticks(xticks)
    for ax in [ax2,ax3,ax4]:
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.set_xlabel("")

    plt.show()
    return fig

sdir = "../../figures/decision_window/heatmap_"

# parameters for decision window
res = 50
model = "lb" # late branching
fb = 100
xlim = 20
ylim = 20
f0 = str(res) + "_fb_"+ str(fb) + "_xlim" + str(xlim) + "_ylim" + str(ylim) + ".csv.gz"
f1 = "../../output/decision_window/hm_" + model + "_arm_gamma_res" + f0
f2 = "../../output/decision_window/hm_" + model + "_chr_gamma_res" + f0
df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)

time = np.arange(0,200,0.5)
sim = late_branching("Arm", time, myparams, vir_model_expdecay)
#
fit_name = "latebranch_fit_local"
sim.load_fit(fit_name)
sname = "local"
#
cells, mols = run_infection(sim)
cells.rename(columns = {"value" : "value_bestfit"}, inplace = True)

def get_curvediff(df1,df2, cells,  norm = True):

    df1["Infection"] = "Arm"
    df2["Infection"] = "Cl13"
    df = pd.concat([df1,df2]).reset_index(drop = True)
    cells = cells[["value_bestfit", "time", "species", "Infection"]]

    out = pd.merge(df, cells, how = "left")
    out = out.loc[out.readout.str.contains("cellnumber")]
    out = out.loc[out.species == "Tfh_all"]
    out["curvediff"] = np.abs((out["value"] - out["value_bestfit"])) / out["value_bestfit"]

    out2 = out.loc[out["time"] == out["time"].max(), ["curvediff", "t_start", "t_dur", "species", "Infection"]].drop_duplicates().reset_index(drop = True)
    out2["readout"] = "enddiff"
    out2.rename(columns = {"curvediff" : "value"}, inplace = True)

    # get mean and maximal curve difference
    out = out.groupby(["species", "t_start", "t_dur", "Infection"])["curvediff"].agg(["sum", "max"]).reset_index()
    out = out.melt(id_vars = ["species", "t_start", "t_dur", "Infection"], var_name = "readout")


    out = pd.concat([out, out2])
    if norm:
        out["value"] = out.groupby(["species", "readout"])["value"].transform(lambda x: (x-x.min()) / (x.max() - x.min()))
    return out

out = get_curvediff(df1, df2, cells)


t_start_arr = df1["t_start"].drop_duplicates().values
t_dur_arr = df1["t_dur"].drop_duplicates().values


#fig = plot_single_heatmap(out1, out2, "mean", t_start_arr, t_dur_arr)

fig = plot_multi_heatmap(out, ["sum", "max"], t_start_arr, t_dur_arr)

fig.savefig("../../figures/decision_window/decision_window_heatmap_curvediff.svg")
fig.savefig("../../figures/decision_window/decision_window_heatmap_curvediff.pdf")

def plot_single_heatmap2(df, r, t_start_arr, t_dur_arr, cmap = "rocket_r", xticks = [0,25,50], yticks = [0,25,50]):
    """
    quick and dirty fix to plot end difference
    """
    df1 = df.loc[df["Infection"] == "Arm"]
    df2 = df.loc[df["Infection"] == "Cl13"]

    vals1 = proc_df(df1, r)
    vals2 = proc_df(df2, r)

    vals1 = vals1.T
    vals2 = vals2.T

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(3, 1.0))
    vmin = 0
    vmax = 1.0
    ylabel = "t (d)"
    xlabel = "perturbation window (d)"

    out1 = ax1.pcolormesh(t_dur_arr, t_start_arr, vals1, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    out2 = ax2.pcolormesh(t_dur_arr, t_start_arr, vals2, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)

    cbar = fig.colorbar(out2, ax=[ax1,ax2])
    cbar.set_label("Curve diff. norm.")
    plt.show()
    return fig


fig = plot_single_heatmap2(out, "enddiff", t_start_arr, t_dur_arr)
