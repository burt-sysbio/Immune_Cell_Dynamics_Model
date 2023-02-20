import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

from models.late_branching import late_branching
from utils_data_model import vir_model_expdecay
from utils_data_model import run_infection
from analysis.late_branching.model_params import myparams as myparams
import sys
sys.path.append("../../")

plt.style.use("../paper_theme_python.mplstyle")

time = np.arange(0,70,0.1)
sim = late_branching("Arm", time, myparams, vir_model_expdecay)

sim.load_fit("latebranch_fit_local")
cells, mols = run_infection(sim)
cells.rename(columns = {"value" : "value_bestfit"}, inplace=True)


def proc_df(df, groupvar):
    """
    get maximum and normalized values and append to data frame
    """
    df["val_norm"] = df.groupby(["readout", "species", "model"])["value"].transform(
        lambda x: np.log2(x/x.min())) #(x - x.min()) / (x.max() - x.min()))
    df["valmax"] = df.groupby([groupvar, "readout", "species", "model"])["val_norm"].transform(lambda x: x.max())
    return df

def get_topt(df, groupvar):
    df = df.copy()
    out = df.groupby(["readout", "species", groupvar, "model"]).apply(get_maximum)
    out = out.reset_index()
    return out

def get_maximum(df):
    """
    interpolate maximum
    """
    x = df["t_start"].values
    y = df["val_norm"].values
    #y = df["val_max"].values

    f = InterpolatedUnivariateSpline(x, y, k=4)
    # get Nullstellen
    cr_pts = f.derivative().roots()
    cr_pts = np.append(cr_pts, (x[0], x[-1]))  # also check the endpoints of the interval
    cr_vals = f(cr_pts)
    max_index = np.argmax(cr_vals)

    max_x = cr_pts[max_index]
    max_y = cr_vals[max_index]

    # get the window boundaries for a given percentile
    percentile = 0.75

    x2 = x[y>percentile*max_y]

    window_low = min(x2)
    window_max = max(x2)

    out = pd.DataFrame({"t_opt_x" : [max_x], "t_opt_y" : [max_y], "window_low" : [window_low], "window_max" : [window_max]})
    return out


def pipeline(df, groupvar):
    """
    process df, filter readouts, get maximum by interpolation and respective window (percentile of max)
    """
    df = df.loc[df["species"] == "Th1_all"]
    df = df.loc[df["readout"].str.contains("relative")]

    df["model"] = "gamma"
    df = df.reset_index(drop=True)

    if groupvar == "t_dur":
        df = df.loc[df[groupvar] != 0].copy()
    elif groupvar == "fb_cyto1":
        df = df.loc[df[groupvar] != 1].copy()

    # filter readouts further?
    df = df.loc[df["readout"] == "relative_Th1_all_day_30.0"]
    #df = df.loc[df["readout"] == "curvediff"].copy()
    #df["value"] = df["value"].abs()
    # check optimal times depending on duration
    df2 = proc_df(df, groupvar=groupvar)
    out3 = get_topt(df2, groupvar=groupvar)

    return out3

# parameters for decision window
res = 30
model = "lb" # late branching
fb = 100

df1 = pd.read_csv("../../output/decision_window/hm_" + model + "_arm_gamma_res" + str(res) + "_fb_"+ str(fb) + ".csv")
df2 = pd.read_csv("../../output/decision_window/hm_" + model + "_chr_gamma_res" + str(res) + "_fb_"+ str(fb) + ".csv")

df_list = [df1,df2]

#df1 = pd.read_csv("../../output/decision_window/heatmap_arm_gamma_fb_cyto1_res10_tdur_0.5.csv")
palette = ["0.1", "grey"]
groupvar = "t_dur"

out_list = [pipeline(df, groupvar) for df in df_list]
fig, ax = plt.subplots(figsize = (2.,1.8))

for out, c in zip(out_list, palette):
    ax.scatter(out[groupvar], out["t_opt_x"], s = 2, color = c)
    ax.fill_between(out[groupvar], out["window_low"], out["window_max"], alpha = 0.1)
ax.set_xlabel(groupvar)
ax.set_ylabel("optimal time $t^*$ (h)")

ax.set_xlabel("perturbation window (h)")
ax.set_xticks([0,10,20,30,40])
ax.set_yticks([0,5,10,15,20,25])
if groupvar == "fb_cyto1":
    ax.set_xscale("log")
plt.tight_layout()

plt.text(10, 20, "Chronic", size = 12)
plt.text(1, 10, "Acute", size = 12)

plt.show()
fig.savefig("../../figures/decision_window/decision_window_range.svg")
fig.savefig("../../figures/decision_window/decision_window_range.pdf")

