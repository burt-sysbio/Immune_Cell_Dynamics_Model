import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from utils_data_model import load_pscan_data, proc_pscan
from analysis.late_branching.model_params import fit_lb as best_fit
plt.style.use("../../../paper_theme_python.mplstyle")

mydir = "../../output/param_scans/"
files = os.listdir(mydir)

fit_name = best_fit
files = [f for f in files if fit_name in f]

df_list = [pd.read_csv(mydir + f) for f in files]


for f, df in zip(files, df_list):
    if "Cl13" in f:
        df["Infection"] = "Chronic"
    else:
        df["Infection"] = "Acute"
# df = load_pscan_data(pname, fit_name)

df_all = pd.concat(df_list)

def normfun(x):
    """
    sensitivity coefficients
    """
    delta_x = (x - x.median()) / x.median()
    delta_p = 0.1
    out = delta_x / delta_p
    return out

def proc_df(df_all):
    # change in readout values for change in coefficient
    df_all.loc[:, "val_norm"] = df_all.groupby(["species", "Infection", "param_name", "readout"])["value"].transform(normfun)
    # relative change in coefficient
    df_all.loc[:, "param_val_norm"] = df_all.groupby(["species", "Infection", "param_name", "readout"])[
        "param_value"].transform(lambda x: x / x.median())
    df_all["param_val_norm"] = np.round(df_all["param_val_norm"], 1)

    # focus on these two readout types only in TFH cells
    df_all = df_all.loc[df_all["readout"].str.contains("cellnumber|relative")]
    # only focus on Th1 cells (its symmetric for the most part anyways)

    # split columns to derive time points
    df_all[['a', 'b', "c", "d", "time"]] = df_all['readout'].str.split('_', expand=True)
    df_all["time"] = df_all["time"].astype("float")

    # kick out the middle value (default value)
    df_cellnumbers = df_all.loc[df_all["readout"].str.contains("cellnumber")]
    df_cellnumbers = df_cellnumbers.groupby(["species", "param_name", "Infection", "param_val_norm"]).apply(
        lambda x: np.trapz(x["value"], x["time"])).reset_index()
    df_cellnumbers.columns = ["species", "param_name", "Infection", "param_val_norm", "response_size"]
    df_cellnumbers["response_size_norm"] = df_cellnumbers.groupby(["species", "param_name", "Infection"])["response_size"].transform(lambda x: np.log2(x/x.median()))

    # only focus on Tfh cells
    df_all = df_all.loc[df_all["species"] == "Tfh_all"]
    df_all = df_all.loc[df_all["param_val_norm"] != 1]
    df_all["param_val_norm"] = df_all["param_val_norm"].astype("str")

    return df_all, df_cellnumbers


df_all, df_cellnumbers = proc_df(df_all)
# focus on these time points
df_all = df_all.loc[df_all["time"].isin([9.0, 30.0])]

# rename parameters
df_all.loc[df_all["param_name"].str.contains("deg_chr"), ["param_name"]] = "death_eff"

#df_cellnumbers = df_all.loc[df_all["readout"].str.contains("cellnumber")]
#df_cellnumbers = df_cellnumbers.loc[df_cellnumbers["time"].isin([30.0])]
df_relative = df_all.loc[df_all["readout"].str.contains("relative")]


df_relative["time"] = df_relative["time"].astype("str")
df_relative["title"] = "Tfh (% all) day " + df_relative["time"]

myparams = ["beta_eff", "beta_naive", "death_eff", "death_mem", "p1", "fb_IL10", "prop_ag", "r_mem", "r_chr", "initial_cells"]

labels = [r"$\beta_e$", r"$\beta_{a_1}$", r"$\delta_e$", r"$\delta_m$", r"$\lambda_{0,1}$", "fb_IL10", "prop_ag",
          r"$\beta_m$", r"$\beta_c$", "$n_0$"]

df_relative = df_relative.loc[df_relative.param_name.isin(myparams)]


g = sns.catplot(data = df_relative, x = "param_name", y = "val_norm",
                row = "Infection", hue = "param_val_norm", col = "title",
                row_order= ["Acute", "Chronic"], aspect = 1.,
                kind = "bar", dodge =False, facet_kws= {"margin_titles" : True},
                palette = ["k", "lightgrey"], height = 2)
g.set_titles(row_template = '{row_name}', col_template = '{col_name}')

g.set_xticklabels(rotation = 90)
g.set(xlabel = "", ylabel = "Sensitivity")
sns.despine(top = False, right = False)
plt.tight_layout()
plt.show()

# g.savefig("../../figures/local_sensitivity/sensitivity_barplots_lb.svg")
# g.savefig("../../figures/local_sensitivity/sensitivity_barplots_lb.pdf")


df_relative2 = df_relative.loc[df_relative.Infection == "Chronic"]
df_relative3 = df_relative.loc[df_relative.Infection == "Acute"]

g = sns.catplot(data = df_relative2, x = "param_name", y = "val_norm",
                col = "title", hue = "param_val_norm", order = myparams,
                aspect = 0.6,
                kind = "bar", dodge =True,
                palette = ["k", "white"], height = 1.8, color = "white", edgecolor = "k")
g.set_titles(row_template = '{row_name}', col_template = '{col_name}')

g.set_xticklabels(rotation = 90, labels = labels)
g.set(xlabel = "", ylabel = "Sensitivity",yticks = [-1,0,1])
sns.despine(top = False, right = False)
plt.tight_layout()
plt.show()

g.savefig("../../figures/local_sensitivity/sensitivity_barplots_lb_d30_chronic.svg")
g.savefig("../../figures/local_sensitivity/sensitivity_barplots_lb_d30_chronic.pdf")


# use old style sheet for supplements
plt.style.use("../paper_theme_python.mplstyle")
g = sns.catplot(data = df_relative3, x = "param_name", y = "val_norm",
                col = "title", hue = "param_val_norm",
                aspect = 1.0,
                kind = "bar", dodge =True,
                palette = ["k", "white"], height = 1.8, color = "white", edgecolor = "k")
g.set_titles(row_template = '{row_name}', col_template = '{col_name}')

g.set_xticklabels(rotation = 90, )
g.set(xlabel = "", ylabel = "Sensitivity",yticks = [-1,0,1])
sns.despine(top = False, right = False)
plt.tight_layout()
plt.show()

g.savefig("../../figures/local_sensitivity/sensitivity_barplots_lb_d30_acute.svg")
g.savefig("../../figures/local_sensitivity/sensitivity_barplots_lb_d30_acute.pdf")

