import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.analysis.utils import gamma_cdf, fit_gamma
plt.style.use("../paper_theme_python.mplstyle")
"""
idea: plot fit results for any gene module and study together with actual data
concrete: plot Tbet dynamics, Tbet FACS DATA and Th1 module averages together, show that Tbet gene expression alone is not good
current script uses SEM, can be replaced with SD
"""
#sns.set(context = "poster", style = "ticks")

# read facsdata
data_peine_facs = pd.read_csv("../../data/peine_facsdata.csv", decimal=",", sep='\t')
data_peine_facs["val_norm"] = data_peine_facs.value / data_peine_facs.value.max()

data_peine_facs_gata = pd.read_csv("../../data/peine_facsdata_gata.csv", decimal=",", sep=';')
data_peine_facs_gata["val_norm"] = np.log2(data_peine_facs_gata.value / data_peine_facs_gata.value.min())
data_peine_facs_gata["val_norm"] = data_peine_facs_gata.val_norm / data_peine_facs_gata.val_norm.max()

fit_res_peine_facs_gata = fit_gamma(data_peine_facs_gata.time, data_peine_facs_gata.val_norm, gamma_fun= gamma_cdf,
                                    sigma_abs=False, sigma=None, bounds = (0, np.inf))

fit_res_peine_facs = fit_gamma(data_peine_facs.time, data_peine_facs.val_norm, gamma_fun = gamma_cdf, sigma = None, sigma_abs= False,
                               bounds = (0, np.inf))
tbet_facs_alpha = fit_res_peine_facs[0]
tbet_facs_beta = fit_res_peine_facs[1]

# get actual data points for tbet
datadir = "../../data/data_rtm/Peine_rtm_Th1_invitro.csv"
data_peine_th1 = pd.read_csv(datadir)
data_peine_th1_tbx21 = data_peine_th1.loc[data_peine_th1.gene == "Tbx21"]

# get alpha and beta for tbet fit in peine th1 data
df = pd.read_csv("../../output/fit_summary/fit_res_all.csv")
df_peine_th1 = df.loc[(df.gene == "Tbx21") & (df.study == "Peine_rtm_Th1_invitro")]
tbet_alpha = df_peine_th1.loc[df_peine_th1.model == "gamma", "alpha"].values[0]
tbet_beta = df_peine_th1.loc[df_peine_th1.model == "gamma", "beta"].values[0]
# use alpha and beta to get gamma cdf array
time_arr = np.linspace(0,120,100)
tbet_arr = gamma_cdf(time_arr, tbet_alpha, tbet_beta)
tbet_facs_arr = gamma_cdf(time_arr, tbet_facs_alpha, tbet_facs_beta)

data = pd.read_csv("../../data/data_summary/data_rtm_gene_module.csv")

ftest = pd.read_csv("../../data/data_summary/module_ftest_results.csv")
fit_results = pd.read_csv("../../data/data_summary/module_fit_results.csv")

mymodule = "th1_genes"
myID = "Peine_rtm_Th1_invitro"

data = data.loc[(data["ID"] == myID) & (data["module"] == mymodule)]
ftest = ftest.loc[(ftest["ID"] == myID) & (ftest["gene"] == mymodule)]
fit_results = fit_results.loc[(fit_results["ID"] == myID) & (fit_results["gene"] == mymodule)]


data["SEM"] = data.groupby(["time"])["val_norm"].transform("sem")
data = data[["time", "avg_norm2", "SEM"]].drop_duplicates()

x_idx = data.avg_norm2.values
x_idx = np.where(x_idx == 1)[0][0]
data = data.iloc[:(x_idx + 1), :]

alphas = fit_results["alpha"].values
betas = fit_results["beta"].values

# get gamma cdf vals
print("time arr was set manually to 120 for Peine data")
y_sim = [gamma_cdf(time_arr, alpha, beta) for alpha, beta in zip(alphas, betas)]

y_sim_gamma = y_sim[1]

fig, ax = plt.subplots(figsize = (2.4,2.1))
#sns.scatterplot(data=data, x="time", y="avg_norm2", ax=ax, color="k")
# plot tbet module data
ax.plot(time_arr, y_sim_gamma, color = "tab:blue", ls = "--")
ax.errorbar(data.time, data["avg_norm2"], yerr=data["SEM"], ecolor="tab:blue", fmt=".", capsize=6)
# plot tbet data
ax.plot(time_arr, tbet_arr, color = "tab:grey", ls = "--")
ax.scatter(data_peine_th1_tbx21.time, data_peine_th1_tbx21.avg_norm_rtm2, color = "tab:grey", s = 10, zorder = 100000)
# plt tbet facs data
ax.scatter(data_peine_facs.time, data_peine_facs.val_norm, s = 10, color = "lightgrey")
ax.plot(time_arr, tbet_facs_arr, ls = "--", color = "lightgrey")
# other settings
#ax.set_xlim([time_arr.min(), time_arr.max()])
ax.set_xticks([0,24,48,72,96,120])
ax.set_xlabel("time (h)")
ax.set_ylabel("% of maximum")
plt.tight_layout()
plt.show()

fig.savefig("../../figures/module_quantification/module_timecourse.pdf")
fig.savefig("../../figures/module_quantification/module_timecourse.svg")

#fig.savefig("../../figures/tcell_module/peine_th1_module_timecourse.pdf")
#fig.savefig("../../figures/tcell_module/peine_th1_module_timecourse.svg")
