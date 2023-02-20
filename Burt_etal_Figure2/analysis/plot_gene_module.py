"""
load fit results and match with gene modules
needs thorough cleaning of gene annotations to match with pathways

take fit information from individual genes and compute avg+SEM per module and study
"""

from scipy.special import gammainc
import numpy as np

def gamma_cdf(t, alpha, beta):
    dummy = t * beta
    return gammainc(alpha, dummy)

import pandas as pd
import seaborn as sns
import os
from scipy.stats import gamma
import matplotlib.pyplot as plt
import warnings
#sns.set(context = "paper", style = "ticks")
plt.style.use("../paper_theme_python.mplstyle")
sns.set_palette("deep")
warnings.warn("time unit might not be adjusted between data sets")

df_modules = pd.read_csv("../../genesets_literature/gene_module_summary.csv")

# kick out some categories
#rm_module = ["Differentiation"]
#df_modules = df_modules.loc[~df_modules["module"].isin(rm_module)]
# only keep genes with good gamma fits
rmse_thres = 0.2

ylim = [None,None]
xlim = [None,None]
capsize = 4
linewidth = 1.5
xlabel = r"$\mu$ (h)"
ylabel = "SD (h)"
modelfit = "gamma"


df = pd.read_csv("../../output/fit_summary/fit_res_all.csv")

# convert beta because unit is days for Crawford and Powrie
df.loc[df["study"].str.contains("Craw|Powrie", regex =True), "beta"] = df["beta"]/24

df = df.loc[df["rmse"] < rmse_thres]
df = df.loc[df["model"] == modelfit]
#df = df.loc[df["alpha_err"]<10]
#df = df.loc[df["beta_err"]<10]

# clean gene names
df[["gene2", "iso1", "iso2"]] = df["gene"].str.split(r"_|\.", expand = True)

df.drop(columns = ["gene"], inplace = True)
df.rename(columns = {"gene2" : "gene"}, inplace = True)

df = pd.merge(df, df_modules, how = "inner", on = "gene")

df["mu"] = df["alpha"] / df["beta"]
df["SD"] = np.sqrt(df["alpha"] / df["beta"]**2)

# compute mean and sem for genes across studies and modules
out = df.groupby(["module", "study"]).agg({'mu': ['mean', 'sem'], 'SD': ["mean", "sem"],
                                           'alpha': ['mean', 'sem'], 'beta': ["mean", "sem"]})
out.reset_index(inplace = True)
out.columns = ["module", "study", "avg", "avg_err", "SD", "SD_err", "alpha", "alpha_err", "beta", "beta_err"]

out_th1_th2 = out.loc[out.study.str.contains("Peine")]
out_th1_th2 = out_th1_th2.loc[out_th1_th2.study.str.contains("Th1|Th2", regex = True)]

mymodules = ["Cytokine Receptor", "Cytokines", "Transcription Factor", "Proliferation",
             "IL2_signal", "Differentiation"]
out_general = out.loc[out["module"].isin(mymodules)]

out_general = out_general.loc[out_general.study.str.contains("Peine|Nir|Proserpio", regex =True)]

g = sns.FacetGrid(data = out_general, col = "module", hue = "study", legend_out=True, aspect = 0.7,
                  col_wrap= 6, height = 1.6)
g.map_dataframe(plt.errorbar, x= "avg", y = "SD", yerr = "SD_err", xerr= "avg_err",elinewidth=linewidth, capsize=capsize)
g.set_titles("{col_name}")
g.add_legend()
g.set(xlim = xlim, ylim = ylim, xlabel = xlabel, ylabel = ylabel, xticks = [10,20,30,40])
sns.despine(top = False, right = False)
plt.show()
g.savefig("../../figures/module_quantification/supp_allmodules.pdf")
g.savefig("../../figures/module_quantification/supp_allmodules.svg")


out_th17_nir = out.loc[(out.study == "Nir_rtm_Th17_invitro") & (out.module == "Th17")]
out_th1_peine = out.loc[(out.study == "Peine_rtm_Th1_invitro") & (out.module == "Th1")]
out_th2_peine = out.loc[(out.study == "Peine_rtm_Th2_invitro") & (out.module == "Th2")]
out_th2_proserpio = out.loc[(out.study == "Proserpio_rtm_Th2_parasite") & (out.module == "Th2")]

out_differentiation = pd.concat([out_th17_nir, out_th1_peine, out_th2_peine]).reset_index(drop =True)

tbet_facs_alpha = 3.866
tbet_facs_beta = 0.1296
tbet_facs_mu = tbet_facs_alpha / tbet_facs_beta
tbet_facs_sd = np.sqrt(tbet_facs_alpha / tbet_facs_beta**2)

gata_facs_alpha = 2.091
gata_facs_beta = 0.073
gata_facs_mu = gata_facs_alpha / gata_facs_beta
gata_facs_sd = np.sqrt(gata_facs_alpha / gata_facs_beta**2)

g = sns.FacetGrid(data = out_differentiation, hue = "module", legend_out=True, aspect = 1.1,
                  hue_order= ["Th1", "Th2", "Th17"], palette=["tab:blue", "tab:red", "darkgoldenrod"],
                  height= 1.7)
g.map_dataframe(plt.errorbar, x= "avg", y = "SD", yerr = "SD_err", xerr= "avg_err",elinewidth=linewidth, capsize=capsize)
ax = g.axes.flatten()[0]
ax.errorbar(x = out_th2_proserpio.avg, y = out_th2_proserpio.SD,
            yerr = out_th2_proserpio["SD_err"],
            xerr = out_th2_proserpio["avg_err"],
            capsize = capsize,
            color = "lightcoral")

ax.scatter(gata_facs_mu, gata_facs_sd, color = "tab:red", label = "Th2 FACS")
ax.scatter(tbet_facs_mu, tbet_facs_sd, color = "tab:blue", label = "Th1 FACS")
g.add_legend()
g.set(xlim = xlim, xticks = [10,20,30,40], ylim = ylim, xlabel = xlabel, ylabel = ylabel)
sns.despine(top = False, right = False)
plt.show()
g.savefig("../../figures/module_quantification/main_diffmodules.pdf")
g.savefig("../../figures/module_quantification/main_diffmodules.svg")

out_th1 = out.loc[out["study"] == "Peine_rtm_Th1_invitro"]
mymodules = ["Cytokine Receptor", "Cytokines", "Transcription Factor", "Proliferation", "Th1"]
out_th1 = out_th1.loc[out_th1["module"].isin(mymodules)]
g = sns.FacetGrid(data = out_th1, hue = "module",  aspect = 1.1, legend_out=False,
                  hue_order= ["Th1", "Proliferation","Transcription Factor","Cytokines","Cytokine Receptor"],
                  height = 2.1)
g.map_dataframe(plt.errorbar, x= "avg", y = "SD", yerr = "SD_err", xerr= "avg_err",elinewidth=linewidth, capsize=capsize)
g.add_legend()
g.set(xlim = xlim, ylim = ylim, xlabel = xlabel, ylabel = ylabel)
sns.despine(top = False, right = False)
plt.show()
g.savefig("../../figures/module_quantification/peine_th1cell_modules.pdf")
g.savefig("../../figures/module_quantification/peine_th1cell_modules.svg")



