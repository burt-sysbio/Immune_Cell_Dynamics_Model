"""
load fit results and match with gene modules
needs thorough cleaning of gene annotations to match with pathways

take fit information from individual genes and compute avg+SEM per module and study
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import warnings
sns.set(context = "poster", style = "ticks")
warnings.warn("time unit might not be adjusted between data sets")

# load data and clean gene names from isoforms and keep only gamma fits
df = pd.read_csv("../../output/fit_summary/fit_res_all.csv")
df[["gene2", "iso1", "iso2"]] = df["gene"].str.split(r"_|\.", expand = True)
mygenes = ["Ifng", "Tbx21", "Gata3", "Eomes", "Ly6c1"]
df = df.loc[df["gene"].isin(mygenes)]
df.drop(columns = ["gene"], inplace = True)
df.rename(columns = {"gene2" : "gene"}, inplace = True)
df = df.loc[df["model"] == "gamma"]
# create signature genes


df["mu"] = df["alpha"] / df["beta"]
df["SD"] = np.sqrt(df["alpha"] / df["beta"]**2)

# kick out Craford and Powrie because of timescale
df = df.loc[~df["study"].str.contains("Craw|Powrie", regex =True)]

out = df.groupby(["gene"]).agg({'mu': ['mean', 'sem'], 'SD': ["mean", "sem"]})
out = out.reset_index()
out.columns = ["gene", "mu_mean", "mu_sem", "sd_mean", "sd_sem"]
g = sns.relplot(data = df, x = "mu", y = "SD",
                hue = "study", col_order= mygenes,
                aspect = 0.6, col = "gene", zorder = 3)

axes = g.axes.flatten()
g.set(xlabel = "avg (h)", ylabel = "SD (h)", xticks = [0,20,40,60])
for ax, gene in zip (axes, mygenes):
    mydf = out.loc[out["gene"]==gene]
    ax.axvline(mydf["mu_mean"].iloc[0], c = "k", ls = "--", zorder = 2)
    ax.axhline(mydf["sd_mean"].iloc[0], c = "k", ls = "--", zorder = 1)

plt.show()
g.savefig("../../figures/module_quantification/candidate_genes.pdf")
g.savefig("../../figures/module_quantification/candidate_genes.svg")
