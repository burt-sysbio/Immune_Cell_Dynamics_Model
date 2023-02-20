import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(context = "paper", style = "ticks")
# combine bimodal files into one large set and add filter criteron for true bimodality
# (this still needs to be matched to other category)
# read in all files
month = "feb2021"
readdir = "../../output/gamma_fits/" + month + "/"

filenames = os.listdir(readdir)
fitnames = [name for name in filenames if "fit_GMM_" in name]

studies = [name[8:-4] for name in fitnames]

gmm_fits = [pd.read_csv(readdir + f) for f in fitnames]

for name, df in zip(studies, gmm_fits):
    df["study"] = name

df_all = pd.concat(gmm_fits)

df_all["bimodal"] = False
rmse_thres = 0.2
df_all.loc[(df_all["rmse"] < rmse_thres) & (df_all["success"]), "bimodal"] = True
df_all.to_csv("../../output/bimodal_genes/bimodal_genes_gaussian.csv")

g = sns.catplot(data = df_all, x = "study", y = "rmse", kind = "box", aspect= 1.2)
#g.set(yscale = "log")
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()
g.savefig("../../figures/fit_error_distributions/boxplot_rmsedist_gmmfit.pdf")