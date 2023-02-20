import numpy as np
import pandas as pd
import seaborn as sns
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("../paper_theme_python.mplstyle")

month = "feb2021"
readdir = "../../output/gamma_fits/" + month + "/"

filenames = os.listdir(readdir)

filenames = [f for f in filenames if "fit_summary" in f]

# read files and add study column
fits = []
for n in filenames:
    if "fit_summary_all" not in n:
        fit = pd.read_csv(readdir + n)
        fit["study"] = n[12:-4]
        fits.append(fit)

df_fits = pd.concat(fits)

# load the bimodal genes
df_bimodal = pd.read_csv("../../output/bimodal_genes/bimodal_genes_gaussian.csv")
df_bimodal = df_bimodal[["gene", "study", "bimodal"]]

# merge bimodal genes to the original df
df_fits = pd.merge(df_fits, df_bimodal, on=["gene", "study"], how="left")
df_fits.loc[df_fits["bimodal"] & (df_fits["best_fit"] == "other"), "best_fit"] = "bimodal"

xlabels = ["Craw. 1", "Craw. 2", "Craw. 3", "Craw. 4",
           "Nir 1", "Nir 2",
           "Peine 1", "Peine 2", "Peine 3", "Peine 4",
           "Ilott 1",
           "Pros. 1"]

# get total number of kinetic genes
df = df_fits.groupby(["study", "best_fit"])["gene"].count()
df = df.reset_index()

df_plot = df.groupby(["study"])["gene"].sum().reset_index()
g = sns.catplot(data=df_plot, x="study", y="gene", kind="bar", color="0.5")
g.set(xlabel="", ylabel="genes kinetic")
ax = g.axes[0][0]
ax.set_xticklabels(xlabels)
plt.xticks(rotation=90)
plt.show()

df = df.pivot(index="study", columns="best_fit", values="gene")

sns.set_palette("deep")
colors = sns.color_palette("deep", 10)
mycolors = ["purple", colors[2], colors[0], colors[7]]

# plot total kinetic genes
mysize = 8

# remove longtail genes by assigning them to expo category
df["expo"] = df["expo"] + df["longtail"]
df.drop(["longtail"], axis = 1, inplace = True)

ax = df.plot.bar(stacked=True, color = mycolors, figsize=(2.5,2.1), width = 0.85)
ax.set_ylabel("n kinetic genes")
ax.set_xlabel("")
ax.set_xticklabels(xlabels)

fig = ax.get_figure()
# title
new_title = 'Category'
ax.legend(title=new_title)
plt.tight_layout()
plt.show()

fig.savefig("../../figures/category_assignment/barplot_total_kinetic.pdf")
fig.savefig("../../figures/category_assignment/barplot_total_kinetic.svg")

# plot genes normalized
total = df.sum(axis=1)
df_norm = df.values / np.reshape(total.values, (df.shape[0], 1))

df_norm = pd.DataFrame(df_norm, columns=df.columns)
df_norm.index = df.index

df_norm = df_norm * 100


ax = df_norm.plot.bar(stacked=True, color=mycolors, figsize=(2.5, 2.1), width = 0.85)
ax.set_ylabel("fit assignment (%)")
ax.set_xticklabels(xlabels)
ax.set_ylim(0, 100)
ax.set_xlabel("")

fig = ax.get_figure()
# title
new_title = 'Category'
ax.legend(title=new_title)
plt.tight_layout()
plt.show()

fig.savefig("../../figures/category_assignment/barplot_fit_categories.pdf")
fig.savefig("../../figures/category_assignment/barplot_fit_categories.svg")

# save the fit summary
#df_fits.to_csv("../../output/fit_summary/category_assignment.csv")
