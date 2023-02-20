import pandas as pd
from utils import run_f_test
import numpy as np
"""
fit gamma distribution on average module expression data
note that this is different from fitting individual genes 
and then mathcing with module
"""
data = pd.read_csv("../../data/data_summary/data_rtm_gene_module.csv")

data_summary = data[["time", "SD", "avg_norm2", "ID", "module"]].drop_duplicates()
data_summary = data_summary.rename(columns={"avg_norm2" : "avg_norm_rtm2"})

data_summary["gene"] = data_summary["module"]
data_summary["cell_type"] = data_summary["ID"]
data_summary.drop(["module"], axis = 1, inplace = True)
data_summary["SD"] = np.nan
#
# # apply f test for each group and refactor output
data_grouped = data_summary.groupby(["ID"])
out = data_grouped.apply(run_f_test)
mygroups = data_grouped.groups.keys()
groups = [name for name,unused_df in data_grouped]
out1 = [x[0] for x in out]
out2 = [x[1] for x in out]

for x, y, group in zip(out1, out2, groups):
    x["ID"] = group
    y["ID"] = group

#
out1 = pd.concat(out1)
out2 = pd.concat(out2)

out1.to_csv("../../data/data_summary/module_fit_results.csv", index = False)
out2.to_csv("../../data/data_summary/module_ftest_results.csv", index = False)

#
# out1["mu"] = out1["alpha"] / out1["beta"]
# out1["mu_sd"] = out1["alpha"] / out1["beta"]**2
#
# # get genes for which expo vs gamma was signif
# mygenes = out2.loc[(out2["f-test"] == "sig") & (out2["comp"] == "expo_gamma"), "gene"].values
#
# out1["sig"] = False
# out1.loc[out1["gene"].isin(mygenes), "sig"] = True
#
# # plot alpha and avg values for gamma fit
# out3 = out1.loc[out1["model"] == "gamma"]
# out3 = out3[["gene", "sig", "alpha", "beta", "mu", "mu_sd"]]

#out3 = out3.loc[~out3["gene"].str.contains("Nir", regex = False)]

#out3 = pd.melt(out3, value_vars = ["mu", "mu_sd"], id_vars= ["gene", "sig"])

#g = sns.catplot(data = out3, x = "gene", y = "mu", kind = "bar", aspect = 1.0)
#g.set(ylabel = "Th1 Module mean arrival time")
#ticks = [i for i in range(len(out3["gene"].values))]
#plt.xticks(labels = [str(i) for i in ticks], ticks = ticks)
#plt.show()
