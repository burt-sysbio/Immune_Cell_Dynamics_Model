import pandas as pd
"""
process output from proc_gamma_fit_module
uses the fits from module average expression data
note that alternative analysis exists for individual fits that 
are then matched to gene modules
"""
# search for Th1 genes in data
# use regular expression because some genes might have .1 .2 etc
print("preparing module data")
df_modules = pd.read_csv("../../genesets_literature/gene_module_summary.csv")
data = pd.read_csv("../../data/data_summary/data_rtm_combined.csv", low_memory=False)

# first get the mean üer celltype and timepoint and then the max of mean per celltype
data = data[["symbol", "cell_type", "time", "val_norm", "ID"]]
data.rename(columns= {"symbol" : "gene"}, inplace=True)

data = pd.merge(data, df_modules, how = "inner", on = "gene")

# do not group over genes, want average for all Th1 genes per time point and study
data["avg_norm"] = data.groupby(["ID", "time", "module"])["val_norm"].transform("mean")
# get maximum per group (irrespective of time) and divide
data["mean_max"] = data.groupby(["ID", "module"])["avg_norm"].transform(lambda s: s.max() if s.max() > -s.min() else s.min())

# this is if I really divide by max and not abs(min,max)
data["mean_max2"] = data.groupby(["ID", "module"])["avg_norm"].transform("max")

data["val_norm2"] = data["val_norm"] / data["mean_max"]
data["avg_norm2"] = data["avg_norm"] / data["mean_max"]

data["val_norm3"] = data["val_norm"] / data["mean_max2"]
data["avg_norm3"] = data["avg_norm"] / data["mean_max2"]


data["SD"] = data.groupby(["ID", "time", "module"])["val_norm2"].transform("std")

data.to_csv("../../data/data_summary/data_rtm_gene_module.csv")