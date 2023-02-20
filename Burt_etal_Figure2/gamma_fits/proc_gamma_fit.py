import pandas as pd
import os
import numpy as np
# assign "other" category based on rmse threshold and assign ftest significance for
# genes not in "other" category

def get_genesets(df_fit):
    # take dataframe after filtering for errors and return gene sets for pathway analysis
    col = "f-test"
    val1 = "sig"

    # get genes where fit was bad
    df0 = df_fit.loc[~df_fit["keep_fit"], ["gene"]].drop_duplicates()

    # get genes where fit was good
    df_fit = df_fit.loc[df_fit["keep_fit"]]

    # get number of genes where delay was significant
    df1 = df_fit.loc[(df_fit[col] == val1) & (df_fit["comp"] == "expo_gamma"), ["gene"]]
    df2 = df_fit.loc[(df_fit[col] == val1) & (df_fit["comp"] == "expo_longtail"), ["gene"]]

    # expo are all genes with good fit except those in df1 or df2
    df3 = df_fit.loc[~(df_fit.gene.isin(df1.gene) | df_fit.gene.isin(df2.gene)), ["gene"]]
    df3 = df3.drop_duplicates()

    df0["best_fit"] = "other"
    df1["best_fit"] = "gamma"
    df2["best_fit"] = "longtail"
    df3["best_fit"] = "expo"

    print(np.sum([n in df0["gene"] for n in df3["gene"]]))

    df = pd.concat([df0, df1, df2, df3])
    return df

# read in all files
month = "feb2021"
readdir = "../../output/gamma_fits/" + month + "/"
genesetdir = "../../output/genesets/" + month + "/"
filenames = os.listdir(readdir)

files2 = [f for f in filenames if "ftest" in f]
fnames = [f[6:] for f in files2]

files1 = ["fit_res_" + f for f in fnames]

fit_res_all = [pd.read_csv(readdir + f) for f in files1]
ftest_all = [pd.read_csv(readdir + f) for f in files2]

for fit_res, ftest, fname in zip(fit_res_all, ftest_all, fnames):

    # get genes where there is one good fit (low rmse) for at least one model (gamma/longtail)
    rmse_thres = 0.2
    fit_res["keep_fit"] = fit_res.groupby("gene")["rmse"].transform(lambda x: (x < rmse_thres).any())

    # annotate good fit genes in ftest data
    ftest["keep_fit"] = False
    genes_good_fit = fit_res.loc[fit_res["keep_fit"], ["gene"]]
    ftest.loc[ftest.gene.isin(genes_good_fit.gene), "keep_fit"] = True

    # save gene sets with delayed and nondelayed genes separately for pathway analysis
    df = get_genesets(ftest)
    df.to_csv(readdir + "fit_summary_" + fname)
