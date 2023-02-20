import seaborn as sns
import numpy as np
import pandas as pd
import os

# read in all files
month = "feb2021"
readdir = "../../output/gamma_fits/" + month + "/"
savedir = "../../output/fit_summary/"
filenames = os.listdir(readdir)

files2 = [f for f in filenames if "ftest" in f]
fnames = [f[6:] for f in files2]
fnames2 = [x[:-4] for x in fnames]


files1 = ["fit_res_" + f for f in fnames]
files3 = ["fit_summary_" + f for f in fnames]

fit_res_all = [pd.read_csv(readdir + f) for f in files1]
ftest_all = [pd.read_csv(readdir + f) for f in files2]
fit_summary_all = [pd.read_csv(readdir + f) for f in files3]

for df_list in [fit_res_all, ftest_all, fit_summary_all]:
    for name, df in zip(fnames2, df_list):
        df["study"] = name

fit_res_all = pd.concat(fit_res_all)
ftest_all = pd.concat(ftest_all)
fit_summary_all = pd.concat(fit_summary_all)

fit_res_all.to_csv(savedir + "fit_res_all.csv", index = False)
ftest_all.to_csv(savedir + "ftest_all.csv", index = False)
fit_summary_all.to_csv(savedir + "fit_summary_all.csv", index = False)
