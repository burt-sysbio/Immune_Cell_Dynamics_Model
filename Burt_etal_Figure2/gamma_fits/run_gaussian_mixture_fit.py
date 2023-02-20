import numpy as np
import pandas as pd
from lmfit import minimize, Parameters
import os
from utils import gmm_cdf
import joblib

def residual(params, x, data, eps_data):
    mean1 = params["mean1"]
    mean2 = params["mean2"]
    SD = params["SD"]
    #SD2 = params["SD2"]

    model = gmm_cdf(x, mean1, mean2, SD)
    return (data-model) / eps_data


def clean_data(data):
    # kick out data with NANs for expression values (Proserpio has some (from rlog?))
    genes_na = data["gene"][data["avg_norm_rtm2"].isna()]
    data = data[~data["gene"].isin(genes_na)]
    return data

def fit_res_to_frame(fit_res, gene, xdata, ydata):
    fitparams = fit_res.params.valuesdict()
    df = pd.DataFrame([fitparams])
    df["chisqr"] = fit_res.chisqr
    df["aic"] = fit_res.aic
    df["bic"] = fit_res.bic
    df["redchi"] = fit_res.redchi
    df["success"] = fit_res.success
    df["gene"] = gene

    y_sim = gmm_cdf(xdata,
                    fitparams["mean1"],
                    fitparams["mean2"],
                    fitparams["SD"])

    rmse = np.sqrt(np.sum((ydata-y_sim)**2) / len(ydata))
    df["rmse"] = rmse

    return df

def fit_gaussian_mixture_model(data):
    # process data
    print("fitting gaussian mixture model...")
    data = data.drop(["val_norm_rtm2"], axis=1)
    data = data.drop_duplicates()

    fit_results = []

    params = Parameters()
    init_mean = data["time"].max() / 2
    init_sd = init_mean
    params.add("mean1", value = init_mean, min = 0, max = data["time"].max())
    params.add("mean2", value = init_mean, min = 0, max = data["time"].max())
    params.add("SD", value = init_sd, min = 0, max = 100)
    #params.add("SD2", value= init_sd, min = 0, max=100)

    # fit data for each gene and celltype
    for name, group in data.groupby(["gene", "cell_type"]):
        group = group[["time", "gene", "cell_type", "avg_norm_rtm2", "SD"]].drop_duplicates()
        x = group["time"].values
        y = group["avg_norm_rtm2"].values
        eps_data = group["SD"].values

        # check that there are no nans in y
        # check that data is normalized and keep array only until max is reached
        assert (~np.isnan(y).any())
        assert np.max(y) == 1.0
        assert y[0] == 0

        max_idx = np.where(y == 1.0)[0][0]
        y = y[:(max_idx + 1)]
        x = x[:(max_idx + 1)]

        if len(y) > 3:
            # check if error data are available
            if (not np.isnan(eps_data).any()) and (eps_data != 0).any():
                eps_data[eps_data == 0] = np.mean(eps_data[eps_data != 0])
                eps_data = eps_data[:(max_idx + 1)]
            else:
                eps_data = np.ones_like(y)

            out = minimize(residual, params, args=(x, y, eps_data))
            out_df = fit_res_to_frame(out, group["gene"].iloc[0], x, y)
            fit_results.append(out_df)

    fit_results = pd.concat(fit_results)
    fit_results = fit_results.reset_index(drop=True)
    return fit_results

# read in all files
month = "feb2021"
datadir = "../../data/data_rtm/"
readdir = "../../output/gamma_fits/" + month + "/"

datanames = os.listdir(datadir)

# get filenames for data and corresponding summary files containing fit categories
datanames = [name for name in datanames if ".csv" in name]
data_all = [pd.read_csv(datadir + f) for f in datanames]
savenames = [readdir + "fit_GMM_" + f for f in datanames]


data_list = [clean_data(data) for data in data_all]

n = len(data_list)
fit_results = joblib.Parallel(n_jobs=n)(joblib.delayed(fit_gaussian_mixture_model)(data) for data in data_list)

for df, savename in zip(fit_results, savenames):
    df.to_csv(savename)