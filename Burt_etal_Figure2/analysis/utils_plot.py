#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:46:15 2020

@author: burt
"""
import seaborn as sns
import pandas as pd
import numpy as np
from utils import gamma_cdf, gmm_cdf

def prep_fit(gene: str, data, fit_res_gamma, fit_res_gmm):
    """
    purpose: plot gene expression time course from data with corresponding fits
    data: df from data_rtm processed data
    fit : df from gamma fits generated from data
    """
    # get fit vals but only until 1 is reached
    exp = data.loc[gene]
    x_idx = exp.avg_norm_rtm2.values
    x_idx = np.where(x_idx == 1)[0][0]
    exp = exp.iloc[:(x_idx + 1), :]

    alphas = fit_res_gamma.loc[gene].alpha.values
    betas = fit_res_gamma.loc[gene].beta.values

    # get gamma cdf vals
    data_x = exp.time.drop_duplicates().values
    x_sim = np.linspace(0, max(data_x) + 20, 100)
    y_sim = [gamma_cdf(x_sim, alpha, beta) for alpha, beta in zip(alphas, betas)]

    # gaussian model fit data
    m1 = fit_res_gmm.loc[gene]["mean1"]
    m2 = fit_res_gmm.loc[gene]["mean2"]
    SD = fit_res_gmm.loc[gene]["SD"]
    #sd2 = fit_res_gmm.loc[gene]["SD2"]
    y_sim_gmm = gmm_cdf(x_sim, m1, m2, SD)

    df_sim = pd.DataFrame({"time": x_sim, "fit1": y_sim[0], "fit2": y_sim[1], "fit3": y_sim[2], "fit4": y_sim_gmm})
    return df_sim, exp


def plot_single_fit(gene, data, fit_res_gamma, fit_res_gmm, ax, show_rmse=True, show_title=True,
                    my_categories = ["gamma", "expo", "longtail", "bimodal"], lw = 1.5):
    """
    plot gamma expo bimodal etc together
    """
    # get simulation values for the gene
    df1, df2 = prep_fit(gene, data, fit_res_gamma, fit_res_gmm)

    sns.scatterplot(data=df2, x="time", y="avg_norm_rtm2", ax=ax, color="k", zorder = 1000, s = 10)
    # add some error bars if available
    if not np.isnan(df2.SD).any():
        ax.errorbar(df2.time, df2.avg_norm_rtm2, yerr=df2.SD, ecolor="k", fmt="none", zorder = 1000, capsize = 2)

    # for the best fit, use large font and solid lines, otherwise small and dashed
    colors = ["tab:green", "tab:blue", "tab:orange", "purple"]
    categories = ["expo", "gamma", "longtail", "bimodal"]
    myfits = ["fit1", "fit2", "fit3", "fit4"]
    zorders = [4,3,2,1]

    for myfit, color, fit_type, zorder in zip(myfits, colors, categories, zorders):
        if fit_type in my_categories:
            sns.lineplot(data = df1, x = "time", y = myfit, color = color, ax = ax, lw = lw, zorder = zorder)

    ax.set_ylabel("expr. norm.")
    ax.set_xlabel("time (h)")

    if show_rmse:
        rmse = fit_res_gamma.loc[gene]["rmse"].tolist()
        rmse_gmm = fit_res_gmm.loc[gene]["rmse"].tolist()
        rmse.append(rmse_gmm)

        names = ["exp", "gamma", "long", "bimodal"]

        rmse = [str(np.round(val, 2)) + n for val, n in zip(rmse, names)]
        title = " ".join(rmse)
        ax.text(5,0.05,title)

    if show_title:
        ax.set_title(gene, style = "italic")

    return ax