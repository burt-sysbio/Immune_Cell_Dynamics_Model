
import seaborn as sns
import pandas as pd
from src.analysis.utils_plot import plot_single_fit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gammainc
import random
plt.style.use("../paper_theme_python.mplstyle")


def gamma_cdf(t, alpha, beta):
    dummy = t * beta
    return gammainc(alpha, dummy)

def gamma_cdf1(t, beta):
    return gamma_cdf(t, 1, beta)

def fit_gamma(x, y, sigma):
    # dummy output in case fit does not work (e.g. gene kinetics are bimodal)
    # run fit catch runtime exception for bad fit
    max_idx = np.where(y == np.max(y))[0][0]
    y = y[:(max_idx + 1)]
    x = x[:len(y)]
    sigma = sigma[:len(y)]
    print(x,y, sigma)
    fit_val, fit_err = curve_fit(f=gamma_cdf, xdata=x, ydata=y, sigma=sigma,
                                 absolute_sigma=True)
    # error of fitted params (not used atm)
    err = np.sqrt(np.diag(fit_err))

    # assign fit values depending on gamma_fun (exponential fit only fits beta not alpha)
    alpha_fit, beta_fit = fit_val
    alpha_err, beta_err = err

    # compute chi sqr and get residuals
    # need to change pipeline function in R to write sigma as 1 instead of NA
    nexp = gamma_cdf(x, *fit_val)
    rss = np.sum((y-nexp)**2)
    # root mean squared
    rmse = np.sqrt(rss/len(y))

    out = [alpha_fit, beta_fit, rmse]
    return out

# read in all files
month = "feb2021"
fitdir = "../../output/gamma_fits/" + month + "/"
datadir = "../../data/data_rtm/"

df_modules = pd.read_csv("../../genesets_literature/gene_module_summary.csv")
#df_modules = df_modules.loc[df_modules["module"].isin(["Cytokines", "Cytokine Receptor"])]

study = "Proserpio_rtm_Th2_parasite.csv"
study2 = study[:-4]

fit_results = pd.read_csv(fitdir + "fit_res_" + study)
fit_res_gamma = fit_results.loc[fit_results["model"] == "gamma"]
fit_res_gamma = fit_res_gamma.loc[fit_res_gamma["rmse"]<0.05]
fit_res_gamma = fit_res_gamma.loc[fit_res_gamma["alpha"]<30]
fit_res_gamma = fit_res_gamma.loc[fit_res_gamma["alpha"]>=5]
fit_res_gamma = fit_res_gamma.loc[fit_res_gamma["alpha_err"]<10]
fit_res_gamma = fit_res_gamma.loc[fit_res_gamma["beta_err"]<10]

fit_res_expo = fit_results.loc[fit_results["model"] == "expo"]
fit_res_expo = fit_res_expo.loc[fit_res_expo["rmse"]<0.05]
fit_res_expo = fit_res_expo.loc[fit_res_expo["beta_err"]<10]

# read data and filter for genes in expo and gamma categories with "smooth" fit statistics
data = pd.read_csv(datadir + study)

# drop duplicates? I dont know under which circumstances exactly those duplicates arose in the first place
data = data.drop_duplicates()
data_reduced_gamma = data.loc[data["gene"].isin(fit_res_gamma["gene"].drop_duplicates().values)]
data_reduced_expo = data.loc[data["gene"].isin(fit_res_expo["gene"].drop_duplicates().values)]


def myfit(df):
    df = df[["time", "avg_norm_rtm2", "SD"]].drop_duplicates().copy()
    df = df.sort_values(["time"]).reset_index(drop = True).copy()
    x = df["time"].values
    y = df["avg_norm_rtm2"].values
    sigma = df["SD"].values

    out = fit_gamma(x,y, sigma)
    return out



def plot_fit(df_data, df_fit, gene, ax, mycolor = "blue"):
    df = df_data.loc[df_data.gene == gene].copy()
    df_fit = df_fit.loc[df_fit.gene == gene].copy()

    df = df.sort_values(["time"]).copy()
    # run fit
    #alpha_fit, beta_fit, rmse = myfit(df)
    alpha_fit, beta_fit, rmse = df_fit[["alpha", "beta", "rmse"]].values.flat
    # based on fit, compute corresponding Erlang alpha and beta
    mu_fit = alpha_fit/beta_fit
    alpha_round = round(alpha_fit)
    beta_round = alpha_round / mu_fit

    x = df["time"].values
    y = df["avg_norm_rtm2"].values
    SD = df["SD"].values

    max_idx = np.where(y == np.max(y))[0][0]
    y_red = y[:(max_idx + 1)]
    x_red = x[:len(y_red)]

    nexp = gamma_cdf(x_red, alpha_round, beta_round)
    rss_round = np.sum((y_red-nexp)**2)
    # root mean squared
    rmse_round = np.sqrt(rss_round/len(y_red))

    x_arr = np.linspace(x[0],x[-1],100)

    y_arr_gamma = gamma_cdf(x_arr, alpha_fit, beta_fit)
    y_arr_erlang = gamma_cdf(x_arr, alpha_round, beta_round)

    ax.plot(x_arr, y_arr_gamma, c = "k", lw = 3.5)
    ax.plot(x_arr, y_arr_erlang, c = "lightgrey", lw = 1.2, ls = "--")

    #ax.text(20,0.1,f"E({alpha_round},{round(beta_round,3)}), rmse={round(rmse_round,3)}",size = 6)
    #ax.text(20,0.,fr"$\gamma$({round(alpha_fit,3)},{round(beta_fit,3)}), rmse={round(rmse,3)}", size = 6)
    ax.text(20,0.0,f"E({alpha_round},{round(beta_round,3)})",size = 7)
    ax.text(20,0.18,fr"$\gamma$({round(alpha_fit,3)},{round(beta_fit,3)})", size = 7)

    ax.text(0,0.9, gene, size = 8, style = "italic")
    ax.errorbar(x,y, yerr =SD, fmt = ".", color = mycolor)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("")
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(["","",""])
    ax.set_xticks([0,24,48,72,96])
    ax.set_ylim([-0.1,1.2])


gene_list_gamma = list(data_reduced_gamma.gene.sample(3))
gene_list_expo = list(data_reduced_expo.gene.sample(3))
gene_selection_expo = ["Ranbp10", "Prrg4", "Tasl"]
gene_selection_gamma = ["Sulf2", "Timm22", "Rsrc1"]

mypal = sns.color_palette("deep")
color1 = mypal[0]
color2 = mypal[2]
figsize = (4.,1.1)
fig, axes = plt.subplots(1,3, figsize = figsize)
for gene, ax in zip(gene_selection_gamma, axes):
    plot_fit(data_reduced_gamma, fit_res_gamma, gene, ax, color1)

axes[0].set_ylabel("expr. norm.")
axes[0].set_yticklabels(["0", "0.5", "1.0"])
#plt.tight_layout()
plt.show()


fig.savefig("../../figures/gamma_vs_erlang/gamma_vs_erlang_bestfit_gamma.pdf", transparent = True)
fig.savefig("../../figures/gamma_vs_erlang/gamma_vs_erlang_bestfit_gamma.svg", transparent = True)

fig, axes = plt.subplots(1,3, figsize = figsize)
for gene, ax in zip(gene_selection_expo, axes):
    plot_fit(data_reduced_expo, fit_res_expo, gene, ax, color2)

axes[0].set_ylabel("expr. norm.")
axes[0].set_yticklabels(["0", "0.5", "1.0"])
#plt.tight_layout()
plt.show()

fig.savefig("../../figures/gamma_vs_erlang/gamma_vs_erlang_bestfit_expo.pdf", transparent = True)
fig.savefig("../../figures/gamma_vs_erlang/gamma_vs_erlang_bestfit_expo.svg", transparent = True)

