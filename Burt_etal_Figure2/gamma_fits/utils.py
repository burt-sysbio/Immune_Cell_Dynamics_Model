from scipy.special import gammainc
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import gammainc
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
def gamma_cdf(t, alpha, beta):
    dummy = t * beta
    return gammainc(alpha, dummy)


def gamma_cdf1(t, beta):
    return gamma_cdf(t, 1, beta)


def gmm_cdf(x, mean1, mean2, SD, w1 = 0.5, w2 = 0.5):
    gmm1 = w1 * stats.norm.cdf(x, loc = mean1, scale = SD)
    gmm2 = w2 * stats.norm.cdf(x, loc = mean2, scale = SD)
    return gmm1 + gmm2


def conv_cols(df):
    #df.columns = df.columns.droplevel()
    df.columns = [str(col) for col in df.columns]
    df = df.reset_index()
    return (df)


def prep_data(df):
    # check column names because old version used gene name
    if "gene_name" in df.columns:
        df = df.rename(columns={"gene_name": "gene"})

    df_err = df[["cell_type", "gene", "time", "SD"]]
    df_err = df_err.drop_duplicates()

    df_val = df[["cell_type", "gene", "time", "avg_norm_rtm2"]]
    df_val = df_val.drop_duplicates()

    # need to make wide again for fit
    df_list = [df_val, df_err]

    cols = ["avg_norm_rtm2", "SD"]
    df_list = [pd.pivot(data=df,
                              index=["cell_type", "gene"],
                              columns="time", values = col) for col, df in zip(cols,df_list)]


    # extract data and errors
    timepoints = np.asarray(df_list[0].columns)
    n_timepoints = len(timepoints)

    # convert cols to str and reset the index
    df_list = [conv_cols(df) for df in df_list]

    genes = df_list[0].gene.values
    vals = df_list[0].iloc[:, -n_timepoints:].values
    errs = df_list[1].iloc[:, -n_timepoints:].values

    return genes, vals, errs, timepoints


def fit_kinetic(df, gamma_fun, bounds):
    genes, vals, errs, x = prep_data(df)

    fit_res = []

    # fit gamma dist for each gene
    for i, gene in zip(range(len(vals)), genes):
        y = vals[i, :]
        sigma = errs[i, :]

        # check that there are no nans in y
        # check that data is normalized and keep array only until max is reached
        assert (~np.isnan(y).any()), gene
        assert np.abs((np.max(y) - 1) < 1e-8), y
        assert np.abs((y[0] - 0) < 1e-8), y

        max_idx = np.where(y == np.max(y))[0][0]
        y = y[:(max_idx + 1)]

        # only focus on arrays where y has 4 time points
        if len(y) > 3:
            xdata = x[:len(y)]

            # check if there are nans or if all vals are 0 in SD array
            if (not np.isnan(sigma).any()) and (sigma != 0).any():

                # in readcount data SD can be 0 if two readcounts are identical
                # in this case, I set SD to the average sigma
                sigma[sigma == 0] = np.mean(sigma[sigma != 0])

                sigma = sigma[:(max_idx+1)]
                sigma_abs = True
            else:
                sigma = None
                sigma_abs = False

            # run fit and append output including gene name
            out = fit_gamma(xdata, y, sigma, sigma_abs, gamma_fun, bounds)
            out = [gene] + out
            fit_res.append(out)

    # convert fit res to dataframe
    colnames = ["gene", "alpha", "beta", "rss", "rmse", "alpha_err", "beta_err"]
    df_fit_res = pd.DataFrame(fit_res, columns=colnames)

    return df_fit_res


def fit_gamma(x, y, sigma, sigma_abs, gamma_fun, bounds):
    # dummy output in case fit does not work (e.g. gene kinetics are bimodal)
    out = [np.nan, np.nan, np.nan]
    try:
        # run fit catch runtime exception for bad fit
        fit_val, fit_err = curve_fit(f=gamma_fun, xdata=x, ydata=y, sigma=sigma,
                                     absolute_sigma=sigma_abs, bounds=bounds)

        # error of fitted params (not used atm)
        err = np.sqrt(np.diag(fit_err))

        # assign fit values depending on gamma_fun (exponential fit only fits beta not alpha)
        if gamma_fun == gamma_cdf:
            alpha_fit, beta_fit = fit_val
            alpha_err, beta_err = err
        else:
            beta_fit = fit_val[0]
            alpha_fit = 1
            alpha_err = np.nan
            beta_err = err[0]

        # compute chi sqr and get residuals
        # need to change pipeline function in R to write sigma as 1 instead of NA
        nexp = gamma_fun(x, *fit_val)
        rss = np.sum((y-nexp)**2)
        # root mean squared
        rmse = np.sqrt(rss/len(y))

        out = [alpha_fit, beta_fit, rss, rmse, alpha_err, beta_err]

    except RuntimeError:
        print("max calls reached no good fit")

    return out


# compare nested models by F test for two least squares fits (f1 and f2 are degrees of freedom from two nested models)
def f_test(rss1, rss2, n_obs, f1=1, f2=2):
    d1 = f2 - f1
    d2 = n_obs - f2
    F = ((rss1 - rss2) / d1) / (rss2 / d2)
    pval = 1 - stats.f.cdf(F, d1, d2)
    return pval


def get_pvals(fit1, fit2, n):
    """
    run f test for two fit results from function fit kinetic
    """
    # store comparison, this needs to be upfront
    comp = fit1["model"][0] + "_" + fit2["model"][0]
    # keep only those columns for merging and merge because some fits have nas generated
    fit1 = fit1[["gene", "rss"]]
    fit2 = fit2[["gene", "rss"]]

    fit = fit1.merge(fit2, how = "inner", on = "gene")

    # run f test on merged genes rss values after removing nans
    fit = fit.dropna()
    #fit.fillna(1e30)
    pvals = f_test(fit.rss_x.values, fit.rss_y.values, n_obs=n)

    # perform fdr correction and gerate output dataframe
    #padj = fdrcorrection(pvals)[1]
    df = pd.DataFrame({"gene": fit.gene.values, "p": pvals})

    # add some additional columns
    df["comp"] = comp
    df["f-test"] = "ns"
    df["f-test"][df["p"] <= 0.05] = "sig"

    return df


def run_f_test(df, gamma_1=gamma_cdf1, gamma_2=gamma_cdf):
    # edge casing for old col names
    if "gene_name" in df.columns:
        df = df.rename(columns={"gene_name": "gene"})

    print("expo fit...")
    fit1 = fit_kinetic(df, gamma_1, bounds=(0, np.inf))  # expo fit
    print("gamma fit...")
    fit2 = fit_kinetic(df, gamma_2, bounds=([1, 0], [np.inf, np.inf]))  # gamma fit a>1
    print("longtail fit...")
    fit3 = fit_kinetic(df, gamma_2, bounds=([0, 0], [1, np.inf]))  # gamma fit a<1

    if (fit1.empty or fit2.empty or fit3.empty):
        return pd.DataFrame(), pd.DataFrame()

    else:
        # add names to model fit
        fits = [fit1, fit2, fit3]
        names = ["expo", "gamma", "longtail"]
        assert (len(fits) == len(names))
        for f, n in zip(fits, names):
            f["model"] = n

        # run f test to compare gamma and longtail fits vs exponential model
        timepoints = df.time.drop_duplicates().values
        n_timepoints = len(timepoints)
        ftest_gamma = get_pvals(fit1, fit2, n_timepoints)
        ftest_longtail = get_pvals(fit1, fit3, n_timepoints)

        df1 = pd.concat(fits)
        df2 = pd.concat([ftest_gamma, ftest_longtail])
        return df1, df2