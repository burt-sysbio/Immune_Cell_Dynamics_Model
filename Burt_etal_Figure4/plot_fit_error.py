"""
visualize different fit statistics for individual parameters and globally
"""
import pickle
import numpy
import pandas as pd
import seaborn as sns
import pandas
import sys
sys.path.append("../../")
from analysis.late_branching.model_params import fit_names_local as fit_names
import matplotlib.pyplot as plt

plt.style.use("../../../paper_theme_python.mplstyle")

fit_names = [fit_names[0], fit_names[2]]
fit_dir = "../../output/fit_results/"

fit_reports = []

for fit_name in fit_names:
    with open(fit_dir + fit_name + '_fit_report.p', 'rb') as fp:
        fit_result = pickle.load(fp)
    fit_reports.append(fit_result)


df_list = []
chisqr = []
akaike = []

for fit_report, name in zip(fit_reports, fit_names):

    chisqr.append(fit_report.chisqr)
    akaike.append(fit_report.aic)
    mydict = fit_report.params
    params = mydict.keys()
    fit_report_errors = [mydict[p].stderr for p in params]
    fit_report_values = [mydict[p].value for p in params]
    mydf = pd.DataFrame({"param_value" : fit_report_values, "param_std" : fit_report_errors, "param_name" : params})
    mydf["name"] = name

    # modify different parameters of early and late topology to plot together
    if "early" in name:
        mydf = mydf.loc[mydf.param_name != "r_chr"]
        mydf.loc[mydf.param_name == "rate_chr1", "param_name"] = "r_chr"
    df_list.append(mydf)


# df_chisqr = pd.DataFrame({"name" : fit_names, "chisqr": chisqr})
# df_aic = pd.DataFrame({"name" : fit_names, "aic": akaike})
# df_aic["aic_norm"] = df_aic["aic"] + 50
#
# g = sns.catplot(data = df_chisqr, x = "name", y = "chisqr", kind = "bar", color = "grey", height = 1.6)
# g.set_xticklabels(rotation = 90, labels = ["constr", "free"])
# g.set(xlabel ="")
# #plt.show()
#
# g.savefig("../../figures/confidence_intervals/barplot_chisqr_constrained_unconstrained.svg", dpi = 300)
# g.savefig("../../figures/confidence_intervals/barplot_chisqr_constrained_unconstrained.pdf", dpi = 300)
#
#
# g = sns.catplot(data = df_aic, x = "name", y = "aic_norm", kind = "bar", color = "grey", height = 1.6)
# g.set_xticklabels(rotation = 90, labels = ["constr", "free"])
# g.set(xlabel ="")
# #plt.show()
#
# g.savefig("../../figures/confidence_intervals/barplot_AIC_constrained_unconstrained.svg", dpi = 300)
# g.savefig("../../figures/confidence_intervals/barplot_AIC_constrained_unconstrained.pdf", dpi = 300)


fit_report_summary = pd.concat(df_list).reset_index()
#fit_report_summary.loc[fit_report_summary["param_name"] == "rate_chr1", "param_name"] = "r_chr"
#fit_report_summary = fit_report_summary.loc[fit_report_summary["param_name"] != "r_chr"]
fit_report_summary["CV"] = fit_report_summary["param_std"] / fit_report_summary["param_value"]

fit_report_allparams = fit_report_summary.loc[fit_report_summary.name.str.contains("allparams")]
fit_report_fitparams = fit_report_summary.loc[fit_report_summary.name.str.contains("fitparams")]

mypal = sns.color_palette("dark")
mypal2 = ["goldenrod", "k"]


# g = sns.catplot(data = fit_report_summary, x = "param_name", y = "CV", hue = "name", kind = "bar",
#                 height = 1.6, palette = mypal2)
# g.set(yscale = "log", xlabel = "", ylabel = "Standard error")
# g.set_xticklabels(rotation = 90)
# plt.tight_layout()
# plt.show()
#
# g.savefig("../../figures/confidence_intervals/barplot_confidence_intervals_constrained_unconstrained_allparams.svg", dpi = 300)
# g.savefig("../../figures/confidence_intervals/barplot_confidence_intervals_constrained_unconstrained_allparams.pdf", dpi = 300)



fit_report_summary_red = fit_report_summary.loc[fit_report_summary["param_name"].isin(fit_report_fitparams["param_name"])]
fit_report_summary_red = fit_report_summary_red.loc[fit_report_summary["param_name"]!="p2"]

myparams = ["p1", "fb_IL10", "prop_ag", "r_mem", "r_chr", "initial_cells"]

labels = [r"$\lambda_{\mathrm{Th1},0}$", r"$\xi_\mathrm{IL10}$", r"$\lambda_{e,\mathrm{ag}}$",
          r"$\beta_m$", r"$\beta_c$", "$n_0$"]


g = sns.catplot(data = fit_report_summary_red, x = "param_name", y = "CV", hue = "name", kind = "bar",
                height = 1.6, palette = mypal2, aspect = 1.5, order = myparams)
g.set(yscale = "log", xlabel = "", ylabel = "relative error")
g.set_xticklabels(rotation = 90, labels = labels)
plt.show()

g.savefig("../../figures/confidence_intervals/barplot_confidence_intervals_constrained_unconstrained.svg", dpi = 300)
g.savefig("../../figures/confidence_intervals/barplot_confidence_intervals_constrained_unconstrained.pdf", dpi = 300)

fit_report_summary_red["id"] = fit_report_summary_red.groupby(["param_name"]).ngroup()

fit_report_summary_red = fit_report_summary_red.loc[fit_report_summary_red["param_name"] != "initial_cells"]

# import numpy as np
# fit_report_summary_red["id_dodge"] = fit_report_summary_red["id"] + np.random.normal(0,0.2,12)
#
# g = sns.FacetGrid(fit_report_summary_red, hue = "name")
# g.map_dataframe(plt.errorbar, x = "id_dodge", y = "param_value", fmt = "o", yerr = "param_std", alpha = 0.6)
# g.set_xticklabels(rotation = 90)
# plt.show()
#

