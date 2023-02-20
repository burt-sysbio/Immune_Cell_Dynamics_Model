#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

pscan = pd.read_csv("output_fig2C.csv")

pscan = pscan[pscan.readout != "Decay"]

# divide by alpha to get mean instead of rate
pscan.p_val = pscan.p_val.apply(lambda x : x/7.)

pscan["norm2"] = pscan.groupby(["readout", "name"])["read_val"].transform(lambda x: np.log2(x/x.min()))
# rename some readouts
pscan.loc[pscan["readout"] == "Area", "readout"] = "Response Size"
pscan.loc[pscan["readout"] == "Peak", "readout"] = "Peak Height"
pscan.loc[pscan["readout"] == "Peaktime", "readout"] = "Peak Time"

g = sns.relplot(data = pscan, x = "p_val", y = "norm2", col = "name", hue = "readout",
                kind = "line", facet_kws = {"despine" : False})
g.set(ylim = (-5,30), xlim = (1e-1, 1e1), xlabel = "divisions per day",
      ylabel = "effect size", xscale = "log")

g.set_titles("{col_name}")
for ax in g.axes:
    for a in ax:
        a.axvline(x = 15.2/7, linewidth = 2., ls = "--", color = "k", zorder = 1000)

plt.show()

# %%
