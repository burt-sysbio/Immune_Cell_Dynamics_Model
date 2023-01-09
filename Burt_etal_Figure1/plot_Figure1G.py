#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:38:50 2020

@author: burt
one simple diff x--> y with differentiation
keep death rate constant and vary feedback for prolif within a
range so that it does not top the death rate
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_heatmap

sns.set_palette("deep")

cmap = "rocket_r"
xlabel = "perturbation start (d)"
ylabel = "perturbation length (d)"

titles = ["Delay", "NoDelay"]

for title in titles:

    df = pd.read_csv("output/fig1/cyto_blockade_heatmap_" + title + ".csv")
    readouts = df["readout"].drop_duplicates().values
    print("available readouts:" + readouts)
    readouts = ["peak_eff2"]
    for r in readouts:
        fig, z = plot_heatmap(df,
                              value_col= "value",
                              readout= r,
                              log_color = False,
                              cmap = cmap,
                              xlabel = xlabel,
                              ylabel = ylabel,
                              cbar_label= "%ThX day 3",
                              title = title,
                              log_axes= False,
                              vmin = 0.5,
                              vmax = 1.0,
                              figsize= (2.7,2.3),
                              cbar_ticks = [0.5,0.75,1.0],
                              xticks = [0,1,2],
                              yticks = [0,0.5,1])

        plt.show()
