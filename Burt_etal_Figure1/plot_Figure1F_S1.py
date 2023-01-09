# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
mydir = "output/"

sns.set_palette("deep")
### load data
delay_endstate = pd.read_csv(mydir + "cyto_blockade_delay.csv")
mcarlo_endstates = pd.read_csv(mydir + "cyto_blockade_mcarlo.csv")
cells = pd.read_csv(mydir + "cyto_blockade_timecourse.csv")
cells_control = pd.read_csv(mydir + "cyto_blockade_timecourse_control.csv")
pscan_blockstart = pd.read_csv(mydir + "cyto_blockade_perturbation_start.csv")

# some type annotations, parameters and colors
# this needs to correspond to actual t0 entries
vals = delay_endstate["t0"].drop_duplicates().values
block_dur = 0.5

palette = ["0.6", "k", "olivedrab"]

delay_endstate["t0"] = delay_endstate["t0"].astype("str")
mcarlo_endstates["t0"] = mcarlo_endstates["t0"].astype("str")
cells["t0"] = cells["t0"].astype("str")
cells_control["t0"] = cells_control["t0"].astype("str")


# %%
# plot scan for perturbation start
ylabel = "Curve diff. norm."
xlim = [0,4]
g = sns.relplot(data = pscan_blockstart, x = "t0", y = "val_INT", style = "name", kind = "line", aspect = 1.,
                color = "k", legend = False, height = 1.6, lw = 1.5)
g.set(xlabel = "perturbation start (d)", ylabel = ylabel, ylim = [0,0.4], xlim = [0,3])
sns.despine(top = False, right = False)

ax = g.axes[0][0]
for val, col in zip(vals, palette):
    ax.add_patch(Rectangle((val, 0.38), block_dur, 0.02, facecolor = col, edgecolor= "white"))

plt.show()

########################################################################################
########################################################################################
########################################################################################
# plot timecourse control + one perturbation

perturb_start = 0.7
cells_perturb = cells.loc[cells["t0"] == str(perturb_start)]

# %%
fig, ax = plt.subplots(figsize = (1.8,1.5))
sns.lineplot(data = cells_control, x = "time", y = "value", style = "name", hue = "cell",
             palette= ["grey", "grey"], legend= False)
sns.lineplot(data = cells_perturb, x = "time", y = "value", style = "name", hue = "cell", legend= False)

ax.add_patch(Rectangle((perturb_start, 90), block_dur, 5, facecolor = "k", edgecolor= "white"))
ax.set_ylim([0,100])
ax.set_xlim([0,4])
ax.set_xlabel("time (d)")
ax.set_ylabel("cells (% of total)")
plt.show()

# make cartoon figure, only RTM and SSM for one celltype
# %%
xticks = [0,1,2,3,4]
fig, ax = plt.subplots(figsize = (1.8,1.6))
cells_control_cartoon = cells_control.loc[cells_control.cell == "eff2"]
sns.lineplot(data = cells_control_cartoon, x = "time", y = "value", hue = "name",legend= False, lw = 2,
             palette = ["k", "purple"])

df_filled = cells_control_cartoon.loc[cells_control_cartoon["name"] == "delay"]
ax.fill_between(df_filled["time"],df_filled["value"], color = "0.8")
ax.set_ylim([0,40])
ax.set_xlim([0,4])
ax.set_xticks(xticks)
ax.set_xlabel("time (d)")
ax.set_ylabel("ThX cells (a.u.)")
plt.tight_layout()
plt.show()

# plot delay scan
# %%
g = sns.relplot(data = delay_endstate, x = "SD", y = "val_INT", hue = "t0", palette=palette,
                legend = False, hue_order= ["0.1", "1.5", "0.7"], aspect = 1.1, height = 1.6)
ax = g.axes.flat
sns.lineplot(data = delay_endstate, x = "SD", y = "val_INT", hue = "t0", palette = palette, lw = 1,
             legend=False, hue_order= ["0.1", "1.5", "0.7"])
g.set(xlabel = "step parameter k", ylabel = ylabel, yticks = [0,0.2, 0.4], xticks = np.arange(1,11))
sns.despine(top = False, right = False)

plt.show()


# plot monte carlo simulation
# %%

g = sns.catplot(data = mcarlo_endstates, x = "name", y = "val_INT", hue = "t0",
                kind = "bar",legend = False, palette = palette, ci = 95,
                aspect= 1, alpha = 1, height = 1.7, capsize = 0.1, errwidth = 0.3, errcolor = "k")
sns.despine(top =False, right = False)
g.set(xlabel = "", ylabel = ylabel)
labels = ["early", "late", "int"]*2
tickpos = [-0.25, 0, 0.25, 0.75, 1, 1.25]
plt.xticks(ticks = tickpos, labels = labels, rotation = 90)
plt.show()



# plot difference vs control
# %%
cells_control = cells_control.rename(columns = {"value" : "val_cntrl"})
cells_control = cells_control[["time", "name", "val_cntrl", "cell"]]

cells_singleplot = pd.merge(cells_singleplot, cells_control, how = "left")
cells_singleplot["val_norm"] = cells_singleplot["value"] - cells_singleplot["val_cntrl"]

g = sns.relplot(data = cells_singleplot, x = "time", y = "val_norm", hue = "t0",
                row = "name", kind = "line", legend = False, palette = palette,
                hue_order= ["0.1", "1.5", "0.7"], height = 0.85, aspect = 1.4, lw = 2)
g.set(xlim = xlim, yticks= [0,10,20], ylim = [0, 27], ylabel = "Tfh (vs cntrl)", xlabel = "time (d)", xticks = xticks)
for val, col in zip(vals, palette):
    #g.axes[0,0].add_patch(Rectangle((val, 27), block_dur, 2, facecolor = col, edgecolor= "white"))
    g.axes[1,0].add_patch(Rectangle((val, 20), block_dur, 2, facecolor=col, edgecolor="white"))
sns.despine(top = False, right = False)
g.set_titles("")
plt.tight_layout()
plt.show()

