import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../paper_theme_python.mplstyle')

# read in all files
month = "feb2021"
readdir = "../../output/gamma_fits/" + month + "/"
genesetdir = "../../output/genesets/" + month + "/"
filenames = os.listdir(readdir)

pat1 = "fit_res"
pat2 = "ftest"

files1 = [f for f in filenames if pat1 in f]
files2 = [f for f in filenames if pat2 in f]

fits_all = [pd.read_csv(readdir+ f1) for f1 in files1]
ftest_all = [pd.read_csv(readdir+ f2) for f2 in files2]
fnames = [f2[6:-4] for f2 in files2]

# store as excel sheet
store_table = True
if store_table:
    print("storing supplement table, set to False if not needed")
    with pd.ExcelWriter('Burt_etal_Supplementary_Table_Transcriptome_Analysis_Gamma_Fits.xlsx') as writer:
        for df_xlsx, fname in zip(fits_all, fnames):
            fname = str.split(fname, "_")
            if fname[0] == "Powrie":
                fname[0] = "Ilott"
            fname = fname[0] + "_" + fname[2]
            df_xlsx = df_xlsx.iloc[:,1:].copy()
            df_xlsx.to_excel(writer, sheet_name=fname, index = False, na_rep="NA")

for f,n in zip(fits_all,fnames):
    f["study"] = n

df_dist = pd.concat(fits_all)

xlabels = ["Craw. 1", "Craw. 2", "Craw. 3", "Craw. 4",
           "Nir 1", "Nir 2",
           "Peine 1", "Peine 2", "Peine 3", "Peine 4",
           "Ilott 1",
           "Pros. 1"]

colors = sns.color_palette("deep", 10)
palette = [colors[0], colors[2]]

df_dist_red = df_dist.loc[df_dist["model"] != "longtail"]
g = sns.catplot(data = df_dist_red, x = "study", y = "rmse", hue = "model", kind = "box",
                fliersize = 0, hue_order= ["gamma", "expo"], whis=[5, 95], legend_out=True,
                palette = palette, aspect= 1.5, height = 1.9)

g.set(ylim = (-0.1,0.8), ylabel = "fit error (RMSE)", xlabel = "")
g.set_xticklabels(rotation=90, labels= xlabels)

sns.despine(top = False, right = False)
#plt.tight_layout()
plt.show()

savedir = "../../figures/fit_error_distributions/"
g.savefig(savedir + "rmse_dist.pdf")
g.savefig(savedir + "rmse_dist.svg")

df_dist_red = df_dist_red.reset_index()

df_dist_red = df_dist_red.loc[df_dist_red["rmse"] > 1e-3]
g = sns.displot(data = df_dist_red, x = "rmse", hue = "model", palette = palette, bins = 30,
                hue_order= ["gamma", "expo"], legend = True, alpha = 0.5, aspect = 1.2, height = 1.8)
g.set(xlim = (0,0.8), xticks = [0,0.2,0.4,0.6,0.8], ylim = (0,12000), xlabel = "fit error (RMSE)", ylabel = "n genes")
sns.despine(top = False, right = False)
g.savefig(savedir + "rmse_dist_comb.pdf")
g.savefig(savedir + "rmse_dist_comb.svg")
plt.show()