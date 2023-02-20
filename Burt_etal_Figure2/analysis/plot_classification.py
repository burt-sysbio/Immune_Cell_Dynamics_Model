import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# load classification data
df_win = pd.read_csv("../../output/fit_summary/category_assignment_winners.csv")
df_modules = pd.read_csv("../../genesets_literature/gene_module_summary.csv")
# load tfs, cytos, cytoR, th2 th1 genes
plt.style.use("../paper_theme_python.mplstyle")
sns.set_palette("deep")

def barplot(df, geneset, df_modules):
    """
    make barplot that counts number of gamma/expo, longtail etc
    genes for a specific gene set (TFs, cytos etc)
    geneset needs to be a dataframe with a "gene" column w gene symbols
    """
    if geneset != "all kinetic":
        mylist = df_modules.loc[df_modules["module"] == geneset, "gene"]
        df = df.loc[df.gene.isin(mylist)]

    # count hits per gene across studies
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize = (6,6))
    sns.countplot(data=df, x="winner",
                  order=["gamma", "expo", "bimodal", "longtail", "other"],
                  palette=["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:grey"],
                  ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("n genes")
    ax.set_title(geneset)
    plt.xticks(rotation=90)
    plt.close()
    #plt.show()
    #fig.savefig("../../figures/barplot_genelists_delays/delays_histplot_" + geneset +".pdf")

    out = df.groupby(["winner"])["gene"].count()
    out = out.to_frame()
    out["group"] = geneset
    return out

# plot barplot for each gene set in df_modules
genesets = df_modules["module"].drop_duplicates().values
genesets = np.concatenate((genesets, ["all kinetic"]))
out = [barplot(df_win, geneset, df_modules) for geneset in genesets]

# summarize output and plot together
out = pd.concat(out)
out = out.reset_index()
out["sum"] = out.groupby(["group"])["gene"].transform("sum")
out["relval"] = (out["gene"] / out["sum"])*100


colors = sns.color_palette("deep", 10)
palette = [colors[0], colors[2], colors[7], "purple"]

# assign longtail as expo
print("assigning longtail as expo")

out.loc[out.winner == "longtail", "winner"] = "expo"

fig, ax = plt.subplots(figsize = (1.7,1.3))
g = sns.histplot(x = 'group', hue = 'winner',weights= 'relval',
                multiple = 'stack',data=out,shrink = 0.75,
                 palette = palette, alpha = 1,
                 hue_order = ["gamma", "expo", "other", "bimodal"]
                 )
ax.set_xlabel("")
ax.set_ylabel("category assignment (%)")
ax.set_ylim([0,100])
plt.xticks(rotation = 90)

plt.show()
fig.savefig("../../figures/barplot_genelists_delays/barplot_stacked_kinetic_groups.pdf")
fig.savefig("../../figures/barplot_genelists_delays/barplot_stacked_kinetic_groups.svg")

fig, ax = plt.subplots(figsize = (1.4,1.5))
g = sns.histplot(y = 'group', hue = 'winner',weights= 'relval',
                multiple = 'stack',data=out,shrink = 0.9,
                 palette = palette, alpha = 1,
                 hue_order = ["gamma", "expo", "other", "bimodal"])
ax.set_ylabel("")
ax.set_xlabel("fit assignment (%)")
ax.set_xlim([0,100])

plt.show()
fig.savefig("../../figures/barplot_genelists_delays/barplot_stacked_kinetic_groups_rotated.pdf")
fig.savefig("../../figures/barplot_genelists_delays/barplot_stacked_kinetic_groups_rotated.svg")

# out2 = out.loc[out["winner"].isin(["expo", "gamma"]),:]
#
# fig, ax = plt.subplots(figsize = (6,6))
# g = sns.barplot(data = out2, hue= "winner", y = "relval", x = "group",
#                 palette= ["tab:green", "tab:blue"])
# ax.set_ylim([0,70])
# ax.set_ylabel("fit assignment (%)")
# ax.set_xlabel("")
#
# plt.xticks(rotation = 90)
# ax.get_legend().remove()
# plt.show()
# fig.savefig("../../figures/barplot_genelists_delays/barplot_expo_vs_gamma.pdf")
# fig.savefig("../../figures/barplot_genelists_delays/barplot_expo_vs_gamma.svg")

keep_modules = ["Cytokines", "Cytokine Receptor"]
df_modules_red = df_modules.loc[df_modules["module"].isin(keep_modules)]
df_sign_genes = df_win.loc[df_win["gene"].isin(df_modules_red["gene"]), ["gene", "gamma"]]

n_keep = 30
df_sign_genes_top = df_sign_genes.sort_values(["gamma"], ascending=False).iloc[:n_keep,:]
df_sign_genes_top["gamma"] = df_sign_genes_top["gamma"] * 100 / 11

g = sns.catplot(data = df_sign_genes_top, x = "gamma", y = "gene", kind = "bar", color = "0.5",
                aspect = 0.6, height = 3.5)
g.set(xlabel = "gamma distributed (% of studies)", ylabel = "", xticks = [0,50,100])
g.set_yticklabels(style = "italic")
plt.show()

g.savefig("../../figures/barplot_genelists_delays/barplot_stacked_signature_genes.pdf")
g.savefig("../../figures/barplot_genelists_delays/barplot_stacked_signature_genes.svg")
