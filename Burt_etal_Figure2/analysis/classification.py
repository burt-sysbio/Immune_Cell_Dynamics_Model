import pandas as pd

"""
take summary of fit categories and do some postprocessing
first, clean up isoforms
second, filter genes that are expressed at least n times across studies
for those genes, assign "winner" based on how often gene was classified as gamma, expo, bimodal etc
store output
"""

month = "feb2021"
readdir = "../../output/fit_summary/"
df_fits = pd.read_csv(readdir + "category_assignment.csv")

# kick out crawford data set for this analysis because the kinetics are so late?
#df_fits = df_fits[df_fits['study'].str.contains('Crawford')==False]

df_fits[["gene2", "iso1", "iso2"]] = df_fits["gene"].str.split(r"_|\.", expand = True)
df_fits = df_fits.sort_values(by = "gene")
df_fits = df_fits[["gene", "best_fit", "study", "gene2"]]

# get the number of times a genes is found to be kinetic across samples
# note that lambda expr is used coz now multiple isoforms are reduced to gene name
# so study could show up more than once fur each gene
# "nunique" as string also works
df_fits["n_obs"] = df_fits.groupby(["gene2"])["study"].transform(lambda s: s.nunique())

# keep only genes with at least n hits across studies
# this filter seems to affect mainly number of kinetic genes but not the ratios (gamma/expo/etc)
df_fits = df_fits.loc[df_fits["n_obs"] >= 5, :]

# for each gene, find out the highest number of best fit hits
# i.e Eomes --> gamma 4 hits, expo 1 hit --> assign gamma
# problem arises for equal number of hits!
df_win = df_fits.groupby(["gene2", "best_fit"])["study"].nunique()
df_win = df_win.reset_index()

# src taken instead of idxmax from stack overflow
df_win = df_win.pivot(columns = "best_fit", values = "study", index = "gene2")
a = df_win.copy()
df_win['winner'] = a.idxmax(axis=1)
s = a.eq(a.max(axis=1), axis=0).sum(axis=1)
df_win['winner'] = df_win['winner'].mask(s > 1, 'Equality')

# kick out the category that is unclear
df_win = df_win.loc[df_win["winner"] != "Equality"]
df_win = df_win.reset_index()

# save the fit summary
df_win = df_win.rename(columns = {"gene2" : "gene"})
df_win.to_csv("../../output/fit_summary/category_assignment_winners.csv")
