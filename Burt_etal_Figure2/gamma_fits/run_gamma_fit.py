#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:24:52 2020

@author: burt
fit gamma dist for output generated from R should work on all kinetic data together
also performs f test
"""

import pandas as pd
from utils import run_f_test
import os

month = "feb2021"
readdir = "../../data/data_rtm/"
savedir = "../../output/gamma_fits/" + month + "/"

# load all files
filenames = os.listdir(readdir)

# grab rtm files
pattern = "_rtm_"
filenames = [f for f in filenames if pattern in f]

# use only 1 file for testing?
#filenames = [filenames[0]]


# run gamma and expo fit, then compute f test statistic
for filename in filenames:
    df = pd.read_csv(readdir + filename)

    # if rna seq data, kick out nans generated through rlog trafo
    if "Proserpio" in filename:
        genes_na = df.gene[df.SD.isna()]
        df = df[~df.gene.isin(genes_na)]

    # kick out NAs, at least in properio there are some nas for some reason
    # I need to double check this
    output = run_f_test(df)

    # save output
    out_names = ["fit_res_", "ftest_"]
    for df, n in zip(output, out_names):
        df.to_csv(savedir+n+filename)