from lmfit import Parameters
import copy

# params1 = Parameters()
# these are my default parameters (used for global fit)
# params1.add("prop_ag", value=0.15, min=0.05, max=0.25)
# params1.add("r_chr", value=0.01, vary = True)
# params1.add("r_mem", value=0.01, vary = True)
# params1.add("r_IL10_chr", value=100, min = 0, max = 10000)
# params1.add("deg_IL10", value=5000000, min = 0, max = 10000000)
# params1.add("initial_cells", value = 2000, min = 100, max = 5000)
# params1.add("p1", value=0.7, min=0.3, max=1.0)
# params1.add("p2", expr = "1-p1")
# params1.add("p1_ag", value= 0.5, min=0, max=100.0)
# params1.add("p2_IL10", value = 10, min = 0, max = 100.0)
# params1.add("fb_IL10", value = 0.3, min = 0.01, max = 0.7)
# params1.add("fbfc_ag_chr", value = 500, min = 1, max = 10000)

# these are the parameters for which I got a good local fit
params1 = Parameters()
params1.add("prop_ag", value=0.15, min=0.1, max=0.45)
params1.add("r_chr", value=0.005, vary = True, min = 0, max = 0.01)
params1.add("r_mem", value=0.005, vary = True, min = 0.001, max = 0.01)
params1.add("initial_cells", value = 3000, min = 2000, max = 4000)
params1.add("p1", value=0.6, min=0.5, max=0.8)
params1.add("p2", expr = "1-p1")
params1.add("fb_IL10", value = 0.2, min = 0.1, max = 0.4)
#params1.add("p1_ag", value= 0.5, min=0, max=100.0)
#params1.add("p2_IL10", value = 10, min = 0, max = 100.0)
#params1.add("r_IL10_chr", value=100, min = 0, max = 10000)
#params1.add("fbfc_ag_chr", value = 500, min = 1, max = 10000)

# parameters where r_chronic comes from intermediate cells
params2 = copy.deepcopy(params1)
params2["r_chr"].set(vary = False, value = 0)
params2.add("rate_chr1", value = 0.005, vary = True, min = 0.001, max = 10)

# include additional parameters that I had kept fixed previously (data parameters as fit parameters
params3 = copy.deepcopy(params1)
params3.add("beta_naive", value = 9.7, vary = True, min = 8, max = 12)
params3.add("beta_eff", value = 15.2, vary = True, min = 10, max = 21)
params3.add("death_eff", value = 0.24, vary = True, min = 0.05, max = 0.3)
params3.add("death_prec", expr = "death_eff")
params3.add("deg_chr_th1", expr = "death_eff")
params3.add("deg_chr_tfh", expr = "death_eff")
params3.add("death_mem", value = 0.009, vary = True, min = 0.001, max = 0.1)

# additional parameters with intermediate r_chronic
params4 = copy.deepcopy(params3)
params4["r_chr"].set(vary = False, value = 0)
params4.add("rate_chr1", value = 0.005, vary = True, min = 0.001, max = 10)

# summary:
# params1: constrained model latebranch
# params2: constrained model earlybranch
# params3: unconstrained model latebranch
# params4: unconstrained model earlybranch



