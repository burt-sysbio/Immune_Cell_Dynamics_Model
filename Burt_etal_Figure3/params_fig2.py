from scipy.constants import N_A
d = {
    "b" : 0,
    "initial_cells" : 1.0,
    "alpha" : 22,
    "beta_naive" : 15.0,
    "alpha_prec" : 7,
    "beta_prec" : 15.2,
    "alpha_p" : 7,
    "beta_p" : 15.2,
    "d_eff" : 0.24,
    "d_prec" : 0,
    "d_naive" : 0,
    "rate_il2_naive" : 1 * 3600 * 24,  # il2 model
    "rate_il2_prec": 150 * 3600 * 24,  # il2 model
    "up_il2": 4 * 3600 * 24,       # il2 model
    "deg_il2": 0,      # il2 model
    "up_carry" : 0.001,
    "K_carry": 0.1,        # saturated uptake
    "K_il2_cons" : 7.5*N_A*20e-6*10e-12, # saturated uptake
    "deg_myc" : 0.37,   # timer model
    "hill" : 12,
    "EC50_il2" : 5,
    "EC50_myc" : 0.1,
    "EC50_carry" : 0.5,
    "n_crit" : 1e3,
    "ag_dose" : 0,
    "deg_restim" : 1,
    "il2_stimulation" : 1,
    }

