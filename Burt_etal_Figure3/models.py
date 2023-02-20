# -*- coding: utf-8 -*-
"""
keep all ode models here
"""
import numpy as np
from scipy.constants import N_A
# =============================================================================
# linear models
# =============================================================================
def repeated_stimulation(time, state, model, d, vir_model):

    il2_global = state[-1]
    myc = state[-2]
    carry = state[-3]
    restim = state[-4]

    eff_idx = d["alpha"]+d["alpha_prec"]
    naive = state[:d["alpha"]]
    prec = state[d["alpha"]:eff_idx]
    eff = state[eff_idx:-4]

    n_naive = np.sum(naive)
    n_prec = np.sum(prec)
    n_eff = np.sum(eff)

    il2_production = d["rate_il2_naive"] * n_naive + d["rate_il2_prec"] * n_prec 
    dt_il2 = il2_production - d["up_il2"] * (n_eff) * (il2_global/(il2_global+d["K_il2_cons"]))

    il2_effective = il2_global * (1e12/(20e-6*N_A))

    ncells = n_naive+n_prec+n_eff
    beta_p = model(ncells, myc, il2_effective, carry, d)


    d_eff = d["d_eff"]
    div_naive = 1 + d["beta_p"] / d["beta_naive"]
    div_prec = 1 + d["beta_p"] / d["beta_prec"]

    influx_naive = 0
    influx_prec = naive[-1] * d["beta_naive"] * div_naive * 2
    influx_eff = prec[-1] * d["beta_prec"] * div_prec + eff[-1] * beta_p * 2

    dt_naive = diff_chain(naive, influx_naive, d["beta_naive"], 0)
    dt_prec = diff_chain(prec, influx_prec, d["beta_prec"], 0)
    dt_eff = diff_chain(eff, influx_eff, beta_p, d_eff)

    dt_myc = -d["deg_myc"]*myc
    dt_restim = -d["deg_restim"]*restim
    dt_carry = -d["up_carry"] * (n_eff + n_prec) * (carry/(carry+d["K_carry"]))
    dt_state = np.concatenate((dt_naive, dt_prec, dt_eff, [dt_restim], [dt_carry], [dt_myc], [dt_il2]))

    return dt_state


def diff_chain(state, influx, beta, outflux):

    dt_state = np.zeros(len(state))
    dt_state[0] = influx - (beta + outflux) * state[0]
    for i in range(1,len(state)):
            dt_state[i] = beta * state[i - 1] - (beta + outflux) * state[i]

    return dt_state

# =============================================================================
# homeostasis models
# =============================================================================

def menten_signal(x, K = 0.1, hill = 12):
    out = x**hill / (x**hill + K**hill)
    if out < 1e-12:
        out = 0
    return out


def null_model(ncells, myc, il2, conc_C, d):
    beta_p = d["beta_p"]

    if d["beta_p"] !=0:
        if ncells > 1e20:
            d["beta_p"] = 0

    return beta_p


def timer_prolif(ncells, myc, il2, conc_C, d):
    beta_p = d["beta_p"] * menten_signal(myc, d["EC50_myc"], d["hill"])
    return beta_p


def il2_prolif(ncells, myc, il2, conc_C, d):
    beta_p = d["beta_p"] * menten_signal(il2, d["EC50_il2"], d["hill"])
    return beta_p


def carry_prolif(ncells, myc, il2, conc_C, d):

    if d["beta_p"] !=0:
        if ncells > d["n_crit"]:
            d["beta_p"] = 0

    return d["beta_p"]


