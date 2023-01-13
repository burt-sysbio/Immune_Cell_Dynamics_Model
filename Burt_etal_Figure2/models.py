# -*- coding: utf-8 -*-
"""
keep all ode models here
"""
import numpy as np
from numba import jit
from scipy.constants import N_A
from scipy.interpolate import interp1d
# =============================================================================
# linear models
# =============================================================================
def repeated_stimulation(time, state, model, d, vir_model):

    # global IL2 is an extra variable!
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

    #ag = vir_model(time)
    #print(time, ag, il2_global)
    #ag_effective = (ag**3 / (ag**3 + 1)) * d["rate_il2_prec"] * n_eff * 0.01

    if (model.__name__ == "restim") | (model.__name__ == "restim_timer_il2"):
        restim_effective = restim**3 / (restim**3 + 1)
        il2_production = d["rate_il2_naive"] * n_naive + d["rate_il2_prec"] * n_prec + d["rate_il2_prec"] * n_eff * restim_effective * 0.01
        # np exp(-1*time) is explicit timer for restimulation
    else:
        il2_production = d["rate_il2_naive"] * n_naive + d["rate_il2_prec"] * n_prec # (n_naive+n_prec) #+ ag_effective
    dt_il2 = il2_production - d["up_il2"] * (n_eff) * (il2_global/(il2_global+d["K_il2_cons"]))

    il2_effective = il2_global * (1e12/(20e-6*N_A))

    ncells = n_naive+n_prec+n_eff
    beta_p = model(ncells, myc, il2_effective, carry, d)


    d_eff = d["d_eff"]
    # check homeostasis criteria
    # differentiation
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

def generate_poisson_process(mu, num_events):
    time_intervals = -np.log(np.random.random(num_events)) / mu
    total_events = time_intervals.cumsum()

    return total_events

def spiketrain(mu):
    n_events = 1000
    res = 1000

    event_times = generate_poisson_process(mu, n_events)
    yarr = np.zeros(res)
    xarr = np.linspace(0, event_times[0], res)
    for i in range(len(event_times)-1):
        x = np.linspace(event_times[i], event_times[i+1], res)
        y = (x-x[0])**20 * np.exp(-10*(x-x[0]))
        y = stepfun(x, hi = 1, lo = 0, start = x[0], end = x[0]+0.5)
        xarr = np.concatenate((xarr, x))
        yarr = np.concatenate((yarr, y))

    f = interp1d(xarr, yarr)

    return f

def stepfun(x, hi, lo, start, end, s=10):
    assert start <= end
    assert hi >= lo
    """
    nice step fun that return value "hi" is start<x<end, else "lo"
    """
    out = 0.5 * (hi+lo + (hi-lo) * np.tanh((x-start)*s) * np.tanh((end-x)*s))
    return out


def diff_chain(state, influx, beta, outflux):

    dt_state = np.zeros(len(state))
    dt_state[0] = influx - (beta + outflux) * state[0]
    for i in range(1,len(state)):
            dt_state[i] = beta * state[i - 1] - (beta + outflux) * state[i]

    return dt_state


def menten(conc, vmax, K, hill):   
    # make sure to avoid numeric errors for menten
    conc = conc if conc > 0 else 0
    out = (vmax*conc**hill) / (K**hill+conc**hill)
    
    return out

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

    # add some artificial buffer to the null model (aka make the Null model like
    # a very high carrying capacity
    if d["beta_p"] !=0:
        if ncells > 1e20:
            d["beta_p"] = 0

    return beta_p

def restim(ncells, myc, il2, conc_C, d):
    beta_p = d["beta_p"] * menten_signal(il2, d["EC50_il2"], d["hill"])
    return beta_p

def restim_timer_il2(ncells, myc, il2, conc_C, d):
    beta_p = d["beta_p"] * menten_signal(il2, d["EC50_il2"], d["hill"]) * menten_signal(myc, d["EC50_myc"], d["hill"])
    return beta_p


def timer_prolif(ncells, myc, il2, conc_C, d):
    beta_p = d["beta_p"] * menten_signal(myc, d["EC50_myc"], d["hill"])
    return beta_p

def timer_il2_prolif(ncells, myc, il2, conc_C, d):
    beta_p = d["beta_p"] * menten_signal(myc,d["EC50_myc"], d["hill"]) * menten_signal(il2, d["EC50_il2"], d["hill"])
    return beta_p


def il2_prolif(ncells, myc, il2, conc_C, d):
    beta_p = d["beta_p"] * menten_signal(il2, d["EC50_il2"], d["hill"])
    return beta_p

def carry_prolif(ncells, myc, il2, conc_C, d):
    #beta_p = d["beta_p"] * menten_signal(conc_C, d["EC50_carry"], d["hill"])

    if d["beta_p"] !=0:
        if ncells > d["n_crit"]:
            d["beta_p"] = 0

    return d["beta_p"]


