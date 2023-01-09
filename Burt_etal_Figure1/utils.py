import numpy as np
import pandas as pd
from scipy.stats import lognorm as log_pdf
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
from itertools import product
from scipy.integrate import odeint

def simple_chain(state, time, d):
    # split states
    naive = state[:d["alpha_naive"]]
    eff1 = state[d["alpha_naive"]:(d["alpha_naive"]+d["alpha_eff1"])]
    eff2 = state[(d["alpha_naive"]+d["alpha_eff1"]):-3]
    myc = state[-1]
    cyto2 = state[-2]
    cyto1 = state[-3]

    # cytokine ODE with perturbation
    t_start = d["block_fb_start"]
    t_end = d["block_fb_start"] + d["block_dur"]

    if d["use_fb_regular"]:
        rate_cyto2 = 1
        fb_regular = prob_fb(sum(eff2)*rate_cyto2, d["fb_eff1"], 1)
        p2 = d["p1"] * fb_regular
    else:
        fb_eff1 = stepfun(time, hi=d["fb_eff1"], lo=1, start=t_start, end=t_end)
        p2 = d["p1"] * fb_eff1
    p1 = 1-d["p1"]
    p1, p2 = norm_prob(p1, p2)

    dt_cyto1 = 0
    dt_cyto2 = 0

    dt_myc = -d["deg_myc"]*myc

    # compute influx into next chain
    influx_naive = 0

    # algebraic relations timer
    beta_p = d["beta_p"]#*pos_fb(myc, d["EC50_myc"])
    #print(beta_p)
    beta = d["beta"]
    influx_eff1 = naive[-1] * beta * p1
    influx_eff2 = naive[-1] * beta * p2

    death = d["d_eff"]

    dt_naive = diff_chain(naive, influx_naive, beta, d["d_naive"], d["div_naive"])
    dt_eff_1 = diff_chain(eff1, influx_eff1, beta_p, death, d["div_eff"])
    dt_eff_2 = diff_chain(eff2, influx_eff2, beta_p, death, d["div_eff"])

    dt_state = np.concatenate((dt_naive, dt_eff_1, dt_eff_2, [dt_cyto1], [dt_cyto2],[dt_myc]))

    return dt_state

class Sim:
    """
    model simulation class
    initialize with a name (str), mode (model specific)
    params (dict), time to sim (arr) and core(model specific)
    """

    def __init__(self,
                 name,
                 params,
                 time,
                 core=simple_chain):
        # type of homeostasis model
        self.name = name
        self.params = dict(params)
        self.time = time
        self.core = core
        self.cells = None
        self.molecules = None
        self.default_params = dict(params)

    def init_model(self):
        """
        set initial conditions for ODE solver
        """
        # +3 for myc, cyto1, cyto2
        y0 = np.zeros(self.params["alpha_naive"] + self.params["alpha_eff1"] + self.params["alpha_eff2"] + 3)
        y0[0] = 1
        # set myc conc.
        y0[-1] = 1

        return y0

    def run_model(self):
        y0 = self.init_model()
        block_start = self.params["block_fb_start"]
        block_end = block_start + self.params["block_dur"]
        state = odeint(self.core, y0, self.time, args=(self.params,),
                       hmax=0.005)
        return state

    def run_sim(self):
        state = self.run_model()
        d = self.params
        time = self.time

        naive = state[:, :d["alpha_naive"]]
        naive = np.sum(naive, axis=1)

        eff1 = state[:, d["alpha_naive"]:(d["alpha_naive"] + d["alpha_eff1"])]
        eff1 = np.sum(eff1, axis=1)

        eff2 = state[:, (d["alpha_naive"] + d["alpha_eff1"]):-3]
        eff2 = np.sum(eff2, axis=1)

        cyto1 = state[:, -3]
        cyto2 = state[:, -2]
        myc = state[:, -1]

        cells = np.stack([naive, eff1, eff2], axis=-1)
        molecules = np.stack([cyto1, cyto2, myc], axis=-1)

        def modify(df, colnames, name):
            df = pd.DataFrame(data=df, columns=colnames)
            df.loc[:, "time"] = time
            df = pd.melt(df, id_vars=["time"], var_name="cell")
            df.loc[:,"name"] = name
            return df

        cells = modify(cells, ["naive", "eff1", "eff2"], self.name)
        molecules = modify(molecules, ["cyto1", "cyto2", "myc"], self.name)

        self.cells = cells
        self.molecules = molecules
        return cells, molecules

    def set_params(self, pnames, arr):
        for pname, val in zip(pnames, arr):
            self.params[pname] = val

    def reset(self):
        self.params = self.default_params.copy()

def norm_prob(p1,p2):
    """
    should be transitioned to norm_probs
    """
    #warnings.warn("norm_prob should transition to norm_probs")
    s = p1+p2
    p1 = p1/s
    p2 = p2/s
    return p1,p2

def diff_chain(state, influx, beta, death, n_div):
    """
    Parameters
    ----------
    state : arr
        arr to intermediate states.
    influx : float
        flux into first state of chain.
    beta : float
        DESCRIPTION.
    death : float
        DESCRIPTION.

    Returns
    -------
    dt_state : array
        DESCRIPTION.

    """
    dt_state = np.zeros_like(state)
    for i in range(len(state)):
        if i == 0:
            dt_state[i] = influx - (beta + death) * state[i] + 2 * n_div * beta * state[-1]
        else:
            dt_state[i] = beta * state[i - 1] - (beta + death) * state[i]

    return dt_state



def stepfun(x, hi, lo, start, end, s=100):
    assert start <= end
    assert hi >= lo
    """
    nice step fun that return value "hi" is start<x<end, else "lo"
    """
    if start == end:
        out = np.ones_like(x) * lo
    else:
        out = 0.5 * (hi + lo + (hi - lo) * np.tanh((x - start) * s) * np.tanh((end - x) * s))
    return out



def draw_new_params(sim, param_names, heterogeneity, use_CV = False):
    """
    heterogeneity.
    can be either coeff. of variation, then set use_CV = True
    and provide CV (typical values are between 0.1 and 1)
    """
    assert "params" in dir(sim)
    assert all(key in sim.params for key in param_names)

    for param in param_names:
        mean = sim.params[param]
        if use_CV:
            std = mean * heterogeneity
        else:
            std = heterogeneity

        sigma, scale = lognorm_params(mean, std)
        sample = log_pdf.rvs(sigma, 0, scale, size=1)
        sim.params[param] = sample[0]


def lognorm_params(mode, stddev):
    """
    Given the mode and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    p = np.poly1d([1, -1, 0, 0, -(stddev/mode)**2])
    r = p.roots
    sol = r[(r.imag == 0) & (r.real > 0)].real
    shape = np.sqrt(np.log(sol))
    scale = mode * sol
    return shape, scale



def prob_fb(x, fc, EC50, hill=3):
    out = (fc * x**hill + EC50**hill) / (x**hill + EC50**hill)
    return out

def pos_fb(cyto, EC50):
    cyto = np.asarray(cyto)
    cyto[cyto<0]=0

    out = cyto ** 3 / (cyto ** 3 + EC50 ** 3)
    return out

def fb_fc(cyto, gamma, EC50):
    cyto = np.asarray(cyto)
    cyto[cyto<0]=0

    out = (gamma * cyto ** 3 + EC50 ** 3) / (cyto ** 3 + EC50 ** 3)
    return out


#def prob_fb(cyto, gamma, EC50):
#    """
#    deprecated use fb_fb
#    """
#    warnings.warn("prob_fb is deprecated, use fb_fb")
#    cyto = cyto if cyto > 0 else 0
#    out = (gamma * cyto ** 3 + EC50 ** 3) / (cyto ** 3 + EC50 ** 3)
#    return out


def plot_heatmap(df, value_col, readout, log_color, xlabel = None, ylabel = None,
                 vmin=None, vmax=None, cmap="Reds", log_axes=True, cbar_label = None,
                 title = None, figsize = (8.0, 6), xticks = None,
                 yticks = None, cbar_ticks = None):
    """
    NOTE that I changed this function to apply to data models works with pscan2d from proc branch rtm
    df needs to have a column named readout and a column named value or similar
    take df generated from 2dscan and plot single heatmap for a given readout
    note that only effector cells are plotted
    value_col: could be either val norm or value as string
    log_color: color representation within the heatmap as log scale, use if input arr was log scaled
    or if variation across multiple scales is expected
    """
    # process data (df contains all readouts and all cells
    df = df.loc[df["readout"] == readout,:]
    arr1 = df["param_val1"].drop_duplicates()
    arr2 = df["param_val2"].drop_duplicates()
    assert (len(arr1) == len(arr2))

    # arr1 and arr2 extrema are bounds, and z should be inside those bounds
    z_arr = df[value_col].values
    z = z_arr.reshape((len(arr1), len(arr2)))
    z = z.T

    z = z[:-1, :-1]

    # check if color representation should be log scale
    sm, norm = get_colorscale(log_color, cmap, vmin, vmax)

    # plot data
    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(arr1, arr2, z, norm = norm, cmap=cmap, rasterized = True)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # adjust scales
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    if xlabel is None:
        xlabel = ax.set_xlabel(df.pname1.iloc[0])
    ax.set_xlabel(xlabel)

    if ylabel is None:
        ylabel = ax.set_ylabel(df.pname2.iloc[0])
    ax.set_ylabel(ylabel)

    cbar = plt.colorbar(sm, ax=ax)
    if cbar_label is None:
        cbar_label = readout
    cbar.set_label(cbar_label)

    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    return fig, z


def get_colorscale(hue_log, cmap, vmin = None, vmax = None):
    if hue_log:
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # make mappable for colorbar
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    return sm, norm

def pscan2d(sim, prange1, prange2, pname1, pname2, res, rangefun = np.linspace,
            timepoints = [9], cellnames = ["Tfh_all", "nonTfh"],
            cellnames_maxima =  ["Tfh_all", "nonTfh"]):
    inputs = get_2dinputs(prange1, prange2, res, rangefun)
    outputs = []

    for i, input in enumerate(inputs):
        p1, p2 = input
        sim.params[pname1] = p1
        sim.params[pname2] = p2
        sim.run_sim()

        r = get_readouts_data(sim.cells,
                              timepoints = timepoints,
                              cellnames = cellnames,
                              cellnames_maxima = cellnames_maxima)
        r["param_val1"] = p1
        r["param_val2"] = p2

        outputs.append(r)

    outputs = pd.concat(outputs)
    outputs["name"] = sim.name
    outputs["pname1"] = pname1
    outputs["pname2"] = pname2

    sim.reset()
    return outputs


def get_readouts_data(cells, timepoints=[5.0, 10.0, 30.0, 60.0], cellnames=["Tfh_all", "nonTfh"],
                      cellnames_maxima = ["Th1_eff", "Tfh_eff", "Tr1_all", "Tfh_chr"]):
    """
    returns a tidy dataframe with two columns: readout and value
    every readout function should have this format
    readouts here are the cell maxmima cell the cells inv ariable cellnames_max
    and the relative fraction of Tfh cells
    """
    # get cell maxima
    df_maxima = get_cellmaxima(cells, cellnames_maxima)

    # check that cellnames provided and timepoints provided are in cells df
    assert cells["time"].drop_duplicates().isin(timepoints).sum() == len(timepoints)
    assert cells["cell"].drop_duplicates().isin(cellnames).sum() == len(cellnames)

    # reduce df to provided timepoints and celltypes
    cells = cells[cells["time"].isin(timepoints)]
    cells = cells[cells["cell"].isin(cellnames)]

    # get cell ratio: compute sum and normalize
    cells.loc[:, "tot"] = cells.groupby('time')['value'].transform('sum')
    cells.loc[:, "value"] = cells["value"] / cells["tot"]
    cells.loc[:, "readout"] = "relTfh"
    cells = cells.loc[cells["cell"] == cellnames[0], :]
    cells["readout"] = cells["readout"] + "_day" + cells["time"].astype(str)
    # cells["readout"] = cells[["readout", "time"]].agg("_".join, axis=1)
    cells = cells.loc[:, ["readout", "value"]]

    df = pd.concat([df_maxima, cells])
    df = df.reset_index(drop=True)
    return df


def get_cellmaxima(cells, cellnames):
    # get cell maxima fpr given cellnames
    # check that the number of celltypes provided matches the number found in cells df
    assert cells["cell"].drop_duplicates().isin(cellnames).sum() == len(cellnames)
    cells = cells[cells["cell"].isin(cellnames)]
    df = cells.groupby(["cell"])["value"].max()
    df = df.reset_index()
    df.loc[:, "readout"] = "peak"
    df["readout"] = df[["readout", "cell"]].agg("_".join, axis=1)
    df = df.loc[:, ["readout", "value"]]
    return df



def get_2dinputs(prange1, prange2, res, rangefun=np.linspace):
    """
    p1 : tuple (min, max) of param range for pname1
    p2 : tuple (min, max) of param range for pname2
    post process with plot heatmap
    """
    # generate arrays and get cartesian product
    arr1 = rangefun(prange1[0], prange1[1], res)
    arr2 = rangefun(prange2[0], prange2[1], res)
    inputs = product(arr1, arr2)
    return inputs