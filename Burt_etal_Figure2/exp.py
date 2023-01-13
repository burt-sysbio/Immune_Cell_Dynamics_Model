# -*- coding: utf-8 -*-
"""
simulation class minimal models with repeated stimulation
"""
import readout_module as readouts
import models as model

import numpy as np
from scipy.integrate import odeint, solve_ivp
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import copy
import itertools
import matplotlib.ticker as ticker
from scipy.optimize import minimize_scalar
from matplotlib.colors import LogNorm
from scipy.stats import lognorm as log_pdf
import warnings
from scipy.constants import N_A

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


def change_param(simlist, pname, arr):
    assert len(arr) == len(simlist)
    for sim, val in zip(simlist,  arr):

        sim.name = val
        sim.parameters[pname] = val
    
    return simlist

    
def make_sim_list(Simulation, n = 20):
    sim_list = [copy.deepcopy(Simulation) for i in range(n)]
    return sim_list


class Simulation:
    """
    model simulation class
    initialize with a name (str), mode (model specific)
    parameters (dict), time to sim (arr) and core(model specific)
    """
    def __init__(self, name, mode, parameters, start_times, vir_model):
        self.name = name
        self.mode = mode
        self.parameters = dict(parameters)
        self.default_params = dict(parameters)
        self.time = None
        self.state = None
        self.cells = None
        self.mols = None
        self.cell_arr = None
        self.mol_arr = None
        self.start_times = start_times
        self.vir_model = vir_model

        
    def init_model(self):
        """
        set initial conditions for ODE solver. Not that this includes multiple starting points of the ODE
        """
        n_molecules = 4 # myc and carrying capacity molecule
        y0 = np.zeros(self.parameters["alpha"] + self.parameters["alpha_prec"] + 1 * self.parameters["alpha_p"] + n_molecules)

        # init myc concentration
        y0[-2] = 1. # this should matter only for the first cell population at t0, other naive cells are not initialized yet and timer is reset
        y0[-3] = 1 # carrying capacity!
        y0[-4] = self.parameters["il2_stimulation"]
        # multiply y0 as often as there are stimulations
        #y0_tile = np.tile(y0, len(self.start_times))
        #y0_tile = np.tile(y0, 1) # the tile was only used the naive cells were restimulated, now I stimulate effector cells to produce IL2

        # add a global IL2 concentration at the end
        il2_global_init = 6e7
        y0[-1] = il2_global_init
        #y0_tile = np.concatenate((y0_tile, [il2_global_init]))
        y0[0] = self.parameters["initial_cells"] # only initialize the first naive cell population
        return y0
    
    
    def compute_cellstates(self, **kwargs):
        """
        summarize effector cells and naive cells
        """
        self.run_model(**kwargs)
        cell_arr = self.cell_arr
        mol_arr = self.mol_arr
        assert cell_arr is not None

        idx1 = self.parameters["alpha"]
        idx2 = self.parameters["alpha"] + self.parameters["alpha_prec"]

        tnaive = cell_arr[:, :idx1]
        tprec = cell_arr[:, idx1:idx2]
        teff = cell_arr[:, idx2:]

        teff = np.sum(teff, axis = 1)
        tnaive = np.sum(tnaive, axis = 1)
        tprec = np.sum(tprec, axis = 1)
        cd4_all = np.sum(cell_arr, axis = 1)

        cells = np.stack((tnaive, tprec, teff, cd4_all), axis = -1)
        cells = pd.DataFrame(data = cells, columns = ["naive", "prec", "eff", "CD4_all"])

        cells.loc[:,"time"] = self.time
        cells = pd.melt(cells, id_vars = ["time"], var_name = "species")
        cells.loc[:,"name"] = self.name

        cells["model"] = self.mode.__name__

        mols = pd.DataFrame(data = mol_arr, columns = ["Restim", "Carry", "Timer", "IL2"])
        mols.loc[:,"time"] = self.time
        mols = pd.melt(mols, id_vars = ["time"], var_name = "species")
        mols.loc[:,"name"] = self.name
        mols["model"] = self.mode.__name__

        self.cells = cells
        self.mols = mols
        return cells    

    def get_cells(self, cell_list):

        cells = self.compute_cellstates()
        cells = cells.loc[cells["species"].isin(cell_list),:]
        return cells

    def run_model(self, **kwargs):
        """
        should generate same
        run one large ODE with different starting times
        needs parameters that indicate different starting times...
        return: should return same as run_model function
        """
        if "max_step" in kwargs:
            print("running model with step size " + str(kwargs["max_step"]))
        else:
            print("running model, no step size set")
        start_times = self.start_times

        y0 = self.init_model()

        mode = self.mode
        params = dict(self.parameters)
        vir_model = self.vir_model

        d = self.parameters
        # using solve_ivp instead of ODEINT for stiff problem.
        ts = []
        ys = []

        for i in range(len(start_times)):
            tstart, tend = start_times[i]

            sol = solve_ivp(fun = model.repeated_stimulation,
                            t_span = (tstart, tend), y0 = y0,
                            args=(mode, d, vir_model), **kwargs,
                            method = 'LSODA')

            # append all simulation data except for the last time step because this will be included in next simulation
            ts.append(sol.t[:-1])
            y0 = sol.y[:,-1].copy()

            # reset restim timer (not myc) and make IL2 production timing dependent
            y0[-4] = 1

            ys.append(sol.y[:,:-1])


        state = np.hstack(ys).T
        time = np.hstack(ts).T
        self.time = time

        #state = odesol.y
        # # with new solve_ivp method, I need to transpose, because output is switched compared to ODEint
        # state = state.T
        #
        # # factor out global IL2 before splitting array and summing across all stimulations
        state[:,-1] = state[:,-1] * (1e12 / (20e-6*N_A))
        #state = state[:,:-1]
        #state = np.split(state, len(1), axis = 1) # exchange len(1) with len(start_times) for previous version
        #state = np.sum(state, axis = 0)
        #
        cell_arr = state[:,:-4]
        mol_arr = state[:,-4:] # contains timer and carry information

        # # reset the IL2 from individual simulations (IL2 external, not used atm)
        # # with global IL2, need to reshape IL2 array for proper concatenation
        #mol_arr = np.hstack((mol_arr,il2[:,None]))
        #
        self.cell_arr = cell_arr
        self.mol_arr = mol_arr
        return cell_arr, mol_arr

    
    def plot_cells(self, ylim = [1e-1, 1e7],  **kwargs):
        #self.compute_cellstates()
        g = sns.relplot(data = self.cells, x = "time", y = "value", kind = "line",
                        col = "species", facet_kws= {"sharey" : True})
        g.set(yscale = "log", ylim = ylim)
        plt.show()
        return g
    

    def plot_molecules(self, **kwargs):
        #self.compute_cellstates()
        g = sns.relplot(data = self.mols, x = "time", y = "value", kind = "line",
                        col = "species", facet_kws= {"sharey" : False})
        plt.show()
        return g

    def get_readouts(self):
        """
        get readouts from state array
        """
        state = self.cells
        state = state.loc[state["species"] == "eff",:]

        peak = readouts.get_peak_height(state.time, state.value)
        area = readouts.get_area(state.time, state.value)
        tau = readouts.get_peaktime(state.time, state.value)
        decay = readouts.get_duration(state.time, state.value)
        
        reads = [peak, area, tau, decay]
        read_names = ["Peak", "Area", "Peaktime", "Decay"]
        data = {"readout" : read_names, "read_val" : reads}
        reads_df = pd.DataFrame(data = data)
        reads_df["name"] = self.name
        
        if "menten" in self.mode.__name__ :
            modelname = "menten"
        else:
            modelname =  "thres"
            
        reads_df["model_name"] = modelname
        
        return reads_df

            
    def vary_param(self, pname, arr, normtype = "first", normalize = True, **kwargs):

        readout_list = []
        edge_names = ["alpha", "alpha_p"]

        # edgecase for distributions
        dummy = None
        if pname in edge_names:
            dummy = "beta" if pname == "alpha" else "beta_p"
            arr = np.arange(2, 20, 2)

        for val in arr:
            # edgecase for distributions
            if pname in edge_names:
                self.parameters[dummy] = val
                
            self.parameters[pname] = val
            self.compute_cellstates(**kwargs)
            read = self.get_readouts()

            read["p_val"] = val
            readout_list.append(read)

            self.reset_params()
        if normalize:
            df = self.vary_param_norm(readout_list, arr, edge_names, normtype, pname)
        else:
            df = readout_list
        return df


    def vary_param_norm(self, readout_list, arr, edge_names, normtype, pname):
        """
        take readout list and normalize to middle or beginning of array
        Parameters
        ----------
        readout_list : list
            readouts for diff param values.
        arr : array
            parameter values.
        edgenames : list of strings
            parameter names.
        normtype : string, should be either "first" or "middle"
            normalize to middle or beginning of arr.

        Returns
        -------
        df : data frame
            normalized readouts
        """
        
        df = pd.concat(readout_list)
        df = df.reset_index(drop = True)
        
        # merge df with normalization df    
        norm = arr[int(len(arr)/2)]

        assert normtype in ["first", "middle"]
        if normtype == "first":
            norm = arr[0]
        elif normtype == "middle":
            # get the value in arr, which is closest to the median
            norm = arr[np.argmin(np.abs(arr-np.median(arr)))]

        df2 = df[df.p_val == norm]

        df2 = df2.rename(columns = {"read_val" : "ynorm"})
        df2 = df2.drop(columns = ["p_val"])
        df = df.merge(df2, on=['readout', 'name', "model_name"], how='left')
        
        # compute log2FC
        logseries = df["read_val"]/df["ynorm"]
        logseries = logseries.astype(float)

        df["log2FC"] = np.log2(logseries)
        df = df.drop(columns = ["ynorm"])
        
        # add xnorm column to normalise x axis for param scans
        df["xnorm"] = df["p_val"] / norm
        df["pname"] = pname
        
        if pname in edge_names:
            df["p_val"] = df["p_val"] / (df["p_val"]*df["p_val"])
            
        return df
    
    
    def norm(self, val, pname, norm):
        """
        optimization function
        calculate difference between simulated response size and wanted response size
        val : parameter value
        pname: str, parameter name
        norm : wanted response size
        returns float, difference in abs. values between wanted resp. size and calc. response size
        """
        self.parameters[pname] = float(val)
        state = self.state
        area = readouts.get_area(state.time, state.cells)

        return np.abs(area-norm)

    
    def norm_readout(self, pname, norm, bounds = None):
        """
        adjust parameter to given normalization condition (area = norm)
        pname: str of parameter name to normalize
        norm : value of area to normalize against
        bounds : does not work well - if bounds provided, only scan for paramter in given range
        returns: adjusted parameter value
        """
        if bounds != None:
            method = "Bounded"
        else:
            method = "Brent"
        
        # minimize norm function for provided values
        out = minimize_scalar(self.norm, method = method, bounds=bounds, args=(pname, norm, ))  
        #if pname == "beta_p":
        #    print(out)
        #    print(self.mode)
        
        dummy = np.nan
        print(out.success, out.fun, out.x)
        # check results of optimization object
        if out.success == True and out.fun < 1e-2:
            dummy = out.x
            
        return dummy


    def get_heatmap(self, arr1, arr2, name1, name2, norm_list = None):
        
        """
        make a heatmap provide two arrays and two parameter names as well
        as readout type by providing readout function
        can also provide normalization value for log2 presentation
        """
        area_grid = []
        peaktime_grid = []
        peak_grid = []
        grids = [area_grid, peaktime_grid, peak_grid]
        readout_funs = [readouts.get_area, readouts.get_peaktime, readouts.get_peak_height]
        old_params = dict(self.parameters)

        assert (len(readout_funs) == len(grids))
        # if no normalization was provided do not normalize so use None for each readout
        if norm_list is None:
            norm_list = [None for _ in readout_funs]

        for val1, val2 in itertools.product(arr1, arr2):
            # loop over each parameter combination and get readouts
            self.parameters[name1] = val1
            self.parameters[name2] = val2
            self.compute_cellstates()
            df = self.cells.loc[self.cells["species"] == "CD4_all"]
            # get each readout and normalize, append to grid
            for readout_fun, grid, norm_val in zip(readout_funs, grids, norm_list):
                readout = readout_fun(df.time, df.value)
                if norm_val is not None:
                    readout = np.log2(readout/norm_val)
                grid.append(readout)

            self.parameters = dict(old_params)
            #print(len(z))

        # process output, combine readouts
        df = pd.DataFrame(grids)
        df = df.T
        df.columns = ["Area", "Peak Time", "Peak Height"]

        mylist = [(x, y) for x, y in itertools.product(arr1, arr2)]
        df[["param_val1", "param_val2"]] = mylist
        df=df.melt(id_vars=["param_val1", "param_val2"], var_name = "readout")
        df["name"] = self.name
        df["pname1"] = name1
        df["pname2"] = name2
        return df
 
    
    def plot_heatmap(self, arr1, arr2, name1, name2, readout_fun, norm = None, 
                     vmin = None, vmax = None, title = None, 
                     label1 = None, label2 = None, cmap = "bwr", log = True,
                     cbar_label = "change response size"):
    
        arr1, arr2, val = self.get_heatmap(arr1, arr2, name1, name2, readout_fun, norm)

        fig, ax = plt.subplots(figsize = (6,4))
        color = cmap
        cmap = ax.pcolormesh(arr1, arr2, val, cmap = color, vmin = vmin, vmax = vmax,
                             rasterized = True)

        
        loc_major = ticker.LogLocator(base = 10.0, numticks = 100)
        loc_minor = ticker.LogLocator(base = 10.0, 
                                      subs = np.arange(0.1,1,0.1),
                                      numticks = 12) 
        
        if log == True:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(loc_major)
            ax.xaxis.set_minor_locator(loc_minor)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_title(title)
        cbar = plt.colorbar(cmap)
        cbar.set_label(cbar_label)
            
        plt.tight_layout()
        # previously returned fig

        return fig
    
    def gen_lognorm_params(self, pname, std, n = 20):
        """
        deprecated use set_params_lognorm
        """
        mean = self.parameters[pname]
        sigma, scale = lognorm_params(mean, std)
        sample = log_pdf.rvs(sigma, 0, scale, size = n)
        
        return sample
    

    def get_lognorm_array(self, pname, res, CV = 0.1):
        """
        generate lognorm array with given CV
        res: number of parameter values sampled from lognorm
        mean of lognorm taken from current value of simulation object pname

        """
        # need to check if tuple was provided, if yes then parameters need to be the same for this analysis!

        myval = self.parameters[pname]
        assert myval != 0
        std = myval * CV
        sigma, scale = lognorm_params(myval, std)
        sample = log_pdf.rvs(sigma, 0, scale, size=res)
        return sample

    def set_params_lognorm(self, params : list, CV = 0.1):
        """
        take a list of params and shuffle lognorm style
        """
        for pname in params:
            assert pname in self.parameters.keys()
            sample = self.get_lognorm_array(pname, 1, CV)
            self.parameters[pname] = sample[0]


    def draw_new_params(self, param_names, heterogeneity):
        """
        deprecated use set_params_lognorm
        """
        for param in param_names:
            mean = self.parameters[param] 
            std = mean*(heterogeneity/100.)
            print("watch out, dividing by 100 for some ancient mistaken? reason...")
            sigma, scale = lognorm_params(mean, std)
            sample = log_pdf.rvs(sigma, 0, scale, size = 1)
            # I only want single float, not array
            sample = sample[0]
            self.parameters[param] = sample

    def reset_params(self):
        """
        reset parameters to default state
        """
        self.parameters = dict(self.default_params)

    def gen_arr(self, pname, scales = (1,1), use_percent = False, n = 30):
        """
        scales could be either 1,1 for varying one order of magnitude
        or 0.9 and 1.1 to vary by 10 %
        """
        edge_names = ["alpha", "alpha_1", "alpha_p"]
        if pname in edge_names:
            arr = np.arange(2, 20, 2)
        else:
            params = dict(self.parameters)
            val = params[pname]

            if use_percent:
                val_min = scales[0] * val
                val_max = scales[1] * val
                arr = np.linspace(val_min, val_max, n)
            else:
                val_min = 10**(-scales[0])*val
                val_max = 10**scales[1]*val
                arr = np.geomspace(val_min, val_max, n)

        return arr
        

class SimList:
       
    def __init__(self, sim_list):
        self.sim_list = sim_list
 
    
    def reduce_list(self, cond):
        sim_list_red = [sim for sim in self.sim_list if np.abs(sim.get_area()-cond) < 1.]
        return SimList(sim_list_red)
    
    
    def get_readout(self, name):
        readout_list = []
        for sim in self.sim_list:
            df = sim.get_readouts()
            # check that readout name is actually available since I change readouts sometimes
            assert name in df.readout.values
            out = float(df.read_val[df.readout == name])
            readout_list.append(out)
        return readout_list

         
    def pscan(self, pnames, arr = None, scales = (1,1), n = None, normtype = "first", use_percent = False, **kwargs):
        pscan_list = []
        for sim in self.sim_list:
            for pname in pnames:
                if arr is None:
                    assert n is not None, "need to specific resolution for pscan array"
                    arr = sim.gen_arr(pname = pname, scales = scales, n = n, use_percent= use_percent)

                readout_list = sim.vary_param(pname, arr, normtype, **kwargs)
                
                pscan_list.append(readout_list)
        
        df = pd.concat(pscan_list)
        return df
    
    
    def run_timecourses(self):
        df_list = [sim.get_cells(["CD4_all"]) for sim in self.sim_list]
        df = pd.concat(df_list)

        return df
    
    
    def normalize(self, pname, norm, bounds):
        out_list = []
        for sim in self.sim_list:
            out = sim.norm_readout(pname, norm, bounds = bounds)
            out_list.append(out)
        return out_list


    def get_param_arr(self, pname):
        out = [sim.parameters[pname] for sim in self.sim_list]
        return out

    
    def plot_timecourses(self, arr, arr_name, log = True, log_scale = False, xlim = (None, None),
                         ylim = (None, None), cmap = "cividis_r", cbar_scale = 1.,
                         il2_max = False, ticks = None, data = None, norm_arr = None):
        """
        plot multple timecourses with colorbar
        can provide a data argument from run timecourses
        """
        # parameter for scaling of color palette in sns plot
        # run the time courses to generate data

        #cmap =  # sns.color_palette("light:grey_r", as_cmap=True)
        if data is None:
            print("hi there")
            data = self.run_timecourses()
            data = data.reset_index(drop = True)
            if norm_arr is not None:
                data.loc[:,"name"] = data.loc[:,"name"] / norm_arr

        if norm_arr is not None:
            arr = arr/norm_arr
        vmin = np.min(arr)
        vmax = np.max(arr)
        if log == True:
            norm = matplotlib.colors.LogNorm(vmin = vmin, vmax = vmax)
            hue_norm = LogNorm(vmin = vmin, vmax = vmax)
        else:
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
            hue_norm = None
        
        # make mappable for colorbar
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])


        
        # hue takes the model name, so this should be a scalar variable
        # can be generated by change_param function
        g = sns.relplot(x = "time", y = "value", kind = "line", data = data, hue = "name",
                        hue_norm = hue_norm, col = "model", palette = cmap,
                        height = 5,legend = False, aspect = 1.2, 
                        facet_kws = {"despine" : False})

        
        g.set(xlim = xlim, ylim = ylim)
        ax = g.axes[0][0]
        ax.set_ylabel("cell dens. norm.")
        g.set_titles("{col_name}")
        
        # if ticks are true take the upper lower and middle part as ticks
        # for colorbar
        if ticks == True:
            if log == True:
                ticks = np.geomspace(np.min(arr), np.max(arr), 3)
            else:
                ticks = np.linspace(np.min(arr), np.max(arr), 3)

            cbar = g.fig.colorbar(sm, ax = g.axes, ticks = ticks)
            cbar.ax.set_yticklabels(np.round(cbar_scale*ticks,2))
        else:
            cbar = g.fig.colorbar(sm, ax = g.axes, ticks = ticks)
        # add colorbar
        
        cbar.set_label(arr_name)
        #print(100*arr)

        
        cbar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        
        if log_scale == True:
            g.set(yscale = "log", ylim = (0.1, None))    
        

        return g, data
    
    
    def plot_pscan(self, pnames):
        data = self.pscan(pnames)
        g = sns.relplot(data = data, x = "xnorm", hue = "model_name", y = "log2FC", col = "readout",
                    row = "pname", kind = "line")
        g.set(xscale = "log")
        
        return g



