# -*- coding: utf-8 -*-
"""
simulation class minimal models with repeated stimulation
"""
import models as model
import numpy as np
import copy
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.integrate import solve_ivp
from scipy.constants import N_A
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib.colors import LogNorm

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

        # add a global IL2 concentration at the end
        il2_global_init = 6e7
        y0[-1] = il2_global_init
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
                            method = "LSODA")

            # append all simulation data except for the last time step because this will be included in next simulation
            ts.append(sol.t[:-1])
            y0 = sol.y[:,-1].copy()

            # reset restim timer (not myc) and make IL2 production timing dependent
            y0[-4] = 1

            ys.append(sol.y[:,:-1])

        state = np.hstack(ys).T
        time = np.hstack(ts).T
        self.time = time

        # # factor out global IL2 before splitting array and summing across all stimulations
        state[:,-1] = state[:,-1] * (1e12 / (20e-6*N_A))
        cell_arr = state[:,:-4]
        mol_arr = state[:,-4:] # contains timer and carry information

        self.cell_arr = cell_arr
        self.mol_arr = mol_arr
        return cell_arr, mol_arr

        return g

    def get_readouts(self):
        """
        get readouts from state array
        """
        state = self.cells
        state = state.loc[state["species"] == "eff",:]

        peak = get_peak_height(state.time, state.value)
        area = get_area(state.time, state.value)
        tau = get_peaktime(state.time, state.value)
        decay = get_duration(state.time, state.value)
        
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
    
    def run_timecourses(self):
        df_list = [sim.get_cells(["CD4_all"]) for sim in self.sim_list]
        df = pd.concat(df_list)

        return df
    
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
    
    def plot_timecourses(self, arr, arr_name, log = True, log_scale = False, xlim = (None, None),
                         ylim = (None, None), cmap = "cividis_r",
                         data = None, norm_arr = None):
        """
        plot multple timecourses with colorbar
        can provide a data argument from run timecourses
        """
        if data is None:
            print("running time courses, please wait")
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
        
        cbar = g.fig.colorbar(sm, ax = g.axes)
        cbar.set_label(arr_name)  
    
        if log_scale == True:
            g.set(yscale = "log", ylim = (0.1, None))    
        

        return g, data


### helper functions to calculate readouts

def get_maximum(x, y):
    """
    interpolate maximum
    """
    f = InterpolatedUnivariateSpline(x, y, k=4)
    cr_pts = f.derivative().roots()
    cr_pts = np.append(cr_pts, (x[0], x[-1])) 
    cr_vals = f(cr_pts)
    max_index = np.argmax(cr_vals)

    max_x = cr_pts[max_index]
    max_y = cr_vals[max_index]

    return max_x, max_y


def get_tau(time, cells):
    """
    get halftime of peak
    """
    cells = cells.array 
    crit = check_criteria(cells)
    
    if crit == True:
    
        peak_idx = np.argmax(cells)
        # get max value
        peak = cells[peak_idx]
        peak_half = peak / 2.
        cells = cells[:(peak_idx+1)]
        time = time[:(peak_idx+1)]
        # assert that peak is not at beginning
        if peak_idx <= 3:
            tau = np.nan
        # assert that peak half is in new cell array
        elif np.all(peak_half<cells):
            tau = np.nan
        else:
            f = interp1d(cells, time)           
            tau = f(peak_half)
            tau = float(tau)
            
    else:
        tau = np.nan
            
    return tau    



def get_peak_height(time, cells):
    """
    get height of peak
    """
    cells = cells.array
    time = time.array
    crit = check_criteria(cells)

    if crit == True:
        peaktime, peak_val = get_maximum(time, cells)
    else: 
        peak_val = np.nan
        
    return peak_val



def get_peaktime(time, cells):
    """
    get time of peak
    """
    cells = cells.array
    time = time.array
    crit = check_criteria(cells)
    
    if crit == True:
        peaktime, peak_val = get_maximum(time, cells)
    else:
        peaktime = np.nan
    
    return peaktime



def get_duration(time, cells):
    """
    get total time when cells reach given threshold
    """
    cells = cells.array
    time = time.array
    crit = check_criteria(cells)
    thres = 0.001
    if crit == True and (cells > thres).any():
        # get times where cells are > value
        time2 = time[cells > thres]
        #use last element 
        dur = time2[-1]

    else:
        dur = np.nan
        
    return dur
        

def get_decay(time, cells):
    """
    get the half-time of decay
    """
    
    cells = cells.array 
    crit = check_criteria(cells)
    cellmax = np.amax(cells)
    cellmin = cells[-1] 
    
    if crit == True:
        peak_id = np.argmax(cells)
        cells = cells[peak_id:]
        time = time[peak_id:]
        
        # make sure there are at least two values in the array
        assert len(cells) > 1
        
        # interpolate to get time unter half of diff between max and arr end is reached
        celldiff = (cellmax - cellmin) / 2
        celldiff = cellmax - celldiff
        f = interp1d(cells, time)
        #print(cellmax, cellmin, celldiff)
        tau = f(celldiff)
    else:
        tau = np.nan
    
    return float(tau)


def get_area(time, cells):
    
    cells = cells.array 
    crit = check_criteria(cells)
    
    if crit == True:
        area = np.trapz(cells, time)
    else: 
        area = np.nan
        
    return area


def check_criteria(cells): 
    cellmax = np.amax(cells)
    cellmin = cells[-1]
    last_cells = cells[-10]
    # test first if peak id is at the end of array
    peak_id = np.argmax(cells)
    if peak_id >= len(cells) - 12:
        return False
    
    # check if cells increase and decrease around peak monotonically
    arr_inc = np.diff(cells[(peak_id-10):peak_id]) > 0
    arr_dec = np.diff(cells[peak_id:(peak_id+10)]) < 0
    crit4 = arr_inc.all() and arr_dec.all()

    # check difference between max and endpoint
    crit1 = np.abs(cellmax-cellmin) > 1e-3
    # check that max is higher than endpoint
    crit2 = cellmax > cellmin
    # check that something happens at all
    crit3 = np.std(cells) > 0.001
    # check that last cells are close to 0
    crit5 = (last_cells < 1e-1).all()
    crit5 = np.std(last_cells) < 1e-1

    criteria = [crit1, crit2, crit3, crit4, crit5]
    crit = True if all(criteria) else False

    return crit