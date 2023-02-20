import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from utils_data_model import lognorm_params
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import lognorm as log_pdf
import os
from numba import jit
import json

class Simulation:
    cells = None
    cells_acute = None
    cells_chr = None
    molecules = None
    molecules_acute = None
    molecules_chr = None
    default_params = None
    fit_report = None
    readouts = None
    fit_params = None
    cells_best_fit = None

    def __init__(self, name, time, params, virus_model):
        # type of homeostasis model
        self.name = name
        self.params = dict(params)
        self.time = time
        self.default_params = dict(params)
        self.virus_core = virus_model

    def ode(self, state, time, params):
        pass

    def init_model(self):
        pass

    def fit_model(self):
        pass
    def get_ode_args(self):
        pass


    def run_model(self):
        y0 = self.init_model()
        self.virus_model = self.virus_core(self.time, self.params)

        args = self.get_ode_args()
        state = odeint(self.ode, y0, self.time, args=args)

        return state

    def param_scan2(self, pname, arr,
                   infection = "",
                   cell_names=["Th1_all", "Tfh_all"],
                   timepoints=[5.0, 9.0, 15.0, 30.0, 60.0],
                   use_fit=None,
                   save = True,
                   scan_type = "param_scans"):
        """
        pname: str or tuple, parameter name(s)
        arr: list or np.array, parameter array to scan
        pscan_name : str,
        cell_names : list of strings, should be celltypes
        timepoints: list of floats at which times cellnumbers should be returned
        return dataframe with readouts for each parameter value
        """
        assert (scan_type == "param_scans") | (scan_type == "mcarlo")

        pname = (pname,) if type(pname) != tuple else pname

        for p in pname:
            assert p in self.params.keys()

        # if fit ID is provided update parameters accordingly and set fit ID to savename
        if use_fit is not None:
            #self.load_fit(use_fit) this does not work bc then chronic setting is overwritten
            fit_ID = use_fit
        else:
            fit_ID = ""

        mylist = []
        # get readouts for each parameter value
        for x in arr:
            # if tuple was provided (for example for proliferation rates, then adjust both to same value)
            for p in pname:
                self.params[p] = x

            # if probabilitiy is used then need to update other probabilities for norm procedures
            if pname[0] in ["p1", "p2", "p3", "p4"]:
                self.norm_init_probs()

            self.compute_cellstates()
            out = self.get_readouts(timepoints=timepoints, cell_names=cell_names)
            out["param_value"] = x
            mylist.append(out)

        # process output
        reads = pd.concat(mylist)

        # concatenate param names if more than one param_name provided
        param_name = "_".join(pname)
        reads["param_name"] = param_name

        # reset and store pscan output
        savedir = "../../output/" + scan_type + "/"
        savename = savedir + "scan_" + param_name + "_" + fit_ID + infection + ".csv"
        if save:
            reads.to_csv(savename, index=False)

        self.reset_params()
        out = pd.concat(mylist)
        return out

    def param_scan(self, pname, arr,
                   cell_names=["Th1_all", "Tfh_all"],
                   timepoints=[5.0, 9.0, 15.0, 30.0, 60.0], name = "Arm"):
        """
        pname: str or tuple, parameter name(s)
        arr: list or np.array, parameter array to scan
        pscan_name : str,
        cell_names : list of strings, should be celltypes
        timepoints: list of floats at which times cellnumbers should be returned
        return dataframe with readouts for each parameter value
        """

        pname = (pname,) if type(pname) != tuple else pname

        for p in pname:
            assert p in self.params.keys()

        mylist = []
        # get readouts for each parameter value
        for x in arr:
            # if tuple was provided (for example for proliferation rates, then adjust both to same value)
            for p in pname:
                self.params[p] = x

            self.compute_cellstates()
            out = self.get_readouts(timepoints=timepoints, cell_names=cell_names)
            out["param_value"] = x
            mylist.append(out)

        # process output
        reads = pd.concat(mylist)

        # concatenate param names if more than one param_name provided
        param_name = "_".join(pname)
        reads["param_name"] = param_name

        self.reset_params()
        out = pd.concat(mylist)
        out["param_name"] = param_name
        out["model"] = name

        return out

    def reset_params(self):
        """
        reset parameters to default state
        """
        self.params = dict(self.default_params)


    @staticmethod
    @jit(nopython=True)
    def diff_chain(state, influx, beta, outflux):

        dt_state = np.zeros(len(state))
        dt_state[0] = influx - (beta + outflux) * state[0] #+ 2 * n_div * beta * state[-1]
        for i in range(1,len(state)):
                dt_state[i] = beta * state[i - 1] - (beta + outflux) * state[i]

        return dt_state



    def plot_cells(self, name="", cell_list=None, **kwargs):

        assert self.cells is not None
        mycolumns = ["time", "species", "value", "name"]
        assert (self.cells.columns == mycolumns).all()
        x = self.cells

        if cell_list is not None:
            x = x.loc[x["species"].isin(cell_list)]
        g = sns.relplot(data=x,
                        x=x["time"],
                        y=x["value"],
                        col="species", col_wrap=5,
                        facet_kws={"sharey": False},
                        kind="line")
        g.set(yscale="log", ylim=[1, None],
              xlabel = "time (days)",
              ylabel = "cells")
        g.set_titles("{col_name}")
        #plt.savefig("figures/" + str(name)+"cells")
        sns.despine(top = False, right = False)
        plt.show()
        return g


    def plot_molecules(self, name="", mol_list=None, **kwargs):

        assert self.molecules is not None

        mycolumns = ["time", "species", "value", "name"]
        assert (self.molecules.columns == mycolumns).all()
        x = self.molecules

        if mol_list is not None:
            x = x.loc[x["species"].isin(mol_list)]
        g = sns.relplot(data=x,
                        x=x["time"],
                        y=x["value"],
                        col="species",
                        col_wrap=2,
                        kind="line",
                        facet_kws={"sharey": False})
        #plt.savefig("figures/" + str(name)+"molecules")

        return g

    def plot_all(self, name="", cell_list=None, mol_list=None):
        self.plot_cells(name=name, cell_list=cell_list)
        self.plot_molecules(name=name, mol_list=mol_list)

    def draw_new_params(self, param_names, heterogeneity, use_CV=False):
        """
        deprecated
        """
        assert "params" in dir(self)
        assert all(key in self.params for key in param_names)

        for param in param_names:
            mean = self.params[param]
            if use_CV:
                std = mean * heterogeneity
            else:
                std = heterogeneity
            # std = mean * (heterogeneity / 100.)
            sigma, scale = lognorm_params(mean, std)
            sample = log_pdf.rvs(sigma, 0, scale, size=1)
            self.params[param] = sample[0]

    def compute_cellstates(self):
        pass

    def get_residuals(self, data):
        """

        data : dataframe that contains variance and average values
        data df should contain columns "cell" and "time" to merge to simulation data
        the celltypes in data should correspond to fit_cells
        fit_cells should contain the names of the celltypes to be fitted
        returns residual array used for fitting
        """
        # cells should be data frame with only time and cell
        cells, mols = self.compute_cellstates()
        #print("check data format and if cellstates returns cells and molecules")
        df = pd.merge(cells, data, on=["time", "species"], how="inner")
        # print(df)
        #resid = (df["value_x"].values - df["value_y"].values) / df["eps"].values  # Do we need to take the absolut value here?
        resid = np.log10(df["value_x"].values) - np.log10(df["value_y"].values)
        return resid

    def get_cells(self, mycells: list):
        assert self.cells is not None
        cells = self.cells
        cells = cells.loc[cells["species"].isin(mycells)]
        return cells

    def set_params(self, fit_result : dict):
        for key, val in fit_result.items():
            assert key in fit_result.keys()
            self.params[key] = val
            self.default_params[key] = val

    def store_fit(self, out, fit_name : str):
        # store fit result
        print("adjust path for load and store fit")
        fit_dir = "../../output/fit_results/"

        with open(fit_dir + fit_name + '.p', 'wb') as fit_result:
            pickle.dump(out, fit_result, protocol=pickle.HIGHEST_PROTOCOL)

    def load_fit(self, fit_name : str):
        # set model parameters to fit
        # load fit result
        print("adjust path for load and store fit")
        fit_dir = "../../output/fit_results/"
        with open(fit_dir + fit_name + '.p', 'rb') as fp:
            fit_result = pickle.load(fp)

        print("updating fit result, parameters and default parameters!")
        # checking if dict for fit result was stored (backwards compatibility
        if type(fit_result) != dict:
            self.fit_params = fit_result.params.valuesdict()
            self.fit_report = fit_result
        else:
            self.fit_params = fit_result
        self.set_params(self.fit_params)

        self.default_params = dict(self.params)


    def get_features(self, cell_list):
        """
        get readouts from state array
        """
        cells = self.cells
        cells = cells.loc[cells["species"].isin(cell_list)]
        assert cells is not None

        def myfun(df):
            peak_x, peak_y = self.get_maximum(df.time.array, df.value.array)
            read_names = ["peak_x", "peak_y"]
            reads = [peak_x, peak_y]
            s = pd.Series(reads, index=read_names)
            return s

        # for each cell get x and y data of maximum peak and make tidy
        out = cells.groupby("species").apply(myfun)
        out = out.reset_index().melt(var_name="readout", id_vars="species")

        return out


    def get_best_fit(self, fit_name = "latebranch_fit_local"):
        params = dict(self.params) # store to reset later
        self.load_fit(fit_name)
        cells, mols = self.compute_cellstates(update_species = False)
        self.cells_best_fit = cells.copy()
        self.set_params(params)


    def get_curvediff(self):
        if self.cells_best_fit is None:
            self.get_best_fit()

        assert self.cells is not None
        cells = self.cells
        cells2 = self.cells_best_fit
        # rename best fit simulation
        cells2.rename(columns={"value": "value_bestfit"}, inplace=True)

        out = pd.merge(cells, cells2, how="left", on=["time", "species", "name"])
        out = out.loc[out.species.isin(["Tfh_all"])].copy()

        out.loc[:,"curvediff"] = np.abs((out["value"] - out["value_bestfit"])) / out["value_bestfit"]

        out2 = out.groupby(["species"])["curvediff"].mean().reset_index()
        out2["readout"] = "curvediff"
        out2.rename(columns = {"curvediff" : "value"}, inplace=True)
        return out2

    def get_gradient(self, cell_list):

        cells_rel = self.get_relative_cells()
        cells_rel = cells_rel.loc[cells_rel["species"].isin(cell_list)]
        df = cells_rel.groupby(["species"])["value"].agg(lambda x: np.sum(np.abs(np.diff(x))))
        df = df.reset_index()
        df["readout"] = "heterogeneity_" + df["species"]
        return df

    def get_maximum(self, x, y):
        """
        interpolate maximum
        """

        f = InterpolatedUnivariateSpline(x, y, k=4)
        # get Nullstellen
        cr_pts = f.derivative().roots()
        cr_pts = np.append(cr_pts, (x[0], x[-1]))  # also check the endpoints of the interval
        cr_vals = f(cr_pts)
        max_index = np.argmax(cr_vals)

        max_x = cr_pts[max_index]
        max_y = cr_vals[max_index]

        return max_x, max_y

    def get_celldata(self, timepoints: list, cellnames: list):
        """
        return relative and absolute cell numbers for given timepoints and cells
        """
        assert self.cells is not None
        # check that cellnames provided and timepoints provided are in cells df
        assert self.cells["time"].drop_duplicates().isin(timepoints).sum() == len(timepoints)
        assert self.cells["species"].drop_duplicates().isin(cellnames).sum() == len(cellnames)

        cells_rel = self.get_relative_cells()
        cells = self.cells

        def myfun(cells, readout_name):
            # reduce df to provided timepoints and celltypes
            cells = cells[cells["time"].isin(timepoints)]
            cells = cells[cells["species"].isin(cellnames)]
            cells = cells[["time", "species", "value"]]
            cells["readout"] = readout_name + "_" + cells["species"] + "_day_" + cells["time"].astype(str)
            #cells.drop(columns="time", inplace=True)
            return cells

        readout_names = ["cellnumber", "relative"]
        out = [myfun(x, y) for x, y in zip([cells, cells_rel], readout_names)]
        out = pd.concat(out)

        #readout_names = "cellnumber"
        #out = myfun(cells, readout_names)

        return out

    def get_readouts(self, timepoints, cell_names):
        """
        return dataframe with readouts at given timepoints and for given cells
        """
        reads1 = self.get_features(cell_names)
        reads2 = self.get_celldata(timepoints, cell_names)
        reads3 = self.get_curvediff()
        #reads3 = self.get_gradient(cell_names)
        out = pd.concat([reads1, reads2, reads3])
        #out = reads2
        self.readouts = out
        return out

    def get_relative_cells(self):
        pass

    def norm_init_probs(self):
        pass

    def get_pscan_array(self, pname, res, myrange):
        """
        produce an array for parameter scan ranging across multiple orders of magnitude
        """
        if type(pname) == tuple:
            assert self.params[pname[0]] == self.params[pname[1]]
            myval = self.params[pname[0]]
        else:
            myval = self.params[pname]

        assert myval != 0
        if myrange == "10perc":
            arr = np.linspace(0.9*myval, 1.1*myval, res)
        else:
            arr = np.geomspace(myval * 10 ** (-myrange), myval * 10 ** myrange, res)

        return arr


    def get_lognorm_array(self, pname, res, CV = 0.1, use_SD = False):
        """
        generate lognorm array with given CV
        res: number of parameter values sampled from lognorm
        mean of lognorm taken from current value of simulation object pname

        """
        # need to check if tuple was provided, if yes then parameters need to be the same for this analysis!
        if type(pname) == tuple:
            assert self.params[pname[0]] == self.params[pname[1]]
            myval = self.params[pname[0]]
        else:
            myval = self.params[pname]

        assert myval != 0
        if use_SD:
            std = CV
        else:
            std = myval * CV
        sigma, scale = lognorm_params(myval, std)
        sample = log_pdf.rvs(sigma, 0, scale, size=res)
        return sample


    def set_params_lognorm(self, params, CV = 0.1, use_SD = False):
        """
        take a list of params and shuffle lognorm style
        use_SD says that instead of CV SD is provided
        """
        # make sure that array is provided is using SD as input, otherwise create one
        if use_SD:
            assert len(CV) == len(params)
            CV_arr = CV
        else:
            CV_arr = [CV for _ in params]

        for pname, val in zip(params, CV_arr):
            sample = self.get_lognorm_array(pname, 1, val, use_SD)

            # if pname is tuple set all values to this value
            pname = (pname,) if type(pname) != tuple else pname
            for p in pname:
                self.params[p] = sample[0]


        self.norm_params()


    def norm_params(self):
        probs = np.array([self.params["p1"], self.params["p2"]])
        probs = probs / np.sum(probs)
        self.params["p1"] = probs[0]
        self.params["p2"] = probs[1]

        if self.params["prop_ag"] > 1:
            self.params["prop_ag"] = 1


    def set_params_multivariate_normal(self):
        """
        from fit report draw parameters from multivariate normal dist based on covariance matrix,
        then adjust model params
        """
        assert self.fit_report is not None
        popt = list(self.fit_report.params.valuesdict().values())
        pcov = self.fit_report.covar
        params = np.random.multivariate_normal(popt, pcov, 1)
        params[params<0] = 1e-5
        param_names = list(self.fit_report.params.valuesdict().keys())

        mydict = dict(zip(param_names, params))

        self.set_params(mydict)

    def set_params_uniform(self, params : list, sigma_list : list):
        """
        sigma is standard error from fit output aka 68% conf interval
        """
        for pname, sigma in zip(params, sigma_list):
            if type(pname) == tuple:
                assert self.params[pname[0]] == self.params[pname[1]]
                myval = self.params[pname[0]]
            else:
                myval = self.params[pname]
            assert myval != 0
            lower = (myval-sigma) if (myval-sigma) > 0 else 0
            upper = myval + sigma
            newval = np.random.uniform(lower, upper)

            pname = (pname,) if type(pname) != tuple else pname
            for p in pname:
                self.params[p] = newval

    def global_sensitivtity(self, param_names, fname, infection, res = 100, CV = 0.1,
                            sampling = "CV"):
        """
        shuffle all params lognorm style
        """
        assert sampling in ["random", "CV", "SD"]
        assert all([x in self.params.keys() for x in param_names if type(x) != tuple])
        # if fit ID is provided update parameters accordingly and set fit ID to savename

        mylist = []
        for i in range(res):

            if sampling == "random":
                # in this case, CV should be a list of standard error values for each parameter
                self.set_params_uniform(param_names, CV)
                sname = "uniform/"

            elif sampling == "multivariate":
                self.set_params_multivariate_normal()
                sname = "multivariate/"

            elif sampling == "CV":
                self.set_params_lognorm(param_names, CV, use_SD = False)
                sname = "CV" + str(CV) + "/"

            elif sampling == "SD":
                self.set_params_lognorm(param_names, CV, use_SD = True)
                sname = "SD/"

            self.compute_cellstates()

            out = self.get_readouts(timepoints = np.arange(0.5,70,0.5),
                                    cell_names = ["Th1_all", "Tfh_all"])

            out["run_ID"] = i
            mylist.append(out)
            self.reset_params()

        df = pd.concat(mylist)
        savedir = "../../output/global_sensitivity/" + sname

        df.to_csv(savedir + "mcarlo_global_" + fname + "_" + infection + ".csv", index = False)
        return df

    def run_decision_window(self, t_start_arr, t_dur_arr):
        mylist = []

        for t_start in t_start_arr:
            for t_dur in t_dur_arr:
                self.params["t_start"] = t_start
                self.params["t_end"] = t_start + t_dur
                self.compute_cellstates()
                out = self.get_readouts(cell_names=["Th1_all", "Tfh_all"], timepoints = np.arange(0.5,140.5,0.5))
                out["t_start"] = t_start
                out["t_dur"] = t_dur
                mylist.append(out)

        df = pd.concat(mylist)
        self.reset_params()

        return df

    def run_decision_window_pscan(self, t_start_arr, param_arr, param_name, t_dur):
        mylist = []

        for t_start in t_start_arr:
            for val in param_arr:
                assert param_name in self.params.keys()
                self.params[param_name] = val
                self.params["t_start"] = t_start
                self.params["t_end"] = t_start + t_dur
                self.compute_cellstates()
                out = self.get_readouts(cell_names=["Th1_all", "Tfh_all"], timepoints=[5.0, 9.0, 15.0, 30.0, 60.0], )
                out["t_start"] = t_start
                out[param_name] = val
                mylist.append(out)

        df = pd.concat(mylist)
        self.reset_params()

        return df