import numpy as np
from models.simulation_class import Simulation
import pandas as pd

from scipy import constants
from utils_data_model import get_chain_indexes, diff_chain, diff_chain2, stepfun
from numba import njit


class late_branching(Simulation):
    """
    model simulation class
    initialize with a name (str), mode (model specific)
    params (dict), time to sim (arr) and core(model specific)
    """

    def get_ode_args(self):
        d = self.params
        out = (d["alpha_naive"], d["alpha_prec"], d["alpha_eff"],
               d["beta_naive"], d["beta_eff"], d["prop_ag"],
               d["death_naive"], d["death_prec"], d["death_eff"], d["deg_chr_th1"], d["deg_chr_tfh"], d["death_mem"],
               d["div_naive"], d["div_prec"], d["div_eff"],
               d["r_cyto1"], d["r_cyto2"], d["deg_cyto1"], d["deg_cyto2"],
               d["fb_cyto1"], d["fb_cyto2"],
               d["r_IL2_naive"], d["r_IL2_eff"], d["up_IL2"], d["K_IL2_cons"], d["deg_IL2"],
               d["myc_EC50"], d["deg_myc"],
               d["r_chr"], d["K_ag_chr"], d["K_ag_IL10"], d["K_ag_prob"], d["K_ag_prec"], d["r_mem"],
               d["model"], d["n_tregs"],
               d["r_IL10_t"], d["r_IL10_chr"], d["deg_IL10"],
               d["K_IL10_chr"], d["K_IL10_prob"], d["K_IL10_prec"],
               d["p1"], d["p1_ag"], d["p2"], d["p2_IL10"], d["p3"], d["fb_IL10"],
               d["vir_decay"], d["vir_load"], d["nstage_th1"], d["nstage_tfh"], d["up_IL2_eff"],
               d["hill"], d["fbfc_ag_chr"], d["K_fbfc_ag"],
               d["t_start"], d["t_end"], d["rate_chr1"], d["fb_decwindow"])
        return out

    #  # put this below staticmethod?
    @staticmethod
    @njit
    def ode(state, time,
            alpha_naive, alpha_prec, alpha_eff,
            beta_naive, beta_eff, prop_ag,
            death_naive, death_prec, death_eff, deg_chr_th1, deg_chr_tfh, death_mem,
            div_naive, div_prec, div_eff,
            r_cyto1, r_cyto2, deg_cyto1, deg_cyto2,
            fb_cyto1, fb_cyto2,
            r_IL2_naive, r_IL2_eff, up_IL2, K_IL2_cons, deg_IL2,
            myc_EC50, deg_myc,
            r_chr, K_ag_chr, K_ag_IL10, K_ag_prob, K_ag_prec, r_mem,
            model, n_tregs,
            r_IL10_t, r_IL10_chr, deg_IL10,
            K_IL10_chr, K_IL10_prob, K_IL10_prec,
            p1, p1_ag, p2, p2_IL10, p3, fb_IL10,
            vir_decay, vir_load, nstage_th1, nstage_tfh, up_IL2_eff,
            hill, fbfc_ag_chr, K_fbfc_ag,
            t_start, t_end, rate_chr1, fb_decwindow):
        """
        ode for naive -> prec -> eff1 and eff2
        il_2_const = 3600 * 24
        up_IL2 = up_IL2 * il2_const
        r_IL2_eff = r_IL2_eff * il_2_const
        r_IL2_naive = r_IL2_naive * il_2_const
        """
        naive_idx, prec_idx, eff1_idx, eff2_idx, chr1_idx, chr2_idx = get_chain_indexes(alpha_naive, alpha_prec,
                                                                                        alpha_eff, nstage_th1,
                                                                                        nstage_tfh)

        # split states
        naive = state[:naive_idx]
        prec = state[naive_idx: prec_idx]
        eff1 = state[prec_idx: eff1_idx]
        chr1 = state[eff1_idx: chr1_idx]
        eff1_t = state[chr1_idx]
        eff2 = state[(chr1_idx + 1):eff2_idx]
        chr2 = state[eff2_idx: chr2_idx]
        eff2_t = state[chr2_idx]

        # molecules indexes
        ag = state[-1]
        myc = state[-2]
        cyto2 = state[-3]
        cyto1 = state[-4]
        IL10 = state[-5]
        IL2 = state[-6]

        tfh_mem = state[-7]
        chr2_t = state[-8]
        th1_mem = state[-9]
        chr1_t = state[-10]

        n_th1 = sum(eff1) + eff1_t
        n_tfh = sum(eff2) + eff2_t
        n_chr1 = sum(chr1) + chr1_t

        n_eff = n_th1 + n_tfh
        # antigen
        dt_ag = -vir_decay * ag

        ag_eff = ag ** hill / (ag ** hill + 0.1 ** hill)
        # ODEs molecules
        dt_cyto1 = r_cyto1 * n_th1 - deg_cyto1 * cyto1
        dt_cyto2 = r_cyto2 * n_tfh - deg_cyto2 * cyto2
        dt_myc = -deg_myc * myc

        # IL10: Note: r_IL10_1, r_IL10_2 and r_IL10_t are in (0,1) and no rates!
        IL10_prod = (r_IL10_t * eff1_t + r_IL10_chr * n_chr1)  # * ag_eff
        dt_IL10 = IL10_prod - deg_IL10 * IL10

        IL10_pos = IL10 ** hill / (IL10 ** hill + K_IL10_prob ** hill)
        IFNg_pos = cyto1 ** hill / (cyto1 ** hill + K_IL10_prob ** hill)
        IL21_pos = cyto2 ** hill / (cyto2 ** hill + K_IL10_prob ** hill)

        # could also exclude IL2 from terminal effectors
        # but I dont think it makes a difference
        IL2_prod = (r_IL2_naive * (sum(naive) + sum(prec)) + r_IL2_eff * n_eff)  # *ag_eff

        IL2_conc = IL2 * 1e12 / (20e-6 * constants.N_A)
        il2_eff = IL2_conc ** hill / (IL2_conc ** hill + K_IL2_cons ** hill)

        IL2_cons_tregs = up_IL2 * n_tregs * il2_eff
        IL2_cons_eff = up_IL2_eff * n_eff * il2_eff
        IL2_cons = IL2_cons_tregs + IL2_cons_eff
        dt_IL2 = IL2_prod - IL2_cons - deg_IL2 * IL2

        # probabilities
        p1 = p1 + p1_ag * ag_eff + fb_cyto1 * IFNg_pos
        p2 = p2 + p2_IL10 * IL10_pos + fb_cyto2 * IL21_pos

        # decision window
        fb_eff1 = stepfun(time, hi=fb_decwindow, lo=1, start=t_start, end=t_end)
        p1 = p1 * fb_eff1

        probs = np.array([p1, p2])
        probs = (1 - p3) * probs / sum(probs)

        myc_eff = myc ** hill / (myc ** hill + 0.1 ** hill)
        if model == "Timer_IL2":
            prolif_ag_indep = myc_eff * il2_eff
        elif model == "Timer":
            prolif_ag_indep = myc_eff
        else:
            prolif_ag_indep = il2_eff

        fb_IL10 = (fb_IL10 * IL10 ** hill + K_IL10_prec ** hill) / (IL10 ** hill + K_IL10_prec ** hill)
        beta_p_prec = beta_eff * (prop_ag * ag_eff + (1 - prop_ag) * prolif_ag_indep * fb_IL10)

        # naive values
        influx_naive = 0
        dt_naive = diff_chain(naive, influx_naive, beta_naive, death_naive)

        # precursor values
        influx_prec = naive[-1] * beta_naive + beta_p_prec * np.asarray(p3) * div_prec * prec[-1]

        dt_prec = diff_chain(prec, influx_prec, beta_p_prec, death_prec)

        # effector values
        influx_eff = prec[-1] * beta_p_prec * div_eff

        # chain ODEs
        fb_ag_chr = (fbfc_ag_chr * ag ** hill + K_fbfc_ag ** hill) / (ag ** hill + K_fbfc_ag ** hill)
        rate_chr1 = rate_chr1 * fb_ag_chr

        rate_chr = r_chr * fb_ag_chr

        dt_eff1, influx_chr1 = diff_chain2(eff1, influx_eff * probs[0], beta_eff, death_eff, rate_chr1, nstage_th1)
        dt_eff2, influx_chr2 = diff_chain2(eff2, influx_eff * probs[1], beta_eff, death_eff, rate_chr1, nstage_tfh)

        dt_chr1, _ = diff_chain2(chr1, influx_chr1, beta_eff, death_eff, 0, nstage_th1)
        dt_chr2, _ = diff_chain2(chr2, influx_chr2, beta_eff, death_eff, 0, nstage_tfh)

        dt_eff1_t = beta_eff * eff1[-1] * div_eff - (rate_chr + r_mem + death_eff) * eff1_t
        dt_eff2_t = beta_eff * eff2[-1] * div_eff - (rate_chr + r_mem + death_eff) * eff2_t

        influx_chr1_total = rate_chr * eff1_t + beta_eff * chr1[-1] * div_eff  # influx from terminal effector cells and from early chronic cells
        influx_chr2_total = rate_chr * eff2_t + beta_eff * chr2[-1] * div_eff

        dt_chr1_t = influx_chr1_total - deg_chr_th1 * chr1_t
        dt_chr2_t = influx_chr2_total - deg_chr_tfh * chr2_t

        # memory and chronic phenotypes
        dt_th1_mem = r_mem * eff1_t - death_mem * th1_mem
        dt_tfh_mem = r_mem * eff2_t - death_mem * tfh_mem

        # combine cell states
        dt_state = [
            *dt_naive, *dt_prec, *dt_eff1, *dt_chr1, dt_eff1_t, *dt_eff2, *dt_chr2, dt_eff2_t,
            dt_chr1_t, dt_th1_mem, dt_chr2_t, dt_tfh_mem,
            dt_IL2, dt_IL10, dt_cyto1, dt_cyto2, dt_myc, dt_ag]

        return dt_state

    def init_model(self):
        """set initial conditions for ODE solver"""
        # +3 for myc, cyto1, cyto2
        y0 = np.zeros(self.params["alpha_naive"] +
                      self.params["alpha_prec"] +
                      self.params["alpha_eff"] * self.params["nstage_th1"] * 2 +  # multiply by two because chronic cells have now also a chain
                      1 +  # t1
                      self.params["alpha_eff"] * self.params["nstage_tfh"] * 2 +  # *2 because chronic cells have now also a chain before entering terminal chronic state
                      1 +  # t2
                      10)  # last entry are number of molecules (6 myc, cyto1, cyto2, IL2, IL10, ag) + nb of memory phenotypes (2) + nb chronic phenotype (2)
        y0[0] = self.params["initial_cells"]
        # set myc conc.
        y0[-1] = 1 * self.params["vir_load"]  # ag
        y0[-2] = 1  # myc
        # self.norm_init_probs()

        return y0

    def compute_cellstates(self, update_species=True) -> tuple:
        d = self.params
        state = self.run_model()
        naive_idx, prec_idx, eff1_idx, eff2_idx, chr1_idx, chr2_idx = get_chain_indexes(d["alpha_naive"],
                                                                                        d["alpha_prec"], d["alpha_eff"],
                                                                                        d["nstage_th1"],
                                                                                        d["nstage_tfh"])

        naive = state[:, :naive_idx]
        prec = state[:, naive_idx: prec_idx]
        eff1 = state[:, prec_idx: eff1_idx]
        chr1 = state[:, eff1_idx: chr1_idx]
        eff1_t = state[:, chr1_idx]
        eff2 = state[:, (chr1_idx + 1):eff2_idx]
        chr2 = state[:, eff2_idx: chr2_idx]
        eff2_t = state[:, chr2_idx]

        ag = state[:, -1]
        myc = state[:, -2]
        cyto2 = state[:, -3]
        cyto1 = state[:, -4]
        IL10 = state[:, -5]
        IL2 = state[:, -6] * 1e12 / (20e-6 * constants.N_A)
        tfh_mem = state[:, -7]
        tfh_chr = state[:, -8]
        th1_mem = state[:, -9]
        th1_chr = state[:, -10]

        cell_list = [naive, prec, eff1, eff2, chr1, chr2]

        cell_list = [np.sum(x, axis=1) for x in cell_list]

        cell_list_sums = [cell_list[0], cell_list[1],
                          cell_list[2] + eff1_t,
                          cell_list[3] + eff2_t,
                          th1_chr + cell_list[4],
                          tfh_chr + cell_list[5],
                          th1_mem, tfh_mem]

        cells = np.stack(cell_list_sums, axis=-1)

        molecules = np.stack([cyto1, cyto2, IL2, IL10, myc, ag], axis=-1)

        colnames1 = ["Naive", "Precursor", "Th1", "Tfh", "Th1_c", "Tfh_c", "Th1_mem", "Tfh_mem"]
        colnames2 = ["IFNG", "IL21", "IL2", "IL10", "Myc", "Antigen"]
        cells = self.modify(cells, colnames1, self.name, compute_combinations=True)
        molecules = self.modify(molecules, colnames2, self.name)

        if update_species:
            self.cells = cells
            self.molecules = molecules

        return cells, molecules

    def modify(self, df, colnames, name, compute_combinations=False):
        """
        add some additional celltypes
        """
        df = pd.DataFrame(data=df, columns=colnames)

        # add some celltypes if df is cells
        if compute_combinations:
            df = self.compute_cell_combinations(df)

        df.loc[:, "time"] = self.time
        df = pd.melt(df, id_vars=["time"], var_name="species")
        df.loc[:, "name"] = name
        return df

    def get_relative_cells(self):

        cells = self.cells
        assert cells is not None

        # remove th1 all and tfh all because they are postprocessed mix
        cellstate_combinations = ["Th1_all", "Tfh_all", "CD4_All", "CD4_Chronic", "CD4_Eff", "CD4_Mem"]
        cells = cells.loc[~cells["species"].isin(cellstate_combinations)]

        # for each time point get sum across each celltype

        cells_copy = cells.copy()  ## this is only to get rid of the setting with copy warning...
        cells_copy.loc[:, "total"] = cells_copy.groupby("time").transform("sum").loc[:, "value"]
        cells_copy.loc[:, "value_rel"] = cells_copy.loc[:, "value"].values / cells_copy.loc[:, "total"].values

        cells = cells_copy
        # make wide to compute Th1 all and Tfh all (which I kicked out before)
        cells = cells[["time", "species", "value_rel"]]
        cells = cells.pivot(values="value_rel", columns=["species"], index="time").reset_index()

        cells = self.compute_cell_combinations(cells)
        # make tidy again
        cells = cells.melt(id_vars="time")

        return cells

    @staticmethod
    def compute_cell_combinations(cells):
        cells["Th1_all"] = cells["Th1"] + cells["Th1_c"] + cells["Th1_mem"]  # + 0.5 * cells["Precursor"]
        cells["Tfh_all"] = cells["Tfh"] + cells["Tfh_c"] + cells["Tfh_mem"]  # + 0.5 * cells["Precursor"]

        cells["CD4_Chronic"] = cells["Th1_c"] + cells["Tfh_c"]
        cells["CD4_Eff"] = cells["Th1"] + cells["Tfh"]
        cells["CD4_Mem"] = cells["Th1_mem"] + cells["Tfh_mem"]
        # add Naive to not start with 0 at time 0
        cells["CD4_All"] = cells["Naive"] + cells["CD4_Chronic"] + cells["CD4_Eff"] + cells["CD4_Mem"] + cells[
            "Precursor"]
        cells["Tfh_rel"] = cells["Tfh_all"] / (cells["Tfh_all"] + cells["Th1_all"])
        cells["Tfh_rel_fit"] = cells["Tfh_rel"] * cells["CD4_All"]
        return cells
