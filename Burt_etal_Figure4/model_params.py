from scipy.constants import N_A

myparams = dict(alpha_naive=15,  # Polonsky et al (I think two parameterizations existed, one where we took the estimated
                # from the paper directly and one were the data were refitted by us
                beta_naive=9.7,  # Polonsky et al, avg division time first division for alpha = 10, conv to days
                death_naive=0,  # Proserpio et al
                alpha_prec=7,  # Polonsky et al --> need to do the fitting again
                K_ag_prec=0.5,
                K_IL10_prec=0.5,
                death_prec=0.24,  # Proserpio et al
                # proliferation
                alpha_eff=7,  # polonsky
                beta_eff=15.2,  # Polonsky et al, 11.2h avg division time was measured in vitro, convert to days * alpha
                death_eff=0.24,
                # molecules
                r_cyto1=1,  # not used
                r_cyto2=1,  # not used
                div_eff= 2,  # helper param
                div_naive=0,  # helper param
                div_prec=2,  # helper param
                deg_myc=0.26,  # estimated
                deg_cyto1=1,  # not used
                deg_cyto2=1,  # not used
                K_ag_IL10=0.5,
                r_IL10_t=1,
                r_IL10_chr=100,
                deg_IL10=5000000,  # test
                r_IL2_naive=10 * 3600 * 24,
                r_IL2_eff=150 * 3600 * 24,
                K_IL2_cons=5, # * N_A * 20e-6 * 10e-12,
                deg_IL2=2.8e-5,
                up_IL2= 1*3600 * 24,
                n_tregs=1e5,
                # feedbacks
                fb_cyto1=0,  #
                fb_cyto2=0,  #
                fb_decwindow = 1,
                myc_EC50=0.1,  # estimated
                # probabilities
                p1=0.6,  # test
                p1_ag=0.5,
                K_ag_prob=0.5,
                p2=0.4,  # test
                p2_IL10=10,
                K_IL10_prob=2,
                p3=0.9,
                # memory cells
                r_mem=0.005,  # model fit
                death_mem=0.009,  # taken from loehning paper memory clel halflive is around 75 days
                r_chr=0.005,  # test
                K_ag_chr=0.5,
                K_IL10_chr=2,
                deg_chr_th1=0.24,# assume that it is mainly the precursor pop that is maintained
                deg_chr_tfh=0.24, # assume that it is mainly the precursor pop that is maintained
                # model design: vir load feedback switches between 0 and 1
                vir_load=1,
                vir_decay = 0.26,
                model="Timer_IL2",
                # init conditions
                initial_cells=3000,
                prop_ag = 0.15,
                fb_IL10=0.3,
                nstage_th1 = 4,
                nstage_tfh = 4,
                up_IL2_eff = 1*3600*24,
                hill = 3,
                fbfc_ag_chr = 500,
                K_fbfc_ag = 3,
                t_start=1000,
                t_end=1200,
                rate_chr1 = 0,)

myparams_SSM = dict(myparams)
myparams_SSM["alpha_naive"] = 1
myparams_SSM["beta_naive"] = myparams["beta_naive"] / myparams["alpha_naive"]
myparams_SSM["alpha_prec"] = 1
myparams_SSM["alpha_eff"] = 1
myparams_SSM["beta_eff"] = myparams["beta_eff"] / myparams["alpha_eff"]

#fit_lb = "latebranch_fit_global"
fit_lb = "latebranch_fit_local"
data_params_lb = ["beta_eff", "beta_naive", ("death_eff", "deg_chr_th1", "deg_chr_tfh"), "death_mem"]

fit_params_lb = ["p1",
                 "p2",
                 "fb_IL10",
                 "r_mem",
                 "r_chr",
                 "initial_cells",
                 "prop_ag"]

fit_names = ["lb_fit_fitparams_latechr", "lb_fit_fitparams_earlychr",
             "lb_fit_allparams_latechr", "lb_fit_allparams_earlychr"]

fit_names_local = [x + "_local" for x in fit_names]
fit_names_global = [x + "_global" for x in fit_names]