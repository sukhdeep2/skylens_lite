import os, sys

sys.path.append("/verafs/scratch/phy200040p/sukhdeep/project/skylens/skylens/")
import dask
import numpy as np
import jax.numpy as jnp
import numpy as np
import warnings, logging
import copy
import multiprocessing, psutil
import sparse
import gc
import dask.bag
from dask import delayed

from skylens.power_spectra import *
from skylens.angular_power_spectra import *
from skylens.wigner_transform import *
from skylens.binning import *
from skylens.tracer_utils import *
from skylens.window_utils import *
from skylens.utils import *
from skylens.parse_input import *

d2r = jnp.pi / 180.0


class Skylens:
    def __init__(
        self,
        use_defaults=True,
        yaml_inp_file=None,
        python_inp_file=None,
        l=None,
        l_cl=None,
        l_bins=None,
        l_bins_center=None,
        bin_cl=None,
        use_binned_l=None,
        do_pseudo_cl=None,
        stack_data=None,
        bin_xi=None,
        do_xi=None,
        theta_bins=None,
        theta_bins_center=None,
        use_binned_theta=None,
        WT_kwargs=None,
        WT=None,
        cl_func_names=None,
        zkernel_func_names=None,
        shear_zbins=None,
        kappa_zbins=None,
        galaxy_zbins=None,
        Ang_PS=None,
        logger=None,
        tracer_utils=None,
        pk_params=None,
        cosmo_params=None,
        z_PS=None,
        nz_PS=None,
        log_z_PS=None,
        use_window=None,
        window_lmax=None,
        window_l=None,
        store_win=None,
        Win=None,
        wigner_files=None,
        name="",
        clean_tracer_window=None,
        f_sky=None,
        wigner_step=None,
        corrs=None,
        corr_indxs=None,
        stack_indxs=None,
        scheduler_info=None,
        njobs_submit_per_worker=None,
    ):

        # self.__dict__.update(locals()) #assign all input args to the class as properties
        self.use_defaults = use_defaults
        if yaml_inp_file is not None:
            inp_args = parse_yaml(file_name=yaml_inp_file, use_defaults=use_defaults)
            self.__dict__.update(inp_args)
        elif python_inp_file is not None:
            inp_args = parse_python(
                file_name=python_inp_file, use_defaults=use_defaults
            )
            self.__dict__.update(inp_args)
        else:
            inp_args = parse_dict(dic=locals(), use_defaults=use_defaults)
            self.__dict__.update(inp_args)
            # del inp_args
        self.l0=self.l*1. #in case of imaster, l is changed to effective bin ell. l0 store original ell in case needed
        if self.l_cl is None: #TODO: cl and pseudo-cl can have different binning and ell ranges. l_cl should store ell for cl calculations and the default ell should be for pseudo-cl.
            self.l_cl=self.l 
        self.l_cl0=self.l_cl*1.
        
        self.set_WT()
        self.set_bin_params()
        self.set_binned_measure(inp_args)  # (locals())
        del inp_args

        if logger is None:  # not really being used right now
            self.logger = logging.getLogger()
            self.logger.setLevel(level=logging.DEBUG)
            logging.basicConfig(
                format="%(asctime)s %(levelname)s:%(message)s",
                level=logging.DEBUG,
                datefmt="%I:%M:%S",
            )

        if tracer_utils is None:
            self.tracer_utils = Tracer_utils(
                shear_zbins=self.shear_zbins,
                galaxy_zbins=self.galaxy_zbins,
                kappa_zbins=self.kappa_zbins,
                logger=self.logger,
                l=self.l_cl,
                scheduler_info=self.scheduler_info,
                zkernel_func_names=self.zkernel_func_names,
            )

        self.set_corr_indxs(corr_indxs=self.corr_indxs, stack_indxs=self.stack_indxs)

        self.window_lmax = 30 if window_lmax is None else window_lmax
        self.window_l = (
            np.arange(self.window_lmax + 1) if self.window_l is None else self.window_l
        )

        self.set_WT_spins()

        self.z_bins = self.tracer_utils.z_bins

        if Ang_PS is None:
            self.Ang_PS = Angular_power_spectra(
                l=self.l_cl,
                logger=self.logger,
                window_l=self.window_l,
                z_PS=self.z_PS,
                nz_PS=self.nz_PS,
                log_z_PS=self.log_z_PS,
                z_PS_max=self.tracer_utils.z_PS_max,
                pk_params=self.pk_params,
                cosmo_params=self.cosmo_params,
            )
            # FIXME: Need a dict for these args
        self.set_cl_funcs()
        
        self.Win0=window_utils(window_l=self.window_l,l=self.l0,l_bins=self.l_bins,l_cl=self.l_cl0,
                               corrs=self.corrs,s1_s2s=self.s1_s2s,scheduler_info=self.scheduler_info,
                               use_window=self.use_window, corr_indxs=self.stack_indxs,z_bins=self.tracer_utils.z_win,
                               window_lmax=self.window_lmax,Win=self.Win,WT=self.WT,do_xi=self.do_xi,
                               do_pseudo_cl=self.do_pseudo_cl,wigner_step=self.wigner_step,
                               skylens_class0=self.skylens0,skylens_class_b=self.skylens_b,skylens_b_xi=self.skylens_b_xi,
                               xi_bin_utils=self.xi_bin_utils,store_win=self.store_win,wigner_files=self.wigner_files,
                               bin_window=self.use_binned_l,bin_theta_window=self.use_binned_theta,
                               njobs_submit_per_worker=self.njobs_submit_per_worker)
        self.Win0.get_Win()
        self.bin_window = self.Win0.bin_window
        win = self.Win0.Win
        self.Win = win
        self.scatter_win()
        self.set_WT_binned()
        self.set_binned_measure(None, clean_up=True)
        workers, self.nworkers, self.njobs_submit = get_workers_njobs(
            self.scheduler_info, self.njobs_submit_per_worker
        )

        if clean_tracer_window:
            self.tracer_utils.clean_z_window()

        print("Window done. Size:", get_size_pickle(self.Win))
        self.clean_setup()

    def clean_setup(self):
        """
        remove some objects from self that are not required after setup.
        """
        atrs = [
            "kappa_zbins",
            "galaxy_zbins",
            "shear_zbins",
            "WT_kwargs",
            "pk_params",
            "cosmo_params",
        ]
        for atr in atrs:
            if hasattr(self, atr):
                delattr(self, atr)

    def clean_non_essential(self):
        """
        remove some objects from that may not be required after some calculations.
        """
        atrs = ["z_bins"]
        for atr in atrs:
            if hasattr(self, atr):
                delattr(self, atr)
            if hasattr(self.tracer_utils, atr):
                delattr(self.tracer_utils, atr)

    def set_cl_funcs(
        self,
    ):
        if self.cl_func_names is None:
            self.cl_func_names = {}  # we assume it is a dict below.
        self.cl_func = {}
        for corr in self.corrs:
            #            self.cl_func[corr]=self.calc_cl
            if self.cl_func_names.get(corr) is None:
                if self.cl_func_names.get(corr[::-1]) is None:
                    self.cl_func_names[corr] = "calc_cl"
                    self.cl_func_names[corr[::-1]] = "calc_cl"
                else:
                    self.cl_func_names[corr] = self.cl_func_names[corr[::-1]]
            if self.cl_func.get(corr) is None:
                #                 if hasattr(self,self.cl_func_names[corr]):
                #                     self.cl_func[corr]=getattr(self,self.cl_func_names[corr])
                #                 elif hasattr(self.Ang_PS.PS,self.cl_func_names[corr]):
                #                     self.cl_func[corr]=getattr(self.Ang_PS.PS,self.cl_func_names[corr])
                #                 else:
                self.cl_func[corr] = globals()[self.cl_func_names[corr]]
            if not callable(self.cl_func[corr]):
                raise Exception(self.cl_func[corr], "is not a callable function")
            self.cl_func[corr[::-1]] = self.cl_func[corr]

    def set_binned_measure(
        self, local_args, clean_up=False
    ):  # FIXME: needs cleanup, refactoring
        """
        If we only want to run computations at effective bin centers, then we
        need to bin the windows and wigner matrices properly, for which unbinned
        quantities need to be computed once. This function sets up the unbinned
        computations, which are used later for binning window coupling and wigner
        matrices.
        This is useful when running multiple computations for chains etc. For
        covariance and one time calcs, may as well just do the full computation.
        """
        if clean_up:
            if self.use_binned_l or self.use_binned_theta:
                del self.skylens0, self.skylens_b, self.c_ell0, self.c_ell_b
            return
        if self.use_binned_l or self.use_binned_theta:
            inp_args = {}
            client = client_get(self.scheduler_info)
            for k in local_args.keys():
                if (
                    k == "self"
                    or k == "client"
                    or "yaml" in k
                    or "python" in k
                    or "Win" in k
                ):
                    continue
                inp_args[k] = copy.deepcopy(
                    self.__dict__[k]
                )  # when passing yaml, most of input_args are updated. use updated ones
            if self.l_bins_center is None:
                self.l_bins_center = jnp.int32(
                    (self.l_bins[1:] + self.l_bins[:-1]) * 0.5
                )
            inp_args["use_binned_l"] = False
            inp_args["use_binned_theta"] = False
            inp_args["use_window"] = False
            inp_args["bin_xi"] = False
            inp_args["name"] = "S0"

            inp_args2 = copy.deepcopy(inp_args)

            self.skylens0 = Skylens(**inp_args)  # to get unbinned c_ell and xi
            #             self.skylens0.bin_xi=False #we want to get xi_bin_utils

            inp_args2["l"] = self.l_bins_center
            inp_args2["l_cl"] = self.l_bins_center
            inp_args2["name"] = "S_b"
            inp_args2["l_bins"] = None
            inp_args2["bin_cl"] = False
            inp_args2["do_xi"] = False
            self.skylens_b = Skylens(**inp_args2)  # to get binned c_ell
            self.skylens_b_xi = None
            if self.do_xi and self.use_binned_theta:
                theta_bins = inp_args["theta_bins"]
                if self.theta_bins_center is None:
                    self.theta_bins_center = (
                        theta_bins[1:] + theta_bins[:-1]
                    ) * 0.5  # FIXME:this may not be effective theta of meaurements
                inp_args_xi = copy.deepcopy(inp_args)
                inp_args_xi["name"] = "S_b_xi"
                inp_args_xi["bin_xi"] = True
                inp_args_xi["do_pseudo_cl"] = False
                inp_args_xi["use_window"] = self.use_window
                inp_args_xi["Win"] = self.Win
                #                 inp_args_xi['WT'].reset_theta_l(theta=self.theta_bins_center)#FIXME
                self.skylens_b_xi = Skylens(**inp_args_xi)  # to get binned xi.

                self.xi0 = self.skylens0.xi_tomo()["xi"]
                self.xi_b = self.skylens_b_xi.xi_tomo()["xi"]
            self.l = self.l_bins_center * 1.0
            self.l_cl = self.l_bins_center * 1.0
            self.c_ell0 = self.skylens0.cl_tomo()["cl"]
            self.c_ell_b = self.skylens_b.cl_tomo()["cl"]
            print("set binned measure done")
        else:
            self.skylens_b = self
            self.skylens0 = self
            self.skylens_b_xi = None

    def set_corr_indxs(self, corr_indxs=None, stack_indxs=None):
        """
        set up the indexes for correlations. indexes= tracer and bin ids.
        User can input the corr_indxs which will be the ones computed (called stack_indxs later).
        However, when doing covariances, we may need to compute the
        aiddtional correlations, hence those are included added to the corr_indxs.
        corr_indxs are used for constructing full compute graph but only the stack_indxs
        is actually computed when stack_dat is called.
        """
        self.stack_indxs = stack_indxs
        self.corr_indxs = corr_indxs

        if self.corrs is None:
            if self.stack_indxs is not None:
                self.corrs = list(self.stack_indxs.keys())
            else:
                nt = len(self.tracer_utils.tracers)
                self.corrs = [
                    (self.tracer_utils.tracers[i], self.tracer_utils.tracers[j])
                    for i in np.arange(nt)
                    for j in np.arange(i, nt)
                ]

        if self.corr_indxs is None:
            self.corr_indxs = self.stack_indxs if self.stack_indxs else {}
        else:
            print("not setting corr_indxs", bool(self.corr_indxs))
            return

        for tracer in self.tracer_utils.tracers:
            self.corr_indxs[(tracer, tracer)] = [
                j
                for j in itertools.combinations_with_replacement(
                    np.arange(self.tracer_utils.n_bins[tracer]), 2
                )
            ]

        for tracer1 in self.tracer_utils.tracers:  # zbin-indexs for cross correlations
            for tracer2 in self.tracer_utils.tracers:
                if tracer1 == tracer2:  # already set above
                    continue
                if self.corr_indxs.get((tracer1, tracer2)) is not None:
                    continue
                self.corr_indxs[(tracer1, tracer2)] = [
                    k
                    for l in [
                        [(i, j) for i in np.arange(self.tracer_utils.n_bins[tracer1])]
                        for j in np.arange(self.tracer_utils.n_bins[tracer2])
                    ]
                    for k in l
                ]

        if self.stack_indxs is None:
            self.stack_indxs = self.corr_indxs

    def set_WT_spins(self):
        """
        set the spin factors for tracer pairs, used for wigner transforms.
        """
        self.s1_s2s = {}
        for tracer1 in self.tracer_utils.tracers:  # zbin-indexs for cross correlations
            for tracer2 in self.tracer_utils.tracers:
                self.s1_s2s[(tracer1, tracer2)] = [
                    (self.tracer_utils.spin[tracer1], self.tracer_utils.spin[tracer2])
                ]
        if "shear" in self.tracer_utils.tracers:
            self.s1_s2s[("shear", "shear")] = [(2, 2), (2, -2)]
        self.s1_s2s[("window")] = [(0, 0)]

    def set_WT(self):
        """
        Setup wigner transform based on input args, if not transform
        class is not passed directly.
        """
        if self.WT is not None or self.WT_kwargs is None or not self.do_xi:
            return
        self.WT = wigner_transform(**self.WT_kwargs)

    def set_WT_binned(self):
        """
        If we only want to compute at bin centers, wigner transform matrices need to be binned.
        """
        if self.WT is None:
            return
        client = client_get(self.scheduler_info)
        #         self.WT.scatter_data(scheduler_info=self.scheduler_info)
        WT = self.WT
        self.WT_binned = {corr: {} for corr in self.corrs}  # intialized later.
        self.inv_WT_binned = {corr: {} for corr in self.corrs}  # intialized later.
        self.WT_binned_cov = {corr: {} for corr in self.corrs}

        if self.use_binned_theta or self.use_binned_l:
            WT.set_binned_theta(theta_bins=self.theta_bins)
            WT.set_binned_l(l_bins=self.l_bins)
        if self.do_xi and (self.use_binned_l or self.use_binned_theta):
            for corr in self.corrs:
                s1_s2s = self.s1_s2s[corr]
                self.WT_binned[corr] = {s1_s2s[im]: {} for im in np.arange(len(s1_s2s))}
                self.inv_WT_binned[corr] = {
                    s1_s2s[im]: {} for im in np.arange(len(s1_s2s))
                }
                self.WT_binned_cov[corr] = {
                    s1_s2s[im]: {} for im in np.arange(len(s1_s2s))
                }
                for indxs in self.corr_indxs[corr]:
                    cl0 = client.compute(self.c_ell0[corr][indxs]).result()
                    cl_b = client.compute(self.c_ell_b[corr][indxs]).result()
                    wt0 = cl0
                    wt_b = 1.0 / cl_b
                    if jnp.all(cl_b == 0):
                        wt_b[:] = 0
                    for im in np.arange(len(s1_s2s)):
                        s1_s2 = s1_s2s[im]
                        win_xi = None
                        if self.use_window:  # and self.xi_win_approx:
                            win_xi = client.gather(
                                self.Win["cl"][corr][indxs]
                            )  # ['xi']#this will not work for covariance
                            win_xi = win_xi["xi"]
                        self.WT_binned[corr][s1_s2][indxs] = delayed(
                            self.binning.bin_2d_WT
                        )(
                            wig_mat=self.WT.wig_d[s1_s2],
                            wig_norm=self.WT.wig_norm,
                            wt0=wt0,
                            wt_b=wt_b,
                            bin_utils_cl=self.cl_bin_utils,
                            bin_utils_xi=self.xi_bin_utils[s1_s2],
                            win_xi=win_xi,
                            use_binned_theta=self.use_binned_theta,
                        )
                        self.WT_binned[corr][s1_s2][indxs] = client.compute(
                            self.WT_binned[corr][s1_s2][indxs]
                        ).result()
                        self.WT_binned[corr][s1_s2][indxs] = client.scatter(
                            self.WT_binned[corr][s1_s2][indxs]
                        )

                        if self.do_xi and self.use_binned_theta:
                            xi0 = client.compute(self.xi0[corr][s1_s2][indxs]).result()
                            xi_b = client.compute(
                                self.xi_b[corr][s1_s2][indxs]
                            ).result()
                            wt0_inv = xi0
                            wt_b_inv = 1.0 / xi_b
                            if jnp.all(xi_b == 0):
                                wt_b_inv[:] = 0
                            self.inv_WT_binned[corr][s1_s2][indxs] = delayed(
                                self.binning.bin_2d_inv_WT
                            )(
                                wig_mat=self.WT.wig_d[s1_s2],
                                wig_norm=self.WT.inv_wig_norm,
                                wt0=wt0_inv,
                                wt_b=wt_b_inv,
                                bin_utils_cl=self.cl_bin_utils,
                                bin_utils_xi=self.xi_bin_utils[s1_s2],
                                win_xi=None,
                                use_binned_l=self.use_binned_l,
                            )
                            self.inv_WT_binned[corr][s1_s2][indxs] = client.compute(
                                self.inv_WT_binned[corr][s1_s2][indxs]
                            ).result()
                            self.inv_WT_binned[corr][s1_s2][indxs] = client.scatter(
                                self.inv_WT_binned[corr][s1_s2][indxs]
                            )

    def update_zbins(self, z_bins={}, tracer="shear"):
        """
        If the tracer bins need to be updated. Ex. when running chains with varying photo-z params.
        """
        self.tracer_utils.set_zbins(z_bins, tracer=tracer)
        self.z_bins = self.tracer_utils.z_bins
        return

    def set_bin_params(self):
        """
        Setting up the binning functions to be used in binning the data
        """
        self.binning = binning()
        self.cl_bin_utils = None
        client = client_get(self.scheduler_info)
        if self.bin_cl or self.use_binned_l:
            self.cl_bin_utils = self.binning.bin_utils(
                r=self.l0, r_bins=self.l_bins, r_dim=2, mat_dims=[1, 2]
            )
            #             self.cl_bin_utils={k:client.scatter(self.cl_bin_utils[k]) for k in self.cl_bin_utils.keys()}
            self.cl_bin_utils = scatter_dict(
                self.cl_bin_utils, scheduler_info=self.scheduler_info, broadcast=True
            )
        self.xi_bin_utils = None
        if self.do_xi and self.bin_xi:
            self.xi_bin_utils = {}
            for s1_s2 in self.WT.s1_s2s:
                self.xi_bin_utils[s1_s2] = delayed(self.binning.bin_utils)(
                    r=self.WT.theta_deg[s1_s2],
                    r_bins=self.theta_bins,
                    r_dim=2,
                    mat_dims=[1, 2],
                )
                self.xi_bin_utils[s1_s2] = client.compute(
                    self.xi_bin_utils[s1_s2]
                ).result()
                self.xi_bin_utils[s1_s2] = scatter_dict(
                    self.xi_bin_utils[s1_s2],
                    scheduler_info=self.scheduler_info,
                    broadcast=True,
                )

    def calc_cl(
        self,
        zbin1={},
        zbin2={},
        corr=("shear", "shear"),
        cosmo_params=None,
        Ang_PS=None,
    ):  # FIXME: this can be moved outside the class.thenwe don't need to serialize self.
        """
        Compute the angular power spectra, Cl between two source bins
        zs1, zs2: Source bins. Dicts containing information about the source bins
        """
        clz = Ang_PS.clz
        cls = clz["cls"]
        f = Ang_PS.cl_f
        sc = zbin1["kernel_int"] * zbin1["kernel_int"]
        dchi = clz["dchi"]
        cl = jnp.dot(cls.T * sc, dchi)
        # cl*=2./jnp.pi #FIXME: needed to match camb... but not CCL
        return cl

    def bin_cl_func(
        self, cl=None, cov=None
    ):  # moved out of class. This is no longer used
        """
        bins the tomographic power spectra
        results: Either cl or covariance
        bin_cl: if true, then results has cl to be binned
        bin_cov: if true, then results has cov to be binned
        Both bin_cl and bin_cov can be true simulatenously.
        """
        cl_b = None
        if not cl is None:
            if self.use_binned_l or not self.bin_cl:
                cl_b = cl * 1.0
            else:
                cl_b = self.binning.bin_1d(xi=cl, bin_utils=self.cl_bin_utils)
            return cl_b

    def calc_pseudo_cl(self,cl,Win):# moved outside the class. Not used now.
        pcl=cl@Win['M']
        return  pcl
        
    def cl_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,
                corrs=None,bias_kwargs={},bias_func=None,stack_corr_indxs=None,
                z_bins=None,Ang_PS=None,stack_file_write=None):#FIXME: redundant with tomo short.
        """
        Computes full tomographic power spectra and covariance, including shape noise. output is
        binned also if needed.
        Arguments are for the power spectra  and sigma_crit computation,
        if it needs to be called from here.
        source bins are already set. This function does set the sigma crit for sources.
        """

        l = self.l
        if corrs is None:
            corrs = self.corrs
        if stack_corr_indxs is None:
            stack_corr_indxs = self.stack_indxs
        if Ang_PS is None:
            Ang_PS = self.Ang_PS
        if z_bins is None:
            z_bins = self.z_bins
        #         if z_params is None:
        #             z_params=z_bins
        client = client_get(self.scheduler_info)
        tracers = np.unique([j for i in corrs for j in i])

        corrs2 = corrs.copy()

        Ang_PS.angular_power_z(
            cosmo_h=cosmo_h, pk_params=pk_params, cosmo_params=cosmo_params
        )

        if cosmo_h is None:
            cosmo_h = Ang_PS.PS  # .cosmo_h

        zkernel = {}
        AP = client.scatter(Ang_PS, broadcast=True)
        for tracer in tracers:
            zkernel[tracer] = self.tracer_utils.set_kernels(
                Ang_PS=AP, tracer=tracer, z_bins=z_bins[tracer], delayed_compute=True
            )
            if "galaxy" in tracers:
                if bias_func is None:
                    bias_func = "constant_bias"
                    bias_kwargs = {"b1": 1, "b2": 1}

        zkernel = client.compute(zkernel).result()
        zkernel = scatter_dict(
            zkernel, scheduler_info=self.scheduler_info, broadcast=True
        )

        cosmo_params = scatter_dict(
            cosmo_params, scheduler_info=self.scheduler_info, broadcast=True
        )

        out = {}
        cl = {corr: {} for corr in corrs2}
        cl.update({corr[::-1]: {} for corr in corrs2})
        pcl = {corr: {} for corr in corrs2}
        pcl.update({corr[::-1]: {} for corr in corrs2})  # pseudo_cl
        cl_b = {corr: {} for corr in corrs2}
        cl_b.update({corr[::-1]: {} for corr in corrs2})
        pcl_b = {corr: {} for corr in corrs2}
        pcl_b.update({corr[::-1]: {} for corr in corrs2})
        cov = {}

        print("cl_tomo, Win:", self.Win)

        for corr in corrs2:
            corr2 = corr[::-1]
            corr_indxs = self.corr_indxs[(corr[0], corr[1])]  # +self.cov_indxs
            for (
                i,
                j,
            ) in (
                corr_indxs
            ):  # FIXME: we might want to move to map, like covariance. will be useful to define the tuples in forzenset then.
                cl[corr][(i, j)] = delayed(self.cl_func[corr])(
                    zbin1=zkernel[corr[0]][i],
                    zbin2=zkernel[corr[1]][j],
                    corr=corr,
                    cosmo_params=cosmo_params,
                    Ang_PS=AP,
                )
                cl_b[corr][(i, j)] = delayed(bin_cl_func)(
                    cl=cl[corr][(i, j)],
                    use_binned_l=self.use_binned_l,
                    bin_cl=self.bin_cl,
                    cl_bin_utils=self.cl_bin_utils,
                )
                if (
                    self.use_window
                    and self.do_pseudo_cl
                    and (i, j) in self.stack_indxs[corr]
                ):
                    if not self.bin_window:
                        pcl[corr][(i, j)] = delayed(calc_pseudo_cl)(
                            cl[corr][(i, j)], Win=self.Win["cl"][corr][(i, j)]
                        )
                        pcl_b[corr][(i, j)] = delayed(bin_cl_func)(
                            cl=pcl[corr][(i, j)],
                            use_binned_l=self.use_binned_l,
                            bin_cl=self.bin_cl,
                            cl_bin_utils=self.cl_bin_utils,
                        )
                    else:
                        pcl[corr][(i, j)] = None
                        pcl_b[corr][(i, j)] = delayed(calc_cl_pseudo_cl)(
                            zbin1=zkernel[corr[0]][i],
                            zbin2=zkernel[corr[1]][j],
                            corr=corr,
                            cosmo_params=cosmo_params,
                            Ang_PS=AP,
                            Win=self.Win["cl"][corr][(i, j)],
                        )
                #                         pcl_b[corr][(i,j)]=delayed(calc_pseudo_cl)(None,cl_b[corr][(i,j)],Win=self.Win.Win['cl'][corr][(i,j)])
                else:
                    pcl[corr][(i, j)] = cl[corr][(i, j)]
                    pcl_b[corr][(i, j)] = cl_b[corr][(i, j)]
                cl[corr2][(j, i)] = cl[corr][
                    (i, j)
                ]  # useful in gaussian covariance calculation.
                pcl[corr2][(j, i)] = pcl[corr][
                    (i, j)
                ]  # useful in gaussian covariance calculation.
                cl_b[corr2][(j, i)] = cl_b[corr][
                    (i, j)
                ]  # useful in gaussian covariance calculation.
                pcl_b[corr2][(j, i)] = pcl_b[corr][
                    (i, j)
                ]  # useful in gaussian covariance calculation.

        print("cl graph done")
        cosmo_params = gather_dict(cosmo_params, scheduler_info=self.scheduler_info)

        out_stack = delayed(self.stack_dat)(
            {"pcl_b": pcl_b, "est": "pcl_b"},
            corrs=corrs,
            corr_indxs=stack_corr_indxs,
            stack_file_write=stack_file_write,
        )
        return {
            "stack": out_stack,
            "cl_b": cl_b,
            "cl": cl,
            "pseudo_cl": pcl,
            "pseudo_cl_b": pcl_b,
            "zkernel": zkernel,
        }  # ,'clz':clz}

    def gather_data(self):
        client = client_get(self.scheduler_info)
        keys = ["xi_bin_utils", "cl_bin_utils", "Win", "WT_binned", "z_bins", "SN"]
        for k in keys:
            if hasattr(self, k):
                self.__dict__[k] = gather_dict(
                    self.__dict__[k], scheduler_info=self.scheduler_info
                )
        self.Ang_PS.clz = client.gather(self.Ang_PS.clz)
        self.tracer_utils.gather_z_bins()
        if self.WT is not None:
            self.WT.gather_data()

    def scatter_win(self):
        if self.Win is None or not isinstance(self.Win, dict):
            return
        print("scattering window. Size: ", get_size_pickle(self.Win))

        self.Win["cl"] = scatter_dict(
            self.Win["cl"], scheduler_info=self.scheduler_info, depth=0, broadcast=True
        )
        return

    def scatter_data(self):
        client = client_get(self.scheduler_info)
        keys = ["xi_bin_utils", "cl_bin_utils", "WT_binned"]
        for k in keys:
            if hasattr(self, k):
                self.__dict__[k] = scatter_dict(
                    self.__dict__[k],
                    scheduler_info=self.scheduler_info,
                    depth=1,
                    broadcast=True,
                )
        self.scatter_win()
        if self.WT is not None:
            self.WT.scatter_data()

    def tomo_short(
        self,
        cosmo_h=None,
        cosmo_params=None,
        pk_lock=None,
        WT_binned=None,
        WT=None,
        corrs=None,
        bias_kwargs={},
        bias_func=None,
        stack_corr_indxs=None,
        z_bins=None,
        Ang_PS=None,
        zkernel=None,
        Win=None,
        cl_bin_utils=None,
        xi_bin_utils=None,
        pk_params=None,
    ):
        """ """
        if corrs is None:
            corrs = self.corrs
        if stack_corr_indxs is None:
            stack_corr_indxs = self.stack_indxs

        tracers = np.unique([j for i in corrs for j in i])

        Ang_PS.angular_power_z(
            cosmo_h=cosmo_h,
            pk_params=pk_params,
            pk_lock=pk_lock,
            cosmo_params=cosmo_params,
        )
        if cosmo_h is None:
            cosmo_h = Ang_PS.PS  # .cosmo_h

        if zkernel is None:
            zkernel = {}
            for tracer in tracers:
                zkernel[tracer] = self.tracer_utils.set_kernels(
                    Ang_PS=Ang_PS,
                    tracer=tracer,
                    z_bins=z_bins[tracer],
                    delayed_compute=False,
                )
                if "galaxy" in tracers:
                    if bias_func is None:
                        bias_func = "constant_bias"
                        bias_kwargs = {"b1": 1, "b2": 1}
        if self.do_xi:
            return self.xi_tomo_short(
                corrs=corrs,
                stack_corr_indxs=stack_corr_indxs,
                zkernel=zkernel,
                Ang_PS=Ang_PS,
                Win=Win,
                WT_binned=WT_binned,
                WT=WT,
                xi_bin_utils=xi_bin_utils,
            )
        else:
            return self.cl_tomo_short(corrs=corrs,stack_corr_indxs=stack_corr_indxs,
                                      zkernel=zkernel,Ang_PS=Ang_PS,Win=Win,cl_bin_utils=cl_bin_utils)
    
    def cl_tomo_short(self,corrs=None,stack_corr_indxs=None,Ang_PS=None,zkernel=None,
                        cosmo_params=None,Win=None,cl_bin_utils=None):
        """
        Same as cl_tomo, except no delayed is used and it only returns a stacked vector of binned pseudo-cl.
        This function is useful for mcmc where we only need to compute pseudo-cl, and want to reduce the
        dask overheard. You should run a parallel mcmc, where each call to this function is placed inside
        delayed.
        """
        if Win is None:
            Win = self.Win
        l = self.l
        out = {}
        pcl_b = []
        for corr in corrs:
            corr_indxs = stack_corr_indxs[corr]  # +self.cov_indxs
            for (
                i,
                j,
            ) in (
                corr_indxs
            ):  # FIXME: we might want to move to map, like covariance. will be useful to define the tuples in forzenset then.
                if self.use_window and self.do_pseudo_cl:
                    if not self.bin_window:
                        cl = self.cl_func[corr](
                            zbin1=zkernel[corr[0]][i],
                            zbin2=zkernel[corr[1]][j],
                            corr=corr,
                            cosmo_params=cosmo_params,
                            Ang_PS=Ang_PS,
                        )  # clz=Ang_PS.clz)

                        pcl = calc_pseudo_cl(cl, Win=Win["cl"][corr][(i, j)])
                        pcl_b += [
                            bin_cl_func(
                                cl=pcl,
                                use_binned_l=self.use_binned_l,
                                bin_cl=self.bin_cl,
                                cl_bin_utils=cl_bin_utils,
                            )
                        ]
                    else:
                        pcl = None
                        pcl_b += [
                            calc_cl_pseudo_cl(
                                zbin1=zkernel[corr[0]][i],
                                zbin2=zkernel[corr[1]][j],
                                corr=corr,
                                cosmo_params=cosmo_params,
                                Ang_PS=Ang_PS,
                                Win=Win["cl"][corr][(i, j)],
                            )
                        ]
                else:
                    pcl = self.cl_func[corr](
                        zbin1=zkernel[corr[0]][i],
                        zbin2=zkernel[corr[1]][j],
                        corr=corr,
                        cosmo_params=cosmo_params,
                        Ang_PS=Ang_PS,
                    )
                    pcl_b += [
                        bin_cl_func(
                            cl=pcl,
                            use_binned_l=self.use_binned_l,
                            bin_cl=self.bin_cl,
                            cl_bin_utils=self.cl_bin_utils,
                        )
                    ]
        pcl_b = jnp.concatenate(pcl_b).ravel()
        return pcl_b

    def get_xi(self, cls={}, s1_s2=[], corr=None, indxs=None, Win=None):
        cl = cls[corr][indxs]  # this should be pseudo-cl when using window
        wig_m = None
        if self.use_binned_l or self.use_binned_theta:
            wig_m = self.WT_binned[corr][s1_s2][indxs]
        th, xi = self.WT.projected_correlation(
            l_cl=self.l, s1_s2=s1_s2, cl=cl, wig_d=wig_m
        )
        xi_b = xi

        if (
            self.bin_xi and not self.use_binned_theta
        ):  # wig_d is binned when use_binned_l
            if self.use_window:
                xi = xi * Win["xi"]
            xi_b = self.binning.bin_1d(xi=xi, bin_utils=self.xi_bin_utils[s1_s2])

        if self.use_window:  # and self.xi_win_approx:
            xi_b /= Win["xi_b"]
        return xi_b

    def xi_tomo(
        self,
        cosmo_h=None,
        cosmo_params=None,
        pk_params=None,
        corrs=None,
        bias_kwargs={},
        bias_func=None,
        stack_corr_indxs=None,
        z_bins=None,
        Ang_PS=None,
        stack_file_write=None,
    ):  # FIXME: redundant with tomo short.
        """
        Computed tomographic angular correlation functions. First calls the tomographic
        power spectra and covariance and then does the hankel transform and  binning.
        """
        """
            For hankel transform is done on l-theta grid, which is based on s1_s2. So grid is
            different for xi+ and xi-.
            In the init function, we combined the ell arrays for all s1_s2. This is not a problem
            except for the case of SSV, where we will use l_cut to only select the relevant values
        """

        if cosmo_h is None:
            cosmo_h = self.Ang_PS.PS  # .cosmo_h
        if corrs is None:
            corrs = self.corrs
        if stack_corr_indxs is None:
            stack_corr_indxs = self.stack_indxs

        # Donot use delayed here. Leads to error/repeated calculations
        cls_tomo_nu = self.cl_tomo(
            cosmo_h=cosmo_h,
            cosmo_params=cosmo_params,
            pk_params=pk_params,
            corrs=corrs,
            bias_kwargs=bias_kwargs,
            bias_func=bias_func,
            stack_corr_indxs=stack_corr_indxs,
            z_bins=z_bins,
            Ang_PS=Ang_PS,
            stack_file_write=stack_file_write,
        )

        cl = cls_tomo_nu["cl"]  # Note that if window is turned off, pseudo_cl=cl
        #         clz=cls_tomo_nu['clz']
        cov_xi = {}
        xi = {}
        out = {}
        zkernel = None
        client = client_get(self.scheduler_info)
        AP = client.scatter(self.Ang_PS, broadcast=True)
        if self.use_binned_theta:
            wig_norm = 1
            wig_l = self.WT.l_bins_center
            wig_grad_l = self.WT.grad_l_bins
            wig_theta = self.WT.theta_bins_center
            wig_grad_theta = self.WT.grad_theta_bins
        else:
            wig_norm = self.WT.wig_norm
            wig_l = self.WT.l
            wig_grad_l = self.WT.grad_l
            wig_theta = self.WT.theta[(0, 0)]
            wig_grad_theta = 1

        for corr in corrs:
            s1_s2s = self.s1_s2s[corr]
            xi[corr] = {}
            xi[corr[::-1]] = {}
            for im in np.arange(len(s1_s2s)):
                s1_s2 = s1_s2s[im]
                xi[corr][s1_s2] = {}
                xi[corr[::-1]][s1_s2] = {}
                xi_bin_utils = None
                if not self.use_binned_theta:
                    wig_d = self.WT.wig_d[s1_s2]
                if self.bin_xi:
                    xi_bin_utils = self.xi_bin_utils[s1_s2]
                for indx in self.corr_indxs[corr]:
                    if self.Win is None:
                        win = None
                    else:
                        win = self.Win["cl"][corr][indx]
                    if self.use_binned_theta:
                        wig_d = self.WT_binned[corr][s1_s2][indx]

                    #                     xi[corr][s1_s2][indx]=delayed(self.get_xi)(cls=cl,corr=corr,indxs=indx,
                    #                                                         s1_s2=s1_s2,Win=win)
                    xi[corr][s1_s2][indx] = delayed(get_xi)(
                        cl=cl[corr][indx],
                        wig_d=wig_d,
                        wig_norm=wig_norm,
                        xi_bin_utils=xi_bin_utils,
                        bin_xi=self.bin_xi,
                        use_binned_theta=self.use_binned_theta,
                        Win=win,
                    )
                    xi[corr[::-1]][s1_s2][indx[::-1]] = xi[corr][s1_s2][
                        indx
                    ]  # FIXME: s1_s2 should be reversed as well?...

        print("Done xi graph")

        out["stack"] = delayed(self.stack_dat)(
            {"xi": xi, "est": "xi"}, corrs=corrs, stack_file_write=stack_file_write
        )
        out["xi"] = xi
        out["cl"] = cls_tomo_nu
        out["zkernel"] = zkernel
        return out

    def xi_tomo_short(
        self,
        corrs=None,
        stack_corr_indxs=None,
        Ang_PS=None,
        zkernel=None,
        cosmo_params=None,
        Win=None,
        WT_binned=None,
        WT=None,
        xi_bin_utils=None,
    ):
        """
        Same as xi_tomo / cl_tomo_short, except no delayed is used and it only returns a stacked vector of binned xi.
        This function is useful for mcmc where we only need to compute xi, and want to reduce the
        dask overheard. You should run a parallel mcmc, where each call to this function is placed inside
        delayed.
        """
        if Win is None:
            Win = self.Win
        if Ang_PS is None:
            Ang_PS = self.Ang_PS
        l = self.l
        cl = {corr: {} for corr in corrs}
        out = {}
        for corr in corrs:
            corr_indxs = stack_corr_indxs[corr]  # +self.cov_indxs
            for (
                i,
                j,
            ) in (
                corr_indxs
            ):  # FIXME: we might want to move to map, like covariance. will be useful to define the tuples in forzenset then.
                cl[corr][(i, j)] = self.cl_func[corr](
                    zbin1=zkernel[corr[0]][i],
                    zbin2=zkernel[corr[1]][j],
                    corr=corr,
                    cosmo_params=cosmo_params,
                    Ang_PS=Ang_PS,
                )

        xi_b = []
        if self.use_binned_theta:
            wig_norm = 1
            wig_l = WT.l_bins_center
            wig_grad_l = WT.grad_l_bins
        else:
            wig_norm = WT.norm * WT.grad_l
            wig_l = WT.l
            wig_grad_l = WT.grad_l
        for corr in corrs:
            s1_s2s = self.s1_s2s[corr]
            for im in np.arange(len(s1_s2s)):
                s1_s2 = s1_s2s[im]
                if not self.use_binned_theta:
                    wig_d = WT.wig_d[s1_s2]
                for indx in stack_corr_indxs[corr]:
                    if self.use_binned_theta:
                        wig_d = WT_binned[corr][s1_s2][indx]
                    win = None
                    if self.use_window:
                        win = Win["cl"][corr][indx]

                    xi = get_xi(
                        cl=cl[corr][indx],
                        wig_d=wig_d,
                        wig_norm=wig_norm,
                        xi_bin_utils=xi_bin_utils[s1_s2],
                        bin_xi=self.bin_xi,
                        use_binned_theta=self.use_binned_theta,
                        Win=win,
                    )
                    #                     xi_bi=bin_xi_func(xi=xi,xi_bin_utils=self.xi_bin_utils[s1_s2],bin_xi=self.bin_xi,use_binned_theta=self.use_binned_theta,Win=Win)
                    xi_b += [xi]  # [xi_bi]
        xi_b = jnp.concatenate(xi_b).ravel()
        return xi_b

    def stack_dat(self, dat, corrs, corr_indxs=None, stack_file_write=None):
        """
        outputs from tomographic caluclations are dictionaries.
        This fucntion stacks them such that the cl or xi is a long
        1-d array and the covariance is N X N array.
        dat: output from tomographic calculations.
        XXX: reason that outputs tomographic bins are distionaries is that
        it make is easier to
        handle things such as binning, hankel transforms etc. We will keep this structure for now.
        """

        if corr_indxs is None:
            corr_indxs = self.stack_indxs

        est = dat["est"]
        if est == "xi":
            if self.bin_xi:
                len_bins = len(self.theta_bins) - 1
            else:
                k = list(self.WT.theta.keys())[0]
                len_bins = len(self.WT.theta[k])
        else:
            # est='cl_b'
            if self.bin_cl:
                len_bins = len(self.l_bins) - 1
            else:
                len_bins = len(self.l)

        n_bins = 0
        for corr in corrs:
            n_s1_s2 = 1
            if est == "xi":
                n_s1_s2 = len(self.s1_s2s[corr])
            n_bins += (
                len(corr_indxs[corr]) * n_s1_s2
            )  # jnp.int64(nbins*(nbins-1.)/2.+nbins)
        D_final = jnp.zeros(n_bins * len_bins)
        i = 0
        for corr in corrs:
            n_s1_s2 = 1
            if est == "xi":
                s1_s2 = self.s1_s2s[corr]
                n_s1_s2 = len(s1_s2)

            for im in np.arange(n_s1_s2):
                if est == "xi":
                    dat_c = dat[est][corr][s1_s2[im]]
                else:
                    dat_c = dat[est][
                        corr
                    ]  # [corr] #cl_b gets keys twice. dask won't allow standard dict merge.. should be fixed

                for indx in corr_indxs[corr]:
                    D_final = D_final.at[i * len_bins : (i + 1) * len_bins].set(
                        dat_c[indx]
                    )
                    # D_final[i*len_bins:(i+1)*len_bins]=dat_c[indx]
                    #                     if not jnp.all(jnp.isfinite(dat_c[indx])):
                    #                         print('stack data not finite at',corr,indx,dat_c[indx])
                    i += 1
        print("stack got 2pt")
        out = {}
        out[est] = D_final
        return out


def calc_cl(
    zbin1={}, zbin2={}, corr=("shear", "shear"), cosmo_params=None, Ang_PS=None
):
    """
    Compute the angular power spectra, Cl between two source bins
    zs1, zs2: Source bins. Dicts containing information about the source bins
    """
    clz = Ang_PS.clz
    cls = clz["cls"]
    f = clz["cl_f"]
    sc = zbin1["kernel_int"] * zbin2["kernel_int"]
    dchi = clz["dchi"]
    cl = jnp.dot(cls.T * sc, dchi)
    #     cl/=f**2 #accounted for in kernel
    # cl*=2./jnp.pi #FIXME: needed to match camb... but not CCL
    return cl


def calc_pseudo_cl(cl, Win):
    pcl = cl @ Win["M"]
    return pcl


def calc_cl_pseudo_cl(
    zbin1={},
    zbin2={},
    corr=("shear", "shear"),
    cosmo_params=None,
    Ang_PS=None,
    Win=None,
):  # FIXME: this can be moved outside the class.thenwe don't need to serialize self.
    """
    Combine calc_cl and calc_pseudo_cl functions
    """
    clz = Ang_PS.clz
    cls = clz["cls"]
    f = clz["cl_f"]
    sc = zbin1["kernel_int"] * zbin2["kernel_int"]
    dchi = clz["dchi"]
    cl = jnp.dot(cls.T * sc, dchi)
    pcl = cl @ Win["M"]
    # cl*=2./jnp.pi #FIXME: needed to match camb... but not CCL
    return pcl


def bin_cl_func(cl, use_binned_l=False, bin_cl=False, cl_bin_utils=None):
    """
    bins the tomographic power spectra
    results: Either cl or covariance
    bin_cl: if true, then results has cl to be binned
    bin_cov: if true, then results has cov to be binned
    Both bin_cl and bin_cov can be true simulatenously.
    """
    cl_b = None
    if use_binned_l or not bin_cl:
        cl_b = cl * 1.0
    else:
        cl_b = bin_1d(xi=cl, bin_utils=cl_bin_utils)
    return cl_b


def get_xi(
    cl=None,
    wig_d=None,
    cl_kwargs={},
    wig_norm=1,
    xi_bin_utils=None,
    bin_xi=None,
    use_binned_theta=None,
    Win=None,
):

    xi = projected_correlation(cl=cl, wig_d=wig_d, norm=wig_norm)
    xib = bin_xi_func(
        xi=xi,
        Win=Win,
        xi_bin_utils=xi_bin_utils,
        bin_xi=bin_xi,
        use_binned_theta=use_binned_theta,
    )

    return xib


def bin_xi_func(
    xi=[], Win=None, xi_bin_utils=None, bin_xi=True, use_binned_theta=False
):
    xi_b = xi
    if (
        bin_xi and not use_binned_theta
    ):  # wig_d is binned when use_binned_l#FIXME: Need window correction when use_binned_l
        if Win is not None:
            xi = xi * Win["xi"]
        xi_b = bin_1d(xi=xi, bin_utils=xi_bin_utils)

    if Win is not None:  # win is applied when binning wig_d in case of use_binned_theta
        xi_b /= Win["xi_b"]
    return xi_b
