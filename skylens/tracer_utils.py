"""
Class with utility functions for defining tracer properties, e.g. redshift kernels, galaxy bias, IA amplitude etc.
"""

import os, sys
import copy
from skylens import *
from skylens.utils import *
from skylens.cosmology import *
import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import interp1d
from scipy.integrate import quad as scipy_int1d
from scipy.special import loggamma
from dask import delayed
from dask.distributed import Client, get_client

d2r = jnp.pi / 180.0


class Tracer_utils:
    def __init__(
        self,
        shear_zbins=None,
        galaxy_zbins=None,
        kappa_zbins=None,
        logger=None,
        l=None,
        scheduler_info=None,
        zkernel_func_names=None,
    ):
        self.logger = logger
        self.l = l
        # Gravitaional const to get Rho crit in right units
        self.G2 = G.to(u.Mpc / u.Msun * u.km**2 / u.second**2)
        self.G2 *= 8 * jnp.pi / 3.0

        self.SN = {}
        self.scheduler_info = scheduler_info
        self.n_bins = {}
        self.z_bins = {}
        self.spin = {"galaxy": 0, "kappa": 0, "shear": 2}

        self.set_zbins(z_bins=shear_zbins, tracer="shear")
        self.set_zbins(z_bins=galaxy_zbins, tracer="galaxy")
        self.set_zbins(z_bins=kappa_zbins, tracer="kappa")
        self.tracers = list(self.z_bins.keys())
        if not self.tracers == []:
            print("Tracer utils has tracers: ", self.tracers)
            self.set_z_PS_max()
        else:
            print("Tracer utils has no tracers")
        self.set_z_window()
        self.scatter_z_bins()
        self.zkernel_func_names = zkernel_func_names
        self.set_zkernel_funcs()

    def set_zkernel_funcs(
        self,
    ):
        if self.zkernel_func_names is None:
            self.zkernel_func_names = {}  # we assume it is a dict below.
        self.zkernel_func = {}
        for tracer in self.tracers:
            if self.zkernel_func_names.get(tracer) is None:
                self.zkernel_func_names[tracer] = "set_kernel"

            if self.zkernel_func.get(tracer) is None:
                self.zkernel_func[tracer] = globals()[self.zkernel_func_names[tracer]]
            if not callable(self.zkernel_func[tracer]):
                raise Exception(self.zkernel_func[tracer], "is not a callable function")

    def set_z_PS_max(self):
        """
        Get max z for power spectra computations.
        This is used by Ang_PS to generate z at which
        to compute the power spectra, p(k,z).
        """
        self.z_PS_max = 0
        z_max_all = jnp.array([self.z_bins[tracer]["zmax"] for tracer in self.tracers])
        self.z_PS_max = jnp.amax(z_max_all).item() + 0.01

    def set_zbins(self, z_bins={}, tracer=None):
        """
        Set tracer z_bins in the class z_bins property
        """
        if z_bins is not None:
            self.z_bins[tracer] = copy.deepcopy(z_bins)
            self.n_bins[tracer] = self.z_bins[tracer]["n_bins"]

    def scatter_z_bins(self):
        """
        Scatter the z bins into the dask distributed memory
        """
        if self.scheduler_info is None:
            return
        for tracer in self.tracers:
            nb = self.z_bins[tracer][
                "n_bins"
            ]  # We need this sometimes outisde of dask futures
            self.z_bins[tracer] = scatter_dict(
                self.z_bins[tracer], scheduler_info=self.scheduler_info
            )
            self.z_bins[tracer]["n_bins"] = nb

    def gather_z_bins(self):
        """
        Gather the z bins from the dask distributed memory, back into the main process
        """
        client = client_get(self.scheduler_info)
        for tracer in self.tracers:
            self.z_bins[tracer] = client.gather(self.z_bins[tracer])
            if self.z_win is not None:
                self.z_win[tracer] = client.gather(self.z_win[tracer])

    def set_z_window(
        self,
    ):
        """
        Read in the windows and set a separate dictionary for windows only.
        This used by the window_utils to compute coupling matrices, etc.
        Removed after the window_utils is done.
        """
        self.z_win = {}
        if self.scheduler_info is not None:
            client = get_client(address=self.scheduler_info["address"])
        for tracer in self.tracers:
            self.z_win[tracer] = {}
            for i in np.arange(self.n_bins[tracer]):
                self.z_win[tracer][i] = {}
                for k in self.z_bins[tracer][i].keys():
                    if "window" in k:
                        self.z_win[tracer][i][k] = self.z_bins[tracer][i][k]
        if self.scheduler_info is not None:
            self.z_win = scatter_dict(
                self.z_win, scheduler_info=self.scheduler_info, broadcast=True
            )

    def clean_z_window(
        self,
    ):
        """
        Remove the windows dictionary once coupling matrices are done and
        it is no longer required. This saves memory, especially if
        dask tries to make several copies of the class.
        """
        if self.scheduler_info is not None:
            client = get_client(address=self.scheduler_info["address"])
            for tracer in self.tracers:
                pass
        self.z_win = None

    def get_z_bins(self, tracer=None):
        """
        return z_bins for a tracer.
        """
        return self.z_bins[tracer]

    def reset_z_bins(self):
        """
        Reset cosmology dependent values for each source bin
        """
        for tracer in self.tracers:
            for i in np.arange(self.z_bins[tracer]["n_bins"]):
                self.z_bins[tracer][i]["kernel"] = None
                self.z_bins[tracer][i]["kernel_int"] = None

    def set_kernels(self, Ang_PS=None, tracer=None, z_bins=None, delayed_compute=True):
        """
        Set the tracer kernels. This includes the local kernel, e.g. galaxy density, IA and also the lensing
        kernel. Galaxies have magnification bias.
        """
        if z_bins is None:
            z_bins = self.get_z_bins(tracer=tracer)
            if not delayed_compute:
                z_bins = gather_dict(z_bins, scheduler_info=self.scheduler_info)
        n_bins = z_bins["n_bins"]
        kernel = {}
        for i in np.arange(n_bins):
            if delayed_compute:
                kernel[i] = delayed(self.zkernel_func[tracer])(
                    Ang_PS=Ang_PS, tracer=tracer, z_bin=z_bins[i]
                )
            else:
                kernel[i] = self.zkernel_func[tracer](
                    Ang_PS=Ang_PS, tracer=tracer, z_bin=z_bins[i]
                )
        return kernel


def set_kernel(Ang_PS=None, tracer=None, z_bin=None):
    """
    Set the tracer kernels. This includes the local kernel, e.g. galaxy density, IA and also the lensing
    kernel. Galaxies have magnification bias.
    """
    #     print('set_kernel')
    kernel = {}
    kernel = set_lensing_kernel(
        Ang_PS=Ang_PS, tracer=tracer, z_bin=z_bin, kernel=kernel
    )
    kernel = set_galaxy_kernel(Ang_PS=Ang_PS, tracer=tracer, z_bin=z_bin, kernel=kernel)
    kernel["kernel_int"] = kernel["Gkernel_int"] + kernel["gkernel_int"]
    #     del kernel['Gkernel_int'],kernel['gkernel_int']
    return kernel


def sigma_crit(zl=[], zs=[], cosmo_h=None):
    """
    Inverse of lensing kernel.
    """
    ds = cosmo_h.comoving_transverse_distance(zs)
    dl = cosmo_h.comoving_transverse_distance(zl)
    ddls = 1.0 - jnp.outer(1.0 / ds, dl)  # (ds-dl)/ds
    x = ds == 0
    # ddls[x,:]=0
    ddls = ddls.at[x, :].set(0)
    w = sigma_crit_norm100 * (1 + zl) * dl
    sigma_c = 1.0 / (ddls * w)
    x = ddls <= 0  # zs<zl
    # sigma_c[x]=jnp.inf
    sigma_c = sigma_c.at[x].set(jnp.inf)
    return sigma_c  # .value


def NLA_amp_z(l, z=[], z_bin={}, cosmo_h=None):
    """
    Redshift dependent intrinsic alignment amplitude. This is assumed to be a function of ell in general,
    though here we set it as constant.
    """
    AI = z_bin["AI"]
    AI_z = z_bin["AI_z"]
    return jnp.outer(
        AI * (1 + z) ** AI_z, jnp.ones_like(l)
    )  # FIXME: This might need to change to account


def constant_bias(l, z=[], z_bin={}, cosmo_h=None):
    """
    Galaxy bias, assumed to be constant (ell and z independent).
    """
    b = z_bin["b1"]
    #         lb_m=z_bin['lm']
    lm = jnp.ones_like(l)
    #         x=self.l>lb_m  #if masking out modes based on kmax. lm is attribute of the z_bins, that is based on kmax.
    #         lm[x]=0
    return b * jnp.outer(jnp.ones_like(z), lm)


def linear_bias_z(l, z=[], z_bin={}, cosmo_h=None):
    """
    linear Galaxy bias, assumed to be constant in ell and specified at every z.
    """
    b = jnp.interp(
        z, z_bin["z"], z_bin["bz1"], left=0, right=0
    )  # this is linear interpolation

    lb_m = z_bin["lm"]
    lm = jnp.ones_like(l)
    #         x=self.l>lb_m  #if masking out modes based on kmax. lm is attribute of the z_bins, that is based on kmax.
    #         lm[x]=0
    return jnp.outer(b, lm)


def linear_bias_powerlaw(l, z_bin={}, cosmo_h=None):
    """
    Galaxy bias, assumed to be constant (ell independent). Varies as powerlaw with redshift. This is useful
    for some photo-z tests.
    """
    b1 = z_bin["b1"]
    b2 = z_bin["b2"]
    lb_m = z_bin["lm"]
    lm = jnp.ones_like(l) + l / l[-1]
    #         x=self.l>lb_m
    #         lm[x]=0
    return jnp.outer(
        b1 * (1 + z_bin["z"]) ** b2, lm
    )  # FIXME: This might need to change to account


def spin_factor(l, tracer=None):
    """
    Spin of tracers. Needed for wigner transforms and pseudo-cl calculations.
    """
    if tracer is None:
        return jnp.nan
    if tracer == "galaxy":
        s = 0
        return jnp.ones_like(l, dtype="float32")

    if tracer == "shear":
        s = 2  # there is (-1)**s factor, so sign is same of +/- 2 spin
    if tracer == "kappa":
        s = 1  # see Kilbinger+ 2017

    F = loggamma(l + s + 1)
    F -= loggamma(l - s + 1)
    F = jnp.exp(
        1.0 / s * F
    )  # units should be l**2, hence sqrt for s=2... comes from angular derivative of the potential
    F /= (
        l + 0.5
    ) ** 2  # when writing potential to delta_m, we get 1/k**2, which then results in (l+0.5)**2 factor
    x = l - s < 0
    # F[x]=0
    F = F.at[x].set(0)

    return F


def set_lensing_kernel(Ang_PS=None, tracer=None, z_bin=None, kernel=None):
    """
    Compute rho/Sigma_crit for each source bin at every lens redshift where power spectra is computed.
    cosmo_h: cosmology to compute Sigma_crit
    """
    #     kernel=z_bin
    #     rho=Rho_crit(cosmo_h=cosmo_h)*cosmo_h.Om0
    cosmo_h = Ang_PS.PS  # .cosmo_h
    zl = Ang_PS.z
    l = Ang_PS.l

    rho = Rho_crit100 * cosmo_h.Om
    mag_fact = 1
    spin_tracer = tracer
    #     for i in np.arange(n_bins):
    if tracer == "galaxy":
        mag_fact = z_bin["mag_fact"]
        spin_tracer = "kappa"
    spin_fact = spin_factor(l, tracer=spin_tracer)
    kernel["Gkernel"] = (
        mag_fact * rho / sigma_crit(zl=zl, zs=z_bin["z"], cosmo_h=cosmo_h)
    )
    kernel["Gkernel_int"] = jnp.dot(z_bin["pzdz"], kernel["Gkernel"])
    kernel["Gkernel_int"] /= z_bin["Norm"]
    kernel["Gkernel_int"] *= z_bin["shear_m_bias"]
    if z_bin["Norm"] == 0:  # FIXME
        kernel["Gkernel_int"][:] = 0
    kernel["Gkernel_int"] = jnp.outer(spin_fact, kernel["Gkernel_int"])
    #     del kernel['Gkernel']
    return kernel


def set_galaxy_kernel(Ang_PS=None, tracer=None, l=None, z_bin=None, kernel=None):
    """
    set the galaxy position kernel (also for IA). This is function is form bias(z)*dn/dz*f(\chi).
    """
    cosmo_h = Ang_PS.PS  # .cosmo_h
    zl = Ang_PS.z
    l = Ang_PS.l

    IA_const = 0.0134 * cosmo_h.Om
    b_const = 1

    if (
        tracer == "shear" or tracer == "kappa"
    ):  # kappa maps can have AI. For CMB, set AI=0 in the z_bin properties.
        bias_func = NLA_amp_z
        b_const = IA_const
    if tracer == "galaxy":
        bias_func_t = z_bin.get("bias_func")
        bias_func = constant_bias if bias_func_t is None else globals()[bias_func_t]
        if z_bin["bz1"] is not None:
            bias_func = linear_bias_z if bias_func_t is None else globals()[bias_func_t]
    #             bias_func=self.constant_bias #FIXME: Make it flexible to get other bias functions

    spin_fact = spin_factor(l, tracer=tracer)

    dzl = jnp.gradient(zl)
    cH = cosmo_h.Dh / cosmo_h.efunc(zl)
    cH = cH
    kernel["gkernel"] = b_const * bias_func(l, z=zl, z_bin=z_bin, cosmo_h=cosmo_h)
    kernel["gkernel"] = (
        kernel["gkernel"].T / cH
    ).T  # cH factor is for conversion of n(z) to n(\chi).. n(z)dz=n(\chi)d\chi
    kernel["gkernel"] *= spin_fact

    if len(z_bin["pz"]) == 1:  # interpolation doesnot work well when only 1 point
        bb = jnp.digitize(z_bin["z"], zl)

        pz_zl = jnp.zeros_like(zl)
        if bb < len(pz_zl):
            pz_zl[bb] = z_bin["pz"]  # assign to nearest zl
            pz_zl /= jnp.sum(pz_zl * dzl)
    else:
        pz_zl = jnp.interp(
            zl, z_bin["z"], z_bin["pz"], left=0, right=0
        )  # this is linear interpolation
        if not jnp.sum(pz_zl * dzl) == 0:  # FIXME
            pz_zl /= jnp.sum(pz_zl * dzl)
        else:
            print("Apparently empty bin", zl, z_bin["z"], z_bin["pz"])

    kernel["gkernel_int"] = kernel["gkernel"].T * pz_zl  # dzl multiplied later
    kernel["gkernel_int"] /= jnp.sum(pz_zl * dzl)

    if jnp.sum(pz_zl * dzl) == 0:  # FIXME
        kernel["gkernel_int"][:] = 0
    del kernel["gkernel"]
    return kernel
