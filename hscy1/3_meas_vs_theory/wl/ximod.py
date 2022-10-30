import camb
import numpy as np
from . import fftlog
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as ius

class Interp1d(object):
    def __init__(self, angle, spec, bounds_error=False):
        """Makes an interpolation of a function
        """
        if np.all(spec > 0):
            self.interp_func = interp1d(np.log(angle), np.log(
                spec), bounds_error=bounds_error, fill_value=-np.inf)
            self.interp_type = 'loglog'
            self.x_func = np.log
            self.y_func = np.exp
        elif np.all(spec < 0):
            self.interp_func = interp1d(
                np.log(angle), np.log(-spec), bounds_error=bounds_error, fill_value=-np.inf)
            self.interp_type = 'minus_loglog'
            self.x_func = np.log
            self.y_func = lambda y: -np.exp(y)
        else:
            self.interp_func = interp1d(
                np.log(angle), spec, bounds_error=bounds_error, fill_value=0.)
            self.interp_type = "log_ang"
            self.x_func = np.log
            self.y_func = lambda y: y

    def __call__(self, angle):
        interp_vals = self.x_func(angle)
        try:
            spec = self.y_func( self.interp_func(interp_vals) )
        except ValueError:
            interp_vals[0] *= 1+1.e-9
            interp_vals[-1] *= 1-1.e-9
            spec = self.y_func( self.interp_func(interp_vals) )
        return spec

def Extrap1d(x,y):
    """Makes an extrapolation of a function
    """
    def func(x_new):
        if isinstance(x_new, (int, float)):
            x_new = np.atleast_1d(x_new)
        ans = np.zeros(x_new.size)
        # smaller x
        sel = x_new < x.min()
        ans[sel] = np.exp( np.log(y[1]/y[0])/np.log(x[1]/x[0]) * np.log(x_new[sel]/x[0]) + np.log(y[0]) )
        # larger x
        sel = x_new > x.max()
        ans[sel] = np.exp( np.log(y[-2]/y[-1])/np.log(x[-2]/x[-1]) * np.log(x_new[sel]/x[-1]) + np.log(y[-1]) )
        # x in between
        sel = np.logical_and(x.min()<= x_new, x_new <= x.max())
        ans[sel] = np.exp(ius(np.log(x),np.log(y))(np.log(x_new[sel])))
        return ans
    return func

def efficiency_q_over_chi(zs, pzs, cosmo, form='func', zl_max=None, zl_bin=800):
    """Gets lensing efficiency.

    Args:
        zs (ndarray):           source redshift
        pzs (ndarray):          source redshift distribution
        cosmo (cosmo):          astropy cosmology instance
    Returns:
        q_over_chi (ndarray):   lensing efficiency divided by comoving
                                distance at lens redshift.
    """
    if zl_max is None:
        zl_max = zs.max()
    zl = np.linspace(1e-4, zl_max, zl_bin)
    chil = cosmo.comoving_distance(zl).value
    H0 = cosmo.H0.value/3e5 # [1/Mpc]
    prefactor = 3.0/2.0*cosmo.Om0*H0**2*(1+zl)

    # integrate of nz
    # pzs_norm = simps(pzs, zs)
    pzs_norm = np.sum(pzs)
    chis = cosmo.comoving_distance(zs).value

    q = []
    for _chil in chil:
        integrand = _chil*(chis-_chil)/chis*pzs/pzs_norm
        integrand[integrand<0] = 0.0
        q.append(np.sum(integrand))
    q = prefactor*np.array(q)

    if form == 'func':
        return ius(chil, q/chil, ext=3)
    elif form == 'arr':
        return q/chil


def get_pk(l, chil, pkfunc_list):
    """Reads pkfunc_list and give P(l+0.5/chil, zl).

    Args:
        l (ndarray):        array of ells
        chil (ndarray):     array of comoving distance [Mpc]
        pkfunc_list (list): an array of power spectrum at redshift planes
    Returns:
        out (ndarray):      an array of power spectrum k=l+0.5/chil
    """
    _v = l+0.5
    pk = []
    for i, _chil in enumerate(chil):
        _pk = pkfunc_list[i](_v/_chil)[0]
        pk.append(_pk)
    out =   np.array(pk)
    return out

def get_pkfunc(zl, k, linear_model, h, halofit, do_corr_shell=False):
    """Computes nonlinear matter power spectrum from a given linear matter
    power spectrum.

    Args:
        zl (ndarray):           lens redshift
        k  (ndarray):           k bin on which pklin0 is computed.
        linear_model (object):  linear model instance
        h (float):              hubble parameter [100km/Mpc/sec]
        halofit (object):       halofit instance
        do_corr_shell (bool):   whether do shell thickness correction
    Returns:
        pkfunc_list (list):     list of inter/extra-polated function of
                                nonlinear matter power spectrum.
    """
    if do_corr_shell:
        c1,c2 = 9.5171e-4, 5.1543e-3
        alpha1, alpha2, alpha3 = 1.3063, 1.1475, 0.62793
        cor_shell   =   (1+c1*k**-alpha1)**alpha1/(1.+c2*k**-alpha2)**alpha3
    else:
        cor_shell   =   1.
    pkfunc_list = []
    for _zl in zl:
        pklin = linear_model.get_pklin_array_from_z(k, _zl)
        halofit.set_pklin(k, pklin, _zl)
        pkhalofit = halofit.get_pkhalo()
        pkfunc_list.append(Extrap1d(k*h, cor_shell*pkhalofit/h**3))
    return pkfunc_list

def angular_power_spectrum(l, cosmo, linear_model,
                           q1_over_chi_func, q2_over_chi_func, halofit,
                           zl_scale='loglin', zl_bin=[40, 40]):
    """Computes angular power spectrum of cosmic shear. zl_scale='loglin',
    zl_bin=[40,40] is sufficient setting to give 0.5% accuracy.

    Args:
        cosmo (object):             astropy cosmology instance
        linear_model (object):      linear_model instance
        q1_over_chi_func (ndarray): lensing efficiency function q1 divided by chi
        q2_over_chi_func (ndarray): lensing efficiency function q2 divided by chi
        halofit (object):           halofit instance
        zl_sclae (ndarray):         scale to sample lens redshift for integration.
        zl_bin (int):               number of lens redshift bin for integration.
    Returns:
        cl (ndarray):               cosmic shear angular power spectrum at l.
    """

    # init integration bin
    if zl_scale == 'lin':
        zl = np.linspace(1e-6, 7.0, zl_bin)
    elif zl_scale == 'log':
        zl = np.logspace(-6, np.log10(3.0), zl_bin)
    elif zl_scale == 'loglin':
        zl = np.concatenate([np.logspace(-6,-1.01, zl_bin[0]), np.linspace(1e-1, 3.0, zl_bin[1])])
    else:
        raise ValueError('z_scale can only be lin, log or loglin.')

    # compute the comoving distance at lens redshift.
    chil = cosmo.comoving_distance(zl).value
    # init linear power
    linear_model.init_pklin_array(zl)
    # get efficiency
    q1_over_chil = q1_over_chi_func(chil)
    q2_over_chil = q2_over_chi_func(chil)


    # get linear matter power spectrum, k bin below is sufficient.
    k = np.logspace(-6, 6, 2000)
    # get list of the functions of interpolated nonlinear matter power spectrum.
    pkfunc_list = get_pkfunc(zl, k, linear_model, cosmo.h.copy(), halofit)
    cl_sparse = []
    l_sparse =  np.logspace(-1, 6, 200)

    for _l in l_sparse:
        pk = get_pk(_l, chil, pkfunc_list)
        _cl = simps(q1_over_chil*q2_over_chil*pk, chil)
        cl_sparse.append(_cl)

    cl = Extrap1d(l_sparse, np.array(cl_sparse))(l)

    return cl

def angular_power_spectrum_finite_shell(l, cosmo, linear_model,
                           q1_over_chi_func, q2_over_chi_func, halofit):
    """Computes angular power spectrum of cosmic shear.
    NOTE:
    zl_scale='loglin', zl_bin=[40,40] is sufficient setting to give 0.5%
    accuracy.

    Args:
        cosmo (object):             astropy cosmology instance
        linear_model (object):      linear_model instance
        q1_over_chi_func (ndarray): lensing efficiency function q1 divided by chi
        q2_over_chi_func (ndarray): lensing efficiency function q2 divided by chi
        halofit (object):           halofit instance
    Returns:
        cl (ndarray):               cosmic shear angular power spectrum at l
    """

    z   =   np.logspace(-3.8,0.78,1000)
    #
    chi =   cosmo.comoving_distance(z).value
    chi2z = ius(chi, z, ext = 2)
    del z,chi
    # get linear matter power spectrum, k bin below is sufficient.
    k   =   np.logspace(-6, 5, 1000)
    # calculate the effective lensing kernel (for finite number of shells)
    N   =   38
    chil=   np.zeros(N)
    zl  =   np.zeros(N)
    chieff= np.zeros(N)
    dchi=   150./cosmo.h
    for i in range(N):
        chil[i] = (i+0.5)*dchi
        zl[i]= chi2z(chil[i])
        c1= i*dchi
        c2= (i+1)*dchi
        chieff[i] = 3.0/4.0*(c2**4-c1**4)/(c2**3-c1**3)
    del chi2z

    # init linear pk
    linear_model.init_pklin_array(zl)
    # get list of the functions of interpolated nonlinear matter power spectrum.
    pkfunc_list = get_pkfunc(zl, k, linear_model, cosmo.h.copy(), \
            halofit, do_corr_shell=True)
    # get efficiency q
    q1 = q1_over_chi_func(chil)*chil
    q2 = q2_over_chi_func(chil)*chil

    cl_sparse = []
    l_sparse = np.logspace(-1, 6, 500)
    for _l in l_sparse:
        pk = get_pk(_l, chil, pkfunc_list)
        _cl = simps(q1*q2*pk/chieff**2.,chil)
        cl_sparse.append(_cl)
    cl = Extrap1d(l_sparse, np.array(cl_sparse))(l)
    return cl

def cl2xipm(l, cl, N_extrap_low=0, N_extrap_high= None):
    """Converts cl to xi_{+/-} using fftlog.

    Args:
        l (ndarray):        must be equally spaced in logarithmic.
        cl (ndarray):       angular power specturm on l.
        N_extrap_low (int): extrapolation
    Returns:
        tp (ndarray):       theta for xi_+, [arcmin]
        xip (ndarray):      xi_+
        tm (ndarray):       theta for xi_-, [arcmin]
        xim (ndarray):      xi_-
    """
    hankel = fftlog.hankel(l, l**2*cl, 1.01,
                           N_extrap_low=N_extrap_low, N_extrap_high=N_extrap_high,
                           c_window_width=0.25)
    tp, xip = hankel.hankel(0)
    tm, xim = hankel.hankel(4)
    xip /= 2*np.pi
    xim /= 2*np.pi
    # change unit of theta to arcmin
    tp = np.rad2deg(tp)*60. # arcmin
    tm = np.rad2deg(tm)*60. # arcmin
    return tp, xip, tm, xim

# power spectrum from CAMB
class camb_class:
    def __init__(self):
        # WMAP2013 cosmolog by default
        cparam = np.array([0.02254,0.11417,0.721,3.083548,0.97,-1.])
        self.set_cosmology(cparam)
        return

    def set_cosmology(self, cparam, mnu=0.06, omk=0.0):
        """Setups cosmology model

        Args:
            cparam (ndarray):   cosmology parameters
            mnu (float):        neutrino mass
            omk (float):        omega_k
        """
        omnuh2 = 0.00064*(mnu/0.06)
        ombh2 = cparam[0]
        omch2 = cparam[1]
        Om_L = cparam[2]
        h = ( (ombh2+omch2+omnuh2)/(1-Om_L) )**0.5
        self.h = h
        H0 = 100.*h
        As = np.exp(cparam[3])*1.e-10
        ns = cparam[4]
        w  = cparam[5]

        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
        self.pars.InitPower.set_params(As=As, ns=ns)
        self.pars.set_dark_energy(w=w, cs2=1.0, wa=0, dark_energy_model='fluid')
        self.cparam = cparam
        self.mnu = mnu
        self.omk = omk
        return

    def init_pklin_array(self, z, kmax=5e2, minkh=1e-5, maxkh=5e2):
        """Initializes linear power function at different redshift snapshots
        """
        if not 0.0 in z:
            redshift =  np.array([0.0] + list(z))
        else:
            redshift =  z
        self.pars.set_matter_power(redshifts=redshift, kmax=kmax/self.h)
        self.pars.NonLinear=camb.model.NonLinear_none
        results=camb.get_results(self.pars)
        ans =   results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=250)
        self.k_camb      =  np.atleast_1d(ans[0])
        self.z_camb      =  np.atleast_1d(ans[1])
        self.pklin_camb  =  np.atleast_1d(ans[2])
        self.sigma8_camb =  results.get_sigma8()[-1]
        # redshifts have been re-sorted so that z=0 is at the end of the
        # array
        self.results_camb= results
        return

    def get_pklin_array_from_idx(self, k, zidx):
        pk = self.pklin_camb[zidx, :]
        return Extrap1d(self.k_camb, pk)(k)

    def get_pklin_array_from_z(self, k, z):
        zidx = np.argmin(np.abs(self.z_camb-z))
        return self.get_pklin_array_from_idx(k, zidx)
