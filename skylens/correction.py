import camb
import pyhalofit
import numpy as np
from scipy.integrate import simps
from scipy.integrate import trapz
from scipy.interpolate import InterpolatedUnivariateSpline as ius

def convert_decosmo2halofitcosmo(decosmo, mnu=0.06):
    """
    convert dark emulator cosmology list to pyhalofit cosmology dict.
    """
    omnuh2 = 0.00064*(mnu/0.06)
    decosmo = decosmo.flatten()
    halofitcosmo = dict()
    halofitcosmo['Omega_de0'] = decosmo[2]
    halofitcosmo['w0'] = decosmo[5]
    halofitcosmo['h'] = ((omnuh2+decosmo[0]+decosmo[1])/(1.0-decosmo[2]))**0.5
    halofitcosmo['wa'] = 0
    halofitcosmo['Omega_K0'] = 0
    return halofitcosmo

class camb_class:
    def __init__(self):
        cparam = np.array([[ 0.02225,  0.1198 ,  0.6844 ,  3.094  ,  0.9645 , -1.     ]])
        self.set_cosmology(cparam)

    def set_cosmology(self, cparam, mnu=0.06, omk=0.0):
        cparam = np.reshape(cparam, (1,6)).copy()
        omnuh2 = 0.00064*(mnu/0.06)
        ombh2 = cparam[0][0]
        omch2 = cparam[0][1]
        Om_L = cparam[0][2]
        h = ( (ombh2+omch2+omnuh2)/(1-Om_L) )**0.5
        H0 = 100*h
        As = np.exp(cparam[0][3])*1e-10
        ns = cparam[0][4]
        w  = cparam[0][5]
        self.h = h

        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
        self.pars.InitPower.set_params(As=As, ns=ns)
        self.pars.set_dark_energy(w=w, cs2=1.0, wa=0, dark_energy_model='fluid')
        self.cparam = cparam
        self.mnu = mnu
        self.omk = omk

    def get_cosmology(self):
        return self.cparam.copy()

    def get_mnu(self):
        return self.mnu

    def get_omk(self):
        return self.omk

    def init_pklin_array(self, k, z, kmax=1e2, minkh=1e-4, maxkh=1e2):
        if not 0.0 in z:
            redshift = [0.0] + list(z)
        else:
            redshift = z
        self.pars.set_matter_power(redshifts=redshift, kmax=kmax)
        self.pars.NonLinear = camb.model.NonLinear_none
        results = camb.get_results(self.pars)
        ans = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=250)
        self.k_camb      = np.atleast_1d(ans[0])
        self.z_camb      = np.atleast_1d(ans[1])
        self.pklin_camb  = np.atleast_1d(ans[2])
        self.sigma8_camb = results.get_sigma8()[self.z_camb==0.0]
        self.results_camb = results

    def get_sigma8(self):
        return self.sigma8_camb

    def get_pklin_array_from_idx(self, k, zidx):
        pk = self.pklin_camb[zidx, :]
        return log_extrap_func(self.k_camb, pk)(k)

    def get_pklin_array_from_z(self, k, z):
        zidx = np.argmin(np.abs(self.z_camb-z))
        return self.get_pklin_array_from_idx(k, zidx)

def log_extrap_func(x,y):
    def func(x_new):
        if isinstance(x_new, (int, float)):
            x_new = np.atleast_1d(x_new)
        ans = np.zeros(x_new.size)
        sel = x_new < x.min()
        if np.sum(sel):
            ans[sel] = np.exp( np.log(y[1]/y[0])/np.log(x[1]/x[0]) * np.log(x_new[sel]/x[0]) + np.log(y[0]) )
        sel = x.max() < x_new
        if np.sum(sel):
            ans[sel] = np.exp( np.log(y[-2]/y[-1])/np.log(x[-2]/x[-1]) * np.log(x_new[sel]/x[-1]) + np.log(y[-1]) )
        sel = np.logical_and(x.min()<= x_new, x_new <= x.max())
        ans[sel] = 10**ius(np.log10(x),np.log10(y))(np.log10(x_new[sel]))
        return ans
    return func

def get_pk(_l, chil, h, pkfunc_list, modl=True):#, skip=1):
    """Read pkfunc_list and give P(l/chil, zl), used for cosmis shear.
    """
    pk = []
    if modl:
        nu = _l + 0.5
    else:
        nu = _l
    for i, _chil in enumerate(chil):
        k = nu/_chil # [/Mpc]
        k /= h # [h/Mpc]
        _pk = pkfunc_list[i]( k )[0] # [(Mpc/h)^3]
        _pk /= h**3 #[Mpc^3]
        pk.append(_pk)
    pk = np.array(pk)
    #if skip > 1:
    #    pk = log_extrap_func(chil[::skip], pk)(chil)
    return pk

def angular_power_spectrum_finite_shell(l, cosmo, linear_model,
                           q1_over_chi_func, q2_over_chi_func, halofit=None):
    """
    This function computes angular power spectrum of cosmic shear.
    Args:
        cosmo            : astropy cosmology instance
        linear_model     : linear_model instance
        halofit          : halofit instance
        q1_over_chi_func : lensing efficiency no.1 function divided by chi
        q2_over_chi_func : lensing efficiency no.2 function divided by chi
        zl_bin           : number of lens redshift bin for integration.
    Returns:
        cl               : cosmic shear angular power spectrum at l.
    """

    z = np.linspace(0.0, 7, 1000)
    chi = cosmo.comoving_distance(z).value
    chi2z = ius(chi,z)

    # get linear matter power spectrum, k bin below is sufficient.
    k = np.logspace(-6, 4, 1000)

    # correction for finite shell effect -- step (2),
    # following equation (B1) and (B2) of https://arxiv.org/abs/1901.09488
    N = 38
    chil = np.zeros(N)
    zl = np.zeros(N)
    chieff = np.zeros(N)
    dchi = 150
    for j in range(N):
        i = j+1
        chil[j] = (i-0.5)*dchi
        zl[j] = chi2z(chil[j])
        c1 = i*dchi
        c2 = (i+1)*dchi
        chieff[j] = 3.0/4.0*(c2**4-c1**4)/(c2**3-c1**3)

    # init linear pk
    linear_model.init_pklin_array(k, zl)
    # get list of the functions of interpolated nonlinear matter power spectrum
    pkfunc_list = get_pkfunc_finite_shell(zl, k, linear_model, halofit)
    # get efficiency q
    q1 = q1_over_chi_func(chil)*chil
    q2 = q2_over_chi_func(chil)*chil
    cl_sparse = []
    l_sparse = np.logspace(0.0, 5.0, 70)
    for _l in l_sparse:
        pk = get_pk(_l, chil, cosmo.h.copy(), pkfunc_list)
        _cl = np.sum( dchi/chieff**2*q1*q2*pk)
        cl_sparse.append(_cl)
    cl = log_extrap_func(l_sparse, np.array(cl_sparse))(l)
    return cl

def get_pkfunc_finite_shell(zl, k, linear_model, halofit):
    """
    This function computes nonlinear matter power spectrum
    from a given linear matter power spectrum.
    Args:
        zl           : lens redshift
        k            : k bin on which pk will be computed.
        linear_model : linear model instance.
        h            : hubble parameter [100km/Mpc/sec]
        halofit      : halofit instance
    Returns:
        pkfunc_list  : list of inter/extra-polated function of
                       nonlinear matter power spectrum.
    """
    # correction for finite shell effect -- step (1),
    # following equation (B3) of https://arxiv.org/abs/1901.09488
    c1,c2 = 9.5171e-4, 5.1543e-3
    alpha1, alpha2, alpha3 = 1.3063, 1.1475, 0.62793
    corr_finite_shell = (1+c1*k**-alpha1)**alpha1/(1+c2*k**-alpha2)**alpha3

    pkfunc_list = []
    for _zl in zl:
        pklin = linear_model.get_pklin_array_from_z(k, _zl)
        halofit.set_pklin(k, pklin, _zl)
        pkhalofit = halofit.get_pkhalo()
        pkfunc_list.append(log_extrap_func(k, corr_finite_shell*pkhalofit))
    return pkfunc_list

def get_cl_with_correction(nside, zs_pzs_fname):
    cparam = np.array([[0.02254,
                        0.11417,
                        0.721,
                        3.08354868,
                        0.97,
                        -1.]])
    # instantiate
    mnu = 0.0
    linear_power = camb_class()
    halofit = pyhalofit.halofit()

    # set cosmology
    linear_power.set_cosmology(cparam, mnu=mnu, omk=0.0)
    halofit.set_cosmology(convert_decosmo2halofitcosmo(cparam, mnu=mnu))

    # astropy cosmo
    cosmo = halofit.cosmo

    # load zs,pzs
    d_pzs = np.loadtxt(zs_pzs_fname)
    zs1, pzs1 = d_pzs[:, 0], d_pzs[:, 1]
    zs2, pzs2 = d_pzs[:, 2], d_pzs[:, 3]
    
    q_over_chi_func1 = efficiency_q_over_chi(zs1, pzs1, cosmo)
    q_over_chi_func2 = efficiency_q_over_chi(zs2, pzs2, cosmo)

    #l = np.logspace(0, 5, 1024)
    l = np.arange(3 * nside)
    cl = angular_power_spectrum_finite_shell(l, cosmo, linear_power,
                                  q_over_chi_func1, q_over_chi_func2,
                                  halofit=halofit)

    # correction for resolution effect
    # see equation (26) of https://arxiv.org/abs/1901.09488
    N_SIDE = 8192
    l_sim = 1.6*N_SIDE
    cl = cl/(1+(l/l_sim)**2)
    sel = 3*N_SIDE < l
    cl[sel] = 0.0
    return l, cl


def efficiency_q_over_chi(zs, pzs, cosmo, form='func', zl_max=None, zl_bin=100):
    """
    Args:
        zs         : source redshift
        pzs        : source redshift distribution
        cosmo      : astropy cosmology instance
    Returns:
        q_over_chi : lensing efficiency divided by comoving distance
                    at lens redshift.
    """
    if zl_max is None:
        zl_max = zs.max()
    zl = np.linspace(1e-4, zl_max, zl_bin)
    chil = cosmo.comoving_distance(zl).value
    H0 = cosmo.H0.value/299792.4580 # [1/Mpc]
    prefactor = 3.0/2.0*cosmo.Om0*H0**2*(1+zl)

    pzs_norm = trapz(pzs, zs)

    chis = cosmo.comoving_distance(zs).value

    q = []
    sel = zs>0
    for _chil in chil:
        integrand = np.zeros(zs.shape)
        integrand[sel] = _chil*(chis[sel]-_chil)/chis[sel]*pzs[sel]/pzs_norm
        integrand[integrand<0] = 0.0
        _q = simps(integrand, zs)
        q.append(_q)
    q = prefactor*np.array(q)
    
    if form == 'func':
        return ius(chil, q/chil, ext=3)
    elif form == 'arr':
        return q/chil
