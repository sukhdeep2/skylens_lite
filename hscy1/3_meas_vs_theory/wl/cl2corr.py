import scipy.interpolate
import numpy as np
from . import pyfftlog

# These are the ones the user can use
TRANSFORM_W = "w"
TRANSFORM_XI = "xi"
TRANSFORM_GAMMAT = "gamma"

# If they use xi then it splits into xip and xim.
TRANSFORM_XIP = "xip"
TRANSFORM_XIM = "xim"

DEFAULT_N_TRANSFORM = 8192
DEFAULT_ELL_MIN = 0.0001
DEFAULT_ELL_MAX = 5.0e6
DEFAULT_THETA_MIN = 0.1
DEFAULT_THETA_MAX = 1000.0


TRANSFORMS = [TRANSFORM_W, TRANSFORM_GAMMAT, TRANSFORM_XI]

DEFAULT_SECTIONS = {
    TRANSFORM_XI: ("shear_cl", "shear_xi"),
    TRANSFORM_XIP: ("shear_cl", "shear_xi"),
    TRANSFORM_XIM: ("shear_cl", "shear_xi"),
    TRANSFORM_GAMMAT: ("galaxy_shear_cl", "galaxy_shear_xi"),
    TRANSFORM_W: ("galaxy_cl", "galaxy_xi"),
}

OUTPUT_NAMES = {
    TRANSFORM_W: "bin_{}_{}",
    TRANSFORM_GAMMAT: "bin_{}_{}",
    TRANSFORM_XIP: "bin_{}_{}",
    TRANSFORM_XIM: "bin_{}_{}",
}


# Bias q and order mu parameters for transform
_TRANSFORM_PARAMETERS = {
    TRANSFORM_W: (0.0, 0.0),
    TRANSFORM_XIP: (0.0, 0.0),
    TRANSFORM_XIM: (0.0, 4.0),
    TRANSFORM_GAMMAT: (0.0, 2.0),
}


class LogInterp(object):
    """
    This is a helper object that interpolates into f(x) where x>0.
    If all f>0 then it interpolates log(f) vs log(x).  If they are all f<0 then it
    interpolate log(-f) vs log(x).  If f is mixed or has some f=0 then it just interpolates
    f vs log(x).

    """

    def __init__(self, angle, spec, kind):
        if np.all(spec > 0):
            self.interp_func = scipy.interpolate.interp1d(
                np.log(angle), np.log(spec), kind, bounds_error=False
            )
            self.interp_type = "loglog"
        elif np.all(spec < 0):
            self.interp_func = scipy.interpolate.interp1d(
                np.log(angle), np.log(-spec), kind, bounds_error=False
            )
            self.interp_type = "minus_loglog"
        else:
            self.interp_func = scipy.interpolate.interp1d(
                np.log(angle), spec, kind, bounds_error=False
            )
            self.interp_type = "log_ang"

    def __call__(self, angle):
        if self.interp_type == "loglog":
            spec = np.exp(self.interp_func(np.log(angle)))
        elif self.interp_type == "minus_loglog":
            spec = -np.exp(self.interp_func(np.log(angle)))
        else:
            assert self.interp_type == "log_ang"
            spec = self.interp_func(np.log(angle))
        return spec


class Transformer(object):
    """
    Class to build Hankel Transformers that convert from 2D power spectra to correlation functions.
    Several transform types are allowed, depending whether you are using cosmic shear, clustering, or
    galaxy-galaxy lensing.
    """

    def __init__(
        self,
        transform_type,
        n,
        ell_min,
        ell_max,
        theta_min,
        theta_max,
        lower=1.0,
        upper=-2.0,
    ):

        # We use a fixed ell grid in log space and will interpolate/extrapolate our inputs onto this
        # grid. We typically use a maximum ell very much higher than the range we have physical values
        # for.  The exact values there do not matter, but they must be not have a sharp cut-off to avoid
        # oscillations at small angle.
        self.ell_min = ell_min
        self.ell_max = ell_max
        ell = np.logspace(np.log10(ell_min), np.log10(ell_max), n)
        self.ell = ell
        dlogr = np.log(ell[1]) - np.log(ell[0])

        # pyfftlog has several options about how the theta and ell values used are chosen.
        # This option tells it to pick them to minimize ringing.
        kropt = 1

        # The parameters of the Hankel transform depend on the type.
        # They are defined in a dict at the top of the file
        self.q, self.mu = _TRANSFORM_PARAMETERS[transform_type]

        # Prepare the Hankel transform.
        self.kr, self.xsave = pyfftlog.fhti(n, self.mu, dlogr, q=self.q, kropt=kropt)

        # We always to the inverse transform, from Fourier->Real.
        self.direction = -1

        # Some more fixed values.
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.lower = lower
        self.upper = upper

        # work out the effective theta values.
        nc = 0.5 * (n + 1)
        log_ellmin = np.log(ell_min)
        log_ellmax = np.log(ell_max)
        log_ellmid = 0.5 * (log_ellmin + log_ellmax)
        ell_mid = np.exp(log_ellmid)
        r_mid = self.kr / ell_mid  # radians
        x = np.arange(n)

        # And the effective angles of the output
        self.theta_rad = np.exp((x - nc) * dlogr) * r_mid  # radians
        theta_arcmin = np.degrees(self.theta_rad) * 60.0  # arcmin
        self.range = (theta_arcmin > self.theta_min) & (theta_arcmin < self.theta_max)
        self.theta_arcmin = theta_arcmin[self.range]

    def __call__(self, ell_in, cl_in):
        """Convert the input ell and cl points to the points this transform requires, and then
        transform."""

        # Sample onto self.ell
        cl = self._interpolate_and_extrapolate_cl(ell_in, cl_in)

        if self.q == 0:
            xi = (
                pyfftlog.fht(self.ell * cl, self.xsave, tdir=self.direction)
                / (2 * np.pi)
                / self.theta_rad
            )
        else:
            xi = (
                pyfftlog.fhtq(self.ell * cl, self.xsave, tdir=self.direction)
                / (2 * np.pi)
                / self.theta_rad
            )
        return self.theta_arcmin, xi[self.range]

    def _interpolate_and_extrapolate_cl(self, ell, cl):
        """Extrapolate and interpolate the input ell and cl to the default points for this transform"""
        ell_min = ell[0]
        ell_max = ell[-1]
        interpolator = LogInterp(ell, cl, "linear")
        cl_out = interpolator(self.ell)
        bad_low = np.isnan(cl_out) & (self.ell < ell_min)
        bad_high = np.isnan(cl_out) & (self.ell > ell_max)

        cl_out[bad_low] = cl[0] * (self.ell[bad_low] / ell_min) ** self.lower
        cl_out[bad_high] = cl[-1] * (self.ell[bad_high] / ell_max) ** self.upper

        return cl_out
