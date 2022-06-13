import sys
import traceback

# import pyccl as ccl
import pickle
import camb
import jax.numpy as jnp

# sys.path.insert(0,'../skylens/')

from distributed import LocalCluster
from dask.distributed import Client  # we already had this above

# http://distributed.readthedocs.io/en/latest/_modules/distributed/worker.html

from skylens import *
from skylens.survey_utils import *

# only for python3
import importlib

reload = importlib.reload


LC, scheduler_info = start_client(
    Scheduler_file=None,
    local_directory="../temp/",
    ncpu=None,
    n_workers=1,
    threads_per_worker=None,
    memory_limit="120gb",
    dashboard_address=8801,
)
client = client_get(scheduler_info=scheduler_info)


wigner_files = {}
wig_home = "./tests/"
wigner_files[0] = wig_home + "dask_wig3j_l100_w100_0_reorder.zarr"
wigner_files[2] = wig_home + "dask_wig3j_l100_w100_2_reorder.zarr"

# setup parameters
lmax_cl = 100
lmin_cl = 2
l0 = jnp.arange(lmin_cl, lmax_cl)

lmin_cl_Bins = lmin_cl + 10
lmax_cl_Bins = lmax_cl - 10
Nl_bins = 20
l_bins = jnp.int32(
    jnp.logspace(jnp.log10(lmin_cl_Bins), jnp.log10(lmax_cl_Bins), Nl_bins)
)
lb = jnp.sqrt(l_bins[1:] * l_bins[:-1])

l = jnp.unique(
    jnp.int32(jnp.logspace(jnp.log10(lmin_cl), jnp.log10(lmax_cl), Nl_bins * 20))
)  # if we want to use fewer ell

bin_cl = True

bin_xi = True
theta_bins = jnp.logspace(jnp.log10(1.0 / 60), 1, 20)

nside = 32
window_lmax = nside

use_window = True

do_xi = True
bin_xi = True
bin_cl = True
th_min = 25 / 60
th_max = 250.0 / 60
n_th_bins = 20
th_bins = jnp.logspace(jnp.log10(th_min), jnp.log10(th_max), n_th_bins + 1)
th = jnp.logspace(jnp.log10(th_min), jnp.log10(th_max), n_th_bins * 40)
thb = jnp.sqrt(th_bins[1:] * th_bins[:-1])

# Hankel Transform setup
WT_kwargs = {"l": l0, "theta": th, "s1_s2": [(2, 2), (2, -2), (0, 0), (0, 2), (2, 0)]}
WT = wigner_transform(**WT_kwargs)


# In[24]:


z0 = 1  # 1087
# zs_bin1=source_tomo_bins(zp=[z0],p_zp=jnp.array([1]),ns=30,use_window=use_window,nside=nside)
zs_bin1 = lsst_source_tomo_bins(nbins=2, use_window=use_window, nside=nside, n_zs=50)


corr_ggl = ("galaxy", "shear")
corr_gg = ("galaxy", "galaxy")
corr_ll = ("shear", "shear")
corrs = [corr_ll, corr_ggl, corr_gg]

use_binned_ls = [False, True]

store_wins = [True]  # [False,True] # False is deprecated, needs fixing if to be used.

# In[21]:

bin_cl = True

do_pseudo_cls = [True, False]
do_xis = [False, True]  # [True,True]
use_windows = [True, False]

passed = 0
failed = 0
failed_tests = {}
traceback_tests = {}
nid = 0
for do_xi in do_xis:
    do_pseudo_cl = ~do_xi
    # for do_pseudo_cl in do_pseudo_cls:
    #     if do_xi==do_pseudo_cl:
    #         continue
    for use_window in use_windows:
        for use_binned_l in use_binned_ls:
            for store_win in store_wins:
                s = ""
                s = s + " do_xi " if do_xi else s + " do_cl "
                s = s + " use_window " if use_window else s
                s = s + " use_binned_l " if use_binned_l else s
                s = s + " store_win " if store_win else s
                print("\n", "\n")
                print("passed failed: ", passed, failed, " now testing ", s)
                print("tests that failed: ", failed_tests)
                print("\n", "\n")
                try:
                    kappa0 = Skylens(
                        shear_zbins=zs_bin1,
                        bin_cl=bin_cl,
                        l_bins=l_bins,
                        l=l0,
                        galaxy_zbins=zs_bin1,
                        use_window=use_window,
                        use_binned_l=use_binned_l,
                        wigner_files=wigner_files,
                        store_win=store_win,
                        window_lmax=window_lmax,
                        corrs=corrs,
                        do_xi=do_xi,
                        bin_xi=bin_xi,
                        theta_bins=th_bins,
                        WT=WT,
                        use_binned_theta=use_binned_l,
                        scheduler_info=scheduler_info,
                    )
                    if do_xi:
                        G = kappa0.xi_tomo()
                        xi_bin_utils = client.gather(kappa0.xi_bin_utils)
                    else:
                        G = kappa0.cl_tomo()
                    cc = client.compute(G["stack"]).result()

                    kappa0.gather_data()
                    #                                 kappa0.scatter_data()
                    xi_bin_utils = kappa0.xi_bin_utils
                    cl_bin_utils = kappa0.cl_bin_utils
                    # #                                 kappa0.Ang_PS.clz=client.gather(kappa0.Ang_PS.clz)
                    #                                 kappa0.WT.gather_data()
                    WT_binned = kappa0.WT_binned

                    cS = delayed(kappa0.tomo_short)(
                        cosmo_params=kappa0.Ang_PS.PS.cosmo_params,
                        Win=kappa0.Win,
                        WT=kappa0.WT,
                        WT_binned=WT_binned,
                        Ang_PS=kappa0.Ang_PS,
                        zkernel=G["zkernel"],
                        xi_bin_utils=xi_bin_utils,
                        cl_bin_utils=cl_bin_utils,
                        z_bins=kappa0.z_bins,
                    )
                    cc = client.compute(cS).result()
                    passed += 1

                except Exception as err:
                    print(s, " failed with error ", err)
                    print(traceback.format_exc())
                    failed_tests[failed] = s + " failed with error " + str(err)
                    traceback_tests[failed] = str(traceback.format_exc())
                    failed += 1
                    # crash
                # client.close()
                # client=Client(LC)
                # client.wait_for_workers(n_workers=1)
                client.restart()
                clean_client(scheduler_info=scheduler_info)
                LC, scheduler_info = start_client(
                    Scheduler_file=None,
                    local_directory="../temp/" + str(nid) + "/",
                    ncpu=None,
                    n_workers=1,
                    threads_per_worker=None,
                    memory_limit="120gb",
                    dashboard_address=8801,
                )
                nid += 1
                client = client_get(scheduler_info=scheduler_info)
                client.wait_for_workers(n_workers=1)
                if client.scheduler is None:
                    print("scheduler_info is None", LC)
                    crash

for i in failed_tests.keys():
    print(failed_tests[i])
    print(traceback_tests[i])
