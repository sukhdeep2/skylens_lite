#!/usr/bin/env python
import os
import sys
import argparse
import pickle

import numpy as np
import healpy as hp

import pymaster as nmt

from schwimmbad import MPIPool

with open("../data/mask.pickle", "rb") as f:
    mask = pickle.load(f)
    
SIGMA_E = 0.25
N_SOURCE = 12
ARCMIN2RAD = 1 / 60 * (np.pi / 180)
NOISE_CL = SIGMA_E ** 2 / N_SOURCE * ARCMIN2RAD ** 2


def run_sims(seed):
    nside = 1024
    np.random.seed(seed)
    print("run", seed)
    gaussian_maps = dict()
    
    for i in range(4):
        cl_fname = f"../data/cl_{i}_{i}.txt"
        _, cl = np.loadtxt(cl_fname)
        noise_cl = NOISE_CL * np.ones_like(cl)
        noise_map = hp.sphtfunc.synfast(
            [noise_cl, noise_cl, noise_cl * 0, noise_cl, noise_cl * 0, noise_cl * 0], nside, new=True, pol=True
        )
        gaussian_maps[i] = hp.sphtfunc.synfast(
            [cl, cl, cl * 0, cl, cl * 0, cl * 0], nside, new=True, pol=True
        )
        gaussian_maps[i] += noise_map
        gaussian_maps[i][:, ~mask[i]] = hp.UNSEEN

    Dl_nmt = dict()
    for i in range(4):
        gmi = gaussian_maps[i]
        f2_i = nmt.NmtField(mask[i], [gmi[1], gmi[2]])
        f2_j = nmt.NmtField(mask[i], [gmi[1], gmi[2]])
        Dl = nmt.compute_coupled_cell(f2_i, f2_j)
        Dl_nmt[i] = Dl
        del f2_i, f2_j, Dl

    fname = f"../sims/pcl_{seed}_with_noise.pickle"
    with open(fname, "wb") as f:
        pickle.dump(Dl_nmt, f)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--numSim", default=100, type=int, help="number of simulations to run, e.g. 1"
    )

    args = parser.parse_args()

    refs = range(args.numSim)

    pool = MPIPool()

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    pool.map(run_sims, refs)
    pool.close()
    return


if __name__ == "__main__":
    main()
