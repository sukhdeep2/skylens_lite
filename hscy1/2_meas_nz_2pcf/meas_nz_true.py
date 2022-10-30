#!/usr/bin/env python
#
# Copyright 20220312 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
import os
import schwimmbad
import numpy as np
import astropy.io.fits as pyfits
from argparse import ArgumentParser

class Worker(object):
    def __init__(self):
        self.indir='/hildafs/datasets/shared_phy200017p/HSC_shape_catalog_Y1/catalog_gals_mock/4zbins/'
        self.outdir=os.path.join(os.environ['homeWrk'],'skylens/s16a_mock_nz/')
        assert os.path.isdir(self.indir), '%s does not exist'  %self.indir
        assert os.path.isdir(self.outdir), '%s does not exist' %self.outdir

        zmid=   np.array([0.0506,  0.1023,  0.1553,  0.2097, 0.2657,    0.3233,
            0.3827,  0.4442,  0.5078,  0.5739, 0.6425,  0.714,  0.7885, 0.8664,
            0.9479,  1.0334,  1.1233,  1.2179, 1.3176,  1.423 , 1.5345, 1.6528,
            1.7784,  1.9121,  2.0548,  2.2072, 2.3704,  2.5455, 2.7338, 2.9367,
            3.1559,  3.3932,  3.6507,  3.9309, 4.2367,  4.5712, 4.9382,
            5.3423], dtype=float)
        self.nbins=len(zmid)
        dzs     =   zmid[1:] - zmid[:-1]
        zbounds =   np.zeros(self.nbins+1)
        zbounds[0]=0.01
        zbounds[-1]=zmid[-1]+0.2
        zbounds[1:-1]=zmid[:-1]+dzs/2.
        print(zbounds)
        self.zmid=zmid
        self.zlow=zbounds[:-1]
        self.zhigh=zbounds[1:]
        self.zbounds=zbounds
        self.nz=4
        return

    def run(self,Id):
        isim=Id//21
        irot=Id%21
        ofname= os.path.join(self.outdir,'nz_r%03d_rotmat%02d.fits' %(isim,irot))
        rewrite = True
        if os.path.isfile(ofname) and not rewrite:
            out  =  pyfits.getdata(ofname)
        else:
            print('measuring for ID: %d' %(Id))
            out  =  np.zeros((self.nz,self.nbins))
            for i in range(4):
                znmi= os.path.join(self.indir,'r%03d_rot%02d_zbin%d.fits' %(isim,irot,i+1))
                dd = pyfits.getdata(znmi)
                tmp = np.histogram(dd['z_source_mock'],bins=self.zbounds)[0]
                sumt= np.sum(tmp)
                out[i]= tmp / sumt
                del tmp, sumt
            pyfits.writeto(ofname,out,overwrite=True)
        return out

    def __call__(self,Id):
        print('start ID: %d' %(Id))
        return self.run(Id)

if __name__=='__main__':
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument('--minId', required=True,type=int,
                        help='minimum id number, e.g. 0')
    parser.add_argument('--maxId', required=True,type=int,
                        help='maximum id number, e.g. 4000')
    group   =   parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args    =   parser.parse_args()
    pool    =   schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    worker  =   Worker()
    refs    =   list(range(args.minId,args.maxId))
    outs    =   []
    for r in pool.map(worker,refs):
        outs.append(r)
    outs=np.average(np.stack(outs),axis=0)
    print(outs.shape)
    pyfits.writeto('nz_ture_ave.fits',outs,overwrite=True)
    pool.close()
