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
import treecorr
import schwimmbad
import numpy as np
import astropy.io.fits as pyfits
from argparse import ArgumentParser

class Worker(object):
    def __init__(self):
        self.indir='/hildafs/datasets/shared_phy200017p/HSC_shape_catalog_Y1/catalog_gals_mock/4zbins/'
        self.outdir=os.path.join(os.environ['homeWrk'],'skylens/s16a_mock_2pcf/')
        assert os.path.isdir(self.indir), '%s does not exist'  %self.indir
        assert os.path.isdir(self.outdir), '%s does not exist' %self.outdir
        self.nbins=30
        self.nz=4
        self.nzcross=self.nz#int(self.nz*(self.nz+1)/2)
        return

    def run(self,Id):
        isim =  Id//21
        irot =  Id%21
        ofname=  os.path.join(self.outdir,'2pcf_r%03d_rotmat%02d.fits' %(isim,irot))
        rewrite = False
        if os.path.isfile(ofname) and not rewrite:
            out =   pyfits.getdata(ofname)
        else:
            print('processing ID: %d' %(Id))
            out =   np.zeros(shape=(self.nzcross,2,self.nbins))
            cor =   treecorr.GGCorrelation(nbins=self.nbins,min_sep=0.25,max_sep=360.,sep_units='arcmin')
            ic  =   0
            for i in range(0,self.nz):
                # znmi=   os.path.join(self.indir,'r%03d_rot%02d_zbin%d.fits' %(isim,irot,i+1))
                # ddi =   pyfits.getdata(znmi)
                # msk =   (ddi['e1_mock']**2.+ddi['e2_mock']**2.)<20.
                # ddi =   ddi[msk]
                # # noiseless case
                # catI=   treecorr.Catalog(g1=ddi['shear1_sim'],\
                #         g2=-ddi['shear2_sim'],\
                #         ra=ddi['ra_mock'],dec=ddi['dec_mock'],\
                #         ra_units='deg',dec_units='deg')
                # del msk
                for j in range(i,i+1):
                    znmj=   os.path.join(self.indir,'r%03d_rot%02d_zbin%d.fits' %(isim,irot,j+1))
                    ddj =   pyfits.getdata(znmj)
                    msk =   (ddj['e1_mock']**2.+ddj['e2_mock']**2.)<10.
                    msk =   msk&((ddj['shear1_sim']**2.+ddj['shear2_sim']**2.)<10.)
                    msk =   msk&(ddj['z_source_mock']>0.)
                    ddj =   ddj[msk]
                    del msk
                    # noiseless case
                    catJ=   treecorr.Catalog(g1=ddj['shear1_sim'],\
                            g2=-ddj['shear2_sim'],\
                            ra=ddj['ra_mock'], dec=ddj['dec_mock'],\
                            ra_units='deg', dec_units='deg')

                    del ddj
                    cor.clear()
                    cor.process(catJ,catJ)
                    del catJ
                    out[ic,0,:]=cor.xip
                    out[ic,1,:]=cor.xim
                    cor.clear()
                    ic+=1
            pyfits.writeto(ofname, out, overwrite=True)
        return out

    def __call__(self,Id):
        print('start ID: %d' %(Id))
        try:
            out=self.run(Id)
            return out
        except ValueError:
            print('realization: %d cannot be processed' %Id)
            return None

if __name__=='__main__':
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument('--minId', required=True, type=int,
                        help='minimum id number, e.g. 0')
    parser.add_argument('--maxId', required=True, type=int,
                        help='maximum id number, e.g. 2268')
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
        if r is not None:
            outs.append(r)
    outs=np.average(np.stack(outs),axis=0)
    print(outs.shape)
    pyfits.writeto('2pcf_ture_ave.fits',outs,overwrite=True)
    pool.close()
