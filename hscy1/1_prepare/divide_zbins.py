#!/usr/bin/env python
import gc
import os
import glob
import astropy.io.fits as pyfits
import argparse
import numpy as np
fields=['XMM','VVDS','HECTOMAP','GAMA09H','WIDE12H','GAMA15H']

def prepare_data(mockDir,isim,irot):
    dall=[]
    # append data in every field
    for fieldname in fields:
        dfield=[]
        sDir= os.path.join(mockDir,fieldname,'r%03d'%isim, 'rotmat%d' %irot,'*.fits')
        fnamelist=glob.glob(sDir)
        # append data in every tract
        for fname in fnamelist:
            data=pyfits.getdata(fname)
            dfield.append(data)
            del data
            #os.remove(fname)
        del fnamelist,sDir
        dall.append(np.hstack(dfield))
        del dfield
        gc.collect()
    out=np.hstack(dall)
    out.sort(order='object_id')
    return out

def main(isim,irot):
    nz=4
    zdat=pyfits.getdata('/hildafs/datasets/shared_phy200017p/HSC_shape_catalog_Y1/catalog_gals_mock/ephor_zbest.fits')
    odir='/hildafs/datasets/shared_phy200017p/HSC_shape_catalog_Y1/catalog_gals_mock/4zbins/'
    dDir='/hildafs/projects/phy200017p/share/raytracesim/HSC_S16A/downloads/'
    oname=os.path.join(odir,'r%03d_rot%02d_zbin*.fits' %(isim,irot))
    if len(glob.glob(oname))==nz:
        print('Already have isim:%d irot:%d' %(isim,irot))
        return
    mock=prepare_data(dDir,isim,irot)
    assert np.all(mock['object_id']==zdat['object_id'])
    for iz in range(nz):
        oname=os.path.join(odir,'r%03d_rot%02d_zbin%d.fits' %(isim,irot,iz+1))
        if os.path.isfile(oname):
            continue
        zmin=(iz+1)*0.3
        zmax=(iz+2)*0.3
        msk=(zdat['ephor_photoz_best']>=zmin)&(zdat['ephor_photoz_best']<zmax)
        print(oname)
        pyfits.writeto(oname,mock[msk],overwrite=True)
        del zmin,zmax,msk,oname
    del mock
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, help="the index of the simulation")
    args = parser.parse_args()
    isim=args.id//21
    irot=args.id%21
    if isim>=0 and isim<108:
        main(isim,irot)

