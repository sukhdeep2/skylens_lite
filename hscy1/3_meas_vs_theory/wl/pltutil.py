# Copyright 20220320 Xiangchong Li.
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# python lib
import astropy.io.fits as pyfits
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.colors import SymLogNorm

from matplotlib.ticker import Locator


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()
        majorlocs = np.concatenate((majorlocs, np.array([majorlocs[-1]*10])))

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)
        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


# Hex keys for the color scheme for shear catalog paper
hsc_colors = {
    'GAMA09H'   : '#d73027',
    'GAMA15H'   : '#fc8d59',
    'HECTOMAP'  : '#fee090',
    'VVDS'      : '#000000',
    'WIDE12H'   : '#91bfdb',
    'XMM'       : '#4575b4'
    }

hsc_marker = {
    'GAMA09H'   : 'v',
    'GAMA15H'   : '^',
    'HECTOMAP'  : 's',
    'VVDS'      : 'p',
    'WIDE12H'   : '*',
    'XMM'       : 'o',
    }

colors0=[
    "black",
    "#1A85FF",
    "#D41159",
    "#DE8817",
    "#A3D68A",
    "#35C3D7",
    "#8B0F8C",
    ]

colors=[]
for _ic,_cc in enumerate(colors0):
    cc2= mcolors.ColorConverter().to_rgb(_cc)
    colors.append((cc2[0], cc2[1], cc2[2], 1-0.1*_ic))
    del cc2

cblue=[
    "#004c6d",
    "#346888",
    "#5886a5",
    "#7aa6c2",
    "#9dc6e0",
    "#c1e7ff"
    ]

cred=[
    "#DC1C13",
    "#EA4C46",
    "#F07470",
    "#F1959B",
    "#F6BDC0",
    "#F8D8E3"
    ]

def make_figure_axes(ny=1,nx=1,square=True):
    ''' makes figure and axes
    Args:
        ny (int): number of plot in y
        nx (int): number of plot in x
        square (bool): figure is suqare?
    Returns:
        fig (figure): figure
        axes (list): list of axes
    '''
    if not isinstance(ny, int):
        raise TypeError("ny should be integer")
    if not isinstance(nx, int):
        raise TypeError("nx should be integer")
    axes=[]
    if ny ==1 and nx==1:
        fig=plt.figure(figsize=(6,5))
        ax=fig.add_subplot(ny,nx,1)
        axes.append(ax)
    elif ny==2 and nx==1:
        if square:
            fig=plt.figure(figsize=(6,11))
        else:
            fig=plt.figure(figsize=(6,7))
        ax=fig.add_subplot(ny,nx,1)
        axes.append(ax)
        ax=fig.add_subplot(ny,nx,2)
        axes.append(ax)
    elif ny==1 and nx==2:
        fig=plt.figure(figsize=(11,6))
        for i in range(1,3):
            ax=fig.add_subplot(ny,nx,i)
            axes.append(ax)
    elif ny==1 and nx==3:
        fig=plt.figure(figsize=(18,6))
        for i in range(1,4):
            ax=fig.add_subplot(ny,nx,i)
            axes.append(ax)
    elif ny==1 and nx==4:
        fig=plt.figure(figsize=(20,5))
        for i in range(1,5):
            ax=fig.add_subplot(ny,nx,i)
            axes.append(ax)
    elif ny==2 and nx==3:
        fig=plt.figure(figsize=(15,8))
        for i in range(1,7):
            ax=fig.add_subplot(ny,nx,i)
            axes.append(ax)
    elif ny==2 and nx==4:
        fig=plt.figure(figsize=(20,8))
        for i in range(1,9):
            ax=fig.add_subplot(ny,nx,i)
            axes.append(ax)
    else:
        raise ValueError('Do not have option: ny=%s, nx=%s' %(ny,nx))
    return fig,axes

def make_cosebis_ploddt(nmodes):
    '''
    Args:
        nmodes (int):   number of cosebis modes
    '''
    nzs =   4
    axes=   {}
    fig =   plt.figure(figsize=(10,10))

    label1=r'$E_n [\times 10^{10}]$';label2=r'$B_n [\times 10^{10}]$'
    ll1=r'$E_n$';ll2=r'$B_n$'

    for i in range(nzs):
        for j in range(i,nzs):
            #-----emode
            ax = plt.subplot2grid((10, 10), ((4-i)*2, (3-j)*2), colspan=2,rowspan=2,fig=fig)
            ax.set_title(r'$%d \times %d$' %(i+1,j+1),fontsize=15,y=1.0, pad=-15,x=0.8)
            ax.grid()

            # x-axis
            ax.set_xlim(0.2,nmodes+0.8)
            rr  =   np.arange(1,nmodes+0.8,1)
            if i!=0:
                ax.set_xticks(rr)
                ax.set_xticklabels(['']*len(rr))
            else:
                ax.set_xticks(rr)
            if i==0 and j==1:
                ax.set_xlabel(r'$n$')
            del rr

            # y-axis
            ax.set_ylim(-5.5-i*2,i*10+6.5)
            rr  =   np.arange(int(-4.5-i*2),i*10+5.5,4+i*2)
            if j!=nzs-1:
                ax.set_yticks(rr)
                ax.set_yticklabels(['']*len(rr))
            else:
                ax.set_yticks(rr)
            if j==nzs-1 and i==2:
                ax.set_ylabel(label1)
            axes.update({'%d%d_e'%(i+1,j+1): ax})
            del rr

            #-----bmode
            ax = plt.subplot2grid((10, 10), (i*2, (j+1)*2), colspan=2,rowspan=2,fig=fig)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_title(r'$%d \times %d$' %(i+1,j+1),fontsize=15,y=1.0, pad=-15,x=0.8)
            ax.grid()

            # x-axis
            ax.set_xlim(0.2,nmodes+0.8)
            rr  =   np.arange(1,nmodes+0.8,1)
            if i!=j:
                ax.set_xticks(rr)
                ax.set_xticklabels(['']*len(rr))
            else:
                ax.set_xticks(rr)

            # y-axis
            ax.set_ylim(-4.5,4.5)
            rr  =   np.arange(-4,4.5,2)
            if j!=nzs-1:
                ax.set_yticks(rr)
                ax.set_yticklabels(['']*len(rr))
            else:
                ax.set_yticks(rr)
            if i==2 and j==3:
                ax.set_ylabel(label2)
            axes.update({'%d%d_b'%(i+1,j+1): ax})

    ax = plt.subplot2grid((10, 10), (1*2, 1*2), colspan=2,rowspan=2,fig=fig)
    ax.set_axis_off()
    leg1 = mlines.Line2D([], [], color=colors[0], marker='+',label=ll1,lw=0)
    leg2 = mlines.Line2D([], [], color=colors[0], marker='.',label=ll2,lw=0)
    ax.legend(handles=[leg1,leg2],loc='lower right',fontsize=20, markerscale=2.)

    ax = plt.subplot2grid((10, 10), (0, 0), colspan=2,rowspan=2,fig=fig)
    ax.set_axis_off()
    plt.subplots_adjust(wspace=0.12, hspace=0.12)
    return fig,axes

def make_cosebis_bmode_plot(nmodes):
    '''
    Args:
        nmodes (int):   number of cosebis modes
    '''
    nzs =   4
    axes=   {}
    fig =   plt.figure(figsize=(8,8))

    label2=r'$B_n [\times 10^{10}]$'

    for i in range(nzs):
        for j in range(i,nzs):
            #-----bmode
            ax = plt.subplot2grid((8, 8), (i*2, j*2), colspan=2,rowspan=2,fig=fig)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_title(r'$%d \times %d$' %(i+1,j+1),fontsize=15,y=1.0, pad=-15,x=0.8)
            ax.grid()

            # x-axis
            ax.set_xlim(0.2,nmodes+0.8)
            rr  =   np.arange(1,nmodes+0.8,1)
            if i!=j:
                ax.set_xticks(rr)
                ax.set_xticklabels(['']*len(rr))
            else:
                ax.set_xticks(rr)

            # y-axis
            ax.set_ylim(-4.5,4.5)
            rr  =   np.arange(-4,4.5,2)
            if j!=nzs-1:
                ax.set_yticks(rr)
                ax.set_yticklabels(['']*len(rr))
            else:
                ax.set_yticks(rr)
            if i==2 and j==3:
                ax.set_ylabel(label2)
            axes.update({'%d%d_b'%(i+1,j+1): ax})

    ax = plt.subplot2grid((8, 8), (3*2, 1*2), colspan=2,rowspan=2,fig=fig)
    ax.set_axis_off()
    ll2 = mlines.Line2D([], [], color=colors[0], marker='.',label=r'COSEBIS: $B$-mode',lw=0)
    ax.legend(handles=[ll2],loc='lower right',fontsize=20, markerscale=2.)

    plt.subplots_adjust(wspace=0.12, hspace=0.12)
    return fig,axes

def make_tpcf_plot(title='xi',nzs=4):
    '''Prepares the frames the two-point correlation corner plot
    Args:
        title (str):    title of the figure ['xi', 'thetaxi', 'thetaEB', 'ratio', 'ratio2']
    '''
    axes=   {}
    fig =   plt.figure(figsize=((nzs+1)*2,(nzs+1)*2))
    plt.subplots_adjust(wspace=0, hspace=0)

    if title == 'xi':
        label1=r'$\xi_{+}$'
        label2=r'$\xi_{-}$'
    elif title=='thetaxi':
        label1=r'$\theta\xi_{+} \times 10^4$'
        label2=r'$\theta\xi_{-} \times 10^4$'
    elif title=='thetaEB':
        label1=r'$\theta\xi_{E} \times 10^4$'
        label2=r'$\theta\xi_{B} \times 10^4$'
    elif title=='ratio':
        label1= r'$\delta{\xi_{+}}/\xi_{+}$'
        label2= r'$\delta{\xi_{-}}/\xi_{-}$'
    elif title=='ratio2':
        label1= r'$\delta{\xi_{+}}/\sigma_{+}$'
        label2= r'$\delta{\xi_{-}}/\sigma_{-}$'
    else:
        raise ValueError('title should be xi, thetaxi, thetaEB, ratio or ratio2')

    #-----xip---starts
    for i in range(nzs):
        for j in range(i,nzs):
            ax = plt.subplot2grid(((nzs+1)*2, (nzs+1)*2), ((4-i)*2, (3-j)*2), colspan=2,rowspan=2,fig=fig)
            if title=='thetaxi':
                ax.set_title(r'$%d \times %d$' %(i+1,j+1),fontsize=15,y=1.0, pad=-15,x=0.2)
            else:
                ax.set_title(r'$%d \times %d$' %(i+1,j+1),fontsize=15,y=1.0, pad=-15,x=0.8)
            ax.grid()
            # x-axis
            ax.set_xscale('symlog', linthresh=1e-1)
            ax.xaxis.set_minor_locator(MinorSymLogLocator(1e-1))
            if i!=0:
                ax.set_xticklabels([])
            if i==0 and j==1:
                ax.set_xlabel(r'$\theta$ [arcmin]')
            # y-axis
            if title == 'xi':
                ax.set_ylim(2e-7,2e-3)
                ax.set_yscale('log')
                if j!=nzs-1:
                    ax.set_yticks((1e-6,1e-5,1e-4,1e-3))
                    ax.set_yticklabels(('','','',''))
                else:
                    ax.set_yticks((1e-6,1e-5,1e-4,1e-3))
                if j==nzs-1 and i==2:
                    ax.set_ylabel(label1)
            elif title=='thetaxi':
                ax.set_ylim(-1,(i+2)*2+1.8)
                rr=np.arange(0,(i+2)*2+0.1,2)
                if j!=nzs-1:
                    ax.set_yticks(rr)
                    ax.set_yticklabels(['']*len(rr))
                else:
                    ax.set_yticks(rr)
                if j==nzs-1 and i==2:
                    ax.set_ylabel(label1)
            elif title=='thetaEB':
                ax.set_ylim(-1,(i+2)*2+1.8)
                rr=np.arange(0,(i+2)*2+0.1,2)
                if j!=nzs-1:
                    ax.set_yticks(rr)
                    ax.set_yticklabels(['']*len(rr))
                else:
                    ax.set_yticks(rr)
                if j==nzs-1 and i==2:
                    ax.set_ylabel(label1)
            elif title in ['ratio', 'ratio2']:
                if j!=nzs-1:
                    ax.set_yticklabels([])
                else:
                    pass
                if j==nzs-1 and i==2:
                    ax.set_ylabel(label1)
            else:
                raise ValueError("title should be xi, thetaxi, thetaEB, ratio or ratio2")
            ax.patch.set_alpha(0.1)
            ax.tick_params(direction='out', length=4, width=1., colors='black',
               grid_color='gray', grid_alpha=0.6)
            axes.update({'%d%d_p'%(i+1,j+1): ax})
            del ax
    #-----xip---ends

    #-----xim---starts
    for i in range(nzs):
        for j in range(i,nzs):
            ax = plt.subplot2grid(((nzs+1)*2, (nzs+1)*2), (i*2, (j+1)*2), colspan=2,rowspan=2,fig=fig)
            if title=='thetaxi':
                ax.set_title(r'$%d \times %d$' %(i+1,j+1),fontsize=15,y=1.0, pad=-15,x=0.2)
            else:
                ax.set_title(r'$%d \times %d$' %(i+1,j+1),fontsize=15,y=1.0, pad=-15,x=0.8)
            ax.grid()
            # x-axis
            ax.set_xscale('symlog', linthresh=1e-1)
            ax.xaxis.set_minor_locator(MinorSymLogLocator(1e-1))
            if i!=j:
                ax.set_xticklabels([])

            # y-axis
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            if title == 'xi':
                ax.set_ylim(1e-7,1e-3)
                ax.set_yscale('log')
                if j!=nzs-1:
                    ax.set_yticks((1e-6,1e-5,1e-4,1e-3))
                    ax.set_yticklabels(('','','',''))
                else:
                    ax.set_yticks((1e-6,1e-5,1e-4,1e-3))
                if i==2 and j==3:
                    ax.set_ylabel(label2)
            elif title == 'thetaxi':
                ax.set_ylim(-1,(i+1)*2+1.8)
                rr=np.arange(0,(i+1)*2+0.1,2)
                if j!=nzs-1:
                    ax.set_yticks(rr)
                    ax.set_yticklabels(['']*len(rr))
                else:
                    ax.set_yticks(rr)
                if i==2 and j==3:
                    ax.set_ylabel(label2)
            elif title == 'thetaEB':
                ax.set_ylim(-1,(i+1)*2+1.8)
                rr=np.arange(0,(i+1)*2+0.1,2)
                ax.set_yticks(rr)
                if j!=nzs-1:
                    ax.set_yticklabels(['']*len(rr))
                else:
                    pass
                if i==2 and j==3:
                    ax.set_ylabel(label2)
            elif title in ['ratio', 'ratio2']:
                if j!=nzs-1:
                    ax.set_yticklabels([])
                else:
                    pass
                if i==2 and j==3:
                    ax.set_ylabel(label2)
            else:
                raise ValueError('title should be xi, thetaxi or ratio')
            ax.patch.set_alpha(0.1)
            ax.tick_params(direction='out', length=5, width=1., colors='black',
               grid_color='gray', grid_alpha=0.9)
            axes.update({'%d%d_m'%(i+1,j+1): ax})
            del ax
    #-----xim---ends
    return fig,axes

def plot_cov(covIn,vmin=-3e-12,vmax=3e-11):
    '''makes plot for covariance matrix for cosmic shear
    Args:
        covIn (ndarray):    covariance matrix
        vmin (float):       minimum value
        vmax (float):       maximum value
    Returns:
        fig (figure):       figure for covariance
    '''
    fig=plt.figure(figsize=(12,10))
    norm=SymLogNorm(linthresh=1e-13,vmin=vmin,vmax=vmax)
    ax=sns.heatmap(covIn,cmap="RdBu",norm=norm, square=True,)
    ax.invert_yaxis()
    ny,nx=covIn.shape
    ax.set_xticks(np.arange(0,nx,20))
    ax.set_xticklabels([str(i) for i in np.arange(0,nx,20)])
    ax.set_yticks(np.arange(0,ny,20))
    ax.set_yticklabels([str(i) for i in np.arange(0,ny,20)])
    return fig

def plot_cov_coeff(covIn):
    '''makes plot for covariance matrix for cosmic shear
    Args:
        covIn (ndarray): covariance coefficients
    Returns:
        fig (figure):   figure for covariance coefficients
    '''
    fig, axes = make_figure_axes(nx=1,ny=1)
    ax=axes[0]
    im = ax.imshow(covIn, cmap='bwr', vmin=-1, vmax=1,origin='lower')
    fig.colorbar(im)
    ny,nx=covIn.shape
    ax.set_xticks(np.arange(0,nx+1,20))
    ax.set_yticks(np.arange(0,ny+1,20))
    return fig

def plot_xipm_data(fname,axes,extnms=['xi_plus','xi_minus']):
    """Makes cornor plots for xip and xim from cosmosis data file [fits]
    Args:
        fname (str):    a fits file name
        axes (dict):    a dictionary of axis generated by `make_tpcf_plot`
        extnms (list):  a list of the extension names
    """
    hdul=pyfits.open(fname)
    nzs=4
    for i in range(nzs):
        for j in range(i,nzs):
            ax=axes['%d%d_p'%(i+1,j+1)]
            _mskp=(hdul[extnms[0]].data['BIN1']==(i+1)) & (hdul[extnms[0]].data['BIN2']==(j+1))
            _dp=hdul[extnms[0]].data[_mskp]
            xx=_dp['ang']
            yy=_dp['value']*xx*1e4
            ax.plot(xx,yy,linestyle='--',color=colors[0],linewidth=1.0)
            del xx,yy,_mskp,_dp
            #---
            ax=axes['%d%d_m'%(i+1,j+1)]
            _mskp=(hdul[extnms[1]].data['BIN1']==(i+1)) & (hdul[extnms[1]].data['BIN2']==(j+1))
            _dp=hdul[extnms[1]].data[_mskp]
            xx=_dp['ang']
            yy=_dp['value']*xx*1e4
            ax.plot(xx,yy,linestyle='--',color=colors[0],linewidth=1.0)
            del xx,yy
    return
