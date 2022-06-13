        #TODO: 
        # - Allow windows to be read from a file.
        
import dask
import cProfile,pstats
from dask import delayed
import sparse
from skylens.wigner_transform import *
from skylens.binning import *
import numpy as np
import jax.numpy as jnp
import healpy as hp
from scipy.interpolate import interp1d
import warnings,logging
from distributed import LocalCluster
from dask.distributed import Client,get_client,Semaphore
import zarr
from dask.threaded import get
from distributed.client import Future
import time,gc
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool as Pool
from skylens.utils import *
import pickle
import copy
import psutil
from itertools import islice
# from jax import jit
import jax
import logging

class window_utils():
    def __init__(self,window_l=None,window_lmax=None,l=None,l_bins=None,l_cl=None,corrs=None,s1_s2s=None,use_window=None,#f_sky=None,
                corr_indxs=None,z_bins=None,WT=None,xi_bin_utils=None,do_xi=None,
                store_win=None,Win=None,wigner_files=None,step=None,keep_wig3j_in_memory=None,#xi_win_approx=False,
                client=None,scheduler_info=None,wigner_step=None,
                skylens_b_xi=None,bin_theta_window=None,njobs_submit_per_worker=10,zarr_parallel_read=25,
                skylens_class0=None,skylens_class_b=None,bin_window=None,do_pseudo_cl=None):

        self.__dict__.update(locals()) #assign all input args to the class as properties
        if self.l_cl is None:
            self.l_cl=self.l
        nl=len(self.l)
        nwl=len(self.window_l)*1.0
        
        self.MF=(2*self.l_cl[:,None]+1)# this is multiplied with coupling matrices, before binning.

        workers,self.nworkers,self.njobs_submit=get_workers_njobs(scheduler_info,njobs_submit_per_worker)
        
        self.step=wigner_step
        if self.step is None:
#             self.step=jnp.int32(200.*((2000./nl)**2)*(100./nwl)) #small step is useful for lower memory load
            self.step=jnp.int32(nl/self.nworkers/2)
            if nl/self.step<self.nworkers:
                self.step=jnp.int32(nl/self.nworkers)
            self.step=jnp.int32(min(self.step,nl+1))
            self.step=jnp.int32(max(self.step,1))
            
        print('Win gen: step size',self.step,nl,nwl,self.nworkers,self.use_window,self.Win is None)#,self.lms)    
        self.lms=np.int32(np.arange(nl,step=self.step))
        self.workers_lm={}
        worker_i=0

        for lm in self.lms:
            if scheduler_info is not None:
                self.workers_lm[lm]=workers[worker_i%self.nworkers]
            else:
                self.workers_lm[lm]=None
            worker_i+=1
        
    def get_Win(self,):
        if not self.use_window:
            return
        if self.Win is not None:
            print('window exists, will not compute. Size: ',get_size_pickle(self.Win))
            return
        print('Win not found, computing now')
        self.set_binning()
        
        self.cl_keys=self.set_cl_keys(corrs=None,corr_indxs=None)
        print('get_Win, total cl keys: ',len(self.cl_keys))
        
        client=get_client(address=self.scheduler_info['address'])
        
        wig_3j_2=None
        
        WT_kwargs,xibu=self.get_xi_win_WT()
        
        self.do_cl=True
        Win_cl=self.set_window_cl(WT_kwargs=WT_kwargs,xi_bin_utils=xibu)
        if self.do_xi:
            self.Win=self.get_xi_win(Win_cl=Win_cl)

        if self.do_pseudo_cl:
            if self.do_xi:
                print('Warning: window for xi is different from cl.')
            wig_3j_2=self.set_wig3j(self.keep_wig3j_in_memory)
            self.Win=self.get_cl_win(Win_cl=Win_cl,wig_3j_2=wig_3j_2)

        self.cleanup()
            
    def get_cl_win(self,Win_cl=None,wig_3j_2=None):
        if self.store_win:
            Win=self.set_store_window(corrs=self.corrs,corr_indxs=self.corr_indxs,client=None,
                                  cl_bin_utils=self.cl_bin_utils,Win_cl=Win_cl,wig_3j_2=wig_3j_2)
        else:
            Win=self.set_window_graph(corrs=self.corrs,corr_indxs=self.corr_indxs,
                                  client=None,cl_bin_utils=self.cl_bin_utils,Win_cl=Win_cl,wig_3j_2=wig_3j_2)
        return Win
            
    def get_xi_win_WT(self):
        if not self.do_xi:
            return None,None
        WT_kwargs={'cl':None}
        client=client_get(self.scheduler_info)
        
        s1_s2=(0,0)
        lcl=client.scatter(self.window_l,broadcast=True)
        
        WT_kwargs={'cl':{'l_cl':lcl,'s1_s2':s1_s2,'wig_d':self.WT.wig_d[(0,0)],
                         'wig_l':self.WT.l,'wig_norm':self.WT.wig_norm,'grad_l':self.WT.grad_l}}
        WT_kwargs=scatter_dict(WT_kwargs,broadcast=True,scheduler_info=self.scheduler_info,depth=2)
        
        xibu=None
        if self.xi_bin_utils is not None:
            xibu=self.xi_bin_utils[(0,0)]
        return WT_kwargs,xibu
    
    def get_xi_win(self,Win_cl=None):
        client=client_get(self.scheduler_info)
        Win=delayed(self.combine_coupling_xi)(Win_cl)
        print('Got xi win graph')
        if self.store_win:
            if client is None and self.scheduler_info is None:
                client=get_client()
            elif client is None and self.scheduler_info is not None:
                client=get_client(address=self.scheduler_info['address'])
            Win=client.compute(Win)#.result()
        return Win
    
    def set_binning(self,):
        """
            Set binning if the coupling matrices or correlation function windows need to be binned.
        """
        self.binning=binning()
        self.c_ell0=None
        self.c_ell_b=None

        self.cl_bin_utils=None
        if self.bin_window:
            self.binnings=binning()
            self.cl_bin_utils=self.skylens_class0.cl_bin_utils

            self.c_ell0=self.skylens_class0.cl_tomo()['cl']
            if self.skylens_class_b is not None:
                self.c_ell_b=self.skylens_class_b.cl_tomo()['cl']
            else:
                self.c_ell_b=self.skylens_class0.cl_tomo()['cl_b']
        self.xi0=None
        self.xi_b=None
        if self.bin_theta_window and self.do_xi:
            self.xi0=self.skylens_class0.xi_tomo()['xi']
            self.xi_b=self.skylens_b_xi.xi_tomo()['xi']
        keys=['skylens_class0','skylens_class_b','skylens_b_xi']
        for k in keys:
            if hasattr(self,k):
                del self.__dict__[k]
    
    def wig3j_step_read(self,m=0,lm=None,sem_lock=None):
        """
        wigner matrices are large. so we read them step by step
        """
        step=self.step
        wig_3j=zarr.open(self.wigner_files[m],mode='r')
        if sem_lock is None:
            out=wig_3j.oindex[jnp.int32(self.window_l),jnp.int32(self.l[lm:lm+step]),jnp.int32(self.l_cl)]
        else:
            with sem_lock:
                out=wig_3j.oindex[jnp.int32(self.window_l),jnp.int32(self.l[lm:lm+step]),jnp.int32(self.l_cl)]
        out=out.transpose(1,2,0)
        del wig_3j
        return out

    def set_wig3j_step_multiplied(self,lm=None,sem_lock=None):
        """
        product of two partial migner matrices
        """
        wig_3j_2={}
        wig_3j_1={m1: self.wig3j_step_read(m=m1,lm=lm,sem_lock=sem_lock) for m1 in self.m_s}
        mi=0
        for m1 in self.m_s:
            for m2 in self.m_s[mi:]:
                wig_3j_2[str(m1)+str(m2)]={0:wig_3j_1[m1]*wig_3j_1[m2].astype('float64')} #numpy dot appears to run faster with 64bit ... ????
#                 if m1!=0 or m2!=0:
                mf=self.set_window_pm_step(lm=lm)
                wig_3j_2[str(m1)+str(m2)][2]=wig_3j_2[str(m1)+str(m2)][0]*mf['mf_p']
                wig_3j_2[str(m1)+str(m2)][-2]=wig_3j_2[str(m1)+str(m2)][0]*(1-mf['mf_p'])
            mi+=1
#         del wig_3j_1
#         open_fd = len(psutil.Process().open_files())
        print('got wig3j',lm)#,get_size_pickle(wig_3j_2),thread_count(),open_fd)
        return wig_3j_2

    def set_wig3j_step_spin(self,wig2,mf_pm,W_pm):
        """
        wig2 is product of two wigner matrices. Here multply with the spin dependent factors
        """
        if W_pm==2: #W_+
            mf=mf_pm['mf_p']#.astype('float64') #https://stackoverflow.com/questions/45479363/numpy-multiplying-large-arrays-with-dtype-int8-is-slow
        if W_pm==-2: #W_+
            mf=1-mf_pm['mf_p']
        return wig2*mf

    def set_window_pm_step(self,lm=None):
        """
        Here we set the spin dependent multiplicative factors (X+, X-).
        """
        li1=jnp.int32(self.window_l).reshape(len(self.window_l),1,1)
        li3=jnp.int32(self.l).reshape(1,1,len(self.l))
        li2=jnp.int32(self.l[lm:lm+self.step]).reshape(1,len(self.l[lm:lm+self.step]),1)
        mf=(-1.)**(li1+li2+li3)
        mf=mf.transpose(1,2,0)
        out={}
        out['mf_p']=(1.+mf)/2.
        # out['mf_p']=jnp.int8((1.+mf)/2.)#.astype('bool')
                              #bool doesn't help in itself, as it is also byte size in numpy.
                              #we donot need to store mf_n, as it is simply a 0-1 flip or "not" when written as bool
                              #using bool or int does cost somewhat in computation as numpy only computes with float 64 (or 32 
                              #in 32 bit systems). If memory is not an
                              #issue, use float64 here and then use mf_n=1-mf_p.
#         del mf
        return out

    def set_wig3j(self,set_wig_3j_2=False):
        """
        Set up a graph (dask delayed), where nodes to read in partial wigner matrices, get their products and 
        also the spin depednent multiplicative factors.
        """
        self.wig_3j={}
        if not self.use_window:
            return

        m_s=jnp.concatenate([jnp.abs(jnp.array(i)).flatten() for i in self.s1_s2s.values()])
        self.m_s=jnp.sort(jnp.unique(m_s))
        self.m_s=np.array(self.m_s)

        print('wigner_files:',self.wigner_files)

        if self.store_win:
            client=client_get(scheduler_info=self.scheduler_info) #this seems to be the correct thing to do

        
        self.wig_3j_2={}
        self.wig_3j_1={}
        self.mf_pm={}
#         self.sem_lock = Semaphore(max_leases=self.zarr_parallel_read, name="database",client=client_get(self.scheduler_info))
        wig_3j_2={lm:None for lm in self.lms}
        mf_pm={lm:None for lm in self.lms}
        if set_wig_3j_2:
            for lm in self.lms: #set in get_coupling_lm_all_win
                wig_3j_2[lm]=delayed(self.set_wig3j_step_multiplied)(lm=lm)#,sem_lock=self.sem_lock)
                wig_3j_2[lm]=client.compute(wig_3j_2[lm],worker=self.workers_lm[lm],allow_other_workers=False)
#             self.mf_pm[lm]=delayed(self.set_window_pm_step)(lm=lm)
            
        self.wig_s1s2s={}
        for corr in self.corrs:
            mi=jnp.sort(jnp.absolute(jnp.array(self.s1_s2s[corr])).flatten())
            self.wig_s1s2s[corr]=str(mi[0])+str(mi[1])
        print('wigner done',self.wig_3j.keys())
        return wig_3j_2
    
    def coupling_matrix_large(self,win,wig_3j_2,mf_pm,bin_wt,W_pm,lm,cl_bin_utils=None):
        """
        get the large coupling matrices from windows power spectra, wigner functions and spin dependent 
        multiplicative factors. Also do the binning if called for. 
        This function supports on partial matrices.
        """
        wig=wig_3j_2[W_pm]

        M={}
        for k in win.keys():
            mf=self.MF[lm:lm+self.step,:]
            
            M[k]=coupling_M(wig,win[k],self.window_l,mf)

            if self.bin_window:# and bin_wt is not None:
                M[k]=self.binnings.bin_2d_coupling(M=M[k],bin_utils=cl_bin_utils,
                    partial_bin_side=2,lm=lm,lm_step=self.step,wt0=bin_wt[k]['wt0'],wt_b=bin_wt[k]['wt_b'])
                        #FIXME: Wrong binning for noise.
                
        return M

    def mask_comb(self,win1,win2): 
        """
        combined the mask from two windows which maybe partially overlapping.
        Useful for some covariance calculations, specially SSC, where we assume a uniform window.
        """
        W=win1*win2
        x=jnp.logical_or(win1==hp.UNSEEN, win2==hp.UNSEEN)
        W[x]=hp.UNSEEN
        W[~x]=1. #mask = 0,1
        fsky=(~x).mean()
        return fsky,W#.astype('int16')

    def get_cl_coupling_lm(self,corr_indxs,win,lm,wig_3j_2_lm,mf_pm,cl_bin_utils=None):
        """
        This function gets the partial coupling matrix given window power spectra and wigner functions. 
        Note that it get the matrices for both signal and noise as well as for E/B modes if applicable.
        """
#         indx=self.cl_keys.index(corr_indxs)
        win0=win#[indx]
        win2={}
        i=0
        k=win0['corr']+win0['indxs']

        corr=(k[0],k[1])
        
        wig_3j_2=wig_3j_2_lm[self.wig_s1s2s[corr]]
#         win=win0#[i]#[k]
#             assert win['corr']==corr
        win2={'M':None,'M_noise':None,'M_B':None,'M_B_noise':None,'binning_util':win['binning_util']}
        for kt in ['corr','indxs','s1s2']:
            win2[kt]=win0[kt]
        if lm==0:
            win2=copy.deepcopy(win)
        
        win_M=self.coupling_matrix_large(win[12], wig_3j_2=wig_3j_2,mf_pm=mf_pm,bin_wt=win['bin_wt']
                                     ,W_pm=win['W_pm'],lm=lm,cl_bin_utils=cl_bin_utils)
        win2['M']=win_M['cl']
        if 'N' in win_M.keys():
            win2['M_noise']=win_M['N']
        if win['corr']==('shear','shear') and win['indxs'][0]==win['indxs'][1]: #B mode.
            win_M_B=self.coupling_matrix_large(win[12],wig_3j_2=wig_3j_2,mf_pm=mf_pm,W_pm=-2,bin_wt=win['bin_wt'],
                                            lm=lm,cl_bin_utils=cl_bin_utils)
            win2['M_B_noise']=win_M_B['N']
            win2['M_B']=win_M_B['cl']
        i+=1
        return win2
            
    def combine_coupling_cl(self,result):
        """
        This function combines the partial coupling matrices computed above. It loops over all combinations of tracers
        and returns a dictionary of coupling matrices for all C_ells.
        """
        dic={}
        nl=len(self.l)
        crash
        if self.bin_window:
            nl=len(self.l_bins)-1

        for ii_t in range(len(self.cl_keys)): #list(result[0].keys()):
            ii=ii_t#0#because we are deleting below
            ckt=self.cl_keys[ii_t]
            print('combine_coupling_cl, here1')
            result_ii=result[0][ii]
            corr=result_ii['corr']
            indxs=result_ii['indxs']

            result0={}
            for k in result_ii.keys():
                result0[k]=result_ii[k]
            print('combine_coupling_cl, here2')
            result0['M']=jnp.zeros((nl,nl))
            if  result_ii['M_noise'] is not None:
                result0['M_noise']=jnp.zeros((nl,nl))
            if corr==('shear','shear') and indxs[0]==indxs[1]:
                result0['M_B_noise']=jnp.zeros((nl,nl))
                result0['M_B']=jnp.zeros((nl,nl))
            print('combine_coupling_cl, here3')
            for i_lm in range(len(self.lms)):
                lm=jnp.asscalar(self.lms[i_lm])
                start_i=0 if self.bin_window else lm
                end_i=nl if self.bin_window else lm+self.step
                
                result0['M']=result0['M'].at[start_i:end_i,:].set(result[lm][ii]['M'])

                if  result_ii['M_noise'] is not None:
                    result0['M_noise']=result0['M_noise'].at[start_i:end_i,:].set(result[lm][ii]['M_noise'])
                if corr==('shear','shear') and indxs[0]==indxs[1]:
                    result0['M_B_noise']=result0['M_B_noise'].at[start_i:end_i,:].set(result[lm][ii]['M_B_noise'])
                    result0['M_B']=result0['M_B'].at[start_i:end_i,:].set(result[lm][ii]['M_B'])

            corr21=corr[::-1]
            if dic.get(corr) is None:
                dic[corr]={}
            if dic.get(corr21) is None:
                dic[corr21]={}
            dic[corr][indxs]=result0
            dic[corr[::-1]][indxs[::-1]]=result0
        return dic
    
    def combine_coupling_xi(self,result):
        dic={}
        i=0
        for ii in self.cl_keys: #list(result.keys()):
            result_ii=result[i]
            corr=result_ii['corr']
            indxs=result_ii['indxs']
            corr21=corr[::-1]
            if dic.get(corr) is None:
                dic[corr]={}
            if dic.get(corr21) is None:
                dic[corr21]={}
            dic[corr][indxs]=result_ii
            dic[corr[::-1]][indxs[::-1]]=result_ii
            i+=1
        return dic
    

    def set_cl_keys(self,corrs=None,corr_indxs=None):
        if corrs is None:
            corrs=self.corrs
        if corr_indxs is None:
            corr_indxs=self.corr_indxs
        cl_keys=[corr+indx for corr in corrs for indx in corr_indxs[corr]]
        return cl_keys
    
    def set_window_cl(self,corrs=None,corr_indxs=None,npartitions=100,z_bins=None,
                     WT_kwargs=None,xi_bin_utils=None):
        """
        This function sets the graph for computing power spectra of windows 
        for both C_ell and covariance matrices.
        """
        if corrs is None:
            corrs=self.corrs
        if corr_indxs is None:
            corr_indxs=self.corr_indxs
        if z_bins is None:
            z_bins=self.z_bins
        if WT_kwargs is None:
            WT_kwargs={'cl':None}
        t1=time.time()
        client=client_get(scheduler_info=self.scheduler_info)
        client_func=client.compute #used later
        
        if self.store_win and self.do_cl:
            client_func=client.compute #used later
        
            if self.c_ell0 is not None:
                self.c_ell0=client.compute(self.c_ell0).result()
                self.c_ell_b=client.compute(self.c_ell_b).result()
#                 replicate_dict(self.c_ell0, branching_factor=1,scheduler_info=self.scheduler_info)#doesn't work because we need to change the depth of Future
#                 replicate_dict(self.c_ell_b, branching_factor=1,scheduler_info=self.scheduler_info)
                self.c_ell0=scatter_dict(self.c_ell0,scheduler_info=self.scheduler_info,broadcast=True,depth=2)
                self.c_ell_b=scatter_dict(self.c_ell_b,scheduler_info=self.scheduler_info,broadcast=True,depth=2)

            if self.xi0 is not None:
                self.xi0=client.compute(self.xi0).result()
                self.xi_b=client.compute(self.xi_b).result()
                self.xi0=scatter_dict(self.xi0,scheduler_info=self.scheduler_info,broadcast=True,depth=2)
                self.xi_b=scatter_dict(self.xi_b,scheduler_info=self.scheduler_info,broadcast=True,depth=2)
        
        print('set window_cl: cl0,cl_b done',time.time()-t1)
        WU=client.scatter(self,broadcast=True)
        Win_cl=None
        if self.do_cl:
            def Win_cli():
                i=0
                while i<len(self.cl_keys):
                    ck=self.cl_keys[i]
                    corr=(ck[0],ck[1])
                    indx=(ck[2],ck[3])
                    c_ell0={corr:{indx:self.c_ell0[corr][indx]}} if self.c_ell0 is not None else None
                    c_ell_b={corr:{indx:self.c_ell_b[corr][indx]}}if self.c_ell_b is not None else None
                    yield delayed(get_window_power_cl)(ck,WU,c_ell0=c_ell0,c_ell_b=c_ell_b,
                                                                z_bin1=z_bins[ck[0]][ck[2]],z_bin2=z_bins[ck[1]][ck[3]],
                                                                WT_kwargs=WT_kwargs['cl'],
                                                                xi_bin_utils=xi_bin_utils) 
                    i+=1

            if self.store_win:
                Win_cl=client_func(list(Win_cli()),priority=1)
            else:
                Win_cl=list(Win_cli())
            print('set window_cl: cl done',time.time()-t1,get_size_pickle(self),get_size_pickle(WT_kwargs))
            
        njobs=self.njobs_submit
        if self.store_win:
            if self.do_cl:
                wait_futures(Win_cl)
        print('set_window_cl done',time.time()-t1)
        return Win_cl
        
    def combine_coupling_cl(self,win_lm):
        Win={}
        if self.do_cl:
            win_cl_lm={lmi:win_lm[lmi]['cl'] for lmi in self.lms}
            Win['cl']=self.combine_coupling_cl(win_cl_lm)
        return Win
    
    def combine_coupling_xi(self,win_cl):
        Win={}
        if self.do_cl:
            Win['cl']=self.combine_coupling_xi(win_cl) 
        return Win
    
    def set_store_window(self,corrs=None,corr_indxs=None,client=None,cl_bin_utils=None,
                        Win_cl=None,wig_3j_2={}):
        """
        This function sets and computes the graph for the coupling matrices. It first calls the function to 
        generate graph for window power spectra, which is then combined with the graphs for wigner functions
        to get the final graph.
        """
        if self.store_win:
            client=client_get(scheduler_info=self.scheduler_info) #this seems to be the correct thing to do
        print('setting windows, coupling matrices ',client)
        # if self.do_cl:
        #     print('WU dict size: ',dict_size_pickle(self.__dict__,print_prefact='WU dict size: ',depth=2))
#         self.set_window_cl(corrs=corrs,corr_indxs=corr_indxs,client=client)
#         print('got window cls, now to coupling matrices.',len(self.cl_keys),len(self.cov_keys),self.Win_cl )
        
        Win={}
        WU=client.scatter(self,broadcast=True)
        allow_other_workers=True
        if self.keep_wig3j_in_memory:
            allow_other_workers=False
        if self.do_pseudo_cl: #this is probably redundant due to if statement in init.
            Win_cl_lm={}
            Win_lm={}
            workers=list(client.scheduler_info()['workers'].keys())
            nworkers=len(workers)
            njobs=self.njobs_submit
    
            
            worker_i=0
            job_i=0
            lm_submitted=[]
            
            t1=time.time()
            for lm in self.lms:
                Win_cl_lm[lm]={}
                Win_lm[lm]={}
                
                Win_lm[lm]=delayed(get_coupling_lm_all_win)(WU,Win_cl,lm,wig_3j_2.get(lm),None,cl_bin_utils=cl_bin_utils)
#### Donot delete
                if self.store_win:  
                   client_func=client.compute
                   Win_lm[lm]=client_func(Win_lm[lm],priority=2,workers=self.workers_lm[lm],allow_other_workers=allow_other_workers)
                   worker_i+=1
                   job_i+=1
                   lm_submitted+=[lm]
            Win=delayed(combine_coupling_cl)(WU,Win_lm)

            if self.store_win:
                Win=client.compute(Win,priority=2,workers=self.workers_lm[0],allow_other_workers=allow_other_workers)#.result()
                if len(lm_submitted)>=nworkers:
                    wait_futures(Win_lm,threshold=max(0.5,1-nworkers*1.2/len(lm_submitted) ))
                if self.do_pseudo_cl:
                    self.cleanup()
        print('done combine lm',time.time()-t1)
        return Win
    
    def reduce_win_cl(self,win,win2):
        
        dic=win
        corr=win2['corr']
        corr2=corr[::-1]
        indxs=win2['indxs']
        indxs2=indxs[::-1]
        if dic.get(corr) is None:
            self.Win['cl'][corr]={}
        if dic.get(corr2) is None:
            self.Win['cl'][corr2]={}
        dic[corr2][indxs2]=win2
        dic[corr][indxs]=win2
        return dic
    
    def cleanup(self,): #need to free all references to wigner_3j, mf and wigner_3j_2... this doesnot help with peak memory usage
        pass

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as cf_wait
from concurrent.futures import ALL_COMPLETED
from functools import partial

def get_cl_coupling_lm(WU,lm,wig_3j_2_lm,mf_pm,win,cl_bin_utils=None):
#     print('get_cl_coupling_lm',win,WU,lm)
    return WU.get_cl_coupling_lm(None,win,lm,wig_3j_2_lm,mf_pm,cl_bin_utils=cl_bin_utils)
  
def combine_coupling_cl(WU,win_lm): #win_cl_lm,win_cov_lm):
        t1=time.time()
        self=WU
        Win={}
#         client=client_get(scheduler_info=self.scheduler_info)
        if self.do_cl:
            win_cl_lm={lmi:win_lm[lmi]['cl'] for lmi in self.lms}
            try:
                Win['cl']=self.combine_coupling_cl(win_cl_lm)
            except Exception as err:
                # print('combine_coupling_cl error: ',win_cl_lm[0][0])
                print('combine_coupling_cl error: ',err,Win)
                crash
            del win_cl_lm
        del win_lm,WU
        print('done combine lm',time.time()-t1)
        return Win

    
def get_coupling_lm_all_win(WU,Win_cl,lm,wig_3j_2_lm,mf_pm,cl_bin_utils=None):
    self=WU
    t1=time.time()
    print('get_coupling_lm_all_win',lm)
    if wig_3j_2_lm is None:
        wig_3j_2_lm=self.set_wig3j_step_multiplied(lm=lm)#,sem_lock=self.sem_lock)
        mf_pm=None

    Win_lm={}
    thread_workers=5
    if self.do_cl:
        Win_lm['cl']=[self.get_cl_coupling_lm(None,Wc,lm,wig_3j_2_lm,mf_pm,cl_bin_utils=cl_bin_utils) for Wc in Win_cl]
    print('done lm',lm,time.time()-t1, flush=True)
    return Win_lm

                
                
def get_window_power_cl(corr_indxs,WU,c_ell0=None,c_ell_b=None,z_bin1=None,z_bin2=None,
                       WT_kwargs=None,xi_bin_utils=None):#corr={},indxs={}):
    """
    Get the cross power spectra of windows given two tracers.
    Note that noise and signal have different windows and power spectra for both 
    cases. 
    Spin factors and binning weights if needed are also set here.
    """
#         print('getting window power cl',WT_kwargs)
    self=WU
    corr=(corr_indxs[0],corr_indxs[1])
    indxs=(corr_indxs[2],corr_indxs[3])
    win={}
    win['corr']=corr
    win['indxs']=indxs
    
    s1s2=jnp.absolute(jnp.array(self.s1_s2s[corr])).flatten()
        
    W_pm=0
    if jnp.sum(s1s2)!=0:
        W_pm=2 #we only deal with E mode\
        if corr==('shearB','shearB'):
            W_pm=-2
    if self.do_xi:
        W_pm=0 #for xi estimators, there is no +/-. Note that this will result in wrong thing for pseudo-C_ell.
                #FIXME: hence pseudo-C_ell and xi together are not supported right now

    win[12]={} #to keep some naming uniformity with the covariance window
    win[12]['cl']=jnp.asarray(hp.anafast(map1=z_bin1['window'],map2=z_bin2['window'],
                             lmax=self.window_lmax))[self.window_l]
    
    if corr[0]==corr[1] and indxs[0]==indxs[1]:
        map1=z_bin1['window_N']
        if map1 is None:
            map1=jnp.sqrt(z_bin1['window'])
            mask=z_bin1['window']==hp.UNSEEN
            map1[mask]=hp.UNSEEN        
        win[12]['N']=jnp.asarray(hp.anafast(map1=map1,lmax=self.window_lmax))[self.window_l]

    win['binning_util']=None
    win['bin_wt']=None
    if self.bin_window and self.do_pseudo_cl:
        cl0=c_ell0[corr][indxs]
        cl_b=c_ell_b[corr][indxs]
        win['bin_wt']={}
        win['bin_wt']['cl']={'wt_b':1./cl_b,'wt0':cl0}
        win['bin_wt']['N']={'wt_b':jnp.ones_like(cl_b),'wt0':jnp.ones_like(cl0)}
        if jnp.all(cl_b==0):#avoid nan
            win['bin_wt']['cl']={'wt_b':cl_b*0,'wt0':cl0*0}
            win['bin_wt']['N']={'wt_b':cl_b*0,'wt0':cl0*0}
    win['W_pm']=W_pm
    win['s1s2']=s1s2
    if self.do_xi:
        th,win['xi']=self.WT.projected_correlation(cl=win[12]['cl'],**WT_kwargs)#this is ~f_sky
        win['xi_b']=self.binning.bin_1d(xi=win['xi'],bin_utils=xi_bin_utils) #xi_bin_utils[(0,0)]

    win['M']={} #self.coupling_matrix_large(win['cl'], s1s2,wig_3j_2=wig_3j_2,W_pm=W_pm)*(2*self.l[:,None]+1) #FIXME: check ordering
    win['M_noise']=None
    return win
    
def multiply_window(win1,win2):
    """
    Take product of two windows which maybe partially overlapping and mask it properly.
    """
    W=win1*win2
    x=jnp.logical_or(win1==hp.UNSEEN, win2==hp.UNSEEN)
    W[x]=hp.UNSEEN
    return W

# @jit
def coupling_M(wig,win,window_l,MF):
    # print('coupling_M:',wig.shape,win.shape,window_l.shape,MF.shape)
    # M=wig@(win*(2*window_l+1))
    M=jnp.dot(wig,win*(2*window_l+1))
    M/=4.*jnp.pi
    M*=MF
    return M