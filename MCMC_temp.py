import emcee
from satellites_forMC import *
import os
import corner
import numpy as np
from multiprocessing import Pool
import plots
import time
import numpy as np
import utils
import matplotlib as mpl
from scipy.stats import norm
mpl.use('Agg')
import matplotlib.pylab as plt
import pandas as pd
import dask.dataframe as dd
from satellites_forMC import *
from plots import *
from multiprocessing import Pool
from functools import partial
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
#mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 3.
mpl.rcParams['axes.titlepad'] = 20
#plt.rcParams['axes.linewidth']=5
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5 
mpl.rcParams['axes.titlepad'] = 20 

os.environ["OMP_NUM_THREADS"] = "1"

class sampler:
    
    def __init__(self, priors, prior_type='flat'):
        '''priors is a dictionary that contains the priors. If the priors are flat, it expects upper and lower bound, if gaussian it expects mean and sigma'''
        self.priors = priors
        self.prior_type = prior_type
        self.make = make_satellites_only(use_peak=True)
        self.dict_obs = self.load()
        self.Vol = (250/0.7)**3
        if self.prior_type=='gaussian':
            self.PM11 = partial(norm.logpdf, loc=self.priors['M11'][0], scale=self.priors['M11'][1])
            self.PN11 = partial(norm.logpdf, loc=self.priors['N11'][0], scale=self.priors['N11'][1])
            self.Pgamma11 = partial(norm.logpdf, loc=self.priors['gamma11'][0], scale=self.priors['gamma11'][1])

    
    def lnlike(self,theta):   

        likely=0.
        gamma11,M11, N11 = theta
        dict_SMHM = dict(gamma10=0.57, gamma11= gamma11, beta10=None, beta11=None,\
                    M10=11.95, SHMnorm10=None, M11=M11, SHMnorm11=N11)         
        df_sat = self.make.make_sat(dict_SMHM=dict_SMHM,Mstar_low=11.2, Mstar_up=12, mu=2.5,AK=0.013,sigmaK=0.1, M0=1.5)

        bins = np.append(self.dict_obs['rbins']-self.dict_obs['rbinwidth']/2, self.dict_obs['rbins'][-1]+self.dict_obs['rbinwidth']/2) #restore original bins
        model = np.histogram(df_sat['Re_sat'], bins=bins)[0]/self.Vol/self.dict_obs['rbinwidth']    

        assert len(model)==len(self.dict_obs['rbins'])
        
        for n in range(0,len(model)):
            if (not np.isnan(self.dict_obs['phi'][n]) and (not np.isnan(model[n])) and (self.dict_obs['phi'][n]!=0)):
                sigma2inv=1./self.dict_obs['err'][n]**2
                likely+=( ((self.dict_obs['phi'][n]-model[n])**2)*sigma2inv + np.log(2*np.pi*1./sigma2inv))   
        likely=-0.5*likely 
        return likely

    def IsInRange(self,limits,val):
        if val < limits[1] and val > limits[0]:
            return True
        return False

    def lnprior(self,theta):
        gamma11, M11, N11 = theta
        if self.prior_type == 'flat':
            TM11 = self.IsInRange(self.priors['M11'],M11)
            TN11 = self.IsInRange(self.priors['N11'],N11)
            Tgamma11 = self.IsInRange(self.priors['gamma11'],gamma11)

            if (TM11 and TN11 and Tgamma11):
                return 0.0
            return -np.inf
        elif self.prior_type == 'gaussian':
            priorM11 = self.PM11(M11)
            priorgamma11 = self.Pgamma11(gamma11)
            priorN11 = self.PN11(N11)
            return priorM11+priorgamma11+priorN11
        
    def lnprob(self,theta):

        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)


    def load(self):
        R,P,s=np.loadtxt('/home/lz1f17/data/Rmaj/sat/11.6/Re_all11.6satJKcleaned_ETGs.txt', unpack=True)
        rbinwidth = R[1]-R[0]
        #mask = np.ma.masked_less(P, 1.e-6).mask
        #R = R[mask]
        #P = P[mask]
        #s = s[mask]
        #s = s+0.1*P
        dict_obs = {'rbins':R,'phi':P,'err':s, 'rbinwidth':rbinwidth}

        return dict_obs
    
    
    def sub(self,a):
        return np.abs(a[1]-a[0])/5

    def run_chain(self,n_walkers, n_steps,ntemps, reset = False, name='mcmc', filename='MCMC.h5'):
        ndim = 3
        theta_initial = [0,0.58,-0.014] #gamma11, M11, N11
        if self.prior_type=='flat':
            cov = np.diagflat([self.sub(self.priors['gamma11']), self.sub(self.priors['M11']), self.sub(self.priors['N11'])])
        elif self.prior_type=='gaussian':
            cov = np.diagflat([self.priors['gamma11'][1], self.priors['M11'][1], self.priors['N11'][1]])

        
        walkers_positions=np.zeros((ntemps,n_walkers,ndim))
        lnikely=np.zeros((ntemps,n_walkers,n_steps))

        i=0
        j=0

        while (i < ntemps):
            while (j < n_walkers):
                ran=np.random.multivariate_normal(theta_initial,cov)
                if self.prior_type=='flat':
                    if self.lnprior(ran)==0.0:
                        walkers_positions[i][j]=ran
                elif self.prior_type=='gaussian':
                    walkers_positions[i][j]=ran
                j+=1
            i+=1
            j=0
        



        with Pool() as pool:

            Sampler = emcee.PTSampler(nwalkers = n_walkers, dim=ndim, ntemps=ntemps, logl = self.lnlike, logp=self.lnprob, pool=pool)
            start = time.time()
            Sampler.run_mcmc(walkers_positions, N=n_steps, storechain=True)#,progress=True)#,skip_initial_state_check=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))                    
        chain = np.save('MCMC_PT',Sampler.chain)
        chi2 = np.save('chi2', Sampler.lnprobability)
        return Sampler
    
    


class postprocess:
    def __init__(self, sampler):
        '''sampler: coming from class above. maker: its a make_satellite instance'''
        self.sampler = sampler
        self.samples = self.sampler.flatchain
        self.samples = self.samples.reshape((-1,self.samples.shape[2]))
        print(self.samples.shape)
        ##self.log_prob_samples = self.sampler.get_log_prob(flat=True)
        #self.log_prior_samples = self.sampler.get_blobs(flat=True)
    
    def plot_chain(self):

        labels=[r'$\gamma_z$',r'$M_z$',r'$N_z$']
        #print('autocorrelation tims is: {}'.format(self.sampler.get_autocorr_time()))
        fig = corner.corner(self.samples, labels=labels)
        axes = fig.axes
        for a in axes:
            for tick in a.xaxis.get_major_ticks():
                tick.label.set_fontsize(10)
            for tick in a.yaxis.get_major_ticks():
                tick.label.set_fontsize(10)        
        plt.savefig('Pictures/cornerplot.pdf', bbox_inches='tight')
        plt.close()
        
    def load(self):
        R,P,s=np.loadtxt('/home/lz1f17/data/Rmaj/sat/11.6/Re_all11.6satJKcleaned_ETGs.txt', unpack=True)
        rbinwidth = R[1]-R[0]
        s = s+0.1*P
        dict_obs = {'rbins':R,'phi':P,'err':s, 'rbinwidth':rbinwidth}

        return dict_obs
    
    def check_model(self):
        
        gamma11, M11, N11 = map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]), zip(*np.percentile(self.samples,[16,50,84],axis=0)))

        means = [gamma11[0], M11[0], N11[0]]
        ups = [gamma11[1], M11[1], N11[1]]
        downs = [gamma11[2], M11[2], N11[2]]
        print(means, ups, downs)
    
        dict_obs = self.load()

    
        dict_SMHM = dict(gamma10=0.57, gamma11= gamma11[0], beta10=None, beta11=None,\
                    M10=11.95, SHMnorm10=None, M11=M11[0], SHMnorm11=N11[0])         
      
        maker = make_satellites_only(use_peak=True, use_Multidark=False)
        df_sat = maker.make_sat(dict_SMHM=dict_SMHM,Mstar_low=11.2, Mstar_up=12, mu=2.5,AK=0.013,sigmaK=0.1, M0=1.5)

        bins = np.append(dict_obs['rbins']-dict_obs['rbinwidth']/2, dict_obs['rbins'][-1]+dict_obs['rbinwidth']/2) #restore original bins              
        Vol = (250/0.7)**3
        model = np.histogram(df_sat['Re_sat'], bins=bins)[0]/Vol/dict_obs['rbinwidth']       
        
        
        plt.errorbar(dict_obs['rbins'],dict_obs['phi'],yerr=dict_obs['err'],label='SDSS sat', markersize=15,color='navy', fmt='v',zorder=1)
        plt.plot(dict_obs['rbins'], model, lw=4, label='satellites, MCMC', color='navy', ls='-', zorder=2)
        plt.yscale('log')

        plt.ylabel('$\phi(R_e)$')
        plt.xlabel('$\log{R_e} \ [kpc]$')
        plt.xlim(0.25,2.5)
        plt.ylim(3.e-9,5.e-4)
        
        #plt.ylim(5.e-9,1.e-4)
        plt.legend(frameon=False, fontsize=30)        
        plt.title('Evolving SMHM, MQGs')
        plt.savefig('./Pictures/modelMCMC.pdf', bbox_inches='tight')
        plt.close()
        
