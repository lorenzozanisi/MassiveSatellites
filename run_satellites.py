import numpy as np
import matplotlib.pylab as plt
import utils
import matplotlib as mpl
import pandas as pd
import dask.dataframe as dd
from satellites_forMC import *
from plots import hists, plot_sizefunctions_MonteCarlo
from multiprocessing import Pool
mpl.use('Agg')
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



def run_satellites(Nboot, massrange, dict_SMHM,scatterevol,mu,zinfallrange,ax,ax1,ax2,HostHaloMassRange=None, HostHaloMassRange_axes=None):

    rebins = np.arange(-0.2,2.5,0.1)
    
    dimensions = (Nboot, len(rebins)-1)
    hcen = np.zeros(dimensions)
    hsat = {'all':np.zeros(dimensions),'highz':np.zeros(dimensions),'medz':np.zeros(dimensions),'lowz':np.zeros(dimensions)}
    hsat_highmass = {'all':np.zeros(dimensions),'highz':np.zeros(dimensions),'medz':np.zeros(dimensions),'lowz':np.zeros(dimensions)}
    hsat_medmass = {'all':np.zeros(dimensions),'highz':np.zeros(dimensions),'medz':np.zeros(dimensions),'lowz':np.zeros(dimensions)}
    hsat_lowmass = {'all':np.zeros(dimensions),'highz':np.zeros(dimensions),'medz':np.zeros(dimensions),'lowz':np.zeros(dimensions)}

    hcen_SF = np.zeros(dimensions)
    hsat_SF = {'all':np.zeros(dimensions),'highz':np.zeros(dimensions),'medz':np.zeros(dimensions),'lowz':np.zeros(dimensions)}
    hsat_highmass_SF = {'all':np.zeros(dimensions),'highz':np.zeros(dimensions),'medz':np.zeros(dimensions),'lowz':np.zeros(dimensions)}
    hsat_medmass_SF = {'all':np.zeros(dimensions),'highz':np.zeros(dimensions),'medz':np.zeros(dimensions),'lowz':np.zeros(dimensions)}
    hsat_lowmass_SF = {'all':np.zeros(dimensions),'highz':np.zeros(dimensions),'medz':np.zeros(dimensions),'lowz':np.zeros(dimensions)}

    hcen_Q = np.zeros(dimensions)
    hsat_Q = {'all':np.zeros(dimensions),'highz':np.zeros(dimensions),'medz':np.zeros(dimensions),'lowz':np.zeros(dimensions)}
    hsat_highmass_Q = {'all':np.zeros(dimensions),'highz':np.zeros(dimensions),'medz':np.zeros(dimensions),'lowz':np.zeros(dimensions)}
    hsat_medmass_Q = {'all':np.zeros(dimensions),'highz':np.zeros(dimensions),'medz':np.zeros(dimensions),'lowz':np.zeros(dimensions)}
    hsat_lowmass_Q = {'all':np.zeros(dimensions),'highz':np.zeros(dimensions),'medz':np.zeros(dimensions),'lowz':np.zeros(dimensions)}
    
    ###### to do:  bins of parent halo mass
    
    for n in range(Nboot):
        print(n)
        df_high = make.make_censat(dict_SMHM=dict_SMHM,scatterevol=scatterevol,Mstar_low=massrange[0], Mstar_up=massrange[1], mu=mu, AK=0.013,sigmaK=0.1, AK_SF=0.022, sigmaK_SF=0.13, M0=2.)
        print()
        hcen[n], hsat['all'][n],hsat['highz'][n], hsat['medz'][n], hsat['lowz'][n] = hists(df_high, binsR=rebins, zinfallrange=zinfallrange)
        
        hcen_SF[n], hsat_SF['all'][n],hsat_SF['highz'][n], hsat_SF['medz'][n], hsat_SF['lowz'][n] = hists(df_high[df_high['TType']=='LTGs'], binsR=rebins,zinfallrange=zinfallrange)
        
        hcen_Q[n], hsat_Q['all'][n],hsat_Q['highz'][n], hsat_Q['medz'][n], hsat_Q['lowz'][n] = hists(df_high[df_high['TType']=='ETGs'], binsR=rebins,zinfallrange=zinfallrange)
        
        if HostHaloMassRange is not None:   #REMEMBER TO ADD MPEAK CEN
            
            ############### star forming
            
            _,hsat_highmass_SF['all'][n],hsat_highmass_SF['highz'][n], hsat_highmass_SF['medz'][n], hsat_highmass_SF['lowz'][n] = hists(df_high.query("  mvir_host>{} & TType=='LTGs'".format(HostHaloMassRange[2])), binsR=rebins, zinfallrange=zinfallrange)
            
            _,hsat_medmass_SF['all'][n],hsat_medmass_SF['highz'][n], hsat_medmass_SF['medz'][n], hsat_medmass_SF['lowz'][n] = hists(df_high.query("  {}<mvir_host<{} & TType=='LTGs'".format(HostHaloMassRange[1],HostHaloMassRange[2])), binsR=rebins, zinfallrange=zinfallrange)
            
            _,hsat_lowmass_SF['all'][n],hsat_lowmass_SF['highz'][n], hsat_lowmass_SF['medz'][n], hsat_lowmass_SF['lowz'][n] = hists(df_high.query("  mvir_host<{} & TType=='LTGs'".format(HostHaloMassRange[1])), binsR=rebins, zinfallrange=zinfallrange)
            
            ############### quenched
            _,hsat_highmass_Q['all'][n],hsat_highmass_Q['highz'][n], hsat_highmass_Q['medz'][n], hsat_highmass_Q['lowz'][n] = hists(df_high.query("   mvir_host>{} & TType=='ETGs'".format(HostHaloMassRange[2])), binsR=rebins, zinfallrange=zinfallrange)
            
            _,hsat_medmass_Q['all'][n],hsat_medmass_Q['highz'][n], hsat_medmass_Q['medz'][n], hsat_medmass_Q['lowz'][n] = hists(df_high.query("   {}<mvir_host<{} & TType=='ETGs'".format(HostHaloMassRange[1],HostHaloMassRange[2])), binsR=rebins, zinfallrange=zinfallrange)
            
            _,hsat_lowmass_Q['all'][n],hsat_lowmass_Q['highz'][n], hsat_lowmass_Q['medz'][n], hsat_lowmass_Q['lowz'][n] = hists(df_high.query("  mvir_host<{} & TType=='ETGs'".format(HostHaloMassRange[1])), binsR=rebins, zinfallrange=zinfallrange)          
            
            ############### all
            _,hsat_highmass['all'][n],hsat_highmass['highz'][n], hsat_highmass['medz'][n], hsat_highmass['lowz'][n] = hists(df_high.query('   mvir_host>{}'.format(HostHaloMassRange[2])), binsR=rebins, zinfallrange=zinfallrange)
            
            _,hsat_medmass['all'][n],hsat_medmass['highz'][n], hsat_medmass['medz'][n], hsat_medmass['lowz'][n] = hists(df_high.query('   {}<mvir_host<{}'.format(HostHaloMassRange[1],HostHaloMassRange[2])), binsR=rebins, zinfallrange=zinfallrange)
            
            _,hsat_lowmass['all'][n],hsat_lowmass['highz'][n], hsat_lowmass['medz'][n], hsat_lowmass['lowz'][n] = hists(df_high.query(' mvir_host<{}'.format(HostHaloMassRange[1])), binsR=rebins, zinfallrange=zinfallrange)
        
        
        
        
        

       # hcen.append(hcen_)
       # hsat.append(hsat_)
       # hsat_highz.append(hsat_highz_)
       # hsat_medz.append(hsat_medz_)
       # hsat_lowz.append(hsat_lowz_)

        #hcen_SF.append(hcen_SF_)
        #hsat_SF.append(hsat_SF_)
        #hsat_highz_SF.append(hsat_highz_SF_)
        #hsat_medz_SF.append(hsat_medz_SF_)
        #hsat_lowz_SF.append(hsat_lowz_SF_)

        #hcen_Q.append(hcen_Q_)
        #hsat_Q.append(hsat_Q_)
        #hsat_highz_Q.append(hsat_highz_Q_)
        #hsat_medz_Q.append(hsat_medz_Q_)
        #hsat_lowz_Q.append(hsat_lowz_Q_)

    #plot
    if massrange[0]==11.5 and massrange[1] == 12:
        m = 11.75
    if massrange[0]==11.2 and massrange[1] == 12:
        m = 11.6
    if massrange[0]==11.2 and massrange[1] == 11.5:
        m = 11.3
        
    ax = plot_sizefunctions_MonteCarlo(hcen, hsat['all'],hsat['highz'], hsat['medz'], hsat['lowz'],ax,m=m, TType='All', return_infall_range=True, rebins = rebins, return_legend=True,zinfallrange=zinfallrange)

    ax1 = plot_sizefunctions_MonteCarlo(hcen_SF, hsat_SF['all'],hsat_SF['highz'], hsat_SF['medz'], hsat_SF['lowz'],ax1,m=m, TType='LTGs', return_infall_range=True, rebins = rebins,return_legend=True,zinfallrange=zinfallrange)
    ax2 = plot_sizefunctions_MonteCarlo(hcen_Q, hsat_Q['all'],hsat_Q['highz'], hsat_Q['medz'],hsat_Q['lowz'], ax2,m=m, TType='ETGs', return_infall_range=True, rebins = rebins, return_legend=False,zinfallrange=zinfallrange)


    if HostHaloMassRange is not None:
        if mu is not None:
            ax1_SF, ax2_SF, ax3_SF = HostHaloMassRange_axes[0]
            ax1_Q, ax2_Q, ax3_Q = HostHaloMassRange_axes[1]
            
            ax1_SF = plot_sizefunctions_MonteCarlo(hcen_SF, hsat_lowmass_SF['all'],hsat_lowmass_SF['highz'], hsat_lowmass_SF['medz'], hsat_lowmass_SF['lowz'],ax1_SF,m=m, TType='LTGs', return_infall_range=True, rebins = rebins,return_legend=True,zinfallrange=zinfallrange)
            ax2_SF = plot_sizefunctions_MonteCarlo(hcen_SF, hsat_medmass_SF['all'],hsat_medmass_SF['highz'], hsat_medmass_SF['medz'], hsat_medmass_SF['lowz'],ax2_SF,m=m, TType='LTGs', return_infall_range=True, rebins = rebins,return_legend=True,zinfallrange=zinfallrange)
            ax3_SF = plot_sizefunctions_MonteCarlo(hcen_SF, hsat_highmass_SF['all'],hsat_highmass_SF['highz'], hsat_highmass_SF['medz'], hsat_highmass_SF['lowz'],ax3_SF,m=m, TType='LTGs', return_infall_range=True, rebins = rebins,return_legend=True,zinfallrange=zinfallrange)
            
            ax1_Q = plot_sizefunctions_MonteCarlo(hcen_Q, hsat_lowmass_Q['all'],hsat_lowmass_Q['highz'], hsat_lowmass_Q['medz'], hsat_lowmass_Q['lowz'],ax1_Q,m=m, TType='ETGs', return_infall_range=True, rebins = rebins,return_legend=True,zinfallrange=zinfallrange)
            ax2_Q = plot_sizefunctions_MonteCarlo(hcen_Q, hsat_medmass_Q['all'],hsat_medmass_Q['highz'], hsat_medmass_Q['medz'], hsat_medmass_Q['lowz'],ax2_Q,m=m, TType='ETGs', return_infall_range=True, rebins = rebins,return_legend=True,zinfallrange=zinfallrange)
            ax3_Q = plot_sizefunctions_MonteCarlo(hcen_Q, hsat_highmass_Q['all'],hsat_highmass_Q['highz'], hsat_highmass_Q['medz'], hsat_highmass_Q['lowz'],ax3_Q,m=m, TType='ETGs', return_infall_range=True, rebins = rebins,return_legend=True,zinfallrange=zinfallrange)
            
            return ax,ax1,ax2, [ax1_SF, ax2_SF, ax3_SF], [ax1_Q, ax2_Q, ax3_Q]
        
    return ax,ax1,ax2
        
    
    
def save(fig,ax,massrange, dict_SMHM, scatterevol, choice=None, HaloMassRange=None, TType=None):

    ### Ttype to be use only in conjunction with HaloMassRange to obtain two different plots for LTGs andETGs
    if massrange[0]==11.5 and massrange[1] == 12:
        m = 11.75
    if massrange[0]==11.2 and massrange[1] == 12:
        m = 11.6
    if massrange[0]==11.2 and massrange[1] == 11.5:
        m = 11.3
        
    string = []
    for key in dict_SMHM.keys():
        if dict_SMHM[key] is not None:
            string.extend([key, str(dict_SMHM[key])])
    if scatterevol:
        string.extend(['scatterevol'])
        
    string = '_'.join(string)
    
    if choice=='fquench' and (HaloMassRange is None):
        ax1,ax2 = ax
        fig.tight_layout()
        ax2.set_ylabel('')
        ax1.set_title('LTGs')
        ax2.set_title('ETGs')
        ax1.set_ylim(3.e-9)
        ax2.set_ylim(3.e-9)

        fig.suptitle(str(massrange[0])+r'$<\log{M_{\rm star}}/M_\odot<$'+str(massrange[1])+' '+r'$\mu=$'+str(mu),y=1.1, )
        fig.savefig('./Pictures/fquench/Sizefunct_{}_MPeak_{}_mu{}.png'.format(m,string,mu), bbox_inches='tight')
        fig.clf()     
        
    elif (HaloMassRange is not None) and choice=='fquench':
        ax1,ax2,ax3 = ax
        fig.tight_layout()
        ax1.set_ylim(3.e-9)
        ax2.set_ylim(3.e-9)  
        ax3.set_ylim(3.e-9)  
        fig.suptitle(str(massrange[0])+r'$<\log{M_{\rm star}}/M_\odot<$'+str(massrange[1])+' '+r'$\mu=$'+str(mu)+',{}'.format(TType),y=1.1, )
        
        labels = [str(HaloMassRange[0])+r'$<\log{M_{\rm h,host}}/M_\odot<$'+str(HaloMassRange[1]),str(HaloMassRange[1])+r'$<\log{M_{\rm h,host}}/M_\odot<$'+str(HaloMassRange[2]), r'$\log{M_{\rm h,host}}/M_\odot>$'+str(HaloMassRange[2])]
        
        for a,lab in zip(ax,labels):
            a.set_title(lab)
            
        fig.savefig('./Pictures/fquench/Sizefunct_HaloMassRange_{}_MPeak_{}_{}_mu{}.png'.format(m,string,TType,mu), bbox_inches='tight')
        fig.clf()             
            
        
    else:
         
        fig.tight_layout()
        fig.suptitle(str(massrange[0])+r'$<\log{M_{\rm star}}/M_\odot<$'+str(massrange[1]),y=1.05)
    #    plt.text(2,4.e-7,'satellites\ninitialized\nat peak mass')
        plt.xlim(0.,3)
        plt.ylim(5.e-9)
        fig.savefig('./Pictures/all/Sizefunct_{}_MPeak_{}.png'.format(m,string),bbox_inches='tight')
        fig.clf()
    
    
if __name__=='__main__':
    
    
    for mu in [1.,2.]:
        for gamma11 in [None, 0,0.1]:
            for scatterevol in [True, False]:
                Nboot = 10
                make = make_satellites(use_peak=True)    #initializes everything within the class

                # run parameters

                dict_SMHM = dict(gamma10=0.57, gamma11= gamma11, beta10=None, beta11=None,\
                                M10=11.95, SHMnorm10=None, M11=None, SHMnorm11=None) 
                #scatterevol = False
               # mu = 2.5

                zinfallrange = [1,1.5,1.5]

                massrange = [11.2,12]
                m = 11.6
                fig, ax = plt.subplots(1,1)
                fig1,(ax1,ax2) = plt.subplots(1,2, figsize=(32,16), sharey=True)
                figSF,axSF = plt.subplots(1,3, figsize=(48,16), sharey=True)
                figQ,axQ = plt.subplots(1,3, figsize=(48,16), sharey=True)
                HaloMassRange = [12.5,13.3,14]
                print()
                ax,ax1,ax2, axSF, axQ = run_satellites(Nboot, massrange, dict_SMHM,scatterevol,mu,zinfallrange,ax,ax1,ax2,HaloMassRange, HostHaloMassRange_axes = [axSF,axQ])

                save(fig, ax, massrange, dict_SMHM, scatterevol)
                save(fig1, [ax1,ax2], massrange, dict_SMHM, scatterevol, choice='fquench')
                save(figSF, axSF,massrange,dict_SMHM, scatterevol, choice='fquench', HaloMassRange=HaloMassRange, TType='LTGs' )
                save(figQ, axQ,massrange,dict_SMHM, scatterevol, choice='fquench', HaloMassRange=HaloMassRange, TType='ETGs' )
    raise ValueError
    
    Nboot = 10
    make = make_satellites(use_peak=True)    #initializes everything within the class

    # run parameters
    
    dict_SMHM = dict(gamma10=0.57, gamma11= None, beta10=None, beta11=None,\
                    M10=11.95, SHMnorm10=None, M11=None, SHMnorm11=None) 
    scatterevol = False
    mu = 2.5
    
    zinfallrange = [1,1.5,1.5]
    
    massrange = [11.2,12]
    m = 11.6
    fig, ax = plt.subplots(1,1)
    fig1,(ax1,ax2) = plt.subplots(1,2, figsize=(32,16), sharey=True)
    
    ax,ax1,ax2 = run_satellites(Nboot, massrange, dict_SMHM,scatterevol,mu,zinfallrange,ax,ax1,ax2)
    
    save(fig, ax, massrange, dict_SMHM, scatterevol)
    save(fig1, [ax1,ax2], massrange, dict_SMHM, scatterevol, choice='fquench')
    

    
    
    Nboot = 10
    make = make_satellites(use_peak=True)    #initializes everything within the class

    # run parameters
    
    dict_SMHM = dict(gamma10=0.57, gamma11= None, beta10=None, beta11=None,\
                    M10=11.95, SHMnorm10=None, M11=None, SHMnorm11=None) 
    scatterevol = True
    mu = 2.5
    
    zinfallrange = [1,1.5,1.5]
    
    massrange = [11.2,12]
    m = 11.6
    fig, ax = plt.subplots(1,1)
    fig1,(ax1,ax2) = plt.subplots(1,2, figsize=(32,16), sharey=True)
    
    ax,ax1,ax2 = run_satellites(Nboot, massrange, dict_SMHM,scatterevol,mu,zinfallrange,ax,ax1,ax2)

    save(fig, ax, massrange, dict_SMHM, scatterevol)
    save(fig1, [ax1,ax2], massrange, dict_SMHM, scatterevol, choice='fquench')
    
    
    ###### to do:  bins of parent halo mass
    
    
    
    
    ####################################### second set of parameters ####################
    
    