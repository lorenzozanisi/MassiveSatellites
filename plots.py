import numpy as np
import matplotlib.pylab as plt
import utils
import matplotlib as mpl
import pandas as pd

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


def hists_hostHaloRange(DF_sat, binsR, halo_range, zinfallrange):

    lowz = DF_sat.query(f'{halo_range[0]}<Mpeak_cen<{halo_range[1]} & {zinfallrange[0]}<zscale_sat<{zinfallrange[1]}')['Re_sat']
    medz = DF_sat.query(f'{halo_range[0]}<Mpeak_cen<{halo_range[1]} & {zinfallrange[1]}<zscale_sat<{zinfallrange[2]}')['Re_sat']
    highz = DF_sat.query(f'{halo_range[0]}<Mpeak_cen<{halo_range[1]} & {zinfallrange[2]}<zscale_sat<{zinfallrange[3]}')['Re_sat']
    hsat_highz = np.histogram(highz, bins=binsR)[0]
    hsat_medz = np.histogram(medz, bins=binsR)[0]
    hsat_lowz = np.histogram(lowz, bins=binsR)[0]
    hsat = np.histogram(DF_sat.query(f'{halo_range[0]}<Mpeak_cen<{halo_range[1]}')['Re_sat'], bins=binsR)[0]
    return  hsat,hsat_highz, hsat_medz, hsat_lowz



def hists(DF, binsR, zinfallrange=[0.5,1,1]):
    cen = DF.query('upid==-1')['Re_cen']
    sat = DF.query('upid!=-1')
    hcen = np.histogram(cen, bins=binsR)[0]
    highz = sat.query('zscale>{}'.format(zinfallrange[-1]))
    medz = sat.query('{}<zscale<{}'.format(zinfallrange[0],zinfallrange[1]))
    lowz = sat.query('zscale<{}'.format(zinfallrange[0]))
    hsat_highz = np.histogram(highz['Re_sat'], bins=binsR)[0]
    hsat_medz = np.histogram(medz['Re_sat'], bins=binsR)[0]
    hsat_lowz = np.histogram(lowz['Re_sat'], bins=binsR)[0]
    hsat = np.histogram(sat['Re_sat'], bins=binsR)[0]
    return hcen, hsat,hsat_highz, hsat_medz, hsat_lowz

def plot_sizefunctions(df, ax, m=11.75,TType='All', return_infall_range=True,zinfallrange=[0.5,1,1]):
    
    if TType=='All':
        pass
    else:
        df = df[df['TType']==TType]
    print(len(df))
    Vol = (1000/0.7)**3
    rebins = np.arange(-0.2,2,0.1)
    hcen,hsat,hsat_highz, hsat_medz, hsat_lowz = hists(df,rebins, zinfallrange)
    print(hcen)
    R,P,s=np.loadtxt('/home/lz1f17/data/Rmaj/cen/'+str(m)+'/Re_all'+str(m)+'cenJKcleaned_'+str(TType)+'.txt', unpack=True)
    index=np.where(P >0)[0]
    ax.errorbar(R,P,yerr=s,label='SDSS cen', markersize=15, color='darkorange', fmt='d',zorder=1)
    #R1,P1,s1=np.loadtxt('/home/lz1f17/data//Rmaj/'+str(m)+'/Re_all'+str(m)+'satJKG19.txt', unpack=True)
    R1,P1,s1=np.loadtxt('/home/lz1f17/data//Rmaj/sat/'+str(m)+'/Re_all'+str(m)+'satJKcleaned_'+str(TType)+'.txt', unpack=True)
    ax.errorbar(R1,P1,yerr=s1,label='SDSS sat', markersize=15,color='navy', fmt='v',zorder=1)


    ax.plot(rebins[1:]-0.05, hcen/Vol/0.1, lw=4,label='cen', color='darkorange', ls='--', zorder=2)#, color='firebrick')
    ax.plot(rebins[1:]-0.05, hsat/Vol/0.1, lw=4, label='sat', color='navy', ls='--', zorder=2)#, color='forestgreen')
    if return_infall_range:
        ax.plot(rebins[1:]-0.05, hsat_lowz/Vol/0.1, lw=4,label='sat, $z_{inf}<$'+str(zinfallrange[0]), color='purple', zorder=2)#, color='forestgreen')
        ax.plot(rebins[1:]-0.05, hsat_medz/Vol/0.1,lw=4, label='sat,'+str(zinfallrange[0])+'$<z_{inf}<$'+str(zinfallrange[1]), color='lime', zorder=2)#, color='forestgreen')
        ax.plot(rebins[1:]-0.05, hsat_highz/Vol/0.1, lw=4,label='sat, $z_{inf}>$'+str(zinfallrange[2]), color='teal', zorder=2)#, color='forestgreen')


    ax.set_yscale('log')
    
    ax.set_ylabel('$\phi(R_e)$')
    ax.set_xlabel('$\log{R_e} \ [kpc]$')
    ax.set_xlim(0.25,2.5)
    #plt.ylim(5.e-9,1.e-4)
    ax.legend(frameon=False, fontsize=30)
    
    return ax


def plot_sizefunctions_MonteCarlo(hcen, hsat,hsat_highz, hsat_medz, hsat_lowz,ax,m=11.75, TType='All', return_infall_range=True,rebins = np.arange(-0.2,2.5,0.1), return_legend=False, zinfallrange=[0.5,1,1]):    

    Vol = (1000/0.7)**3

    width = rebins[1]-rebins[0]
    HCEN_L, HCEN_M, HCEN_U = np.percentile(np.array(hcen)/Vol/width, axis=0, q=[16,50,84])
    HSAT_L, HSAT_M, HSAT_U = np.percentile(np.array(hsat)/Vol/width, axis=0, q=[16,50,84])
    HSAT_lowz_L, HSAT_lowz_M, HSAT_lowz_U = np.percentile(np.array(hsat_lowz)/Vol/width, axis=0, q=[16,50,84])
    HSAT_medz_L, HSAT_medz_M, HSAT_medz_U = np.percentile(np.array(hsat_medz)/Vol/width, axis=0, q=[16,50,84])
    HSAT_highz_L, HSAT_highz_M, HSAT_highz_U = np.percentile(np.array(hsat_highz)/Vol/width, axis=0, q=[16,50,84])    
    
    ax.fill_between(rebins[1:]-width/2, HCEN_L, HCEN_U, lw=4,label='cen', color='darkorange',alpha=0.5, zorder=2)#, color='firebrick')
    ax.fill_between(rebins[1:]-width/2, HSAT_L, HSAT_U, lw=4, label='sat', color='navy', alpha=0.5, zorder=2)#, color='forestgreen')
    if return_infall_range:
        ax.fill_between(rebins[1:]-width/2, HSAT_lowz_L, HSAT_lowz_U,  lw=4,label='sat,$z_{inf}<$'+str(zinfallrange[0]), color='purple',  alpha=0.5,zorder=2)#, color='forestgreen')
        ax.fill_between(rebins[1:]-width/2, HSAT_medz_L, HSAT_medz_U, lw=4, label='sat, '+str(zinfallrange[0])+'$<z_{inf}<$'+str(zinfallrange[1]), color='lime',  alpha=0.5,zorder=2)#, color='forestgreen')
        ax.fill_between(rebins[1:]-width/2, HSAT_highz_L, HSAT_highz_U,  lw=4,label='sat, $z_{inf}>$'+str(zinfallrange[2]), color='teal', alpha=0.5, zorder=2)#, color='forestgreen')    
    
    
    R,P,s=np.loadtxt('/home/lz1f17/data/Rmaj/cen/'+str(m)+'/Re_all'+str(m)+'cenJKcleaned_'+str(TType)+'.txt', unpack=True)
    index=np.where(P >0)[0]
    ax.errorbar(R,P,yerr=s,label='SDSS cen', markersize=15, color='darkorange', fmt='d',zorder=1)
    #R1,P1,s1=np.loadtxt('/home/lz1f17/data//Rmaj/'+str(m)+'/Re_all'+str(m)+'satJKG19.txt', unpack=True)
    R1,P1,s1=np.loadtxt('/home/lz1f17/data//Rmaj/sat/'+str(m)+'/Re_all'+str(m)+'satJKcleaned_'+str(TType)+'.txt', unpack=True)
    ax.errorbar(R1,P1,yerr=s1,label='SDSS sat', markersize=15,color='navy', fmt='v',zorder=1)

    ax.set_yscale('log')
    
    ax.set_ylabel('$\phi(R_e)$')
    ax.set_xlabel('$\log{R_e} \ [kpc]$')
    ax.set_xlim(-0.2,2.5)
    #plt.ylim(5.e-9,1.e-4)
    if return_legend:
        ax.legend(frameon=False, fontsize=30, loc='upper right')
    
    return ax