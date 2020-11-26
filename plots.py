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


def hists(DF, binsR):
    cen = DF.query('upid==-1')['Re_cen']
    sat = DF.query('upid!=-1')
    hcen = np.histogram(cen, bins=binsR)[0]
    highz = sat.query('zscale>1')
    medz = sat.query('0.5<zscale<1')
    lowz = sat.query('zscale<0.5')
    hsat_highz = np.histogram(highz['Re_sat'], bins=binsR)[0]
    hsat_medz = np.histogram(medz['Re_sat'], bins=binsR)[0]
    hsat_lowz = np.histogram(lowz['Re_sat'], bins=binsR)[0]
    hsat = np.histogram(sat['Re_sat'], bins=binsR)[0]
    return hcen, hsat,hsat_highz, hsat_medz, hsat_lowz

def plot_sizefunctions(df, ax, m=11.75,TType='All', return_infall_range=True):
    
    if TType=='All':
        pass
    else:
        df = df[df['TType']==TType]
    print(len(df))
    Vol = (1000/0.7)**3
    rebins = np.arange(-0.2,2,0.1)
    hcen,hsat,hsat_highz, hsat_medz, hsat_lowz = hists(df,rebins)
    print(hcen)
    R,P,s=np.loadtxt('/home/lz1f17/data/Rmaj/cen/'+str(m)+'/Re_all'+str(m)+'cenJKG19_'+str(TType)+'.txt', unpack=True)
    index=np.where(P >0)[0]
    ax.errorbar(R,P,yerr=s,label='SDSS cen', markersize=15, color='darkorange', fmt='d',zorder=1)
    #R1,P1,s1=np.loadtxt('/home/lz1f17/data//Rmaj/'+str(m)+'/Re_all'+str(m)+'satJKG19.txt', unpack=True)
    R1,P1,s1=np.loadtxt('/home/lz1f17/data//Rmaj/sat/'+str(m)+'/Re_all'+str(m)+'satJKG19_'+str(TType)+'.txt', unpack=True)
    ax.errorbar(R1,P1,yerr=s1,label='SDSS sat', markersize=15,color='navy', fmt='v',zorder=1)


    ax.plot(rebins[1:]-0.05, hcen/Vol/0.1, lw=4,label='cen', color='darkorange', ls='--', zorder=2)#, color='firebrick')
    ax.plot(rebins[1:], hsat/Vol/0.1, lw=4, label='sat', color='navy', ls='--', zorder=2)#, color='forestgreen')
    if return_infall_range:
        ax.plot(rebins[1:], hsat_lowz/Vol/0.1, lw=4,label='sat, $z_{inf}<0.5$', color='purple', zorder=2)#, color='forestgreen')
        ax.plot(rebins[1:], hsat_medz/Vol/0.1,lw=4, label='sat, $0.5<z_{inf}<1$', color='lime', zorder=2)#, color='forestgreen')
        ax.plot(rebins[1:], hsat_highz/Vol/0.1, lw=4,label='sat, $z_{inf}>1$', color='teal', zorder=2)#, color='forestgreen')


    ax.set_yscale('log')
    
    ax.set_ylabel('$\phi(R_e)$')
    ax.set_xlabel('$\log{R_e} \ [kpc]$')
    ax.set_xlim(0.25,2.5)
    #plt.ylim(5.e-9,1.e-4)
    ax.legend(frameon=False, fontsize=30)
    
    return ax
