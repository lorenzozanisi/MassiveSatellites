import numpy as np
import os
from time import time
import colossus.cosmology.cosmology as cosmo
from colossus.halo import concentration, mass_so
from colossus.lss.mass_function import massFunction
from scipy.interpolate import interp1d
#from hmf_rp16 import hmf_rp16

cosmol=cosmo.setCosmology('planck15')    #cosmology for colossus. By default hmf uses planck15

def fred(x,bpar=1,beta=0.68):   # need to update this with new fit
    
    return 1./(1+beta*1.e12/10**x)

def sizes_from_lambda(R):

    ## Bullock
    lambda0 = -1.459
    sigma = 0.268
    fact = 10**(np.random.normal(lambda0,sigma,size=len(R)))
    return R*fact


def Schechter_like(x,width):

    ###peebles spin
    alpha =-4.126
    beta=0.610
    lambda0 = -2.919

    first = (x/10**lambda0)**(-alpha)
    second = -(x/10**lambda0)**(beta)
    final = first*np.exp(second)

    area = np.sum(final*width)
    return final/area


def Schechter_cumul(x,width):

    y = Schechter_like(x,width)
    cumul = np.cumsum(y)

    return cumul/np.max(cumul)

def extract_lambdas(R):

    x=np.linspace(1.e-4,0.5,1000000)
    width =x[1]-x[0]
    cumul = Schechter_cumul(x,width)
    ff = interp1d(cumul,x)
    xx = np.random.uniform(0,1,size= len(R))
    lambdas = ff(xx)
    return lambdas
    
def sizes_from_lambda_skewed(R):

    lambdas = extract_lambdas(R)
    return np.log10(R*lambdas)

def concentrationDuttonMaccio14(x,z):
    '''accepts M/h'''
    
    b =-0.097 +0.024*z
    a=0.537 + (1.025-0.537)*np.e**(-0.718*z**1.08)
    logc = a+b*(x-12)
    sc = 0.11
    c  = np.random.normal(logc,scale=sc)

    return c

def sizes_from_conc(M,R,gamma,z):
    logc = concentrationDuttonMaccio14(M,z)
    return np.log10(0.012) - gamma*(logc-1) + np.log10(R) #+ np.random.normal(loc=0,scale=0.1)

def get_Rh(halos, redshift):
    halonew = np.array(10**halos)*cosmol.h #converts from Mvir to Mvir/h
    rhalo = mass_so.M_to_R(halonew,redshift,'vir')/cosmol.h  
    return rhalo
    
def get_sizefunction(halos, redshift, V,A_K, sigma_K, stars, masslow,massup,model='K13',sigmaK_evo=False):
    rhalo = get_Rh(halos, redshift)

    if model=='K13':
        inp = rhalo*A_K
        if sigmaK_evo:
            sigma_K= 0.15+0.05*redshift
        Re = Kravtsov(inp, A=A_K, scatt=sigma_K)
        #print(Re)
    if model=='MMW':
        Re = sizes_from_lambda_skewed(rhalo)
        
    mask = np.ma.masked_inside(stars, masslow, massup).mask
    Re = Re[mask]
    
    binwidth = 0.05
    bins = np.arange(-2,3,binwidth)
    try:
        hist = np.histogram(Re, bins=bins)[0]
        return np.array([bins[1:]-0.5*binwidth, hist/V/(0.5*binwidth)])
    except:
        m = (masslow+massup)/2
        print('Could not compute size function for mass='+str(m)+'and z='+str(redshift))
        return np.zeros(len(bins)-1)

def get_mean_size(halos,redshift,A_K,sigma_K,stars,masslow,massup, sigmaK_evo=False):
    rhalo = get_Rh(halos, redshift)
    inp = rhalo*A_K
    if sigmaK_evo:
        sigma_K= 0.15+0.05*redshift
    Re_ = Kravtsov(inp, A=A_K, scatt=sigma_K)
    print(Re_)
    mask = np.ma.masked_inside(stars, masslow, massup).mask
    
    try:
        return 10**np.percentile(Re_[mask],50)  #mean size evolution
    except:
        m = (masslow+massup)/2
        print('Could not compute size evolution for mass='+str(m)+'and z='+str(redshift))
        return np.nan
    
def Kravtsov(RA,A,scatt=0.1, redshift=None):
    RA = np.log10(RA)
    return np.random.normal(RA, scale=scatt)
        
def extract_catalog(N,M):
    f=interp1d(N,M)
    array_cumul=np.arange(min(N),max(N))
    cat=f(array_cumul)
    
    return cat


def get_halos(z, Vol,dlog10m=0.005,hmf_choice='despali16' ):
    ####### set cosmology #########
    cosmol=cosmo.setCosmology('planck15')  
    cosmo.setCurrent(cosmol)
        ##############################
        
    Mvir=10**np.arange(11.,16,dlog10m)
    if hmf_choice=='despali16':
        massfunct =  massFunction(x=Mvir, z=z, mdef='vir', model='despali16', q_out='dndlnM')*np.log(10)  #dn/dlog10M
       
            
    elif hmf_choice=='rodriguezpuebla16':
        massfunct = hmf_rp16(Mvir,z)*Mvir/np.log10(np.exp(1))

    massfunct = massfunct*(cosmol.h)**3 #convert  from massf*h**3
    total_haloMF=massfunct.copy()
    #massfunct =  massFunction(x=Mvir, z=z, q_in='M',mdef='vir', model='despali16', q_out='dndlnM')*np.log(10)  #dn/dlog10M
        
    Mvir=np.log10(Mvir)
    Mvir=Mvir-np.log10(cosmol.h)  #convert from M/h

        
    Ncum=Vol*(np.cumsum((total_haloMF*dlog10m)[::-1])[::-1])
    halos=extract_catalog(Ncum,Mvir)

    return halos


class get_SMHM:
    
    def __init__(self):
        np.random.seed(int(time()+os.getpid()*1000))

    class moster13:
        
        def __init__(self, scatteron=True, scatterevol=False):
            self.scatteron = scatteron
            self.scatterevol = scatterevol
        def make(self, halos, z):
            zparameter = np.divide(z, z+1)
            M10, SHMnorm10, beta10, gamma10, Scatter = 11.590, 0.0351, 1.376, 0.608, 0.15
            M11, SHMnorm11, beta11, gamma11 = 1.195, -0.0247, -0.826, 0.329
        
            M = M10 + M11*zparameter
            N = SHMnorm10 + SHMnorm11*zparameter
            b = beta10 + beta11*zparameter
            g = gamma10 + gamma11*zparameter
            stars =  np.power(10, halos) * (2*N*np.power( (np.power(np.power(10,halos-M), -b) + np.power(np.power(10,halos-M), g)), -1))
    
           
            if self.scatteron:
                if self.scatterevol:
                    if z<0.5:
                        scatt = 0.1
                        print('here')
                    else:
                        scatt =0.3
                else:
                    scatt=0.3
                stars = np.random.normal(np.log10(stars),scale=scatt)
                return stars
            return np.log10(stars)
        
        def __call__(self,halos,z):
            return self.make(halos,z)
        
         
    class grylls19:  
        def __init__(self, scatteron=True,choice='constant'):
            self.scatteron = scatteron
            self.choice = choice
            
        def make(self, halos,z, constant):
            zparameter = np.divide(z-0.1, z+1)   
            if self.choice =='SE':
                M10, SHMnorm10, beta10, gamma10, Scatter = 12.0,0.032,1.5,0.56,0.15 
                M11, SHMnorm11, beta11, gamma11 = 0.6,-0.014,-0.7,0.08
            if self.choice == 'PyMorph':
                print('pymorph')
                M10, SHMnorm10, beta10, gamma10, Scatter = 11.92,0.032,1.64,0.53,0.15
                M11, SHMnorm11, beta11, gamma11 = 0.58,-0.014,-0.69,0.03
            if self.choice == 'cmodel':
                print('cmodel')
                M10, SHMnorm10, beta10, gamma10, Scatter =11.91,0.029,2.09,0.64,0.15
                M11, SHMnorm11, beta11, gamma11 = 0.52,-0.018,-1.03,0.084
        
            if self.choice == 'constant':
                zparameter = 0.
                
            M = M10 + M11*zparameter
            N = SHMnorm10 + SHMnorm11*zparameter
            b = beta10 + beta11*zparameter
            g = gamma10 + gamma11*zparameter
            stars =  np.power(10, halos) * (2*N*np.power( (np.power(np.power(10,halos-M), -b) + np.power(np.power(10,halos-M), g)), -1))
        
            if self.scatteron:
                scatt=0.15
                stars = np.random.normal(np.log10(stars),scale=scatt)
                return stars
            return np.log10(stars)
        
        def __call__(self, halos, z,constant=False):
            return self.make(halos,z, constant)
        
        
    class rodriguezpuebla17:
        
        def __init__(self, scatteron=True, scatterevol =True):
            self.scatteron = scatteron
            self.scatterevol = scatterevol
        def P(self,x,y,z):
            return y*z-x*z/(1+z)

        def a(self,z):
            return 1./(1.+z)

        def nu(self,z):
            nu=np.e**(-4*self.a(z)**2)  
            return nu

        def m1(self,z):
            m1=10**(11.548+self.P(-1.297,-0.026,z)*self.nu(z))
            return m1

        def eps(self,z):
            eps=10**(-1.758+self.P(0.11,-0.061,z)*self.nu(z)+self.P(-0.023,0,z))
            return eps

        def alfa(self,z):
            alfa=1.975+self.P(0.714,0.042,z)*self.nu(z)
            return alfa

        def delta(self,z):
            delta=3.390+self.P(-0.472,-0.931,z)*self.nu(z)
            return delta

        def gamma(self,z):
            gamma=0.498+self.P(-0.157,0,z)*self.nu(z)
            return gamma

        def scattRP17(self,z):
            var=0.1+0.05*z
            scatt=np.sqrt(0.15**2+np.power(var,2))
            return scatt

        def make(self, halos, z, constant):
            if constant:
                z = 0.1
            
            x=halos-np.log10(self.m1(z))
            first=-np.log10(10**(-self.alfa(z)*x)+1)
            second= self.delta(z)*(np.log10(1+np.e**x))**self.gamma(z)
            third=1+np.e**(10**(-x))
            f=first+second/third

            first=-np.log10(10**(-self.alfa(z)*0.)+1)
            second= self.delta(z)*(np.log10(1+np.e**-0.))**self.gamma(z)
            third=1+np.e**(10**(0.))
            f0=first+second/third
    
            stars=np.log10(self.eps(z)*self.m1(z))+ f-f0
        
            
            if self.scatteron:
                if self.scatterevol:
                    scatt = self.scattRP17(z)
                else:
                    scatt = 0.15
                stars = np.random.normal(stars,scale=scatt)
                return stars
            return stars  
        
        def __call__(self, halos,z, constant=False):
            return self.make(halos,z,constant)
        
        
    class Z19_LTGs_ETGs:  #SHMR functional form; Behroozi+2010
        def __init__(self, scatteron=True, choice='All'):
            self.scatteron = scatteron
            self.choice = choice
 
        def g(self, x, a, g, d):
            return (-np.log10(10**(-a*x)+1.) +
                d*(np.log10(1.+np.exp(x)))**g/(1.+np.exp(10**(-x))))
     
        def scatt(self, log10Mvir):
            if self.choice=='All':
                return 0.15 #needs change
            elif self.choice=='LTGs':
                return 0.12
            elif self.choice=='ETGs':
                return 0.14
            elif self.choice=='Z19':
                return 0.16
        
        def make(self,log10Mvir,z=None):
            if self.choice=='LTGs':
                alpha, delta, gamma, log10eps, log10M1 = 1.47439,4.26527,0.314474,-1.70524,11.4935
            elif self.choice=='ETGs':
                alpha, delta, gamma, log10eps, log10M1 = 6.10862,5.2244,0.336961,-2.19148,11.7417
            elif self.choice=='All':
                alpha, delta, gamma, log10eps, log10M1 = 1.72338,4.43454,0.422541,-1.88375,11.5095
            elif self.choice=='Z19':
                alpha, delta, gamma, log10eps, log10M1 = 2.352,3.797,0.6,-1.785,11.632
            x = log10Mvir - log10M1
            g1 = self.g(x, alpha, gamma, delta)
            g0 = self.g(0, alpha, gamma, delta)
            log10Ms = log10eps + log10M1 + g1 - g0
     
            if self.scatteron:
                log10Ms = np.random.normal(log10Ms, self.scatt(log10Mvir))
            return log10Ms
    
        def __call__(self, halos,z=None):
            return self.make(halos,z=None) #z is dummy here for consistency with the call to other SMHM
 
    class constantSMF:
        
        def __init__(self, scatteron=True):
            self.scatteron = scatteron
        def a(self,z):
            return 1./(1.+z)

        def nu(self,z):
            nu=np.e**(-4*self.a(z)**2)
            return nu

        def m1(self,z,M10=11.7845,M1a=1.074,M1z=-1.0596):
            m1=10**(M10+(M1a*(self.a(z)-1)+M1z*z)*self.nu(z))
            return m1

        def eps(self,z,eps0=-1.8456,epsa=-0.9274,epsz=0.1595,epsa2=0.7849):
            eps=10**(eps0+(epsa*(self.a(z)-1)+epsz*z)*self.nu(z)+epsa2*(self.a(z)-1))
            return eps

        def alfa(self,z,alfa0=-2.0105,alfaa=-0.0158):
            alfa=alfa0+(alfaa*(self.a(z)-1))*self.nu(z)
            return alfa

        def delta(self,z,delta0=3.6179,deltaa=-1.3933,deltaz=-2.2852):
            delta=delta0+(deltaa*(self.a(z)-1)+deltaz*z)*self.nu(z)
            return delta

        def gamma(self,z,gamma0=0.4822,gammaa=1.0908,gammaz=0.1906):
            gamma=gamma0+(gammaa*(self.a(z)-1)+gammaz*z)*self.nu(z)
            return gamma
        
        def scatter(self,z):
            var=0.1+0.05*z
            scatt=np.sqrt(0.15**2+np.power(var,2))
            return scatt       


        def make(self,halos,z):
            x=halos-np.log10(self.m1(z))
            first=-np.log10(10**(self.alfa(z)*x)+1)
            second= self.delta(z)*(np.log10(1+np.e**x))**self.gamma(z)
            third=1+np.e**(10**(-x))
            f=first+second/third
            first=-np.log10(10**(self.alfa(z)*0.)+1)
            second= self.delta(z)*(np.log10(1+np.e**-0.))**self.gamma(z)
            third=1+np.e**(10**(0.))
            f0=first+second/third
    
            stars=np.log10(self.eps(z)*self.m1(z))+ f-f0
        
            if self.scatteron:
                scatt = self.scatter(z)
                stars = np.random.normal(stars,scale=scatt)
                return stars
            return stars  
    
        def __call__(self,halos,z):
            return self.make(halos,z)
        

 
        
        
        
        

