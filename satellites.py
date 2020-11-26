import numpy as np
import utils
import pandas as pd

class grylls19:  
    
    def __init__(self, gamma10=None, gamma11= None, beta10=None, beta11=None,\
                M10=None, SHMnorm10=None, M11=None, SHMnorm11=None,scatterevol=False):
        ''' If both gamma10 and gamma11 , are None it return grylls19 Pymorph. Otherwise 
        they are custom.
        '''
        self.scatterevol = scatterevol
        if gamma10 is not None:
            self.gamma10 = gamma10
        else:
            self.gamma10 = 0.53
            
        if gamma11 is not None:
            self.gamma11 = gamma11
        else:
            self.gamma11 = 0.03
            
        if M10 is not None:
            self.M10 = M10
        else:
            self.M10 = 11.92
            
        if SHMnorm10 is not None:
            self.SHMnorm10 = SHMnorm10
        else:
            self.SHMnorm10 = 0.032
            
        if SHMnorm11 is not None:
            self.SHMnorm11=SHMnorm11
        else:
            self.SHMnorm11=-0.014
        if M11 is not None:
            self.M11=M11
        else:
            self.M11 = 0.58
        if beta10 is not None:
            self.beta10 = beta10
        else:
            self.beta10 = 1.64
        if beta11 is not None:
            self.beta11 = beta11
        else:
            self.beta11 = -0.69
            
   
    def make(self, halos,z,scatter, scatterevol=False):
        zparameter = np.divide(z-0.1, z+1)   
       # zparameter = np.divide(z, z+1) 
        M = self.M10 + self.M11*zparameter
        N = self.SHMnorm10 + self.SHMnorm11*zparameter
        b = self.beta10 + self.beta11*zparameter
        g = self.gamma10 + self.gamma11*zparameter
        
      #  if self.orig:
      #      print('pymorph')
      #      gamma10 = 0.53~
      #      gamma11 = 0.03
      #      Scatter = 0.15
      #      g = gamma10 + gamma11*zparameter
            
        stars =  np.power(10, halos) * (2*N*np.power( (np.power(np.power(10,halos-M), -b) +\
                                                       np.power(np.power(10,halos-M), g)), -1))

        if scatterevol:
            scatt = np.sqrt( (0.1*(z-0.1))**2+scatter**2)
            
        else: 
            scatt = scatter
            
        stars = np.random.normal(np.log10(stars),scale=scatt)
            
        return stars
    
    def __call__(self, halos, z,scatter, scatterevol=False):
        return self.make(halos,z,scatter, scatterevol)
    
    
class make_satellites:
    
    def __init__(self,use_acc=True, use_peak=False):
        self.use_acc= use_acc
        self.use_peak = use_peak
        self.df_cen, self.df_sat = self.dark_matter_to_stars()
        self.zscale_cen = self.df_cen['zscale'].min()
        
    
    def dark_matter_to_stars(self):  #use_acc allows to construct parallel mocks with two calls to the class
        
        if self.use_acc:
            data = np.load('/data/cm1n17/MultiDark/npyData/MD_0.093.npy')
            Mpeak = np.log10(data['First_Acc_Mvir']/0.67)
            zscale = 1/data['First_Acc_Scale']-1
        elif self.use_peak:
            data  = np.load('/home/lz1f17/data/MultiDark/MD_0.093.npy')
            Mpeak = np.log10(data['Mpeak']/0.67)
            zscale = 1/data['Mpeak_Scale']-1
        mask = np.ma.masked_greater(Mpeak,11.5).mask
        mvir = np.log10(data['mvir']/0.67)
        mvir = mvir[mask]
        Mpeak = Mpeak[mask]
        zscale = zscale[mask]
        
        rockId = data['id'][mask]
        upid = data['upid'][mask]
        df = pd.DataFrame({"upid":upid, "id":rockId,'Mpeak':Mpeak, 'zscale':zscale,'mvir':mvir})
        
        df_cen = df.query('upid==-1')
        df_sat = df.query('upid!=-1')
        
        SMHM = grylls19()
        df_cen.loc[:,'Mstar_cen'] = SMHM(df_cen.loc[:,'mvir'],np.min(zscale), scatter=0.15)-0.1 # correcting G19 masses
        df_sat.loc[:,'Mstar_sat'] = SMHM(df_sat.loc[:,'Mpeak'],df_sat.loc[:,'zscale'], scatter=0.15)-0.1 # correcting G19 ma
        
        return df_cen, df_sat
    
    def fquench(self, x,z,mu): #z is infall redshift
        beta = 0.68+(1+z)**mu
        return 1./(1+beta*1.e12/10**x.values)
    
    def red_blue(self,DF, mu):   
        DF.loc[:,'TType'] = np.zeros(len(DF))
        halos = DF['Mpeak'].values
        halodf = pd.DataFrame(halos)
        bl = halodf.apply(lambda x: np.random.uniform(size=len(DF)) >  self.fquench(x,DF['zscale'],mu)).values.T[0] 
        DF.loc[bl,'TType'] = 'LTGs'
        red = np.logical_not(bl)
        DF.loc[red,'TType'] = 'ETGs'
        return DF
        
    def make_censat(self, Mstar_low=11.5, Mstar_up=12.5, AK=0.013, mu=None,AK_Q=0.013, AK_SF=0.018,sigmaK=0.08, sigmaK_SF=0.08):  #mu allows exploration of fquench model, the same catalog can be cut in Mstar in different ways
        
        df_cen = self.df_cen.query('{}<Mstar_cen<{}'.format(Mstar_low, Mstar_up))
        df_sat = self.df_sat.query('{}<Mstar_sat<{}'.format(Mstar_low, Mstar_up))
        
        if mu is not None:
            df_cen = self.red_blue(df_cen,mu)
            df_sat = self.red_blue(df_sat,mu)
            
            df_cen_red = df_cen[df_cen['TType']=='ETGs']
            df_sat_red = df_sat[df_sat['TType']=='ETGs']
            
            df_cen_blue = df_cen[df_cen['TType']=='LTGs']
            df_sat_blue = df_sat[df_sat['TType']=='LTGs']
            
            df_cen_red.loc[:,'Rh_cen'] = utils.get_Rh(df_cen_red['mvir'], self.zscale_cen)
            df_cen_blue.loc[:,'Rh_cen'] = utils.get_Rh(df_cen_blue['mvir'], self.zscale_cen)

            df_sat_red.loc[:,'Rh_sat'] = utils.get_Rh(df_sat_red['Mpeak'], df_sat_red['zscale'])
            df_sat_blue.loc[:,'Rh_sat'] = utils.get_Rh(df_sat_blue['Mpeak'], df_sat_blue['zscale'])

            df_cen_red.loc[:,'Re_cen'] = np.random.normal(loc = np.log10(AK*df_cen_red.loc[:,'Rh_cen']), scale=sigmaK)
            df_cen_blue.loc[:,'Re_cen'] = np.random.normal(loc = np.log10(AK_SF*df_cen_blue.loc[:,'Rh_cen']), scale=sigmaK)            
            df_sat_red.loc[:,'Re_sat'] = np.random.normal(loc = np.log10(AK*df_sat_red.loc[:,'Rh_sat']), scale=sigmaK_SF)           
            df_sat_blue.loc[:,'Re_sat'] = np.random.normal(loc = np.log10(AK_SF*df_sat_blue.loc[:,'Rh_sat']), scale=sigmaK_SF)
            
            df_final = pd.concat([df_cen_red,df_cen_blue, df_sat_red, df_sat_blue])
            
        else:
            df_cen.loc[:,'Rh_cen'] = utils.get_Rh(df_cen['mvir'], self.zscale_cen)
            df_sat.loc[:,'Rh_sat'] = utils.get_Rh(df_sat['Mpeak'], df_sat['zscale'])
            df_cen.loc[:,'Re_cen'] = np.random.normal(loc = np.log10(AK*df_cen.loc[:,'Rh_cen']), scale=sigmaK)
            df_sat.loc[:,'Re_sat'] = np.random.normal(loc = np.log10(AK*df_sat.loc[:,'Rh_sat']), scale=sigmaK_SF)
            df_final = pd.concat([df_cen,df_sat])
        
        return df_final #this is the main output of this class