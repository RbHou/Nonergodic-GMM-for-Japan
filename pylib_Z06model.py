# from utils import *
import os, time
import numpy as np
import matplotlib.pyplot as plt 

def GetKey(key):
    return '%.3f'%(key)

#
class Zhao06:
    """
    Class for Zhao et al. 2016 Japan model 
    """
    def __init__(self):
        self.filepth = os.path.join(os.path.dirname(__file__)) 
        self.CoefFile = os.path.join(self.filepth,'pylib_Z06model_param.csv')
        self.Coefs = {}
        self.ReadModelCoefs()
        
        # put some period independent coefs here 
        self.Rref = 10.0 
        self.Mref = 6.5 
        self.Depthref = 1.0
        self.SCs = ['I0','I','II','III','IV', 0, 1, 2, 3, 4]
        self.faults = [0,1,2,3,'U','SS','NM','RV']
    #
    def __call__(self,M,depth,ST,Dist, SC, T, Mech=1, hc = 25, CoefTerms={'NewCoefs':None}):
        """
        Compute IM for single period
        required inputs:
        M, Depth, Dist, SC, T
        rake: rake angle (degree), default is None (Unspecified fault type)
        or give Mech instead of rake
        Mech: 
            1: reverse
            2: strike
            3: normal
            else: 0 unspecified (U=1) (Default)
        
        """
        # ==================
        # Input variables
        # ==================
        self.M = float(M)                           # moment magnitude
        self.depth = min(float(depth),125)          # focal depth
        self.Dist = float(Dist)                     # source distance (km)
        self.ST = ST                                # source type, 1-crustal, 3-interface, 4-slab
        self.SC = SC                                # site class
        self.Mech = Mech
        self.hc = hc

        NewCoefs = CoefTerms['NewCoefs']

        # check inputs
        
        if T in self.periods:
            self.T = T
        else:
            print ('T is not in periods list, try to interpolate')
            raise ValueError
    
        if self.M == None or self.M < 0:
            print ('Moment magnitude must be a postive number')
            raise ValueError

        if self.Dist == None or self.Dist < 0:
            print ('Source distance must be a non-negative number')
            raise ValueError
        
        if self.depth == None or self.depth < 0:
            print ('focal depth must be a non-negative number')
            raise ValueError
        
        
        # if self.Dist < self.depth:
            # print 'Source distance must be larger than focal depth'
            # raise ValueError
        
        if SC not in self.SCs :
            print ("SC must be one of 1, 2, 3, 4 or 'I', 'II', 'III', 'IV'")
            raise ValueError
        
        if Mech != None:
            self.U = 1*(Mech==0)
            self.RV = 1*(Mech==1)
            self.SS = 1*(Mech==2)
            self.NM = 1*(Mech==3)
        else:
            print ("Mech must be one of 1, 2, 3. 1 for reverse, 2 for strike, 3 for normal.")
            raise ValueError
        
        # modify the coefficients (only update Coefs given by NewCoefs (at self.T))
        if NewCoefs != None:
            Tkey = GetKey( self.T )
            NewCoefKeys = NewCoefs.keys()
            for key in NewCoefKeys:
                self.Coefs[Tkey][key] = NewCoefs[key]
        
        # ======================
        # begin to compute IM
        # ======================
        #
        # Median ground motion [PGA & SA: g]
        IM = self.compute_im()
        #
        # standard deviation [same unit as median GM]
        sigmaT, tau, sigma = self.compute_std()
        
        return IM, sigmaT, tau, sigma
    #
    # function of reading coefficients
    def ReadModelCoefs(self):
        self.CoefKeys = open(self.CoefFile,'r').readlines()[0].strip().split(',')[1:]
        inputs = np.loadtxt(self.CoefFile,skiprows=1,delimiter=',')
        self.periods = inputs[:,0]
        coefs = inputs[:,1:]
        for i in range( len(self.periods) ):
            T1 = self.periods[i]
            Tkey = GetKey(T1)
            
            # assign to Coefs
            self.Coefs[Tkey] = {}
            for ikey in range(len(self.CoefKeys)):
                key = self.CoefKeys[ikey]
                cmd = "self.Coefs['%s']['%s'] = coefs[%i,%i]"%(Tkey,key,i,ikey)
                exec(cmd)
    #
    def moment_function(self, Tother=None):
        """
        Magnitude scaling
        """
        #
        if Tother != None:
            Ti = GetKey(Tother)
        else:
            Ti = GetKey(self.T)
        #
        a = self.Coefs[Ti]['a']
        e = self.Coefs[Ti]['e']
        S_R = self.Coefs[Ti]['SR']
        S_I = self.Coefs[Ti]['SI']
        S_S = self.Coefs[Ti]['SS']
        # print('a, e, SR, SI, SS', a, e, S_R, S_I, S_S)
        
        term = a*self.M 
        if self.depth >= self.hc: term += e*(self.depth - self.hc)
        if self.ST == 1 and Mech==1: term += S_R                            # crustal earthquake with reverse faulting
        if self.ST == 3: term += S_I
        if self.ST == 4: term += S_S
        
        return term 
    #
    def distance_function(self,Tother=None):
        """
        Distance function
        """
        if Tother != None:
            Ti = GetKey(Tother)
        else:
            Ti = GetKey(self.T)
        #
        b = self.Coefs[Ti]['b']
        c = self.Coefs[Ti]['c']
        d = self.Coefs[Ti]['d']
        SSL = self.Coefs[Ti]['SSL']
        
        #
        r = self.Dist + c*np.exp(d*self.M)
        term = b * self.Dist - np.log(r)
        if self.ST == 4: term += SSL * np.log(self.Dist)
        
        return term
    #
    def site_function(self, SC=None, Tother=None):
        """
        Site Amplification Function
        """
        if SC != None: 
            self.SC = SC 
        #
        if Tother != None: 
            Ti = GetKey( Tother )
            T = Tother
        else: 
            Ti = GetKey(self.T )
            T = self.T 
        #
        # extract coefficients
        CH = self.Coefs[Ti]['CH']
        CI = self.Coefs[Ti]['C1']
        CII = self.Coefs[Ti]['C2']
        CIII = self.Coefs[Ti]['C3']
        CIV = self.Coefs[Ti]['C4']
        
        #
        # compute spectrum
        lnSa0 = self.moment_function()+self.distance_function()
        
        if (self.SC == 'I0'):
            self.Sa_pred = np.exp(lnSa0 + CH)
        elif (self.SC == 'I'):
            self.Sa_pred = np.exp(lnSa0 + CI)
        elif (self.SC == 'II'):
            self.Sa_pred = np.exp(lnSa0 + CII)
        elif (self.SC == 'III'):
            self.Sa_pred = np.exp(lnSa0 + CIII)
        else:                                      # (self.SC == 'IV')
            self.Sa_pred = np.exp(lnSa0 + CIV)
        
        return self.Sa_pred
    #
    def compute_im(self):
        """
        Compute IM based on functional form of BA08 model
        note: for PGA and PSA, IM has unit (g) here 
        """
        
        IM = self.site_function()
        
        return IM
    #
    def compute_std(self):
        # print 'call compute _std'
        Ti = GetKey(self.T)
        
        sigma = self.Coefs[Ti]['sigma']
        tau = self.Coefs[Ti]['tau']
        sigmaT = self.Coefs[Ti]['sigmaT']
        
        return sigmaT, tau, sigma
#
def attenuation_curve(M=6.6, depth=1.0, ST=1, SC = 'II', T=0.005, Mech=1, hc=25):
    # by default, volcanic distance is zero
    
    Dist_arr = np.geomspace(1,400,80)
    
    oneSeries=[]
    for aDist in Dist_arr:
        aZ06GMM = Zhao06()
        values = aZ06GMM(M, depth, ST, aDist, SC, T, Mech=Mech, hc = hc)
        oneSeries.append(values)
    
    Att_data_Arr = np.array(oneSeries)
    Att_data_Arr = np.column_stack((Dist_arr, Att_data_Arr))
    
    return Att_data_Arr
#
#
def Mw_scaling( depth=1.0, ST=1, Dist=10, SC='II', T = 0.005, Mech=1, hc=25):
    #
    Mw_arr = np.linspace(4.5,8.0,30)
    oneSeries=[]
    for M in Mw_arr:
        aZ06GMM = Zhao06()
        values = aZ06GMM( M, depth, ST, Dist, SC, T, Mech=Mech, hc=hc)
        oneSeries.append(values)
    
    Mw_data_Arr = np.array(oneSeries)
    Mw_data_Arr = np.column_stack((Mw_arr, Mw_data_Arr))
    
    return Mw_data_Arr
#
#
def Spec_vs_period(M=6.6, depth=1.0, ST=1, Dist=10, SC='II', Mech=1, T = None, hc=25):
    
    if T is None:
        T = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,
                    0.6,0.7,0.8,0.9,1.0,1.25,1.5,2.0,
                    2.5,3.0,4.0,5.0]
    
    oneSeries=[]
    for aTp in T:
        aZ06GMM = Zhao06()
        values = aZ06GMM(M, depth, ST, Dist, SC, aTp, Mech=Mech, hc=hc)
        oneSeries.append(values)
    
    period_data_Arr = np.array(oneSeries)
    period_data_Arr = np.column_stack((T,period_data_Arr))
    
    return period_data_Arr
#
