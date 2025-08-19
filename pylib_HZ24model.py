# from utils import *
import os, time
import numpy as np
import matplotlib.pyplot as plt 

def GetKey(key):
    return '%.3f'%(key)


class GMM_HZ24:
    """
    Hou and Zhao (2024) Ground motion model for shallow crustal and upper mantle earthquakes in Japan
    """
    def __init__(self):
        self.filepth = os.path.join(os.path.dirname(__file__)) 
        self.CoefFile = os.path.join(self.filepth,'pylib_HZ24model_param.csv')
        self.Coefs = {}
        self.ReadModelCoefs()
        
        # put some period independent coefs here 
        self.Mref = 6.5 
        self.Depthref = 1.0
        self.faults = [0,1,2,3,'U','SS','NM','RV']
        self.Rref = 10.0 
        self.Rvocref = 10.0 
        self.Tsref = 0.3
    #
    def __call__(self,M,Dist,Ts,T, depth=0, Mech=3, Ftype=None, vDist=0, CoefTerms={'NewCoefs':None}):
        """
        Compute IM for single period
        required inputs:
        M, Depth, Dist, vDist, Ts, T
        rake: rake angle (degree), default is None (Unspecified fault type)
        or give Mech instead of rake
        Mech: 
            1: reverse
            2: strike
            3: normal
            else: 0 unspecified (U=1) (Default)
        Ftype = 'U', or 'SS', or 'RV', or 'NM'
        """
        # ==================
        # Input variables
        # ==================
        self.M = float(M)               # moment magnitude
        self.depth = float(depth)       # focal depth
        self.Dist = float(Dist)         # source distance (km)
        self.vDist = float(vDist)       # volcanic distance (km)
        self.Ts = Ts                    # site class
        self.Mech = Mech

        #print 'T, M, depth, Dist, vDist, Ts, Mech:', T, M, depth, Dist, vDist, Ts, Mech

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
            # print ('Source distance must be larger than focal depth')
            # raise ValueError
        
        if self.Ts <= 0.01:
            self.Ts = 0.01
            print ("Ts must be lager than 0.01s")
        
        if Mech == None and Ftype == None:
            print ('either (U,SS,NM,RV) or focal mechanics should be provided')
            raise ValueError
        else: 
            if Ftype != None: 
                self.U = 1*(Ftype == 'U')
                self.SS = 1*(Ftype == 'SS')
                self.NM = 1*(Ftype == 'NM')
                self.RV = 1*(Ftype == 'RV')
            if Mech != None:
                self.U = 1*(Mech==0)
                self.RV = 1*(Mech==1)
                self.SS = 1*(Mech==2)
                self.NM = 1*(Mech==3)
        
        # modify the coefficients (only update Coefs given by NewCoefs (at self.T))
        if NewCoefs != None:
            Tkey = GetKey( self.T )
            NewCoefKeys = NewCoefs.keys()
            for key in NewCoefKeys:
                self.Coefs[Tkey][key] = NewCoefs[key]
        
        # ========================================================
        # begin to compute IM
        # ========================================================
        
        # Median ground motion [PGA & SA: g]
        IM_NL, IM_L = self.compute_im()     
        
        # standard deviation [same unit as median GM]
        sigmaT, tau, sigma, siteSigma=self.compute_std()
        
        return IM_NL, IM_L, sigmaT, tau, sigma, siteSigma
    #
    # function of 
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
        Magnitude-Moment scaling
        """
        #
        if Tother != None:
            Ti = GetKey(Tother)
        else:
            Ti = GetKey(self.T)
        #
        ccr = self.Coefs[Ti]['ccr']
        pcr = self.Coefs[Ti]['pcr']
        dcr = self.Coefs[Ti]['dcr']
        FCRN = self.Coefs[Ti]['FCRN']
        FUMRV = self.Coefs[Ti]['FUMRV']
        FUMNS = self.Coefs[Ti]['FUMNS']
        bcr = self.Coefs[Ti]['bcr']
        
        if (self.depth<=25.0):
            if (self.M <=7.1):
                term=bcr*self.depth+FCRN*self.NM+ccr*self.M+pcr*(self.M-6.3)**2
            else:
                term=bcr*self.depth+FCRN*self.NM+ccr*7.1+pcr*(7.1-6.3)**2+dcr*(self.M-7.1)
        else:
            if (self.M <=7.1):
                term=FUMRV*self.RV+FUMNS*self.NM+FUMNS*self.SS+ccr*self.M+pcr*(self.M-6.3)**2
            else:
                term=FUMRV*self.RV+FUMNS*self.NM+FUMNS*self.SS+ccr*7.1+pcr*(7.1-6.3)**2+dcr*(self.M-7.1)
        return term 
    #
    def distance_function(self,Tother=None):
        """
        Distance function
        
        """
        # print 'call compute dist term'
        if Tother != None:
            Ti = GetKey(Tother)
        else:
            Ti = GetKey(self.T)
        #
        c1 = self.Coefs[Ti]['c1']
        c2 = self.Coefs[Ti]['c2']
        gcr = self.Coefs[Ti]['gcr']
        gUM = self.Coefs[Ti]['gUM']
        gcrN = self.Coefs[Ti]['gcrN']
        gcrL = self.Coefs[Ti]['gcrL']
        ecr = self.Coefs[Ti]['ecr']
        eum = self.Coefs[Ti]['eum']
        ecrV = self.Coefs[Ti]['ecrV']
        gamma = self.Coefs[Ti]['gamma']
        
        r = 2.0 + self.Dist + np.exp(c1 + c2*min(7.1,self.M))
        if (self.Dist <= 30.0):
            gnx=gcrN*np.log(2.0+self.Dist+np.exp(c1+c2*6.5))
        else:
            gnx=gcrN*np.log(2.0+30.0+np.exp(c1+c2*6.5))
        
        if (self.depth <= 25.0) :
            term = gcr*np.log(r)+gcrL*np.log(self.Dist+200.0)+gnx+ecr*self.Dist+ecrV*self.vDist+gamma
        else:
            term = gUM*np.log(r)+gcrL*np.log(self.Dist+200.0)+gnx+eum*self.Dist+ecrV*self.vDist+gamma
        
        return term
    #
    def site_function_Ts(self, Ts=None, Sarock=None, Tother=None):
        """
        Site Amplification Function
        """
        # print 'call compute site term'
        if Ts != None: 
            self.Ts = Ts 
        #
        if Tother != None: 
            Ti = GetKey( Tother )
            T = Tother
        else: 
            Ti = GetKey(self.T )
            T = self.T 
        #
        # extract coefficients
        b1L = self.Coefs[Ti]['b1L']
        b2L = self.Coefs[Ti]['b2L']
        b3L = self.Coefs[Ti]['b3L']
        Tb = self.Coefs[Ti]['Tb']
        b1 = self.Coefs[Ti]['b1']
        beta = self.Coefs[Ti]['beta']
        TSL = self.Coefs[Ti]['TSL']
        theta1 = self.Coefs[Ti]['theta1']
        theta2 = self.Coefs[Ti]['theta2']
        b1NL = self.Coefs[Ti]['b1NL']
        b2NL = self.Coefs[Ti]['b2NL']
        b3NL = self.Coefs[Ti]['b3NL']
        b4NL = self.Coefs[Ti]['b4NL']
        b0NL = self.Coefs[Ti]['b0NL']
        theta = self.Coefs[Ti]['theta']
        
        #
        # compute rock spectrum
        self.Sarock = np.exp(self.moment_function()+self.distance_function())
        if Sarock is None:
            Sarock = self.Sarock
        
        #
        # compute linear site amplification ratio
        Anmax=1+b1L/np.sqrt(b2L+(1-b3L*T/self.Ts)**2)
        
        #
        # compute nonlinear site parameters
        Imfav=1.0
        Seffrock=Imfav*Sarock
        
        LnAmax1D = np.log(1+b1NL/(b2NL+(1-b3NL*T/self.Ts)**2)) + b4NL*self.Ts + b0NL
        SF=Anmax/np.exp(LnAmax1D)
        
        alpha = 1+1/(1+(Tb/self.Ts)**b1)
        Ca = -0.01 + theta1*(1-np.exp(theta2*(np.log(max(self.Ts, TSL))-np.log(TSL))**2))
        
        lamb = 1.0
        Smr=Seffrock*SF**(lamb*np.tan(theta/180*3.1415926))
        
        #
        lnAn=np.log(Anmax)+Ca*np.log((Smr**alpha+beta)/beta)
        
        self.FsiteNL = np.exp(lnAn)
        self.FsiteL = Anmax
        
        return Sarock, self.FsiteNL, self.FsiteL
    #
    def compute_im(self):
        """
        Compute IM based on functional form of BA08 model
        note: for PGA and PSA, IM has unit (g) here 
        """

        Sarock, FsiteNL, FsiteL = self.site_function_Ts()
        IM_NL = Sarock*FsiteNL
        IM_L = Sarock*FsiteL
        
        return IM_NL, IM_L
    #
    def compute_std(self):
        # print 'call compute _std'
        Ti = GetKey(self.T)
        
        fai = self.Coefs[Ti]['fai']
        tau = self.Coefs[Ti]['tau']
        sigmaT = self.Coefs[Ti]['sigmaT']
        fs2s = self.Coefs[Ti]['fs2s']
        #
        singlesigma = np.sqrt(sigmaT**2-fs2s**2)
        
        return sigmaT, tau, fai, singlesigma
#
#
def attenuation_curve(M=6.6, depth=1.0, Ts = 0.3, T=0.005, Mech=1):
    # by default, volcanic distance is zero
    
    Dist_arr = np.geomspace(1,400,40)
    
    oneSeries=[]
    for aDist in Dist_arr:
        aHZ24GMM = GMM_HZ24()
        values = aHZ24GMM(M, aDist, Ts, T, depth=depth, vDist=0, Mech=Mech)
        oneSeries.append(values)
    
    Att_data_Arr = np.array(oneSeries)
    Att_data_Arr = np.column_stack((Dist_arr, Att_data_Arr))
    
    return Att_data_Arr
#
#
def Mw_scaling( depth=1.0, Dist=10, vDist=0, Ts=0.3, T = 0.005, Mech=1):
    #
    Mw_arr = np.linspace(4.5,8.0,30)
    oneSeries=[]
    for M in Mw_arr:
        aHZ24GMM = GMM_HZ24()
        values = aHZ24GMM( M, Dist, Ts, T, depth=depth, vDist=vDist, Mech=Mech)
        oneSeries.append(values)
    
    Mw_data_Arr = np.array(oneSeries)
    Mw_data_Arr = np.column_stack((Mw_arr, Mw_data_Arr))
    
    return Mw_data_Arr
#
#
def Ts_scaling( M=6.6, depth=1.0, Dist=10, vDist=0, T = 0.005, Mech=1, Ts_arr=None):
    #
    if Ts_arr is None: Ts_arr = np.linspace(0.01,5.0,50)
    oneSeries=[]
    for aTs in Ts_arr:
        aHZ24GMM = GMM_HZ24()
        values = aHZ24GMM( M, Dist, aTs, T, depth=depth, vDist=vDist, Mech=Mech)
        oneSeries.append(values)
    
    Ts_data_Arr = np.array(oneSeries)
    Ts_data_Arr = np.column_stack((Ts_arr, Ts_data_Arr))
    
    return Ts_data_Arr
#
#
def Spec_vs_period(M=6.6, depth=1.0, Dist=10, vDist=0, Ts=0.3, Mech=1, T = None):
    
    if T is None:
        T = [0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,
                    0.12,0.14,0.15,0.16,0.18,0.2,
                    0.25,0.3,0.35,0.4,0.45,0.5,
                    0.6,0.7,0.8,0.9,1.0,1.25,1.5,2.0,
                    2.5,3.0,3.5,4.0,4.5,5.0]
    
    
    oneSeries=[]
    for aTp in T:
        aHZ24GMM = GMM_HZ24()
        values = aHZ24GMM( M, Dist, Ts, aTp, depth=depth, vDist=vDist, Mech=Mech)
        oneSeries.append(values)
    
    period_data_Arr = np.array(oneSeries)
    period_data_Arr = np.column_stack((T,period_data_Arr))
    
    return period_data_Arr
#

