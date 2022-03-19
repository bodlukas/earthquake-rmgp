# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:47:58 2022
...

@author: Lukas Bodenmann, ETH Zurich
"""
import numpy as np
import pandas as pd
from RMGP import GMM


class GMM_AkkarEtAl2014(GMM):
    
    def __init__(self, column_names: list):
        super().__init__( column_names = column_names, distance_metric = 'Repi')
        self.coeff = pd.read_csv('AkkarEtAl2014_GMPE_coeffs_Repi.csv','deliminiter',';')
        self.column_names = column_names                    
    def get_mu_ref_rock(self, M, R, sof, T):
        
        FN = FR = 0
        if sof == 'N': FN = 1
        if sof == 'R': FR = 1
        
        coef = self.coeff[self.coeff.Period==T].squeeze()
        if M <= coef.c1:
            logMedian = (coef.a1 + coef.a2 * (M - coef.c1) + 
                         coef.a3 * (8.5-M)**2 +
                         ( coef.a4 + coef.a5 * (M - coef.c1)) * 
                         np.log(np.sqrt(R**2 + coef.a6**2)) +
                         coef.a8 * FN + coef.a9 * FR)
        else:
            logMedian = (coef.a1 + coef.a7 * (M - coef.c1) + 
                         coef.a3 * (8.5-M)**2 +
                         ( coef.a4 + coef.a5 * (M - coef.c1)) * 
                         np.log(np.sqrt(R**2 + coef.a6**2)) +
                         coef.a8 * FN + coef.a9 * FR)
        return logMedian
    
    def get_site_amplif(self, MedianRockPGA, Vs30, T):
        coef = self.coeff[self.coeff.Period==T].squeeze()
        logS1 = (coef.b1 * np.log(Vs30 / coef.Vref) + 
                 coef.b2 * np.log( (MedianRockPGA + coef.c * 
                                    np.power(Vs30 / coef.Vref, coef.n) ) /
                                  ( (MedianRockPGA + coef.c) * 
                                   np.power(Vs30 / coef.Vref ,coef.n) ) ) ) 
        
        logS2a = coef.b1 * np.log( Vs30 / coef.Vref )
        logS2b = coef.b1 * np.log( coef.Vcon / coef.Vref )
        
        logS = ((Vs30 <= coef.Vref) * logS1 + 
                ((Vs30 > coef.Vref) & (Vs30 <= coef.Vcon)) * logS2a +
                ((Vs30 > coef.Vref) & (Vs30 > coef.Vcon)) * logS2b)
        
        return logS

    def get_mu(self, M, R, Vs30, sof, T):
        '''
        Calculates the logartihmic median ground-motion intensity measure.
        
        Parameters
        ----------
        M : float (1,)
            Moment magnitude Mw.
        R : Array (N,) 
            Source-to-site distance in km.
        Vs30 : Array (N,)
            Shear-wave velocity in the upper 30 m in m/s.
        sof : String (1,)
            Style of faulting (SS: strike-slip, N: normal, R: reverse).
        T : float (1,)
            Period (T=0 is PGA, T=-1 is PGV).
    
        Returns
        -------
        lnMedian : Array (N,)
            Logarithmic median gm-intensity measure in [g]
        '''
        # Reference rock conditions
        logMedianRock = self.get_mu_ref_rock(M, R, sof, T)
        
        if T!=0: 
            MedianRockPGA = np.exp(self._get_mu_ref_rock(M, R, sof, 0))
        else:
            MedianRockPGA = np.exp(logMedianRock)
        
        logS = self.get_site_amplif(MedianRockPGA, Vs30, T)
        
        return logMedianRock + logS
    
    def get_sigma(self, T):
        coef = self.coeff[self.coeff.Period==T].squeeze()
        deltaW = coef.sd1 #Intraevent
        deltaB = coef.sd2 #Interevent
        return deltaB, deltaW

class GMM_AkkarBommer2010(GMM):
    
    def __init__(self, column_names: list):
        super().__init__( column_names = column_names, distance_metric = 'Repi')
        coeff = pd.read_csv('AB10_PanEuropeanGMPE_coeffs.csv')
        coeff = coeff.drop(coeff[coeff.Period=='PGV'].index)
        coeff = coeff.astype({'Period': 'float64'})
        self.coeff = coeff
                    
    def get_mu_ref_rock(self, M, R, sof, T):
        
        FN = FR = 0
        if sof == 'N': FN = 1
        if sof == 'R': FR = 1
        
        coef = self.coeff[self.coeff.Period==T].squeeze()
        log10Median = (coef.b1 + coef.b2 * M + coef.b3 * M**2 + 
            (coef.b4 + coef.b5 * M) * np.log10( np.sqrt(R**2 + coef.b6)) +
            coef.b9 * FN + coef.b10 * FR)
        return log10Median
    
    def get_site_amplif(self, SoilClassEC, T):
        SA = (SoilClassEC == 'B')
        SS = (SoilClassEC == 'C')
        coef = self.coeff[self.coeff.Period==T].squeeze()
        log10S = coef.b7 * SS + coef.b8 * SA
        return log10S

    def get_mu(self, M, R, SoilClassEC, sof, T):
        '''
        Calculates the logartihmic median ground-motion intensity measure.
        
        Parameters
        ----------
        M : float (1,)
            Moment magnitude Mw.
        R : Array (N,) 
            Source-to-site distance in km.
        Vs30 : Array (N,)
            Shear-wave velocity in the upper 30 m in m/s.
        sof : String (1,)
            Style of faulting (SS: strike-slip, N: normal, R: reverse).
        T : float (1,)
            Period (T=0 is PGA, T=-1 is PGV).
    
        Returns
        -------
        lnMedian : Array (N,)
            Logarithmic median gm-intensity measure in [g]
        '''
        # Reference rock conditions
        log10MedianRock = self.get_mu_ref_rock(M, R, sof, T)        
        log10S = self.get_site_amplif(SoilClassEC, T)        
        log10Median = log10MedianRock + log10S
        # Transform to natural logarithm and unit g
        logMedian = np.log(10) * log10Median + np.log(1 / (100*9.816))        
        return logMedian
    
    def get_sigma(self, T):
        coef = self.coeff[self.coeff.Period==T].squeeze()
        deltaW = coef.Sigma1 * np.log(10) #Intraevent
        deltaB = coef.Sigma2 * np.log(10) #Interevent
        return deltaB, deltaW
    
class GMM_BindiEtAl2011(GMM):
    
    def __init__(self, column_names: list):
        super().__init__( column_names = column_names, distance_metric = 'Repi')
        coef = {'e1': 3.672, 'c1': -1.940, 'c2': 0.413, 'h': 10.322, 
                    'c3': 1.34e-4, 'b1': -0.262, 'b2': -0.0707, 'sA': 0,
                    'sB': 0.162, 'sC': 0.240, 'sD': 0.105, 'sE': 0.570, 
                    'f1': -5.03e-2, 'f2': 1.05e-1, 'f3': -5.44e-2, 'f4': 0,
                    'sigmaB': 0.172, 'sigmaW': 0.290, 'sigmaTot': 0.337,
                    'Mref': 5, 'Rref': 1, 'Mh': 6.75, 'b3': 0}
        self.coef = pd.Series(coef)
                    
    def get_mu_ref_rock(self, M, R, sof, T):
        d = self.coef
        FM = ( (d.b1 * (M - d.Mh) + d.b2 * (M - d.Mh)**2) * ( M <= d.Mh ) 
              + (d.b3 * (M - d.Mh)) * ( M > d.Mh ) )
        FD = ( ( d.c1 + d.c2 * (M - d.Mref) ) * 
              np.log10( np.sqrt( R**2 + d.h**2) / d.Rref ) - 
              d.c3 * ( np.sqrt( R**2 + d.h**2) - d.Rref ) )
        Fsof = (d.f1 * (sof == 'N') + d.f2 * (sof == 'R') + 
                d.f3 * (sof == 'SS') + d.f4 * (sof == 'U') )
        log10Median = d.e1 + FM + FD + Fsof
        return log10Median
    
    def get_site_amplif(self, SoilClassEC, T):
        d = self.coef
        FS = ( d.sA * (SoilClassEC == 'A') + d.sB * (SoilClassEC == 'B') + 
              d.sC * (SoilClassEC == 'C') + d.sD * (SoilClassEC == 'D') + 
              d.sE * (SoilClassEC == 'E') )
        log10S = FS
        return log10S

    def get_mu(self, M, R, SoilClassEC, sof, T):
        '''
        Calculates the logartihmic median ground-motion intensity measure.
        
        Parameters
        ----------
        M : float (1,)
            Moment magnitude Mw.
        R : Array (N,) 
            Source-to-site distance in km.
        Vs30 : Array (N,)
            Shear-wave velocity in the upper 30 m in m/s.
        sof : String (1,)
            Style of faulting (SS: strike-slip, N: normal, R: reverse).
        T : float (1,)
            Period (T=0 is PGA, T=-1 is PGV).
    
        Returns
        -------
        lnMedian : Array (N,)
            Logarithmic median gm-intensity measure in [g]
        '''
        # Reference rock conditions
        log10MedianRock = self.get_mu_ref_rock(M, R, sof, T)        
        log10S = self.get_site_amplif(SoilClassEC, T)        
        log10Median = log10MedianRock + log10S
        logMedian = np.log(10) * log10Median + np.log(1 / (100*9.816))        
        return logMedian
    
    def get_sigma(self, T):
        d = self.coef
        deltaW = d.sigmaW * np.log(10) #Intraevent
        deltaB = d.sigmaB * np.log(10) #Interevent
        return deltaB, deltaW
