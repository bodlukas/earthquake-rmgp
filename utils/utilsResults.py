# -*- coding: utf-8 -*-
"""
@author: Lukas Bodenmann, ETH Zurich, January 2022
"""
import numpy as np

class R_stats(object):
    '''
    Based on sampled damage states, this object calculates and stores the 
    mean and the quantiles (specified in qlist) for each damage state,
    timestep and subregion.
    '''
    def __init__(self, NumClasses: int, n_time_steps: int, n_subregions: int, 
                 qlist: list):
        self.q = qlist # List of quantile to evaluate
        self.n_t = n_time_steps # Number of time steps for which to store res.
        self.n_sr = n_subregions # Number of subregions for which to store res.
        self.C = NumClasses # Number of categories for which to store res.
        self.stats = np.zeros( ( self.C, len(self.q)+1, self.n_sr, self.n_t ) )

    def Allocate(self, y_samples, y_insp, t_I, sr_I):
        '''
        Parameters
        ----------
        y_samples : np.array (r x NumClasses)
            r samples of the number of buildings in each class.
        y_insp : np.array (NumClasses, )
            Number of already inspected buildings in each class.
        t_I : int
            Index of time-step
        sr_I : int
            Index of subregion
        '''
        m = np.mean(y_samples + y_insp, axis = 0)
        quantiles = np.quantile(y_samples + y_insp, q = self.q, axis = 0).T
        self.stats[:, 0, sr_I, t_I] = m
        self.stats[:, 1:, sr_I, t_I] = quantiles
            
    def AllocatePrior(self, stats):
        self.stats[:, :, 0] = stats
        
    def Save(self, resfolder: str, model: str, seed: int):
        np.save(resfolder+'stats_' + model + str(seed), self.stats)
   
class R_scores(object):
    '''
    Based on sampled damage states y, this object calculates and stores:
        Marginal Prediction Error (MPE) for each y, timestep and subregion. 
        Joint Prediction Error (JPE) for each timestep and subregion.
    '''    
    def __init__(self, NumClasses: int, n_time_steps: int, n_subregions: int):
        self.n_t = n_time_steps
        self.n_sr = n_subregions
        self.C = NumClasses
        self.MPE = np.zeros( (self.C, self.n_sr, self.n_t ) )
        self.JPE = np.zeros( (self.n_sr, self.n_t) )
    
    def crps(self, y_samples, y_insp, y_true):
        '''
        Calculates the continuous ranked probability score for each
        category (i.e. damage state). This metric serves as the marginal
        prediction error.
        '''
        y_sum = y_samples + y_insp
        r = y_sum.shape[0]
        crps = []
        for c in np.arange(self.C):           
            term_1 = 1/r * np.sum( np.abs(y_sum[:,c] - y_true[c]) )
            term_2 = 1/(2 * (r-1)) * np.sum( np.abs(np.diff(y_sum[:,c]) ))
            crps.append(term_1 - term_2)
        return np.hstack(crps)
    
    def es(self, y_samples, y_insp, y_true):
        '''
        Calculates the energy score for the joint distribution over all
        categories (i.e. damage states). This metric serves as the joint
        prediction error.
        '''
        y_sum = y_samples + y_insp
        r = y_sum.shape[0]
        term_1 = 1/r * np.sum( np.sqrt( np.sum((y_sum - y_true)**2, axis=1) ) )
        term_2 = 1/(2 * (r-1) ) * np.sum( np.sqrt( np.sum( np.diff(y_sum, axis=0)**2, axis=1) ) )
        return (term_1 - term_2)
    
    def Allocate(self, y_samples, y_insp, y_true, t_I, sr_I):
        '''        
        Parameters
        ----------
        y_samples : np.array (r x NumClasses)
            r samples of the number of buildings in each class.
        y_insp : np.array (NumClasses, )
            Number of already inspected buildings in each class.
        y_true : np.array (NumClasses, )
            Actual number of buildings in each class.            
        t_I : int
            Index of time-step
        sr_I : int
            Index of subregion
        '''
        self.MPE[:, sr_I, t_I] = self.crps(y_samples, y_insp, y_true)
        self.JPE[sr_I, t_I] = self.es(y_samples, y_insp, y_true)
        
    def Save(self, resfolder: str, model: str, seed: int):
        np.save(resfolder+'MPE_' + model + str(seed), self.MPE)
        np.save(resfolder+'JPE_' + model + str(seed), self.JPE)