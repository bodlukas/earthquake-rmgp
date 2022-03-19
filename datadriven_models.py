# -*- coding: utf-8 -*-
"""
@author: Lukas Bodenmann, ETH Zurich, January 2022
"""

import numpy as np
import pandas as pd

#%% Baseclass
class Model(object):
    
    def Sample_damage(self, Xtarget, nSamples: int,
                      aggregate: bool):
        '''
        Samples damage states independently for each target building based on
        the predicted class membership probabilities.
        
        Parameters
        ----------
        Xtarget : np.array (n, d)
            Matrix that collects the d-dimensional input vector for each of
            the n target buildings.
        nSamples : int
            Number of samples to generate.
        aggregate : bool, optional
            Whether to aggregate samples for each damage level. 
            The default is False.

        Returns
        -------
        y_samples : array of type int
            If aggregate is false the shape is (nSamples, N).
            If aggregate is true the shape is (nSamples, cy).

        '''
        N = Xtarget.shape[0]
        rng = np.random.default_rng()
        helpuni = rng.uniform(0,1,(N,nSamples))
        y_samples = np.zeros_like(helpuni)
        p_prob = self.Predict_Prob(Xtarget)
        p_cdf = np.append(np.zeros((N,1)),
                          np.cumsum(p_prob, axis=1), axis=1)
        for y in np.arange(self.C):
            y_samples[(helpuni >= p_cdf[:, y].reshape(-1, 1)) & 
                      (helpuni < p_cdf[:, y+1].reshape(-1, 1))] = y
        
        y_samples = y_samples.T
        if aggregate: 
            y_samples = np.apply_along_axis(lambda x: np.bincount(x, minlength = self.C), 
                                               axis=1, arr=y_samples.astype('int'))
        return y_samples
    
#%% Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import itertools

class RF(Model):
    '''
    Wrapper for the random forest classifier of sklearn.
    See also the corresponding documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    '''
    def __init__(self, NumClasses: int, parameters: dict=None):
        '''
        Parameters
        ----------
        NumClasses : int
            Number of classes for which we would like to do predictions.
        parameters : dict, optional
            Parameters used in grid search. The default is None.
            If None the parameters are identical to the ones used in the
            paper.

        '''
        self.C = NumClasses
        self.m = RandomForestClassifier(oob_score=True, n_estimators=1000, max_samples=0.9)
        if parameters is None: 
            param = {'max_features': [1, 2, 3], 
                          'min_samples_leaf': [1, 10, 15, 20]}
        else: 
            param = parameters
            
        param_combi = list(itertools.product(*(param[Name] for Name in param.keys())))
        self.df_param = pd.DataFrame(param_combi, columns=param.keys())  

    def Inference(self, X, obs_y):
        '''
        Perform grid search over all specified combinations specified.
        Compute the out-of-bag score for each combination.
        Choose the combination with the highest score.

        Parameters
        ----------
        X : np.array (m, d)
            Matrix that collects the d-dimensional input vector for each of
            the m buildings.
        obs_y : np.array (m, )
            Observed damage state y for each of the m buildings.

        '''
        nc = len(self.df_param)
        seeds = np.random.choice(np.arange(100*nc), nc, replace=False)
        self.df_param['random_state'] = seeds
        models = []; scores = []
        m_temp = clone(self.m)
        for idx, row in self.df_param.iterrows():
            m_temp.set_params(**dict(row))
            m_temp.fit(X, obs_y)
            models.append(m_temp)
            scores.append(m_temp.oob_score_)
            m_temp = clone(self.m)
        self.scores = scores
        self.m = models[np.argmax(scores)]
        
    def Predict_Prob(self, Xtarget):
        '''
        Calculates the class membership probability for the inputs Xpred.
            
            p(y_target|X_target, D_I)

        Parameters
        ----------
        Xtarget : np.array (n, d)
            Matrix that collects the d-dimensional input vector for each of
            the n target buildings.

        '''
        p_prob = self.m.predict_proba(Xtarget)
        # If one class was not seen in the training set, add zero probability
        # to this class
        if self.m.classes_.size != self.C:
            prob_temp = np.zeros( (Xtarget.shape[0], self.C) )
            DSdetMask = np.isin(np.arange(self.C), self.m.classes_)
            prob_temp[:, DSdetMask] = p_prob
            p_prob = prob_temp
        return p_prob
    
#%% Ordered linear probit
from OrderedSM import OrderedModel

class OLP(Model):
    '''
    Wrapper for the ordered model of statsmodels
    The ordered model of statsmodels was copied to OrderedSM as a local file,
    because of compatibility issues. 
    
    See also the corresponding documentation:
    https://www.statsmodels.org/devel/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html
    '''
    def __init__(self, NumClasses: int):
        self.C = NumClasses
        self.m = None        
    
    # @ignore_warnings(category=ConvergenceWarning)
    def Inference(self, X, obs_y, maxiter: int=int(3000)):
        '''
        Parameters
        ----------
        X : np.array (m, d)
            Matrix that collects the d-dimensional input vector for each of
            the m buildings.
        obs_y : np.array (m, )
            Observed damage state y for each of the m buildings.

        '''
        # Remove a column if it is identical for each training datapoint
        dropidx = []
        for i in np.arange(X.shape[1]):
            if len(np.unique(X[:,i]))==1:
                dropidx.append(i)
        idx = np.isin(np.arange(X.shape[1]), dropidx, invert=True)
        X = X[:,idx]
        # Initialize the model
        m = OrderedModel(endog = obs_y, exog = X)
        self.m = m.fit(maxiter = maxiter, disp = False)
        self.C_obs = np.unique(obs_y)
        self.idx = idx
    
    def Predict_Prob(self, Xtarget):
        '''
        Calculates the class membership probability for the inputs Xpred.
            
            p(ytarget|Xtarget, D_I)

        Parameters
        ----------
        Xtarget : np.array (n, d)
            Matrix that collects the d-dimensional input vector for each of
            the n target buildings.

        '''
        Xtarget = Xtarget[:, self.idx]
        p_prob = self.m.predict(Xtarget)
        # If one class was not seen in the training set, add zero probability
        # to this class
        if self.C_obs.size != self.C:
            prob_temp = np.zeros( (Xtarget.shape[0], self.C) )
            DSdetMask = np.isin(np.arange(self.C), self.C_obs)
            prob_temp[:, DSdetMask] = p_prob
            p_prob = prob_temp
        return p_prob