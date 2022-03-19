# -*- coding: utf-8 -*-
"""
The different modules used for the risk model informed GP (RMGP)
@author: Lukas Bodenmann, ETH Zurich, January 2022
"""

import gpflow as gpflow
import numpy as np
from scipy.stats import norm
import pandas as pd
from .utilsGPflow import Ordinal_mod, Bernoulli_mod, Identity_mod

#%% Event Characteristics

class EventChar(object):
    '''
    Event characteristics 
    
    Attributes
    ----------
    M : float (1, )
        Moment magnitude. 
    xEpi : float (1, )
        x-coordinate of epicenter in km
    yEpi : float (1, )
        y-coordinate of epicenter in km
    sof : string
        Style of faulting:
            'SS': Strike-slip
            'N': Normal
            'R': Reverse
            'U': Unknown
    depth : float (1, ) (Optional)
        Depth of the hypocenter in km
        (This is not used in the present case studies)
    '''    
    def __init__(self, Magnitude: float, xEpi: float, yEpi: float, 
                 style_of_faulting: str, depth: float=None):
        self.M = Magnitude
        self.xEpi = xEpi
        self.yEpi = yEpi
        self.sof = style_of_faulting
        if depth is not None: self.depth = depth

#%% Ground-Motion Model
    
class GMM(object):
    '''
    Class for empirical ground-motion models (GMMs)
    
    Attributes
    ----------
    dist : str (1, )
        Distance metric, currently only supports epicentral and hypo-
        central distance
    column_names : list 
        Specify the names of dataframe columns required for
        prediction of median IM.
        The first two entries should specify the geo-coordinates
        The last entry should specify the soil class        
        Example: column_names = ['x', 'y', 'SoilClass']
        
    '''
    def __init__(self, column_names: list, distance_metric: str):
        self.dist = distance_metric
        self.column_names = column_names
   
    def _get_epi_distance(self, df_target, EventChar):
        xSite =  df_target[self.column_names[0]].values
        ySite =  df_target[self.column_names[1]].values
        Repi = np.sqrt( (xSite - EventChar.xEpi)**2 + 
                        (ySite - EventChar.yEpi)**2 )
        return Repi
    
    def _get_hypo_distance(self, df_target, EventChar):
        Repi = self.get_epi_distance(df_target)
        Rhypo = np.sqrt( (Repi**2 + EventChar.depth**2) )
        return Rhypo
    
    def get_distance(self, df_target, EventChar):
        if self.dist == 'Repi':
            R = self._get_epi_distance(df_target, EventChar)
        elif self.dist == 'Rhypo':
            R = self._get_hypo_distance(df_target, EventChar)
        return R            
    
    def get_log_median_im(self, df_target: pd.DataFrame, T: float, 
                          EventChar: EventChar):
        '''
        Predicts the logarithmic median IM for the sites in df_target and the
        event characteristics in EventChar. If df_target has N rows the 
        output is an array of dimension (N,)

        Parameters
        ----------
        df_target : pd.DataFrame
            Dataframe with required information about the sites. Specifically,
            it should provide the columns specified in the column_names 
            attribute.
        T : float (1,)
            Fundamental period of vibration at which the GMM should be 
            evaluated. For PGA T=0.
        EventChar : EventChar object
            Event characteristics.

        '''
        R = self.get_distance(df_target, EventChar)
        soil_info = df_target[self.column_names[-1]].values
        return self.get_mu(EventChar.M, R, soil_info, EventChar.sof, T)
    
    def get_sigmas(self, T: float):
        '''
        Returns the logarithmic between-event and within-event standard 
        deviation of the GMM. The output is a tuple where the first entry is
        the between-event std and the second entry is the within-event std.

        Parameters
        ----------
        T : float (1,)
            Fundamental period of vibration at which the GMM should be 
            evaluated. For PGA T=0.

        '''
        deltaB, deltaW = self.get_sigma(T)
        return deltaB, deltaW

#%% Spatial Ground-Motion Model

class SpatialGMM(object):
    '''
    Combines an empirical ground-motion model (GMM) with a spatial
    correlation model and assembles the input matrix, the mean and covariance
    functions of the GP model.
    '''
    def __init__(self, GMM: GMM, corr_length: float, T: float):
        '''
        Parameters
        ----------
        GMM : GMM - object
            Ground motion model object.
        corr_length : float
            Correlation range. This corresponds to the separation distance
            between two sites for which the remaining correlation is 5%.
        T : float
            Fundamental period at which the GMM is evaluated. PGA is 0.

        '''
        self.GMM = GMM
        self.corr_length = corr_length
        self.column_names = GMM.column_names
        self.T = T
    
    def mean_func(self, df_target: pd.DataFrame, EventChar: EventChar):
        X = self.InitInput(df_target, EventChar)
        return self.InitMean_Func()(X)
    
    def cov_func(self, df_target: pd.DataFrame):
        coor = df_target[self.column_names[:-1]].values
        return self.InitCov_Func()(coor)
    
    def InitInput(self, df_target: pd.DataFrame, EventChar: EventChar):
        '''
        Assembles the Input Matrix X. The median IM predicted by the GMM is 
        added to the last column of the input matrix.
        
        Parameters
        ----------
        df_target : pd.DataFrame
            Dataframe with required information about the sites. Specifically,
            it should provide the columns specified in the column_names attribute
            of the GMM.
        EventChar : EventChar - object
            Object with the event characteristics

        Returns
        -------
        X : np.array (N x (D+1))
            Input matrix where N is the number of sites, D is the dimension of
            the input space.

        '''
        num_cols = len(self.column_names[:-1]) + 1
        X = np.zeros( (len(df_target), num_cols) )
        X[:, :-1] = df_target[self.column_names[:-1]].values
        X[:, -1] = self.GMM.get_log_median_im(df_target, self.T, EventChar)
        return X

    def InitMean_Func(self):
        '''
        Initializes the mean function for the GP model.
        '''
        return Identity_mod(1)

    def InitCov_Func(self):
        '''
        Initializes the covariance function for the GP model.
        The order should not be changed. First: k_B and then k_W.
        '''
        # get standard deviations of GMM
        stdB, stdW = self.GMM.get_sigmas(self.T)
        # Constant between-event term
        k_B = gpflow.kernels.Constant(variance = stdB**2)
        # Within-event term (exponential covariance function)
        k_W = gpflow.kernels.Matern12(variance = stdW**2, 
                                      lengthscales = self.corr_length/3,
                                      active_dims = [0,1])
        return k_B + k_W

#%% Damage Estimation Model
class DamageEstimationModule(object):
    '''
    Initializes the likelihoods of the GP model based on vulnerability function
    parameters. cb is the number of vulnerability classes B and cy is the number 
    of damage states Y (including no damage). 
    
    Attributes
    ----------
    etas : np.array (cb x (cy-1))
        Threshold parameters. 
    betas : np.array (cb x 1)
        Dispersion parameters. 
        
    '''    
    def __init__(self, etas: np.array, betas: np.array):
        
        self.etas = etas
        self.betas = betas
        self.c_y = etas.shape[1] + 1

    def InitLikelihood(self):
        '''
        Initializes a GPflow likelihood object composed of cb ordinal 
        likelihood objects. For MAP-estimation, the thresholds etas are scaled 
        by 1/beta. See also the documentation of ordinal_mod. 

        Returns
        -------
        likelihood : GPflow object

        '''
        likelihood_list = []
        for i in np.arange(self.etas.shape[0]):
            etas_tilde = self.etas[i, :] / self.betas[i, 0]
            if self.c_y > 2:
                deltas = np.diff(etas_tilde)
                likTemp = Ordinal_mod(eta_tilde_0 = etas_tilde[0], 
                                      deltas = deltas, 
                                      beta = self.betas[i,0])
            else:
                likTemp = Bernoulli_mod(eta_tilde = etas_tilde, 
                                        beta = self.betas[i,0])
            likelihood_list.append(likTemp)        
        likelihood = gpflow.likelihoods.SwitchedLikelihood(likelihood_list)
        return likelihood

    def CollectEstimatedLikelihoodParams(self, likelihood):
        '''
        Collects the vulnerability function parameters from a likelihood and
        updates the initially assigned parameters etas and betas.
        This is used once the GP model is trained and the MAP estimates of the
        parameters are available.

        Parameters
        ----------
        likelihood : GPflow object

        '''
        etaU = []
        betaU = []
        for liktemp in likelihood.likelihoods:
            eta0up = liktemp.eta_tilde_0.numpy() * liktemp.beta.numpy()
            deltasup = liktemp.deltas.numpy() * liktemp.beta.numpy()
            etaU.append( np.append(eta0up, eta0up + np.cumsum(deltasup) ) )
            betaU.append(liktemp.beta.numpy())
        self.etas = np.vstack(etaU)
        self.betas = np.vstack(betaU)
    
    def CondSampleDamage(self, function_samples, vul_class_samples):
        '''
        Generates samples of damage levels conditional on realizations of the
        GP function (logarithmic IM) and of the vulnerability classes b.
        
        R is the number of simulations and N is the size of the target set. 
        
        Parameters
        ----------
        function_samples : float array (R, N)
            R samples from the GP function f = log(im) evaluated at N target
            points.
        vul_class_samples : integer array (R, N) 
            R sampled vulnerability classes b for N target points.

        Returns
        -------
        sam_y : integer array (R, N)
            Sampled damage levels

        '''
        shape = function_samples.shape
        cdf_y = []
        for i in np.arange(self.c_y - 1):
            etab = self.etas[vul_class_samples, i]
            betab = self.betas[vul_class_samples, 0]
            cdf_y.append(1 - norm.cdf( (function_samples - etab) / betab ) )
        cdf_y = np.stack(cdf_y, axis=-1)
        cdf_y = np.concatenate([np.zeros(np.append(list(shape),1)), 
                                cdf_y, 
                                np.ones(np.append(list(shape),1))],axis = -1)
        sam_y = np.zeros(shape)
        rnd_uni = np.random.uniform(0, 1, shape)
        for i in np.arange(self.c_y):
            sam_y[(rnd_uni > cdf_y[:,:,i]) & (rnd_uni <= cdf_y[:,:,i+1])] = i
        return sam_y
    


#%% Typological Attribution Model

class TypologyAttributionModule(object):
    '''
    Collects the main parameters and methods for the typological attribution. 
    ca is the number of uncertain attribute combinations a.
    
    Attributes
    ----------
    AttMat : np.array (..., ca, ...)
        Matrix that indicates the prior class membership probabilities.
        The first dimension usually specifies different construction periods.
        The last dimension can further indicate different number of stories or
        height classes.
    column_names : list 
        Specify the names of dataframe columns required for
        typological attribution.
        The first two entries should specify the geo-coordinates
        The third entry should specify the construction year or period
        The fourth entry should specify the number of stories or height class        
        Example: column_names = ['x', 'y', 'const_year', 'num_stories']
    column_dtypes : list 
        The dtypes of the columns specified in column_names. This is required
        to separate cases where the exact construction year (or number of 
        stories) is known or only the period (or height class).
    func_assign_vul_class: function
        The deterministic function h that maps sampled attribute combinations
        to vulnerability classes using the input information.
    func_prior_mean_prob: function
        A deterministic function that assigns the prior mean class membership
        probability conditional on the input information and the attribution
        matrix.
        
    '''    
    def __init__(self, AttributionMatrix: np.array, column_names: list,
                 column_dtypes: pd.Series, 
                 func_assign_vul_class, func_prior_mean_prob):
        self.AttMat = AttributionMatrix
        self.c_a = self.AttMat.shape[1]
        self.column_names = column_names
        self.column_dtypes = column_dtypes
        self.func_assign_vul_class = func_assign_vul_class
        self.func_prior_mean_prob = func_prior_mean_prob

    def mean_func(self, df_target: pd.DataFrame):
        X = self.InitInput(df_target)
        return self.InitMean_Func()(X)
    
    def cov_func(self, df_target: pd.DataFrame):
        X = self.InitInput(df_target)
        return self.InitCov_Func()(X)
        
    def InitInput(self, df_target):
        '''
        Assembles the Input Matrix X. The prior mean probabilities are assigned
        in the last ca columns of X.
        For categorical inputs (such as construction periods) one-hot encoding
        is performed.
        
        Parameters
        ----------
        df_target : pd.DataFrame
            Dataframe with required information about the buildings. 
            Specifically, it should provide the columns in column_names.

        Returns
        -------
        X : np.array (N x (D+ca))
            Input matrix where N is the number of buildings, 
            D is the dimension of the input space.

        '''        

        X = df_target[self.column_names[:2]].values
        for i, col in enumerate(self.column_names[2:]):
            if df_target[col].dtype.name == 'category':
                # one-hot encoding
                vals = pd.get_dummies(df_target[col]).values
                X = np.append(X, vals, axis = 1)
            else: 
                vals =  df_target[col].values
                X = np.append(X, vals.reshape(-1,1), axis=1)
        log_prior_mean = np.log(self.func_prior_mean_prob(df_target, self.AttMat))
        X = np.append(X, log_prior_mean, axis = 1)
        return X
        
    def InitMean_Func(self):
        '''
        Initializes the mean function for the GP model.
        '''
        return Identity_mod(self.c_a)

    def InitCov_Func(self):
        '''
        Initializes the covariance function for the GP model.
        For categorical inputs we assign linear func. on the one-hot encoded
        inputs. See also Appendices A and C for a more thorough explanation.
        '''
        v_ls = gpflow.kernels.SquaredExponential(lengthscales=5, 
                                                 variance=0.8, active_dims=[0,1])
        v_ss1 = gpflow.kernels.SquaredExponential(lengthscales=0.15, 
                                                  variance=0.6, active_dims=[0,1])

        if self.column_dtypes[2].name == 'category':            
            v_ss2 = []
            for cp in np.arange(len(self.column_dtypes[2].categories)):
                v_ss2.append(gpflow.kernels.Linear(variance=0.6, 
                                                   active_dims=[2 + cp]))
            v_ss2 = np.sum(np.stack(v_ss2))
            sp = 2 + len(self.column_dtypes[2].categories)
        else:
            v_ss2 = gpflow.kernels.SquaredExponential(lengthscales=5, 
                                                      variance=0.6, active_dims=[2])
            sp = 2 + 1
        if self.column_dtypes[3].name == 'category':
            v_ss3 = []
            for hc in np.arange(len(self.column_dtypes[3].categories)):
                v_ss3.append(gpflow.kernels.Linear(variance=0.6, 
                                                   active_dims=[sp + hc]))
            v_ss3 = np.sum(np.stack(v_ss3))
        else:
            v_ss3 = gpflow.kernels.SquaredExponential(lengthscales=5, 
                                                      variance=0.6, active_dims=[3])

        return v_ls + (v_ss1 * v_ss2 * v_ss3)

    def InitLikelihood(self):
        '''
        Initializes the mean function for the GP model.
        '''
        return gpflow.likelihoods.Softmax(self.c_a)
    
    def AssignVulClass(self, sim_a, df_target):
        return self.func_assign_vul_class(sim_a, df_target)
        
    def CondSampleAttributeCombo(self, function_samples):
        '''
        Generates samples of attribute combinations a conditional on 
        realizations of the GP functions g.
        
        R is the number of simulations and N is the size of the target set. 
        
        Parameters
        ----------
        function_samples : float array (R, N, ca)
            R samples from the ca GP functions g evaluated at N target
            points.

        Returns
        -------
        sam_a : integer array (R, N)
            Sampled attribute combinations

        '''        
        shape = function_samples.shape[:-1] 
        # Softmax 
        p_a = np.exp(function_samples) / np.sum(np.exp(function_samples), axis=-1)[:,:,None]
        cdf_a = np.cumsum( np.append(np.zeros(np.append(list(shape),1)), p_a, axis=-1), axis=-1)
        sam_a = np.zeros(shape)
        rnd_uni = np.random.uniform(0, 1, shape)
        for i in np.arange(self.c_a):
            sam_a[(rnd_uni > cdf_a[:,:,i]) & (rnd_uni <= cdf_a[:,:,i+1])] = i
            
        return sam_a.astype('int')        