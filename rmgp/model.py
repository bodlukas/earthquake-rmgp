"""
The main risk-model informed GP model

@author: Lukas Bodenmann, ETH Zurich, January 2022
"""

import gpflow as gpflow
import numpy as np
import pandas as pd
from gpflow.config import Config, as_context
import tensorflow as tf
from .utilsGPflow import VGP_mod, GammaPrior, NormPrior
from copy import deepcopy
from .modules import (SpatialGMM, TypologyAttributionModule, 
                        DamageEstimationModule, EventChar)
     
    
#%% Combined risk-model informed GP (RMGP)

class RMGP(object):
    '''
    Collects all submodules and methods to predict damage levels and typology
    attribute combinations and to perform inference using data from seismic
    network stations and / or from building inspections.
    
    SpGMM, TAM and DEM are the specified modules and EventChar collects
    the event characteristics. See their documentation for further details.
    
    m_im: GP model for the function f = log(im)
    m_ta: GP model for the functions g (typological attribution)
    '''
    
    def __init__(self, SpGMM: SpatialGMM, TAM: TypologyAttributionModule,
                 DEM: DamageEstimationModule, EventChar: EventChar, jitter = 1e-5):
        self.SpGMM = deepcopy(SpGMM)
        self.TAM = deepcopy(TAM)
        self.DEM = deepcopy(DEM)
        self.jitter = jitter
        self.EventChar = EventChar
    
        m_im = VGP_mod(kernel = self.SpGMM.InitCov_Func(),
                       likelihood = self.DEM.InitLikelihood(),
                       mean_function = self.SpGMM.InitMean_Func(),
                       num_latent_gps = 1)
        self.m_im = m_im
        
        m_ta = VGP_mod(kernel = self.TAM.InitCov_Func(),
                       likelihood = self.TAM.InitLikelihood(),
                       mean_function = self.TAM.InitMean_Func(),
                       num_latent_gps = self.TAM.c_a)
        self.m_ta = m_ta
        
    def Inference_station_data(self, df_station: pd.DataFrame, 
                               recorded_ims: np.array):
        '''
        Performs inference using data D_S = (X_S, z) from Ns seismic network 
        stations.
        Once this function is called, sampling of function f will be with
        respect to the posterior predictive p(f_T | D_S).

        Parameters
        ----------
        df_station : pd.DataFrame
            Dataframe with required information about the seismic stations. 
            Specifically, it should provide the columns specified in 
            column_names of the assigned GMM.
        recorded_ims : np.array (Ns, )
            Recorded ground-motion IM.

        '''
        X_S = self.SpGMM.InitInput(df_station, self.EventChar)
        D_S = (X_S, recorded_ims.reshape(-1,1))
        self.m_im.assign_station_data(StationData = D_S)
        
    def Inference_inspection_data(self, df_insp: pd.DataFrame, 
                                  obs_a: np.array, obs_y: np.array):
        '''
        Performs inference using data D_I = (X_I, y, a) from M inspected 
        buildings.
        Once this function is called, sampling of function f will be w.r.t
        the posterior predictive p(f_T | D_I, D_S) or w.r.t. 
        p(f_T | D_I) if seismic recordings are not available.
        
        For inference of the function f and vulnerability function parameters
        the SciPy implementation of the L-BFGS-B optimizer is used.
        For inference of the functions g, the parameters of the variational
        distribution q are found via natural gradients and the hyper-parameters
        are estimated using a tensorflow implementation of the adam optimizer. 
        See the GPflow documentation for more details.

        Parameters
        ----------
        df_insp : pd.DataFrame
            Dataframe with required information about the inspected bldgs. 
            Specifically, it should provide the columns specified in 
            column_names of the assigned GMM and column_names of the assigned
            typological attribution model.
        obs_a : np.array (M, )
            Typological attribute combinations assigned during inspection.
        obs_y : np.array (M, )
            Damage levels assigned during inspection.

        '''
        
        # ----------------------------
        # Step 1: Function f = log(im)
        # ----------------------------
        
        X_I = self.SpGMM.InitInput(df_insp, self.EventChar)
        # Assign vulnerability classes b based on observed attribute combinations
        obs_b = self.TAM.AssignVulClass(obs_a, df_insp)
        D_I = (X_I, np.hstack([obs_y.reshape(-1,1), obs_b.reshape(-1,1)]))
        self.m_im.assign_insp_data(InspData = D_I)
        # The parameters of the kernel are fixed to the initial values
        gpflow.set_trainable(self.m_im.kernel, False)
        for likTemp in self.m_im.likelihood.likelihoods:
            # Assign a weakly informative normal hyper-prior for the first 
            # threshold parameter 
            likTemp.eta_tilde_0.prior = NormPrior(val=likTemp.eta_tilde_0.numpy(),
                              CoV = 0.8)
            gpflow.set_trainable(likTemp.eta_tilde_0, True)
            if self.DEM.c_y > 2:
                # Assign a weakly informative gamma hyper-prior for the deltas
                # Note that deltas are constrained to be positive.             
                likTemp.deltas.prior = GammaPrior(val = likTemp.deltas.numpy(), 
                                    std = 0.9 * np.ones(len(likTemp.deltas.numpy())) )
                gpflow.set_trainable(likTemp.deltas, True)
            
            gpflow.set_trainable(likTemp.beta, False)
        
        config = Config(jitter = self.jitter)
        # Training step
        opt_im = gpflow.optimizers.Scipy()
        with as_context(config):
            opt_im.minimize(self.m_im.training_loss, 
                            self.m_im.trainable_variables,
                            options=dict(maxiter=1000))
        # Collect MAP estimates of vulnerability function parameters
        self.DEM.CollectEstimatedLikelihoodParams(self.m_im.likelihood)
        
        # -------------------------------------------------
        # Step 2: Functions g (for typological attribution)
        # -------------------------------------------------
                  
        X_I = self.TAM.InitInput(df_insp)
        D_I = (X_I, obs_a.reshape(-1,1))
        self.m_ta.assign_insp_data(InspData = D_I)
        # Parameters of variational distribution are optimized via natgrad
        gpflow.set_trainable(self.m_ta.q_mu, False)
        gpflow.set_trainable(self.m_ta.q_sqrt, False)
        adam_opt = tf.optimizers.Adam(0.1)
        natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.2)
        with as_context(config):
            for k in range(20):
                natgrad_opt.minimize(self.m_ta.training_loss, 
                                     var_list = [(self.m_ta.q_mu, self.m_ta.q_sqrt)])
                adam_opt.minimize(self.m_ta.training_loss, 
                                  var_list = self.m_ta.trainable_variables)        
               
    def Sample_typol_attr(self, df_target: pd.DataFrame, nSamples: int,
                          aggregate: bool=False):
        '''
        Samples typology attribute combinations a in (0,1,...,ca-1) for N target 
        buildings from p(g) or p(g|D_I), depending on whether inspection data 
        D_I is available and processed.

        Parameters
        ----------
        df_target : pd.DataFrame
            Dataframe with required information about the N target bldgs. 
            Specifically, it should provide the columns specified in 
            column_names of the assigned typological attribution model.
        nSamples : int
            Number of samples to generate.
        aggregate : bool, optional
            Whether to aggregate samples for each attribute combination. 
            The default is False.

        Returns
        -------
        a_samples : array of type int
            If aggregate is false the shape is (nSamples, N)
            If aggregate is true the shape is (nSamples, ca)

        '''
        config = Config(jitter = self.jitter)
        X_T = self.TAM.InitInput(df_target)
        with as_context(config):
            g_samples = self.m_ta.predict_f_samples(X_T, nSamples).numpy()
        a_samples = self.TAM.CondSampleAttributeCombo(g_samples)
        if aggregate: 
            a_samples = np.apply_along_axis(lambda x: np.bincount(x, minlength = self.TAM.c_a), 
                                               axis=1, arr=a_samples.astype('int'))
        return a_samples.astype('int')

    def Sample_damage(self, df_target: pd.DataFrame, nSamples: int,
                          aggregate: bool=False):
        '''
        Samples damage levels y in (0,1,...,cy-1) for N target 
        buildings from p(f) or p(f|D_S) or p(f|D_I,D_S) and p(g) or p(g|D_I), 
        depending on whether station data D_S and inspection data D_I is 
        available and processed.        

        Parameters
        ----------
        df_target : pd.DataFrame
            Dataframe with required information about the N target bldgs. 
            Specifically, it should provide the columns specified in 
            column_names of the assigned GMM and column_names of the assigned
            typological attribution model.
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
        config = Config(jitter = self.jitter)
        X_T = self.SpGMM.InitInput(df_target, self.EventChar)
        with as_context(config):
            f_samples = self.m_im.predict_f_samples(X_T, nSamples).numpy()
            f_samples = np.squeeze(f_samples, axis=-1)
        
        a_samples = self.Sample_typol_attr(df_target, nSamples)
        b_samples = self.TAM.AssignVulClass(a_samples, df_target)
        
        y_samples = self.DEM.CondSampleDamage(f_samples, b_samples)
        if aggregate: 
            y_samples = np.apply_along_axis(lambda x: np.bincount(x, minlength = self.DEM.c_y), 
                                               axis=1, arr=y_samples.astype('int'))    
        return y_samples.astype('int')