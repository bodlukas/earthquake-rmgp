# -*- coding: utf-8 -*-
"""
Four main classes:
    - Identity_mod: Modified GPflow mean function.
    - Bernoulli_mod: Modified GPflow Bernoulli likelihood.
    - Ordinal_mod: Modified GPflow Ordinal likelihood.
    - VGP_mod: Modified GPflow variational GP model.

@author: Lukas Bodenmann, ETH Zurich, January 2022

For the original GPFlow code see: 
    https://gpflow.readthedocs.io/en/master/index.html 

"""
# License of GPFlow
# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
from gpflow.base import Parameter
from gpflow.mean_functions import MeanFunction
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.utilities import positive, to_default_int, triangular
from gpflow.likelihoods.base import ScalarLikelihood
from gpflow.likelihoods.utils import inv_probit
from gpflow.config import default_float, default_jitter
from gpflow.kullback_leiblers import gauss_kl
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
from gpflow.conditionals.util import base_conditional
from gpflow.conditionals import conditional
from gpflow import logdensities
from tensorflow_probability import distributions as tfd

#%%

class Identity_mod(MeanFunction):
    """
    GPflow mean Function that takes the last num_func entries of the input 
    vector x as the mean m(x).
    
    Attributes
    ----------
    num_func : int
        Number of (latent) functions.
        
    See also GPflow documentation of MeanFunctions.
    """

    def __init__(self, numClasses: int):
        MeanFunction.__init__(self)
        self.numClasses = numClasses

    def __call__(self, X):
        return tf.multiply(X[:,-self.numClasses:],
                           tf.ones_like(X[:,-self.numClasses:]))
#%%

class Bernoulli_mod(ScalarLikelihood):
    '''
    GPflow likelihood that is a modified version of the built-in Bernoulli 
    likelihood of GPflow.
    
    p( Y=1 | f ) = norm.cdf( (f - eta) / beta)
    
    To enable MAP-estimation of the vulnerability function parameters eta_0 
    and beta, this likelihood works with transformed parameters.    
    
    Attributes
    ----------
    beta : float (1, )
        Dispersion of threshold parameter.
        Constrained to positive values.
    eta_tilde : float (1, )
        Scaled threshold parameter that separates y=0 and y=1.
        eta_tilde = eta / beta
        Unconstrained. 
    
    '''
    def __init__(self, eta_tilde, beta, invlink=inv_probit, **kwargs):
        super().__init__(**kwargs)
        self.eta_tilde = Parameter( tf.reshape(eta_tilde, -1) )
        self.beta = Parameter( beta, transform=positive())
        self.invlink = invlink

    def _scalar_log_prob(self, F, Y):
        Y = to_default_int(Y)
        p = self.invlink( (F/self.betas) - self.eta_tilde)
        return logdensities.bernoulli(Y, p)

    def _predict_mean_and_var(self, Fmu, Fvar):
        if self.invlink is inv_probit:
            p = inv_probit( (Fmu - self.beta * self.eta_tilde) /
                           tf.sqrt(Fvar + self.beta**2) )
            return p, p - tf.square(p)
        else:
            # for other invlink, use quadrature
            return super()._predict_mean_and_var(Fmu, Fvar)

    def _predict_log_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return tf.reduce_sum(logdensities.bernoulli(Y, p), axis=-1)

    def _conditional_mean(self, F):
        return self.invlink(F)

    def _conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - (p ** 2)
#%%

class Ordinal_mod(ScalarLikelihood):
    '''
    GPflow likelihood that is a modified version of the built-in ordinal 
    likelihood of GPflow.
    
    p( Y=0 | f ) = 1 - norm.cdf( (f - eta_0) / beta)

    p( Y=1 | f ) = norm.cdf( (f - eta_0) / beta) - 
                    norm.cdf( (f - eta_1) / beta)    

    p( Y=2 | f ) = norm.cdf( (f - eta_1) / beta) - 
                    norm.cdf( (f - eta_2) / beta)              
                    
    p( Y=3 | f ) = norm.cdf( (f - eta_2) / beta)
    
    To enable MAP-estimation of the vulnerability function parameters, this 
    likelihood works with transformed parameters as explained below and in 
    the manuscript.
    
    Attributes
    ----------
    beta : float (1, )
        Dispersion of threshold parameters.
        Constrained to positive values.
    eta_tilde_0 : float (1, )
        First, scaled threshold parameter that separates y=0 and y=1.
        eta_tilde_0 = eta_0 / beta
        Unconstrained.
    deltas : (cy - 2, )
        Difference between scaled threshold parameters for y>1.
        deltas[0] = eta_1 / beta - eta_0 / beta
        deltas[1] = eta_2 / beta - eta_1 / beta
        deltas[2] = ...
        Constrained to positive values.
    
    '''
    def __init__(self, eta_tilde_0, deltas, beta, **kwargs):
        super().__init__(**kwargs)
        self.beta = Parameter( beta, transform=positive())
        self.eta_tilde_0 = Parameter( tf.reshape(eta_tilde_0, -1) )
        self.deltas = Parameter(deltas, transform=positive())
        self.num_bins = deltas.size + 2
        
    def _scalar_log_prob(self, F, Y):
        '''
        Returns logarithm of p(y|f)
        
        w_left = [-\infty, eta_tilde_0, eta_tilde_0 + deltas[0], 
                  eta_tilde_0 + deltas[0] + deltas[1]]
        w_right = [eta_tilde_0, eta_tilde_0 + deltas[0], 
                  eta_tilde_0 + deltas[0] + deltas[1], +\infty]
        
        log(p(y|f)) = log( norm.cdf( f/beta - w_left[y]) - 
                           norm.cdf( f/beta - w_right[y]) )
        
        '''
        w_left = tf.concat([np.array([-np.inf]), 
                                     self.eta_tilde_0,
                                     self.eta_tilde_0 + tf.cumsum(self.deltas)], 0)
        w_right = tf.concat([self.eta_tilde_0,
                                     self.eta_tilde_0 + tf.cumsum(self.deltas),
                                     np.array([np.inf])], 0)
        Y = to_default_int(Y)
        sel_w_left = tf.gather(w_left, Y)
        sel_w_right = tf.gather(w_right, Y)
        log_p = tf.math.log( inv_probit( (F / self.beta) - sel_w_left) - 
                             inv_probit( (F / self.beta) - sel_w_right) + 
                             1e-6 )
        return log_p
    
    def _make_phi(self, F):
        w_left = tf.concat([np.array([-np.inf]), 
                                     self.eta_tilde_0,
                                     self.eta_tilde_0 + tf.cumsum(self.deltas)], 0)
        w_right = tf.concat([self.eta_tilde_0,
                                     self.eta_tilde_0 + tf.cumsum(self.deltas),
                                     np.array([np.inf])], 0)
        return inv_probit( (tf.reshape(F,(-1,1)) / self.beta) - w_left) -\
            inv_probit( (tf.reshape(F,(-1,1)) / self.beta) - w_right)
    
    def _conditional_mean(self, F):
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype = default_float()), (-1,1))
        return tf.reshape(tf.linalg.matmul(phi,Ys), tf.shape(F))
    
    def _conditional_variance(self, F):
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype = default_float()), (-1,1))
        E_y = phi @ Ys
        E_y2 = phi @ (Ys ** 2)
        return tf.reshape(E_y2 - E_y**2,tf.shape(F))

#%%
class VGP_mod(GPModel, InternalDataTrainingLossMixin):
    '''
    This is a custom GP model that performs two-staged inference:
        (1): Inference using seismic recordings (shake map)
        (2): Inference using observed damage from inspected buildings
    It is closely related to the VGP model of gpflow. See also the
    corresponding documentation.    
    '''   
    def __init__(self, mean_function: MeanFunction, 
                 kernel: Kernel, likelihood: Likelihood, 
                 num_latent_gps: int
    ):
        """
        kernel, likelihood, mean_function are appropriate GPflow objects
        num_latent_gps is the number of (latent) function with a GP prior.
        """
        
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.D_S = None
        self.D_I = None

    def assign_station_data(self, StationData: RegressionData):
        '''
        StationData = (X_S, Z) contains the input points [nS, D] of the seismic 
                        stations and the measured GM-IM [Ns, 1] in units [g]
        
        Once this data is assigned, sampling will be with respect to the posterior
        predictive p(f_T|z) instead of the prior p(f_T).
        The posterior predictive is derived with the method _inference_recorded_im
        explained below.
        '''
        self.D_S = data_input_to_tensor(StationData)

    def assign_insp_data(self, InspData: RegressionData):
        '''
        InspData = (X_I, Y) contains the input points X [nI, D] and the 
                    observations Y [nI, 2]
        The first column of Y stores the observed damage states y and the 
        second column indicates the observed vulnerability class. 
        Once this data is assigned, the variational parameters q_mu, and q_sqrt
        are initialized.
        These parameters are not (yet) trained. Training is described in the 
        RMGP object.
        '''
        self.D_I = data_input_to_tensor(InspData)
        X_I, y = self.D_I        
        num_data = X_I.shape[0]

        self.q_mu = Parameter(np.zeros((num_data, self.num_latent_gps)))
        q_sqrt = np.array([np.eye(num_data) for _ in range(self.num_latent_gps)])
        self.q_sqrt = Parameter(q_sqrt, transform=triangular())

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()
    
    def _inference_recorded_im(self, X_T: InputData) -> MeanAndVariance:
        '''    
        Parameters
        ----------
        X_T : Tensorflow Tensor (n x d)
            Input Tensor of locations where to predict GM-IM.
            First two columns should specify the x and y coordinate in km.
            Last column stores the mean logarithm of the GM-IM at the locations
            as calculated via a GMM trend function.
    
        Returns
        -------
        nu_T : Tensorflow Tensor (n x 1)
            The mean vector of the posterior distribution p(f_T|z).
        Psi_TT : Tensorflow Tensor (n x n)
            The covariance matrix of the posterior distribution p(f_T|z)
    
        '''
        X_S, z = self.D_S
        stdB = tf.sqrt(self.kernel.kernels[0].variance)
        stdW = tf.sqrt(self.kernel.kernels[1].variance)
        C_SS = self.kernel.kernels[1](X_S) / stdW**2
        inv_C_SS = tf.linalg.inv(C_SS)
        Z = tf.ones_like(z)
        psiBsq = 1 / ( (1 / stdB**2) + 
                     tf.matmul( tf.transpose(Z), tf.matmul(inv_C_SS, Z) ) / stdW**2 )
        xiB = psiBsq / stdW**2 * tf.matmul( 
            tf.transpose(Z), tf.matmul(inv_C_SS, 
                                       tf.math.log(z) - self.mean_function(X_S) ) )

        C_TT = self.kernel.kernels[1](X_T) / stdW**2
        C_TS = self.kernel.kernels[1](X_T, X_S) / stdW**2
    
        nu_T = self.mean_function(X_T) + xiB + tf.matmul(
            C_TS, tf.matmul(inv_C_SS, tf.math.log(z) - self.mean_function(X_S) - xiB) )
        Psi_TT = (stdW**2 + psiBsq) * (C_TT - tf.matmul(C_TS, tf.matmul(inv_C_SS, tf.transpose(C_TS) ) ) )
        # np.array([psi2 for _ in range(m_ta.num_latent_gps)])
        return nu_T, Psi_TT
         
    def _sliced_covs(self, Xnew: InputData, full_cov: bool = False):
        '''
        This function calculates following mean vectors and covariance matrices:
            mum, Kmm: Mean vector and covariance matrix of posterior p(f_I|z)
            mun, Knn: Mean vector and covariance matrix of posterior p(f_T|z)
            Kmn: Covariance matrix between input points of inspected buildings
                    X_I and target buildings X_T.
        First evaluates the posterior mean vector and covariance matrix of
        the combined set of input points (X_I u X_T) and then slices the results.
        '''       
        X_I, _ = self.D_I
        num_data = X_I.shape[0]
        X_Tot = tf.concat([X_I, Xnew],axis=0)
        m_Tot, K_Tot = self._inference_recorded_im(X_T = X_Tot)               
        mum = tf.slice(m_Tot, [0, 0], [num_data, 1])
        mun = tf.slice(m_Tot, [num_data ,0], [-1, 1])
        Kmm = tf.slice(K_Tot, [0, 0], [num_data, num_data])
        Kmn = tf.slice(K_Tot, [0, num_data], [num_data,-1])
        Knn = tf.slice(K_Tot, [num_data, num_data], [-1, -1])
        if full_cov == False: Knn = tf.linalg.diag_part(Knn)
        Kmm = Kmm + tf.eye(num_data, dtype=default_float()) * default_jitter()
        return mum, mun, Kmm, Kmn, Knn
      
    def elbo(self) -> tf.Tensor:
        r"""
        This method computes the variational lower bound on the marginal
        likelihood. It is equivalent to the negative variational free energy,
        explained in the manuscript.
            
        See the GPflow documentation for more information.

        """
        
        X_I, y = self.D_I
        num_data = X_I.shape[0]
        # Get prior KL.
        KL = gauss_kl(self.q_mu, self.q_sqrt)

        # Get conditionals
        if self.D_S is None:
            K = self.kernel(X_I)
            muOld = self.mean_function(X_I)
        else:
            muOld, K = self._inference_recorded_im(X_T = X_I)
            
        K = K +  tf.eye(num_data, dtype=default_float()) * default_jitter()
        L = tf.linalg.cholesky(K)
        fmean = tf.linalg.matmul(L, self.q_mu) + muOld  # [NN, ND] -> ND
        q_sqrt_dnn = tf.linalg.band_part(self.q_sqrt, -1, 0)  # [D, N, N]
        L_tiled = tf.tile(tf.expand_dims(L, 0), tf.stack([self.num_latent_gps, 1, 1]))
        LTA = tf.linalg.matmul(L_tiled, q_sqrt_dnn)  # [D, N, N]
        fvar = tf.reduce_sum(tf.square(LTA), 2)

        fvar = tf.transpose(fvar)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, y)

        return tf.reduce_sum(var_exp) - KL

    def predict_f(self, Xnew: InputData,  
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        
        if (self.D_I is None): # Before inspection data is gathered
            if (self.D_S is None): # No data from seismic stations available
                mu = self.mean_function(Xnew)
                var = self.kernel(Xnew, full_cov = full_cov)
            else: # Includes available data from seismic stations
                mu, var = self._inference_recorded_im(X_T = Xnew)
                if full_cov == False: var = tf.linalg.diag_part(var)              
            cov_shape = tf.TensorShape([self.num_latent_gps]).concatenate(var.shape)
            var = tf.broadcast_to(var, cov_shape)
        else: # Inspection data is available
            if (self.D_S is None): # No data from seismic stations available
                X_data, _ = self.D_I
                mu, var = conditional(Xnew, X_data, self.kernel, self.q_mu,
                        q_sqrt=self.q_sqrt, full_cov=full_cov, white=True
                    )
                mu = mu + self.mean_function(Xnew)
            else: # Includes available data from seismic stations
                mum, mun, Kmm, Kmn, Knn = (
                    self._sliced_covs(Xnew=Xnew, full_cov=full_cov)
                    )
                mu, var = base_conditional(
                    Kmn=Kmn, Kmm=Kmm, Knn=Knn, f=self.q_mu, full_cov=full_cov, 
                    q_sqrt=self.q_sqrt, white=True)
                mu = mu + mun
        return mu, var

#%% Helper functions for hyper-prior distributions    

def GammaPrior(val, std):
    '''
    Calculates the parameters alpha and beta of a Gamma distribution and 
    generates a tensorflow probability distribution.
    Parameters are calculate such that mode = val and variance = std**2
    
    The mode is (alpha-1)/beta for alpha >= 1
    The variance is alpha/beta^2
    '''
    a = val/std
    alpha = 0.5 * (a**2 + a * np.sqrt(a**2 + 4) + 2) # Mode=Mean, Variance=std**2 and alpha > 1
    beta = (alpha - 1)/val
    p = tfd.Gamma(concentration=alpha, rate=beta)
    return p

def NormPrior(val, CoV):
    '''
    Calculates the parameters of a Normal distribution and
    generates a tensorflow probability distribution.
    Parameters are calculated such that mean=mode=val and scale = std**2
    '''
    mu = val
    sigma = np.sqrt((val*CoV)**2)
    p = tfd.Normal(loc=mu, scale=sigma)
    return p