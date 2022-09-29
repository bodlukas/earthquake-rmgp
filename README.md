# Earthquake RMGP

543142934

This repository collects the Python code for the risk model informed Gaussian process (RMGP) framework that updates regional damage estimates with early post-earthquake inspection data. The code accompanies the following manuscript: 
> Bodenmann L., Reuland Y. and Stojadinovic B. (2022): *Dynamic Post-Earthquake Updating of Regional Damage Estimates Using Gaussian Processes*; Submitted to Reliability Engineering & System Safety. Preprint available at https://doi.org/10.31224/2205

The manuscript explains the framework and presents its application to one simulated and two real earthquake damage datasets. Feel free to use and enhance the current implementation, but make sure to cite the original work.

## Supplementary Material
The corresponding folder contains additional information for the three case studies analyzed in the above manuscript. In particular, several html files explain the prior risk models and data pre-processing steps employed in the three case studies.

## Installation
The requirements.txt file lists all required Python packages. 
I suggest following steps:
1. Create a new virtual (conda) environment
2. Activate the new environment.
3. Use `pip` to install `gpflow`. This automatically installs `tensorflow` and other dependencies.
4. Use `pip` to install `pandas`, `matplotlib`, `scikit` and `statsmodels`. 

Then you should be set up. Note that `scikit` and `statsmodels` are only required for the Ranfom Forest and the Ordered Linear Probit models, but not for the proposed RMGP.

## Usage
The `example.ipynb` notebook explains the basic workflow and functionalities using the toy dataset from this repo. For additional explanations we refer to the manuscript and the supplementary material.

### Structure
- rmgp: Contains the code for the RMGP framework. 
    - `modules.py`: Components of the prior risk model used in RMGP (ground motion, damage and typological attribution).
    - `model.py`: Main model that combines the different components and is used for as a wrapper for inference and predictions.
    - `utils_gpflow.py`: GPFlow-compatible objects for mean functions, likelihoods and variational inference that are specifically adapted for RMGP.
- utils: Contains further utilities used to genereate random inspection sequences and to collect results.
- other_models: Contains the implementations of Random Forest and Ordered linear probit methods that we used in the manuscript to compare with RMGP.
- data_toyexample: Contains the data for the toy example used in the `example.ipynb` notebook.
