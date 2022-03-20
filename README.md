# Earthquake RMGP

This repository collects the Python code for the risk model informed Gaussian process (RMGP) framwork for post-earthquake updating of regional damage estimates. The code accompanies following manuscript:

Bodenmann L, Reuland Y, Stojadinovic B. (2022): Post-earthquake updating of regional damage estimates. doi.

where the framework is explained and applied to three case studies. Feel free to use and enhance the current implementation, but please cite the original work.

## Installation
The requirements.txt file lists all required Python packages. 
I suggest following steps:
(1) Create a new virtual (conda) environment
(2) Use pip to install gpflow. This automatically installs tensorflow and other dependencies.
(3) Use pip to install pandas, matplotlib, scikit and statsmodels. 
Then you should be set up.

## Usage
The example notebook explains the basic workflow and functionalities using the toy dataset from this repo.
