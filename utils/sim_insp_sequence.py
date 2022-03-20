# -*- coding: utf-8 -*-
"""
@author: Lukas Bodenmann, ETH Zurich, January 2022
"""

import numpy as np

def SimProcess(df, seed, AvTeams, ProdTeam, MaxTimeStep):
    '''
    Simulates an inspection sequence. 
    The first timestep at which buildings get inspected is t=1. 
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the building input information.
        Especially the geo-coordinates in km.
    seed : int
    AvTeams : int
        Number of inspection teams available each timestep.
        Here assumed to be constant.
    ProdTeam : int
        Number of buildings the one team can inspect in one timestep.
        Here assumed to be constant.
    MaxTimeStep : int
        Maximum number of timesteps for which the inspection sequence
        is simulated. 

    Returns
    -------
    InspTimeStep : array of size (N, )
        For each building, the timestep at which it was inspected.
        For buildings that were not inspected we assign a value of 
        (MaxTimeStep + 10).

    '''
    N = len(df)
    rng = np.random.RandomState(seed = seed)
    Coorx = df.x.values
    Coory = df.y.values
    inspind = np.zeros(N, dtype='bool')
    InspTimeStep = np.ones(N, dtype='int32')*int(MaxTimeStep + 10)
    index = np.arange(N)
    for d in np.arange(1, MaxTimeStep + 1):
        mask = (inspind == False)
        ind_init = rng.choice(index[mask], int(AvTeams), replace=False)
        inspind[ind_init] = True
        ind_obs = ind_init
        mask = (inspind == False)
        ind_rem = index[mask]
        for i in ind_init:
            ind_next = ind_rem[np.argsort(
                np.sqrt((Coorx[i] - Coorx[ind_rem])**2 + (Coory[i]-Coory[ind_rem])**2) )[:ProdTeam-1]]
            inspind[ind_next] = True
            ind_obs = np.append(ind_obs, ind_next)
            mask = (inspind == False)
            ind_rem = index[mask]
        InspTimeStep[ind_obs] = d
        if np.sum(mask) <= (AvTeams*ProdTeam): 
            InspTimeStep[ind_rem] = d+1
            break
    return InspTimeStep