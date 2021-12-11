import numpy as np
import random
import pandas as pd
from numba.pycc import CC

cc = CC('init_sim')
@cc.export('init_sim', 'f8[:,:](f8[:,:], i8, i8, f8, f8, f8, f8)')
def init_sim(state_mat, N_STATES, N_TREATED, TREATMENT_YEAR_MU,
            TREATMENT_YEAR_SIGMA, TREATMENT_EFFECT_MU, TREATMENT_EFFECT_SIGMA):

    treat_mat = np.empty((N_STATES, 6))
    
    treat_mat[:,0] = np.random.choice(state_mat[:,0], N_STATES, replace = False)
    # choose random states
    chosen_states_ind = treat_mat[:,0].astype(np.int8) 
    # get region for chosen states
    treat_mat[:,1] = state_mat[chosen_states_ind,1].astype(np.int8) 
    # get initial growth rate for chosen states
    treat_mat[:,2] = state_mat[chosen_states_ind,2]

    # # assign treatment to N_TREATED states
    treat_mat[:,3] = np.nan
    treat_mat[0:N_TREATED,3] = 1.0
    np.random.shuffle(treat_mat[:,3]) 
    
    # # assign treatment year to N_TREATED states
    treat_mat[:,4] = np.random.normal(TREATMENT_YEAR_MU,
                                        TREATMENT_YEAR_SIGMA,
                                        N_STATES).astype(np.int8) * treat_mat[:,3]
    
    # generate the potential treatment effect for all states
    treat_mat[:,5] = np.random.normal(TREATMENT_EFFECT_MU,
                                        TREATMENT_EFFECT_SIGMA,
                                        N_STATES)
    
    return treat_mat
cc.compile()

cc = CC('run_sim')
@cc.export('run_sim', 'f8[:,:](f8[:,:], f8[:], i8, f8, f8, f8, f8, f8)')
def run_sim(init_mat, REGIONAL_GROWTH_PCT_PREDICTORS, N_STEPS,
             EXOG_GROWTH_MU, EXOG_GROWTH_SIGMA, EXOG_CONTROL_EXPLAINED_SIGMA, SECTOR_GROWTH_SIGMA,
             SPATIAL_AUTOCORRELATION_PCT):
    N_STATES = init_mat.shape[0]
    df_out = np.empty((N_STATES * (N_STEPS + 1), 4)) #state_id, t, stateControls_t, GDPcont_t
    for i_Year in range(0, N_STEPS + 1): # I think due to the fact that this is a time-series, there isn't a way to vectorize...
        df_row_idx = (i_Year * N_STATES) + np.arange(0,N_STATES) 
        df_out[df_row_idx, 0] = init_mat[:,0] #State
        df_out[df_row_idx, 1] = i_Year #timestep

        dummy_treated_t = (i_Year == init_mat[:,4]).astype(np.int8)

        # this growth percentage is the mean of each individual state's growth percentage
        # can be thought of as the exogenous component to state level growth - ie, condition
        # of the global economy during year i
        i_Year_baseGrowth = np.random.normal(EXOG_GROWTH_MU,
                                             EXOG_GROWTH_SIGMA, 1)[0]

        # this growth percent varies by state, but drawn from the same distribution centered
        # at the exogenous growth percentage drawn above
        growth_pcts = np.random.normal(i_Year_baseGrowth,
                                        SECTOR_GROWTH_SIGMA,
                                        N_STATES)

        if SPATIAL_AUTOCORRELATION_PCT != 0:
            # if autocorrelated, take the random growth percents generated, and 
            # make them all closer to their mean by region. If autocorrelation is 
            # very high, ie 1, then all states in the region will have the same rate. 
            # if 0, all states are indepedent and region has no effect.
            N_REGIONS = len(np.unique(init_mat[:,1]))
            region_growth_base = np.random.normal(
                                        i_Year_baseGrowth,
                                        SECTOR_GROWTH_SIGMA,
                                        N_REGIONS) * SPATIAL_AUTOCORRELATION_PCT

            growth_pcts = growth_pcts * (1 - SPATIAL_AUTOCORRELATION_PCT)

            for idx, region_id in enumerate(np.unique(init_mat[:,1])):
                dummy_region = (init_mat[:,1] == region_id).astype(np.int8)
                growth_pcts += region_growth_base[idx] * dummy_region

        # This is an index for endogenous variables that help us explain the growth rate
        # for a particular state - if the sigma here is 0, this means that we can explain
        # state level growth entirely since controls == growth pct
        pct_explained = np.random.normal(1,
                            EXOG_CONTROL_EXPLAINED_SIGMA,
                            N_STATES)
        df_out[df_row_idx,2] = growth_pcts * pct_explained

        # If there are regional growth predictors, parallel trends violated - this
        # means a different counterfactual growth rate for the treated and control groups, 
        # making the DID estimate invalid
        # if REGIONAL_GROWTH_PCT_PREDICTORS.sum() > 0:
        #     for region_id in np.unique(init_mat[:,1]):
        #         dummy_region = (region_id == init_mat[:,1]).astype(np.int8)
        #         growth_pcts += REGIONAL_GROWTH_PCT_PREDICTORS[region_id] * dummy_region

        if i_Year > 0:
            growth_pcts_treated = growth_pcts + (init_mat[:,5] * dummy_treated_t)
            df_out[df_row_idx,3] = df_out[df_row_idx - N_STATES,3] + growth_pcts_treated
        else:
            df_out[df_row_idx,3] = init_mat[:,2]
        
    return df_out
cc.compile()
