import numpy as np
import random
import pandas as pd
from numba import jit

sim_params =  {
            'TARGET' : 'GDPcont_t0',
            # 'SPATIAL_AUTOCORRELATION_PCT' : 0,
            # 'N_STEPS' : 10,
            'N_TREATED' : 5,
            'N_STATES' : 10,
            'TREATMENT_YEAR' : (5, 0), #mu, sigma
            # 'TREATMENT_YEAR' : {'mu' : 5, 
            # #                     'sigma' : 0},
            'EXOG_GROWTH_PAR' : (.02,.01),
            'EXOG_CONTROL_EXPLAINED_SIGMA': .3,
            'SECTOR_GROWTH_SIGMA' : 0.0,                   
            # 'SECTOR_GROWTH_PCT_PREDICTORS' : {},
            'TREATMENT_EFFECT' : (.3, 0.002) #mu, sigma
            # 'TREATMENT_EFFECT' : {'mu': .3,
            #                       'sigma' : 0.002},
            # 'TREATMENT_EFFECT_PREDICTORS' : {} 
        }

@jit(nopython = True)
def init_sim(state_mat, N_STATES, N_TREATED, TREATMENT_YEAR_PAR, TREATMENT_EFFECT_PAR):
    
    treat_mat = np.empty((N_STATES, 6))
    
    treat_mat[:,0] = np.random.choice(state_mat[:,0], N_STATES, replace = False)
    # choose random states
    chosen_states_ind = treat_mat[:,0].astype(np.int8) 
    # get region for chosen states
    treat_mat[:,1] = state_mat[chosen_states_ind,1] 
    # get initial growth rate for chosen states
    treat_mat[:,2] = state_mat[chosen_states_ind,2]
    treat_mat[:,3] = np.nan

    # # assign treatment to N_TREATED states
    treat_mat[0:N_TREATED,3] = 1.0
    np.random.shuffle(treat_mat[:,3]) 
    
    # # assign treatment year to N_TREATED states
    year_mu, year_sig = TREATMENT_YEAR_PAR
    treat_mat[:,4] = np.random.normal(year_mu, year_sig,
                                    N_STATES) * treat_mat[:,3] 
    
    # generate the potential treatment effect for all states
    effect_mu, effect_sig = TREATMENT_EFFECT_PAR
    treat_mat[:,5] = np.random.normal(effect_mu, effect_sig, N_STATES)
    
    return treat_mat

@jit(nopython = True)
def run_sim(init_mat, N_STEPS,
             EXOG_GROWTH_PAR, EXOG_CONTROL_EXPLAINED_SIGMA, SECTOR_GROWTH_SIGMA,
             SPATIAL_AUTOCORRELATION_PCT):
            #  , SECTOR_GROWTH_PCT_PREDICTORS,
            #  TREATMENT_EFFECT, TREATMENT_EFFECT_PREDICTORS):
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
        exog_mu, exog_sig = EXOG_GROWTH_PAR
        i_Year_baseGrowth = np.random.normal(exog_mu, exog_sig, 1)[0]

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

            region_growth_base = np.random.normal(i_Year_baseGrowth,
                                        SECTOR_GROWTH_SIGMA,
                                        N_STATES) * SPATIAL_AUTOCORRELATION_PCT

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

        # if len(SECTOR_GROWTH_PCT_PREDICTORS) != 0:
        #     for cov, delta in SECTOR_GROWTH_PCT_PREDICTORS.items():
        #         treatment_effects += data[treatment_effects[cov]] * delta

        if i_Year > 0:
            growth_pcts_treated = growth_pcts + (init_mat[:,5] * dummy_treated_t)
            df_out[df_row_idx,3] = df_out[df_row_idx - N_STATES,3] + growth_pcts_treated
        else:
            df_out[df_row_idx,3] = init_mat[:,2]
        
    return df_out
    
# def sim_data(data, TARGET, N_STEPS, N_TREATED, N_STATES, TREATMENT_YEAR, 
#              EXOG_GROWTH_PCT, EXOG_CONTROL_PCT_EXPLAINED,
#              SPATIAL_AUTOCORRELATION_PCT,
#              SECTOR_GROWTH_PCT, SECTOR_GROWTH_PCT_PREDICTORS,
#              TREATMENT_EFFECT, TREATMENT_EFFECT_PREDICTORS):

# def sim_data(data, TARGET, N_STEPS, N_STATES, N_TREATED, TREATMENT_YEAR, TREATMENT_EFFECT
#              EXOG_GROWTH_PCT, EXOG_CONTROL_PCT_EXPLAINED,
#              SPATIAL_AUTOCORRELATION_PCT,
#              SECTOR_GROWTH_PCT, SECTOR_GROWTH_PCT_PREDICTORS,
#              TREATMENT_EFFECT, TREATMENT_EFFECT_PREDICTORS):

#     data = data.sample(n = N_STATES).sort_values(TARGET, ascending=False).reset_index().to_numpy()
#     states = data[:,0]
    
#     # if len(TREATMENT_EFFECT_PREDICTORS) != 0:
#     #     for cov, delta in TREATMENT_EFFECT_PREDICTORS.items():
#     #         treatment_effects += treatment_effects[cov] * delta

#     treat_mat = init_sim(N_STATES, N_TREATED, TREATMENT_YEAR, TREATMENT_EFFECT) #treated, treatment year, treat effect
    