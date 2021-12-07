import numpy as np
import random
import pandas as pd

sim_params =  {
            'TARGET' : 'GDPcont_t0',
            'N_YEARS' : 50,
            'N_TREATED' : 25,
            'N_STATES' : 50,
            'TREATMENT_YEAR' : {'mu' : 25, 
                                'sigma' : 0},
            'SECTOR_GROWTH_PCT' : {'mu' : .02, 
                                'sigma' : .01},                   
            'SECTOR_GROWTH_PCT_PREDICTORS' : {},
            'TREATMENT_EFFECT' : {'mu': .06,
                                  'sigma' : 0.02},
            'TREATMENT_EFFECT_PREDICTORS' : {} 
        }


def sim_data(data, TARGET, N_YEARS, N_TREATED, N_STATES, TREATMENT_YEAR, 
             SECTOR_GROWTH_PCT, SECTOR_GROWTH_PCT_PREDICTORS,
             TREATMENT_EFFECT, TREATMENT_EFFECT_PREDICTORS):

    data = data.sample(n = N_STATES).sort_values(TARGET, ascending=False)
    
    treated = np.array([np.repeat([np.nan], N_STATES - N_TREATED),
                        np.repeat([1],      N_TREATED)]).flatten()
    np.random.shuffle(treated)

    #determine a treatment date, either the same for all (if sigma = 0), or staggered (if sigma != 0)    
    data.loc[:,'treatment_year'] = np.random.normal(TREATMENT_YEAR['mu'],
                                                    TREATMENT_YEAR['sigma'],
                                                    N_STATES) 
    data.loc[:,'treatment_year'] = data.loc[:,'treatment_year'].astype(int) * treated                                             

    for i_Year in range(1, N_YEARS + 1): # I think due to the fact that this is a time-series, there isn't a way to vectorize...
        # dummy_treated_t = (i_Year > data['treatment_year']).apply(int)
        dummy_treated_t = (i_Year == data['treatment_year']).apply(int)
        treatment_effects = np.random.normal(TREATMENT_EFFECT['mu'],
                                             TREATMENT_EFFECT['sigma'],
                                             N_STATES)

        if len(TREATMENT_EFFECT_PREDICTORS) != 0:
            for cov, delta in TREATMENT_EFFECT_PREDICTORS.items():
                treatment_effects += treatment_effects[cov] * delta

        # this growth percent is the percentage in tech growth shared by all states
        growth_pcts = np.random.normal(SECTOR_GROWTH_PCT['mu'],
                                       SECTOR_GROWTH_PCT['sigma'],
                                       N_STATES)
        
        if len(SECTOR_GROWTH_PCT_PREDICTORS) != 0:
            for cov, delta in SECTOR_GROWTH_PCT_PREDICTORS.items():
                treatment_effects += treatment_effects[cov] * delta

        growth_pcts = growth_pcts + (treatment_effects * dummy_treated_t)
        data.loc[:, 'GDPcont_t' + str(i_Year)] = \
            data.loc[:, 'GDPcont_t' + str(i_Year - 1)] + growth_pcts

        # data.loc[:, 'stateGDP_t' + str(i_Year)] = \
        #     data.loc[:, 'stateGDP_t' + str(i_Year - 1)] * (1 + growth_pcts)
        # data.loc[:, 'stateGDPpct_t' + str(i_Year)] = growth_pcts
    
    return data
