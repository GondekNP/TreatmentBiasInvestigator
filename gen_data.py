import numpy as np
import random
import pandas as pd

sim_params =  {
            'N_YEARS' : 8,
            'N_TREATED' : 2,
            'N_STATES' : 4,
            'TREATMENT_YEAR' : {'mu' : 4, 
                                'sigma' : 1},
            'GDP_GROWTH_PCT' : {'mu' : .02, 
                                'sigma' : .001},                   
            'GDP_GROWTH_PCT_PREDICTORS' : {},
            'TREATMENT_EFFECT' : {'mu': .5,
                                  'sigma' : .0001},
            'TREATMENT_EFFECT_PREDICTORS' : {} 
        }


def sim_data(data, N_YEARS, N_TREATED, N_STATES, TREATMENT_YEAR, 
             GDP_GROWTH_PCT, GDP_GROWTH_PCT_PREDICTORS,
             TREATMENT_EFFECT, TREATMENT_EFFECT_PREDICTORS):

    data = data.sample(n = N_STATES).sort_values('stateGDP_t0', ascending=False)
    
    treated = np.array([np.repeat([np.nan], N_STATES - N_TREATED),
                        np.repeat([1],      N_TREATED)]).flatten()
    np.random.shuffle(treated)

    #determine a treatment date, either the same for all (if sigma = 0), or staggered (if sigma != 0)    
    data.loc[:,'treatment_year'] = np.random.normal(TREATMENT_YEAR['mu'],
                                                    TREATMENT_YEAR['sigma'],
                                                    N_STATES) 
    data.loc[:,'treatment_year'] = data.loc[:,'treatment_year'].astype(int) * treated                                             

    for i_Year in range(1, N_YEARS + 1): # I think due to the fact that this is a time-series, there isn't a way to vectorize...
        dummy_treated_t = (i_Year > data['treatment_year']).apply(int)
        treatment_effects = np.random.normal(TREATMENT_EFFECT['mu'],
                                             TREATMENT_EFFECT['sigma'],
                                             N_STATES)

        if len(TREATMENT_EFFECT_PREDICTORS) != 0:
            for cov, delta in TREATMENT_EFFECT_PREDICTORS.items():
                treatment_effects += treatment_effects[cov] * delta

        growth_pcts = np.random.normal(GDP_GROWTH_PCT['mu'],
                                       GDP_GROWTH_PCT['sigma'],
                                       N_STATES)
        
        if len(GDP_GROWTH_PCT_PREDICTORS) != 0:
            for cov, delta in GDP_GROWTH_PCT_PREDICTORS.items():
                treatment_effects += treatment_effects[cov] * delta

        growth_pcts = growth_pcts + (treatment_effects * dummy_treated_t)

        data.loc[:, 'stateGDP_t' + str(i_Year)] = \
            data.loc[:, 'stateGDP_t' + str(i_Year - 1)] * (1 + growth_pcts)
        data.loc[:, 'stateGDPpct_t' + str(i_Year)] = growth_pcts
    
    return data
