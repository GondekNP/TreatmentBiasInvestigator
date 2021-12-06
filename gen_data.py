import numpy as np
import random
import pandas as pd

sim_params =  {
            'N_YEARS' : 20,
            'TREATMENT_YEAR' : {'mu' : 10, 
                                'sigma' : 0},
            'GDP_GROWTH_PCT' : {'mu' : .02, 
                                'sigma' : .001},                   
            'GDP_GROWTH_PCT_PREDICTORS' : {},
            'TREATMENT_EFFECT' : {'mu': .005,
                                  'sigma' : .0001},
            'TREATMENT_EFFECT_PREDICTORS' : {} 
        }


def sim_data(data, N_YEARS, TREATMENT_YEAR, 
             GDP_GROWTH_PCT, GDP_GROWTH_PCT_PREDICTORS,
             TREATMENT_EFFECT, TREATMENT_EFFECT_PREDICTORS):

    #determine a treatment date, either the same for all (if sigma = 0), or staggered (if sigma != 0)    
    data.loc[:,'treatment_year'] = \
        int(random.gauss(mu = TREATMENT_YEAR['mu'],
                         sigma = TREATMENT_YEAR['sigma']))

    for i_Year in range(1, N_YEARS + 1): # I think due to the fact that this is a time-series, there isn't a way to vectorize...
        dummy_treated = (data['treatment_year'] < i_Year).apply(int)
        treatment_effects = np.random.normal(TREATMENT_EFFECT['mu'],
                                             TREATMENT_EFFECT['sigma'],
                                             50)

        if len(TREATMENT_EFFECT_PREDICTORS) != 0:
            for cov, delta in TREATMENT_EFFECT_PREDICTORS.items():
                treatment_effects += treatment_effects[cov] * delta

        growth_pcts = np.random.normal(GDP_GROWTH_PCT['mu'],
                                       GDP_GROWTH_PCT['sigma'],
                                       50)
        
        if len(GDP_GROWTH_PCT_PREDICTORS) != 0:
            for cov, delta in GDP_GROWTH_PCT_PREDICTORS.items():
                treatment_effects += treatment_effects[cov] * delta

        growth_pcts = growth_pcts + (treatment_effects * dummy_treated)

        data.loc[:, 'stateGDP_t' + str(i_Year)] = \
            data.loc[:, 'stateGDP_t' + str(i_Year - 1)] * (1 + growth_pcts)
        data.loc[:, 'stateGDPpct_t' + str(i_Year)] = growth_pcts
    
    return data
