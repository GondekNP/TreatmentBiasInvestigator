import numpy as np
import random
import pandas as pd

sim_params =  {
            'TARGET' : 'GDPcont_t0',
            'SPATIAL_AUTOCORRELATION_PCT' : 0,
            'N_YEARS' : 10,
            'N_TREATED' : 5,
            'N_STATES' : 10,
            'TREATMENT_YEAR' : {'mu' : 5, 
                                'sigma' : 0},
            'EXOG_GROWTH_PCT' : {'mu' : .02, 
                                'sigma' : .01},
            'EXOG_CONTROL_PCT_EXPLAINED': {'sigma' : .3},
            'SECTOR_GROWTH_PCT' : {'sigma' : 0},                   
            'SECTOR_GROWTH_PCT_PREDICTORS' : {},
            'TREATMENT_EFFECT' : {'mu': .3,
                                  'sigma' : 0.002},
            'TREATMENT_EFFECT_PREDICTORS' : {} 
        }


def sim_data(data, TARGET, N_YEARS, N_TREATED, N_STATES, TREATMENT_YEAR, 
             EXOG_GROWTH_PCT, EXOG_CONTROL_PCT_EXPLAINED,
             SPATIAL_AUTOCORRELATION_PCT,
             SECTOR_GROWTH_PCT, SECTOR_GROWTH_PCT_PREDICTORS,
             TREATMENT_EFFECT, TREATMENT_EFFECT_PREDICTORS):

    data = data.sample(n = N_STATES).sort_values(TARGET, ascending=False)
    # states = data.reset_index().State

    treated = np.array([np.repeat([np.nan], N_STATES - N_TREATED),
                        np.repeat([1],      N_TREATED)]).flatten()
    np.random.shuffle(treated)

    #determine a treatment date, either the same for all (if sigma = 0), or staggered (if sigma != 0)    
    data.loc[:,'treatment_year'] = np.random.normal(TREATMENT_YEAR['mu'],
                                                    TREATMENT_YEAR['sigma'],
                                                    N_STATES) 
    data.loc[:,'treatment_year'] = data.loc[:,'treatment_year'].astype(int) * treated

    treatment_effects = np.random.normal(TREATMENT_EFFECT['mu'],
                                             TREATMENT_EFFECT['sigma'],
                                             N_STATES)
        
    if len(TREATMENT_EFFECT_PREDICTORS) != 0:
        for cov, delta in TREATMENT_EFFECT_PREDICTORS.items():
            treatment_effects += treatment_effects[cov] * delta

    data.loc[:,'treatment_effect'] = treatment_effects

    df_treat = data[['treatment_year','treatment_effect']]
    df_out = pd.DataFrame()

    for i_Year in range(0, N_YEARS + 1): # I think due to the fact that this is a time-series, there isn't a way to vectorize...
        df_long = pd.DataFrame()
        dummy_treated_t = (i_Year == data['treatment_year']).apply(int)

        # this growth percentage is the mean of each individual state's growth percentage
        # can be thought of as the exogenous component to state level growth - ie, condition
        # of the global economy during year i
        i_Year_baseGrowth = np.random.normal(EXOG_GROWTH_PCT['mu'],
                                             EXOG_GROWTH_PCT['sigma'],
                                             1)

        # this growth percent varies by state, but drawn from the same distribution centered
        # at the exogenous growth percentage drawn above
        growth_pcts = pd.Series(np.random.normal(i_Year_baseGrowth,
                                           SECTOR_GROWTH_PCT['sigma'],
                                           N_STATES), index=data.index)
        if SPATIAL_AUTOCORRELATION_PCT != 0:
            # if autocorrelated, take the random growth percents generated, and 
            # make them all closer to their mean by region. If autocorrelation is 
            # very high, ie 1, then all states in the region will have the same rate. 
            # if 0, all states are indepedent and region has no effect.
            autocor = data.loc[:,['Region']]
            autocor['growth_pcts'] = growth_pcts
            bunched = autocor.merge(autocor.groupby('Region').mean().reset_index(),
                                on = 'Region',
                                suffixes=('_state','_region'),
                                ).set_index(autocor.index)
            growth_pcts = \
                bunched.loc[:,'growth_pcts_state'] * (1 - SPATIAL_AUTOCORRELATION_PCT) + \
                bunched.loc[:,'growth_pcts_region'] * SPATIAL_AUTOCORRELATION_PCT
        
        # This is an index for endogenous variables that help us explain the growth rate
        # for a particular state - if the sigma here is 0, this means that we can explain
        # state level growth entirely since controls == growth pct
        pct_explained = np.random.normal(1,
                            EXOG_CONTROL_PCT_EXPLAINED['sigma'],
                            N_STATES)
        df_long.loc[:,'stateControls_t'] = growth_pcts * pct_explained

        if len(SECTOR_GROWTH_PCT_PREDICTORS) != 0:
            for cov, delta in SECTOR_GROWTH_PCT_PREDICTORS.items():
                treatment_effects += data[treatment_effects[cov]] * delta

        if i_Year > 0:
            growth_pcts_treated = growth_pcts + (treatment_effects * dummy_treated_t)
            new_GDPcont = last_year_GDPcont + growth_pcts_treated
            df_long.loc[:, 'GDPcont_t'] = new_GDPcont
            last_year_GDPcont = new_GDPcont
        else:
            df_long.loc[:, 'GDPcont_t'] = data.loc[:,'GDPcont_t0']
            last_year_GDPcont = data.loc[:,'GDPcont_t0']
        
        df_long.loc[:,'t'] = i_Year
        df_out = pd.concat([df_out, df_long])

    df_out.set_index('t', append=True, inplace=True)
    return df_out, df_treat