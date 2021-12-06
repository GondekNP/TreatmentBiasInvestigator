from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import statsmodels.api as sm

def preprocess_dat(data, treatment_years, target, treat_dummy_type = 'invariant'):
    encoder = OneHotEncoder(handle_unknown='ignore')
    id_match = data.set_index(['State','t']).index

    state_encoder = encoder.fit(data[['State']])
    state_array = encoder.transform(data[['State']]).toarray()
    state_labels = state_encoder.get_feature_names(['state'])
    state_sparse = pd.DataFrame(state_array,
                                columns=state_labels,
                                index = id_match)

    time_encoder = encoder.fit(data[['t']])
    time_array = encoder.transform(data[['t']]).toarray()
    time_labels = time_encoder.get_feature_names(['t'])
    time_sparse = pd.DataFrame(time_array,
                                columns=time_labels,
                                index = id_match)

    comp = treatment_years.join(time_sparse).reset_index('t')
    comp['d'] = (comp['t'] > comp['treatment_year']).apply(int)
    comp = comp.set_index('t',append = True).drop(columns = 'treatment_year')
    
    staggered = sum(~pd.isna(pd.unique(treatment_years['treatment_year']))) != 1

    if treat_dummy_type == 'invariant' or not staggered:
        data_out = pd.concat([state_sparse, time_sparse, comp.d], axis = 1)
    else: #staggered treatment
        if treat_dummy_type == 'time_variant':
            d_mat = comp.iloc[:,:-1].apply(lambda x: x*comp.d)
            d_mat = d_mat.loc[:,d_mat.sum(axis = 0)!=0]
            d_mat.columns = d_mat.columns.str.replace('t','d')
        elif treat_dummy_type == 'state_variant':
            d_mat = state_sparse.apply(lambda x: x*comp.d)
            d_mat = d_mat.loc[:,d_mat.sum(axis = 0)!=0]
            d_mat.columns = d_mat.columns.str.replace('state','d')
        data_out = pd.concat([state_sparse, time_sparse, d_mat], axis = 1)

    target = data.set_index(['State','t']).loc[:,target]
    X = sm.add_constant(data_out)
    lm_FixedEffect = sm.OLS(target, X)
    results = lm_FixedEffect.fit()
    baseline = (state_labels[0], time_labels[0])

    return results, data_out, baseline
    # return data_out
    
