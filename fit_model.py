from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import statsmodels.api as sm

def fit_TWFE(data, treatment_years, target, linear_time = False, treat_dummy_type = 'invariant'):
    encoder = OneHotEncoder(handle_unknown='ignore')
    idx_match = data.index
    data = data.reset_index()

    state_encoder = encoder.fit(data[['State']])
    state_array = encoder.transform(data[['State']]).toarray()
    state_labels = state_encoder.get_feature_names(['state'])
    state_sparse = pd.DataFrame(state_array[:,1:],
                                columns=state_labels[1:],
                                index = idx_match)

    time_encoder = encoder.fit(data[['t']])
    time_array = encoder.transform(data[['t']]).toarray()
    time_labels = time_encoder.get_feature_names(['t'])
    time_sparse = pd.DataFrame(time_array[:,1:],
                                columns=time_labels[1:],
                                index = idx_match)
    comp = treatment_years.join(time_sparse).reset_index('t')

    comp['d'] = (comp['t'] >= comp['treatment_year']).apply(int)
    comp = comp.set_index('t',append = True).drop(columns = 'treatment_year')

    staggered = sum(~pd.isna(pd.unique(treatment_years['treatment_year']))) != 1
    baseline = [state_labels[0]]

    if linear_time:
        time_out = data[['t']]
        time_out.index = idx_match
    else:
        time_out = time_sparse
        baseline.append(time_labels[0])

    if treat_dummy_type == 'invariant':
        data_out = pd.concat([state_sparse, time_out, comp.d], axis = 1)
    else: #staggered treatment
        if treat_dummy_type == 'time_variant':
            d_mat = comp.iloc[:,1:-1].apply(lambda x: x*comp.d)
            d_mat = d_mat.loc[:,d_mat.sum(axis = 0)!=0]
            d_mat.columns = d_mat.columns.str.replace('t','d')
        elif treat_dummy_type == 'state_variant':
            d_mat = state_sparse.apply(lambda x: x*comp.d)
            d_mat = d_mat.loc[:,d_mat.sum(axis = 0)!=0]
            d_mat.columns = d_mat.columns.str.replace('state','d')
        data_out = pd.concat([state_sparse, time_out, d_mat], axis = 1)

    target = data.loc[:,target]
    target.index = idx_match
    X = sm.add_constant(data_out)
    lm_FixedEffect = sm.OLS(target, X)
    results = lm_FixedEffect.fit()

    return results, data_out, baseline
    # return data_out
    
