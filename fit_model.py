from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import statsmodels.api as sm
import numpy as np
from numba import jit

# @jit
def fit_TWFE(sim_mat, init_mat, linear_time = False, treat_dummy_type = 'invariant'):
    # encoder = OneHotEncoder(handle_unknown='ignore')
    # idx_match = data.index
    # data = data.reset_index()



    # state_encoder = encoder.fit(data[['State']])
    # state_array = encoder.transform(data[['State']]).toarray()
    # state_labels = state_encoder.get_feature_names(['state'])
    # state_sparse = pd.DataFrame(state_array[:,1:],
    #                             columns=state_labels[1:],
    #                             index = idx_match)

    state_sparse = np.eye(52)[sim_mat[:,0].astype(int)]
    included = np.where(state_sparse.sum(axis = 0) != 0)[0]
    N_STATES = len(included)
    state_sparse = state_sparse[:,included]
    state_labels = np.core.defchararray.add(
                        np.repeat('state_', N_STATES),
                        init_mat[:,0].astype(int).astype(str)) 

    # time_encoder = encoder.fit(data[['t']])
    # time_array = encoder.transform(data[['t']]).toarray()
    # time_labels = time_encoder.get_feature_names(['t'])
    # time_sparse = pd.DataFrame(time_array[:,1:],
    #                             columns=time_labels[1:],
    #                             index = idx_match)
    N_STEPS = sim_mat[:,1].max().astype(int)
    time_labels = np.core.defchararray.add(
                                np.repeat('t_', N_STEPS + 1),
                                np.arange(0, N_STEPS + 1).astype(str))                              
    time_sparse = np.eye(N_STEPS + 1)[sim_mat[:,1].astype(int)]
    n_row = time_sparse.shape[0]

    t_compare = time_sparse * np.arange(1, N_STATES + 2) # offset by 1 so need a plus 2
    d_compare = np.tile(init_mat[:,4], N_STATES + 1).reshape((n_row,1))
    comp = (t_compare > d_compare).astype(np.int)
    comp_invariant = comp.sum(axis = 1).reshape((n_row,1))

    # staggered = sum(~pd.isna(pd.unique(treatment_years['treatment_year']))) != 1
    # baseline = [state_labels[0]]

    if linear_time:
        time_out = sim_mat[:,1]
    else:
        time_out = time_sparse

    one_pad = np.ones((n_row, 1))
    if treat_dummy_type == 'invariant':
        data_out = np.concatenate((one_pad, time_out, state_sparse, comp_invariant), axis = 1)
        treat_labels = np.array(['d'])
    else: #staggered treatment
        if treat_dummy_type == 'time_variant':
            to_include = np.where(comp.sum(axis = 0) != 0)[0]
            comp = comp[:, to_include]
            treat_labels = np.core.defchararray.add(
                            np.repeat('d_t', len(to_include)),
                            to_include.astype(str))  
            # d_mat = comp.iloc[:,1:-1].apply(lambda x: x*comp.d)
            # d_mat = d_mat.loc[:,d_mat.sum(axis = 0)!=0]
            # d_mat.columns = d_mat.columns.str.replace('t','d')
            data_out = np.concatenate((one_pad, time_out, state_sparse, comp), axis = 1)
        elif treat_dummy_type == 'state_variant':
            # d_mat = state_sparse.apply(lambda x: x*comp.d)
            # d_mat = d_mat.loc[:,d_mat.sum(axis = 0)!=0]
            # d_mat.columns = d_mat.columns.str.replace('state','d')
            comp_statevar = state_sparse * comp_invariant
            to_include = np.where(comp_statevar.sum(axis = 0) != 0)[0]
            comp_statevar = comp_statevar[:, to_include]
            treat_labels = np.core.defchararray.add(
                            np.repeat('d_state', len(to_include)),
                            init_mat[to_include,0].astype(int).astype(str))  
            data_out = np.concatenate((one_pad, time_out, state_sparse, comp_statevar), axis = 1)

    data_labels = np.concatenate((np.array(['const']), time_labels, state_labels, treat_labels))
    # target = sim_mat[:,3] #GDPcont column
    # X = np.concatenate(np.ones((data.shape[0], 1)) 
    # X = sm.add_constant(data_out)
    # lm_FixedEffect = sm.OLS(target, X)
    # results = lm_FixedEffect.fit()

    # return results, data_out, baseline
    return data_out, data_labels
    
