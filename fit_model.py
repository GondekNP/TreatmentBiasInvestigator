import numpy as np
from numba.pycc import CC
import statsmodels.api as sm

def preprocess_data(sim_mat, init_mat, linear_time = False, treat_dummy_type = 0):
    '''
    0 = invariant treatment effet
    1 = time variant treatment effect
    2 = state variant treatment effect
    '''
    state_sparse = np.eye(52)[sim_mat[:,0].astype(np.int32)]
    included = np.where(state_sparse.sum(axis = 0) != 0)[0]
    N_STATES = len(included)
    state_sparse = state_sparse[:,included]
    state_labels = np.char.add(
                        np.repeat('state_', N_STATES),
                        init_mat[:,0].astype(np.int32).astype(str)) 
    state_labels = state_labels[1:]

    N_STEPS = int(sim_mat[:,1].max())
    time_labels = np.char.add(
                                np.repeat('t_', N_STEPS + 1),
                                np.arange(0, N_STEPS + 1).astype(str))                              
    time_sparse = np.eye(N_STEPS + 1)[sim_mat[:,1].astype(np.int32)]
    n_row = time_sparse.shape[0]

    t_compare = time_sparse * np.arange(1, N_STEPS + 2) # offset by 1 so need a plus 2

    d_compare = np.tile(init_mat[:,4], N_STEPS + 1).reshape((n_row,1))
    comp = (t_compare > d_compare).astype(np.int32)
    comp_invariant = comp.sum(axis = 1).reshape((n_row,1))

    if linear_time:
        time_out = sim_mat[:,1].reshape((n_row,1))
        time_labels = np.array(['t'])
    else:
        time_out = time_sparse[:,1:]
        time_labels = time_labels[1:]
        # baseline = np.append(baseline, time_sparse[0])

    one_pad = np.ones((n_row, 1))
    if treat_dummy_type == 0:
        data_out = np.concatenate((one_pad, sim_mat[:,[2]], time_out, state_sparse[:,1:], comp_invariant), axis = 1)
        treat_labels = np.array(['d'])

    else: 
        if treat_dummy_type == 1:
            to_include = np.where(comp.sum(axis = 0) != 0)[0]
            comp = comp[:, to_include]
            treat_labels = np.char.add(
                            np.repeat('d_t', len(to_include)),
                            to_include.astype(str))  
            data_out = np.concatenate((one_pad, sim_mat[:,[2]], time_out, state_sparse[:,1:], comp), axis = 1)
        elif treat_dummy_type == 2:
            comp_statevar = state_sparse * comp_invariant
            to_include = np.where(comp_statevar.sum(axis = 0) != 0)[0]
            comp_statevar = comp_statevar[:, to_include]
            treat_labels = np.char.add(
                            np.repeat('d_state', len(to_include)),
                            init_mat[to_include,0].astype(np.int32).astype(str))  
            data_out = np.concatenate((one_pad, sim_mat[:,[2]], time_out, state_sparse[:,1:], comp_statevar), axis = 1)

    data_labels = np.concatenate((np.array(['const', 'state_controls']),
                                  time_labels,
                                  state_labels,
                                  treat_labels))
    target = np.ascontiguousarray(sim_mat[:,3]) #GDPcont column
    return data_out, treat_labels, target
# cc.compile()

cc = CC('compute_OLS')
@cc.export('compute_OLS', 'f8[:](f8[:,:], f8[:], i4)')
def compute_OLS(X, Y, treat_idx = 0):
    XTX = X.T @ X
    est = np.linalg.inv(XTX) @ X.T @ Y
    if treat_idx == 0:
        return est
    else:
        return est[-treat_idx:]
cc.compile()

def compute_sm_OLS(X, Y, treat_idx = 0):
    lm_FixedEffect = sm.OLS(Y, X)
    results = lm_FixedEffect.fit()
    return results.params[-1]