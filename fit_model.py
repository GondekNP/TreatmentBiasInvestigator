from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def preprocess_dat(data, treatment_years, target):
    encoder = OneHotEncoder(handle_unknown='ignore')
    
    state_encoder = encoder.fit(data[['State']])
    state_array = encoder.transform(data[['State']]).toarray()
    state_labels = state_encoder.get_feature_names(['state'])
    state_sparse = pd.DataFrame(state_array[:,1:],
                                columns=state_labels[1:])

    time_encoder = encoder.fit(data[['t']])
    time_array = encoder.transform(data[['t']]).toarray()
    time_labels = time_encoder.get_feature_names(['t'])
    time_sparse = pd.DataFrame(time_array[:,1:],
                                columns=time_labels[1:])

    comp = data[['State', 't']].merge(treatment_years)   
    treated_indicator = (comp['t'] > comp['treatment_year']).apply(int)
    treated_indicator.name = 'treated'

    target = data.loc[:,target]
    data_out = pd.concat([state_sparse, time_sparse, treated_indicator], axis = 1)
    baseline = (state_labels[0], time_labels[0])

    return data_out, target, baseline