from pandas.io.parsers import read_csv
# from mpi4py import MPI
import numpy as np
import pandas as pd
import sqlite3

import fit_model
import init_db
import init_sim
import run_sim
import compute_OLS

meta_sim_params = {
    'N_SIMS_PER_COMBO_THREAD': 10, #Embarassingly Parallel approach
    'EXOG_GROWTH_MU' : .03,
    'N_STATES' : 51,
    'N_TREATED' : 25,
    'N_STEPS' : np.arange(100,25,-10), 
    'TREATMENT_YEAR_MU' : 15, #will not change
    'TREATMENT_YEAR_SIGMA' : np.array([0.0, 1.0, 2.5, 5]), #4 sigmas
    'EXOG_GROWTH_SIGMA' : np.arange(0,.033,.003), #10 sigmas
    'EXOG_CONTROL_EXPLAINED_SIGMA' : np.arange(0,.66,.11),
    'SECTOR_GROWTH_SIGMA' : np.arange(0,.033,.009),
    'TREATMENT_EFFECT_MU' : np.arange(0.01,.1, .01), #9 means, from 1/3 to 3x the average year exog growth
    'TREATMENT_EFFECT_SIGMA' : np.arange(0.01,.055, .005),
    'SPATIAL_AUTOCORRELATION_PCT' : np.array([0]),
    # 'SPATIAL_AUTOCORRELATION_PCT' : np.arange(0,.8,.1),
    'REGIONAL_GROWTH_PREDICTORS' : {'none' : np.array([0,0,0,0,0,0,0,0])}
    # 'REGIONAL_GROWTH_PREDICTORS' : { 'strong' : np.array([2,7,3,5,4,6,1,0]) / 800, #strong parallel trend violation
    #                                  'weak'  : np.array([2,7,3,5,4,6,1,0]) / 4000, #weaker
    #                                  'none' : np.array([0,0,0,0,0,0,0,0])} #no violation
    } 

def TWFE_sim(output_db_path, N_SIMS_PER_COMBO_THREAD, EXOG_GROWTH_MU, N_STEPS, N_STATES,
                 N_TREATED, TREATMENT_YEAR_MU, SECTOR_GROWTH_SIGMA,
                 TREATMENT_YEAR_SIGMA, EXOG_GROWTH_SIGMA, EXOG_CONTROL_EXPLAINED_SIGMA,
                 TREATMENT_EFFECT_MU, TREATMENT_EFFECT_SIGMA, SPATIAL_AUTOCORRELATION_PCT,
                 REGIONAL_GROWTH_PREDICTORS):
    insert_track = \
    '''INSERT INTO sim_results(treatment_year_mu, treatment_year_sigma, exog_growth_sigma, exog_control_explained_sigma, treatment_effect_mu, treatment_effect_sigma, spatial_autocorrelation_pct, regional_growth_predictors)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);'''
 
    insert_beta = \
    '''INSERT INTO beta(sim_id, n_steps, parameter, estimate)
        VALUES (?, ?, ?, ?);'''

    init_db.init_db(output_db_path)
    state_mat = pd.read_csv('location_sim_base.csv').to_numpy()
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank() 
    rank = 2
    connection = sqlite3.connect(output_db_path)
    cursor = connection.cursor()

    for _ in range(N_SIMS_PER_COMBO_THREAD):
        for treatment_year_sigma in TREATMENT_YEAR_SIGMA:
            for exog_growth_sigma in EXOG_GROWTH_SIGMA:
                for exog_control_explained_sigma in EXOG_CONTROL_EXPLAINED_SIGMA:
                    for treatment_effect_mu in TREATMENT_EFFECT_MU:
                        for treatment_effect_sigma in TREATMENT_EFFECT_SIGMA:
                            for spatial_autocorrelation_pct in SPATIAL_AUTOCORRELATION_PCT:
                                for sector_growth_sigma in SECTOR_GROWTH_SIGMA:
                                    for rgv_label, region_growth_vector in REGIONAL_GROWTH_PREDICTORS.items():
                                        
                                        init_mat = init_sim.init_sim(
                                            state_mat, N_STATES, N_TREATED, TREATMENT_YEAR_MU, #common across all sims
                                            treatment_year_sigma, treatment_effect_mu, treatment_effect_sigma)
                                        # print('init_mat')
                                        max_n_steps = N_STEPS.max()
                                        
                                        # for item in [init_mat, region_growth_vector, max_n_steps,
                                        #     EXOG_GROWTH_MU, exog_growth_sigma,
                                        #     exog_control_explained_sigma,
                                        #     sector_growth_sigma,
                                        #     spatial_autocorrelation_pct]:
                                        #     print(type(item))
                                        # print(init_mat)
                                        # print(region_growth_vector)

                                        sim_mat = run_sim.run_sim(
                                            init_mat, region_growth_vector, max_n_steps,
                                            EXOG_GROWTH_MU, exog_growth_sigma,
                                            exog_control_explained_sigma,
                                            sector_growth_sigma,
                                            spatial_autocorrelation_pct)
                                        # print('sim_mat')

                                        # hash_str = str(rank) + str(time.time())
                                        # sim_hash = str(hashlib.md5(hash_str.encode()))
                                        cursor.execute(insert_track,
                                            [TREATMENT_YEAR_MU, treatment_year_sigma,
                                            exog_growth_sigma, exog_control_explained_sigma,
                                            treatment_effect_mu, treatment_effect_sigma,
                                            spatial_autocorrelation_pct, rgv_label])

                                        pk = cursor.execute('SELECT last_insert_rowid();').fetchone()[0]
                                        connection.commit()

                                        for treat_dummy_idx in [0]:
                                        # for treat_dummy_idx in [0,1,2]:

                                            for survey_period in N_STEPS:
                                                max_row_idx = N_STATES * survey_period
                                                step_sim = sim_mat[:max_row_idx,:]
                                                data_out, treat_labels, target = fit_model.preprocess_data(
                                                    step_sim, init_mat, 
                                                    treat_dummy_type=treat_dummy_idx)
                                                treat_label_idx = len(treat_labels)
                                                # print('proc_dat')

                                                weights = compute_OLS.compute_OLS(data_out, target, treat_label_idx)
                                                for i, weight in enumerate(weights):
                                                    cursor.execute(insert_beta,
                                                        [pk, int(survey_period),
                                                        treat_labels[i], weight])
                                                # if len(weights) == 1:
                                                #     print(weights)
                                                #     print(type(weights))
                                                #     cursor.execute(insert_beta,
                                                #     [pk, int(survey_period),
                                                #     treat_labels[0], weights])
                                                # else:
                                                #     print(weights)
                                                #     print(type(weights))
                                                #     for i, weight in enumerate(weights):
                                                #         cursor.execute(insert_beta,
                                                #         [pk, int(survey_period),
                                                #         treat_labels[i], weight])
                                                connection.commit()    
    connection.close()

def main():
    TWFE_sim('results.sql', **meta_sim_params)

if __name__ == '__main__':
    main()