import sqlite3

def init_db(output_db_path):
    '''
    Creates contact database if it does not exist.

    Inputs:
        output_db_path: (str) path to contact sql
    '''
    connection = sqlite3.connect(output_db_path)
    cursor = connection.cursor()
    tabs = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tabs = [tab[0] for (tab) in tabs]

    if 'sim_results' not in tabs:
        cursor.execute("""
        CREATE TABLE sim_results 
        (sim_hash VARCHAR(255) PRIMARY KEY,
        treatment_year_mu REAL,
        treatment_year_sigma REAL,
        exog_growth_sigma REAL,
        exog_control_explained_sigma REAL,
        treatment_effect_mu REAL,
        treatment_effect_sigma REAL,
        spatial_autocorrelation_pct REAL,
        regional_growth_predictors VARCHAR(255),
        comp_time REAL)
        """)
        connection.commit()

    if 'beta' not in tabs:
        cursor.execute("""
        CREATE TABLE beta 
        (sim_hash VARCHAR(255),
        n_steps INTEGER, 
        parameter VARCHAR(255),
        estimate REAL,
        std REAL,
        FOREIGN KEY(sim_hash) REFERENCES sim_results(sim_hash))
        """)
        connection.commit()

    connection.close()
