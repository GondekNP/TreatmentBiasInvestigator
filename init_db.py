import sqlite3
import pandas as pd

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
        (sim_id INTEGER PRIMARY KEY AUTOINCREMENT,
        treatment_year_mu INTEGER,
        treatment_year_sigma INTEGER,
        exog_growth_sigma REAL,
        exog_control_explained_sigma REAL,
        treatment_effect_mu REAL,
        treatment_effect_sigma REAL,
        spatial_autocorrelation_pct REAL,
        regional_growth_predictors VARCHAR(255))
        """)
        connection.commit()

    if 'beta' not in tabs:
        cursor.execute("""
        CREATE TABLE beta 
        (sim_id VARCHAR(255),
        n_steps INTEGER, 
        parameter VARCHAR(255),
        estimate REAL,
        std REAL,
        FOREIGN KEY(sim_id) REFERENCES sim_results(sim_id))
        """)
        connection.commit()

    connection.close()

def sql_to_pd(db_path, tab_name):
    '''
    Reads in sql table from a database as pd.DataFrame.
    
    Inputs:
        db_path: (str) path to sql database
        tab_name: (str) name of desired table

    Output:
        (pd.DataFrame) of desired table
    '''
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    command = f'''
    SELECT * FROM {tab_name};'''
    cursor.execute(command)
    table = cursor.fetchall()
    header = []
    for tup in cursor.description:
        col = tup[0]
        if "." in col:
            col = col[col.find(".")+1:]
        header.append(col)
    df = pd.DataFrame(table)
    connection.close()

    if len(df) > 0:
        df.columns = header
        return df
    else:
        return pd.DataFrame(columns = header)