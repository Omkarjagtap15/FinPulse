"""
SQLite Database Layer for NatWest FinPulse

This module acts as the SQL/database layer for the project, replacing
raw CSV reads with SQL queries to demonstrate SQL skills. It provides
functions to initialize an SQLite database from raw CSVs and a series
of query functions that return pandas DataFrames.
"""

import os
import sqlite3
import pandas as pd

def init_database(data_dir='data', db_path='data/finpulse.db') -> str:
    """
    Initialize the SQLite database by loading CSV data into tables.

    Parameters
    ----------
    data_dir : str, optional
        The directory containing the CSV files. Default is 'data'.
    db_path : str, optional
        The path where the SQLite database will be created. Default is 'data/finpulse.db'.

    Returns
    -------
    str
        The path to the created SQLite database.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    csv_files = [
        'population.csv', 'population_meta.csv', 'segment_summary.csv',
        'forecasts.csv', 'segment_forecasts.csv', 'anomalies.csv', 'forecast_meta.csv'
    ]
    
    print(f"Initializing database at {db_path}...")
    
    with sqlite3.connect(db_path) as conn:
        for file in csv_files:
            table_name = file.split('.')[0]
            file_path = os.path.join(data_dir, file)
            
            if os.path.exists(file_path):
                print(f"Loading {file} into SQLite (table '{table_name}')...")
                df = pd.read_csv(file_path)
                df.to_sql(table_name, conn, if_exists='replace', index=False)
            else:
                print(f"Warning: {file_path} not found. Skipping {table_name}.")
                
        print("Creating indexes...")
        
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_population_customer_date ON population(customer_id, date)")
            print("  ✓ Created index on population(customer_id, date)")
        except sqlite3.OperationalError:
            pass
            
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_customer_date ON forecasts(customer_id, date)")
            print("  ✓ Created index on forecasts(customer_id, date)")
        except sqlite3.OperationalError:
            pass
            
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_customer ON anomalies(customer_id)")
            print("  ✓ Created index on anomalies(customer_id)")
        except sqlite3.OperationalError:
            pass
            
    print("Database initialization complete.")
    return db_path


def get_connection(db_path='data/finpulse.db') -> sqlite3.Connection:
    """
    Get a connection to the SQLite database.

    Parameters
    ----------
    db_path : str, optional
        The path to the SQLite database. Default is 'data/finpulse.db'.

    Returns
    -------
    sqlite3.Connection
        A connection object to the database with sqlite3.Row factory.
        
    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}. Please initialize the database first.")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def query_customers_at_risk(db_path='data/finpulse.db') -> pd.DataFrame:
    """
    Query customers at risk of overdraft based on forecast metadata.

    Parameters
    ----------
    db_path : str, optional
        The path to the SQLite database. Default is 'data/finpulse.db'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing customers at risk.
    """
    query = '''
        SELECT customer_id, segment, overdraft_days, min_forecast_balance 
        FROM forecast_meta 
        WHERE overdraft_days > 0 
        ORDER BY overdraft_days DESC
    '''
    
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn)


def query_segment_risk_summary(db_path='data/finpulse.db') -> pd.DataFrame:
    """
    Query risk summary statistics by segment.

    Parameters
    ----------
    db_path : str, optional
        The path to the SQLite database. Default is 'data/finpulse.db'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing segment risk summary.
    """
    query = '''
        SELECT segment, 
               COUNT(*) as customer_count, 
               AVG(avg_fhs) as avg_des, 
               SUM(CASE WHEN risk_tier = 'High' THEN 1 ELSE 0 END) as high_risk_count, 
               SUM(CASE WHEN risk_tier = 'Low' THEN 1 ELSE 0 END) as low_risk_count 
        FROM population_meta 
        GROUP BY segment 
        ORDER BY avg_des ASC
    '''
    
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn)


def query_critical_anomalies(db_path='data/finpulse.db', limit=20) -> pd.DataFrame:
    """
    Query critical anomalies joined with population metadata.

    Parameters
    ----------
    db_path : str, optional
        The path to the SQLite database. Default is 'data/finpulse.db'.
    limit : int, optional
        The maximum number of rows to return. Default is 20.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing critical anomalies.
    """
    query = f'''
        SELECT a.customer_id, a.date, a.segment, a.actual_balance, a.yhat, 
               a.yhat_lower, a.anomaly_severity, pm.risk_tier 
        FROM anomalies a 
        LEFT JOIN population_meta pm ON a.customer_id = pm.customer_id 
        WHERE a.anomaly_severity = 'CRITICAL_EWS' 
        ORDER BY (a.yhat_lower - a.actual_balance) DESC 
        LIMIT {limit}
    '''
    
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn)


def query_customer_balance_trend(customer_id: str, db_path='data/finpulse.db') -> pd.DataFrame:
    """
    Query balance trend for a specific customer.

    Parameters
    ----------
    customer_id : str
        The ID of the customer to query.
    db_path : str, optional
        The path to the SQLite database. Default is 'data/finpulse.db'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the customer's balance trend.
    """
    query = '''
        SELECT date, balance, fhs, liquidity_runway, spend_velocity_ratio 
        FROM population 
        WHERE customer_id = ? 
        ORDER BY date
    '''
    
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(customer_id,))


def query_segment_exposure_by_week(db_path='data/finpulse.db') -> pd.DataFrame:
    """
    Query segment exposure statistics aggregated by week.

    Parameters
    ----------
    db_path : str, optional
        The path to the SQLite database. Default is 'data/finpulse.db'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing segment exposure by week.
    """
    query = '''
        SELECT segment, forecast_week, 
               AVG(liquidity_exposure_pct) as avg_exposure, 
               AVG(stress_exposure_pct) as avg_stress_exposure, 
               AVG(median_fhs) as avg_des 
        FROM segment_forecasts 
        GROUP BY segment, forecast_week 
        ORDER BY segment, forecast_week
    '''
    
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn)


def query_high_velocity_customers(threshold=1.3, db_path='data/finpulse.db') -> pd.DataFrame:
    """
    Query customers with high spend velocity.

    Parameters
    ----------
    threshold : float, optional
        The minimum spend velocity ratio. Default is 1.3.
    db_path : str, optional
        The path to the SQLite database. Default is 'data/finpulse.db'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing high velocity customers.
    """
    query = '''
        SELECT p.customer_id, p.segment, p.date, p.spend_velocity_ratio, p.balance, p.fhs 
        FROM population p 
        WHERE p.spend_velocity_ratio > ? 
        ORDER BY p.spend_velocity_ratio DESC 
        LIMIT 50
    '''
    
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(threshold,))


if __name__ == '__main__':
    print("Running NatWest FinPulse Database Layer Demo...")
    db_path = 'data/finpulse.db'
    data_dir = 'data'
    
    if os.path.exists(data_dir):
        init_database(data_dir=data_dir, db_path=db_path)
    
        print("\nDemo: Querying Risk Summary")
        try:
            df = query_segment_risk_summary(db_path=db_path)
            print(df.head())
        except Exception as e:
            print(f"Error querying db: {e}")
    else:
        print(f"Data directory '{data_dir}' not found. Please run data generation scripts first.")
