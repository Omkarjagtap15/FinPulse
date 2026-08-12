"""
data_loader.py — Data Loading & Caching
=========================================
Loads all 7 CSV files from the data/ directory.
Uses Streamlit's @st.cache_data to avoid re-reading on every interaction.
"""
import os
import pandas as pd
import streamlit as st

import sqlite3

DATA_DIR = "data"

@st.cache_data(show_spinner="Loading data from SQLite Database…")
def load_all_data():
    """
    Load data from the SQLite database. If the database is missing,
    fallback to loading raw CSVs.
    """
    db_path = os.path.join(DATA_DIR, "finpulse.db")
    
    # --- SQLITE DATABASE LOADING ---
    if os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                return {
                    "population":    pd.read_sql("SELECT * FROM population", conn, parse_dates=["date"]),
                    "meta":          pd.read_sql("SELECT * FROM population_meta", conn),
                    "segment_hist":  pd.read_sql("SELECT * FROM segment_summary", conn, parse_dates=["date"]),
                    "forecasts":     pd.read_sql("SELECT * FROM forecasts", conn, parse_dates=["date"]),
                    "segment_fore":  pd.read_sql("SELECT * FROM segment_forecasts", conn, parse_dates=["date"]),
                    "anomalies":     pd.read_sql("SELECT * FROM anomalies", conn, parse_dates=["date"]),
                    "forecast_meta": pd.read_sql("SELECT * FROM forecast_meta", conn),
                }
        except Exception as e:
            st.error(f"Database error, falling back to CSVs: {e}")
            
    # --- CSV FALLBACK ---
    def _read(filename, **kwargs):
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            return pd.read_csv(path, **kwargs)
        return pd.DataFrame()

    return {
        "population":    _read("population.csv",        parse_dates=["date"]),
        "meta":          _read("population_meta.csv"),
        "segment_hist":  _read("segment_summary.csv",   parse_dates=["date"]),
        "forecasts":     _read("forecasts.csv",          parse_dates=["date"]),
        "segment_fore":  _read("segment_forecasts.csv",  parse_dates=["date"]),
        "anomalies":     _read("anomalies.csv",          parse_dates=["date"]),
        "forecast_meta": _read("forecast_meta.csv"),
    }
