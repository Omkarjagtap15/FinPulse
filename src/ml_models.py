"""
Machine Learning & Statistical Core for NatWest FinPulse

This module contains the core feature engineering, predictive modeling,
forecasting, and anomaly detection algorithms for the banking liquidity
risk dashboard. It includes classification models, clustering, time-series
forecasting, and statistical tests.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import f_oneway, chi2_contingency, ttest_ind, pearsonr, spearmanr


def engineer_features(pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer aggregate features from daily population transaction history.
    
    Parameters
    ----------
    pop_df : pd.DataFrame
        DataFrame containing columns: customer_id, date, segment, balance,
        fhs, liquidity_runway, spend_velocity_ratio
        
    Returns
    -------
    pd.DataFrame
        DataFrame with aggregated features per customer_id.
    """
    if pop_df.empty:
        return pd.DataFrame()
        
    def calculate_trend(y):
        x = np.arange(len(y))
        if len(y) > 1:
            return np.polyfit(x, y, 1)[0]
        return 0.0

    aggs = {
        'balance': ['mean', 'std', 'min', 'max', 'last', calculate_trend],
        'fhs': ['mean', 'std', 'last'],
        'liquidity_runway': ['mean', 'last'],
        'spend_velocity_ratio': ['mean', 'max'],
        'segment': ['first']
    }
    
    grouped = pop_df.groupby('customer_id').agg(aggs)
    
    # Flatten multi-level columns
    grouped.columns = [f"{col[0]}_{col[1]}" if col[1] != 'calculate_trend' else f"{col[0]}_trend" for col in grouped.columns]
    
    # Rename columns to match requirements
    rename_map = {
        'balance_mean': 'bal_mean',
        'balance_std': 'bal_std',
        'balance_min': 'bal_min',
        'balance_max': 'bal_max',
        'balance_last': 'bal_last',
        'balance_trend': 'bal_trend',
        'fhs_mean': 'fhs_mean',
        'fhs_std': 'fhs_std',
        'fhs_last': 'fhs_last',
        'liquidity_runway_mean': 'runway_mean',
        'liquidity_runway_last': 'runway_last',
        'spend_velocity_ratio_mean': 'vel_mean',
        'spend_velocity_ratio_max': 'vel_max',
        'segment_first': 'segment'
    }
    grouped = grouped.rename(columns=rename_map)
    
    # Fill NaN std dev (if only 1 record) with 0
    grouped['bal_std'] = grouped['bal_std'].fillna(0)
    grouped['fhs_std'] = grouped['fhs_std'].fillna(0)
    
    # Calculate Coefficient of Variation (CV)
    grouped['bal_cv'] = np.where(grouped['bal_mean'] != 0, grouped['bal_std'] / np.abs(grouped['bal_mean']), 0)
    
    grouped = grouped.reset_index()
    print(f"  ✓ Feature engineering complete — {grouped.shape[1]-1} features for {len(grouped)} customers")
    
    return grouped


def train_risk_classifier(features_df: pd.DataFrame) -> dict:
    """
    Train classification models to predict customer risk tier.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with aggregated features per customer.
        
    Returns
    -------
    dict
        Dictionary containing trained models, reports, and data splits.
    """
    if features_df.empty:
        return {}
        
    # Create target variable
    def determine_risk(fhs_mean):
        if pd.isna(fhs_mean):
            return 'Medium'
        if fhs_mean < 35:
            return 'High'
        elif fhs_mean < 60:
            return 'Medium'
        else:
            return 'Low'
            
    features_df['risk_tier'] = features_df['fhs_mean'].apply(determine_risk)
    
    feature_cols = ['bal_mean', 'bal_std', 'bal_min', 'bal_max', 'bal_trend', 
                    'bal_cv', 'fhs_mean', 'fhs_std', 'runway_mean', 'vel_mean', 'vel_max']
    
    # Handle missing values in features
    X = features_df[feature_cols].fillna(0)
    y = features_df['risk_tier']
    
    if len(y.unique()) < 2:
        return {} # Need at least 2 classes to train
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train) # Tree based models don't need scaling
    rf_pred = rf.predict(X_test)
    
    rf_importance = dict(zip(feature_cols, rf.feature_importances_))
    
    print(f"  ✓ Risk classification complete — trained on {len(X_train)} samples")
    
    return {
        'lr_model': lr,
        'rf_model': rf,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'lr_report': classification_report(y_test, lr_pred, output_dict=True, zero_division=0),
        'rf_report': classification_report(y_test, rf_pred, output_dict=True, zero_division=0),
        'lr_confusion': confusion_matrix(y_test, lr_pred),
        'rf_confusion': confusion_matrix(y_test, rf_pred),
        'rf_feature_importance': rf_importance,
        'X_test': X_test,
        'y_test': y_test,
        'lr_pred': lr_pred,
        'rf_pred': rf_pred
    }


def train_customer_clusters(features_df: pd.DataFrame, n_clusters: int = 8) -> dict:
    """
    Cluster customers based on financial behavior features.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with aggregated features per customer.
    n_clusters : int, optional
        Number of clusters to form (default is 8).
        
    Returns
    -------
    dict
        Dictionary containing KMeans model, metrics, and labels.
    """
    if features_df.empty or len(features_df) < n_clusters:
        return {}
        
    feature_cols = ['bal_mean', 'bal_std', 'bal_cv', 'fhs_mean', 'runway_mean', 'vel_mean']
    X = features_df[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    
    for k in range(2, min(13, len(X_scaled))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append((k, km.inertia_))
        if len(set(km.labels_)) > 1:
            silhouette_scores.append((k, silhouette_score(X_scaled, km.labels_)))
        else:
            silhouette_scores.append((k, 0))
            
    final_km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = final_km.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
    
    print(f"  ✓ Clustering complete — {n_clusters} clusters formed with silhouette {sil_score:.3f}")
    
    return {
        'model': final_km,
        'scaler': scaler,
        'labels': labels,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'silhouette_final': sil_score,
        'feature_cols': feature_cols
    }


def forecast_segment_balance(segment_history: pd.DataFrame, forecast_days: int = 30) -> dict:
    """
    Forecast median balance for a single segment.
    
    Parameters
    ----------
    segment_history : pd.DataFrame
        Historical median balance for a segment (columns: date, median_balance).
    forecast_days : int, optional
        Number of days to forecast (default 30).
        
    Returns
    -------
    dict
        Dictionary containing forecasts and confidence bounds.
    """
    if segment_history.empty or len(segment_history) < 14:
        return {}
        
    df = segment_history.sort_values('date').copy()
    y = df['median_balance'].values
    
    forecast = None
    residuals = None
    method_used = None
    
    try:
        model = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=7, initialization_method="estimated")
        fit_model = model.fit()
        forecast = fit_model.forecast(forecast_days)
        residuals = fit_model.resid
        method_used = 'holt_winters_seasonal'
    except Exception:
        try:
            model = ExponentialSmoothing(y, trend='add', seasonal=None, initialization_method="estimated")
            fit_model = model.fit()
            forecast = fit_model.forecast(forecast_days)
            residuals = fit_model.resid
            method_used = 'holt_winters_trend'
        except Exception:
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            future_x = np.arange(len(y), len(y) + forecast_days)
            forecast = slope * future_x + intercept
            y_pred = slope * x + intercept
            residuals = y - y_pred
            method_used = 'linear_trend'
            
    residual_std = np.std(residuals)
    margin = 1.28 * residual_std
    
    upper = forecast + margin
    lower = forecast - margin
    
    print(f"  ✓ Forecasting complete using {method_used}")
    
    return {
        'forecast': forecast,
        'upper': upper,
        'lower': lower,
        'residual_std': residual_std,
        'method': method_used
    }


def run_statistical_tests(pop_df: pd.DataFrame, features_df: pd.DataFrame) -> dict:
    """
    Run statistical tests on population and engineered features.
    
    Parameters
    ----------
    pop_df : pd.DataFrame
        Historical transaction data.
    features_df : pd.DataFrame
        Engineered customer features.
        
    Returns
    -------
    dict
        Dictionary containing test statistics and p-values.
    """
    results = {}
    
    # Test 1 - ANOVA: Is mean balance significantly different across segments?
    if not features_df.empty and 'segment' in features_df.columns:
        segments = features_df['segment'].unique()
        arrays = [features_df[features_df['segment'] == s]['bal_mean'].values for s in segments]
        arrays = [arr for arr in arrays if len(arr) > 0]
        if len(arrays) > 1:
            f_stat, p_val = f_oneway(*arrays)
            results['anova'] = {'f_statistic': float(f_stat), 'p_value': float(p_val)}
            
    # Test 2 - Chi-square: Is risk_tier independent of segment?
    if not features_df.empty and 'risk_tier' in features_df.columns and 'segment' in features_df.columns:
        contingency = pd.crosstab(features_df['segment'], features_df['risk_tier'])
        if contingency.size > 0:
            chi2, p_val, dof, _ = chi2_contingency(contingency)
            results['chi_square'] = {'chi2': float(chi2), 'p_value': float(p_val), 'dof': int(dof)}
            
    # Test 3 - T-test: mean DES of 'stable_salaried' vs 'sme_distressed'
    if not features_df.empty and 'segment' in features_df.columns:
        g1 = features_df[features_df['segment'] == 'stable_salaried']['fhs_mean'].dropna()
        g2 = features_df[features_df['segment'] == 'sme_distressed']['fhs_mean'].dropna()
        if len(g1) > 0 and len(g2) > 0:
            t_stat, p_val = ttest_ind(g1, g2, equal_var=False)
            results['ttest_segments'] = {'t_statistic': float(t_stat), 'p_value': float(p_val)}
            
    # Test 4 - Pearson: Correlation between balance and DES from population data
    if not pop_df.empty:
        df_corr = pop_df[['balance', 'fhs']].dropna()
        if len(df_corr) > 1:
            r, p_val = pearsonr(df_corr['balance'], df_corr['fhs'])
            results['pearson_bal_des'] = {'r': float(r), 'p_value': float(p_val)}
            
    # Test 5 - Spearman: spend_velocity_ratio and fhs
    if not pop_df.empty:
        df_spearman = pop_df[['spend_velocity_ratio', 'fhs']].dropna()
        if len(df_spearman) > 1:
            rho, p_val = spearmanr(df_spearman['spend_velocity_ratio'], df_spearman['fhs'])
            results['spearman_vel_des'] = {'rho': float(rho), 'p_value': float(p_val)}
            
    print("  ✓ Statistical testing complete")
    return results


def detect_anomalies_statistical(pop_df: pd.DataFrame, forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect statistical anomalies in forecasts compared to historical distributions.
    
    Parameters
    ----------
    pop_df : pd.DataFrame
        Historical transaction data.
    forecasts_df : pd.DataFrame
        Forecasted balances per customer (columns: customer_id, date, segment, balance_pred, yhat_lower).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with anomalies flagged.
    """
    np.random.seed(42)
    if forecasts_df.empty or pop_df.empty:
        return pd.DataFrame()
        
    results = []
    
    for customer_id, group in forecasts_df.groupby('customer_id'):
        hist = pop_df[pop_df['customer_id'] == customer_id].sort_values('date').tail(7)
        if len(hist) < 2:
            continue
            
        mean_bal = hist['balance'].mean()
        std_bal = hist['balance'].std()
        
        for _, row in group.iterrows():
            # Simulate an actual observation based on history
            simulated_actual = np.random.normal(mean_bal, std_bal + 1e-5)
            severity = 'NORMAL'
            
            if simulated_actual < row.get('yhat_lower', mean_bal - 1.28*std_bal):
                severity = 'CRITICAL_EWS'
            elif abs(simulated_actual - mean_bal) > 2 * std_bal:
                severity = 'HIGH_VARIANCE'
                
            results.append({
                'customer_id': customer_id,
                'date': row['date'],
                'segment': row['segment'],
                'actual_balance': simulated_actual,
                'yhat': row['balance_pred'],
                'yhat_lower': row.get('yhat_lower', mean_bal - 1.28*std_bal),
                'anomaly_severity': severity
            })
            
    res_df = pd.DataFrame(results)
    print(f"  ✓ Anomaly detection complete — {len(res_df)} cases processed")
    return res_df
