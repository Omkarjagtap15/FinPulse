# %% [markdown]
# # Exploratory Data Analysis (EDA) - NatWest FinPulse
# This script performs comprehensive Exploratory Data Analysis for the NatWest FinPulse project,
# a banking liquidity risk dashboard modeling synthetic customers across various segments.

# %% [markdown]
# ### Section 1: Setup & Data Loading
# Import necessary libraries, configure visualization styles, and load data files.
# We also ensure the `../model_results/` directory exists to save our figures.

# %%
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from itertools import cycle

warnings.filterwarnings('ignore')

# Set matplotlib style and seaborn palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')

# Create model_results directory for saving figures
output_dir = '../model_results'
os.makedirs(output_dir, exist_ok=True)

# Data paths
data_dir = '../data/'
files = {
    'population': 'population.csv',
    'population_meta': 'population_meta.csv',
    'forecasts': 'forecasts.csv',
    'anomalies': 'anomalies.csv',
    'forecast_meta': 'forecast_meta.csv',
    'segment_summary': 'segment_summary.csv',
    'segment_forecasts': 'segment_forecasts.csv'
}

dataframes = {}

print("--- Section 1: Setup & Data Loading ---")
try:
    for name, file in files.items():
        path = os.path.join(data_dir, file)
        if os.path.exists(path):
            dataframes[name] = pd.read_csv(path)
            print(f"Loaded {name}: {dataframes[name].shape}")
        else:
            print(f"File not found: {path}. Please ensure data is generated.")
            
    if 'population' in dataframes:
        print("\npopulation.csv Data Types & Missing Values:")
        print(dataframes['population'].info())
        print("\nFirst 5 rows of population.csv:")
        display(dataframes['population'].head()) if 'display' in globals() else print(dataframes['population'].head())
except Exception as e:
    print(f"Error loading data: {e}")

# Note: For the rest of the script to run seamlessly, we will extract the loaded dataframes
df_pop = dataframes.get('population', pd.DataFrame())
df_pop_meta = dataframes.get('population_meta', pd.DataFrame())
df_forecasts = dataframes.get('forecasts', pd.DataFrame())
df_anomalies = dataframes.get('anomalies', pd.DataFrame())
df_forecast_meta = dataframes.get('forecast_meta', pd.DataFrame())
df_seg_sum = dataframes.get('segment_summary', pd.DataFrame())
df_seg_for = dataframes.get('segment_forecasts', pd.DataFrame())

# %% [markdown]
# ### Section 2: Distribution Analysis
# Visualizing the distribution of customer balances and DES (Daily Engagement Score/FHS)
# across different segments to understand baseline segment characteristics.

# %%
if not df_pop.empty:
    print("\n--- Section 2: Distribution Analysis ---")
    
    # Plot 1: Balance distribution per segment
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_pop, x='segment', y='balance')
    plt.title('Balance Distribution per Segment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_01_balance_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Plot 2: DES (fhs) distribution per segment
    plt.figure(figsize=(14, 6))
    sns.violinplot(data=df_pop, x='segment', y='fhs', inner='quartile')
    plt.title('DES (FHS) Distribution per Segment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_02_des_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Plot 3: Histogram of DES scores with RED/AMBER/GREEN band shading
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_pop, x='fhs', bins=50, kde=True, color='gray')
    plt.axvspan(0, 30, color='red', alpha=0.2, label='RED (0-30)')
    plt.axvspan(30, 70, color='orange', alpha=0.2, label='AMBER (30-70)')
    plt.axvspan(70, 100, color='green', alpha=0.2, label='GREEN (70-100)')
    plt.title('Distribution of DES Scores')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_03_des_histogram.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary statistics table grouped by segment
    print("\nSummary Statistics Grouped by Segment:")
    summary_stats = df_pop.groupby('segment')[['balance', 'fhs']].describe()
    print(summary_stats)

# %% [markdown]
# ### Section 3: Correlation Analysis
# Analyzing relationships between key numeric variables such as balance, DES (fhs),
# liquidity runway, and spend velocity to uncover linear dependencies.

# %%
if not df_pop.empty:
    print("\n--- Section 3: Correlation Analysis ---")
    
    numeric_cols = ['balance', 'fhs', 'liquidity_runway', 'spend_velocity_ratio']
    # Filter columns that exist
    cols_to_use = [col for col in numeric_cols if col in df_pop.columns]
    
    if len(cols_to_use) > 1:
        corr_matrix = df_pop[cols_to_use].corr()
        
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', fmt='.2f', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap of Key Variables')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_04_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.show()

        # Plot scatter: balance vs fhs colored by segment
        # Using a sample to avoid overplotting if dataset is large
        sample_df = df_pop.sample(n=min(5000, len(df_pop)), random_state=42)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=sample_df, x='balance', y='fhs', hue='segment', alpha=0.5)
        plt.title('Scatter Plot: Balance vs DES (FHS)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_05_scatter_balance_fhs.png'), dpi=150, bbox_inches='tight')
        plt.show()

        # Pearson and Spearman correlations
        if 'balance' in df_pop.columns and 'fhs' in df_pop.columns:
            pearson_corr, _ = stats.pearsonr(df_pop['balance'].dropna(), df_pop['fhs'].dropna())
            spearman_corr, _ = stats.spearmanr(df_pop['balance'].dropna(), df_pop['fhs'].dropna())
            print(f"Pearson Correlation (Balance vs FHS): {pearson_corr:.3f}")
            print(f"Spearman Correlation (Balance vs FHS): {spearman_corr:.3f}")

# %% [markdown]
# ### Section 4: Time-Series Patterns
# Exploring temporal behaviors across contrasting segments, including multi-day trends,
# day-of-month (salary bumps), and day-of-week spending patterns.

# %%
if not df_pop.empty and 'date' in df_pop.columns:
    print("\n--- Section 4: Time-Series Patterns ---")
    
    df_pop['date'] = pd.to_datetime(df_pop['date'])
    df_pop['day_of_month'] = df_pop['date'].dt.day
    df_pop['day_of_week'] = df_pop['date'].dt.dayofweek

    target_segments = ['stable_salaried', 'stretched_salaried', 'sme_distressed']
    available_targets = [seg for seg in target_segments if seg in df_pop['segment'].unique()]
    
    if available_targets:
        # Plot median daily balance over time
        fig, axes = plt.subplots(len(available_targets), 1, figsize=(14, 4 * len(available_targets)), sharex=True)
        if len(available_targets) == 1:
            axes = [axes]
            
        for i, seg in enumerate(available_targets):
            seg_data = df_pop[df_pop['segment'] == seg].groupby('date')['balance'].median().reset_index()
            axes[i].plot(seg_data['date'], seg_data['balance'], label=seg, color='C'+str(i))
            axes[i].set_title(f'Median Daily Balance: {seg}')
            axes[i].set_ylabel('Balance')
            axes[i].grid(True)
            
        plt.xlabel('Date')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_06_timeseries_balance.png'), dpi=150, bbox_inches='tight')
        plt.show()

    # Salary bump pattern
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df_pop, x='day_of_month', y='balance', hue='segment', errorbar=None)
    plt.title('Average Balance by Day of Month (Salary Bump Pattern)')
    plt.xlabel('Day of Month')
    plt.ylabel('Average Balance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_07_day_of_month.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Weekly spending pattern
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_pop, x='day_of_week', y='balance', hue='segment', errorbar=None)
    plt.title('Average Balance by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Average Balance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_08_day_of_week.png'), dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ### Section 5: Statistical Hypothesis Testing
# Conducting rigorous statistical tests to validate segment differences,
# dependence of risk tiers, and distribution normality.

# %%
if not df_pop.empty and 'fhs' in df_pop.columns:
    print("\n--- Section 5: Statistical Hypothesis Testing ---")
    
    # Test 1 — ANOVA
    segments = df_pop['segment'].unique()
    fhs_arrays = [df_pop[df_pop['segment'] == seg]['fhs'].dropna() for seg in segments]
    
    if len(fhs_arrays) > 1:
        f_stat, p_val = stats.f_oneway(*fhs_arrays)
        print(f"Test 1 (ANOVA) - Is mean DES different across segments?")
        print(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4e}")
        print("Interpretation: Significant differences exist in mean DES across at least one pair of segments." if p_val < 0.05 else "Interpretation: No significant difference in mean DES across segments.")
    
    # Test 2 — T-test
    if 'stretched_salaried' in segments and 'stable_salaried' in segments:
        fhs_stretched = df_pop[df_pop['segment'] == 'stretched_salaried']['fhs'].dropna()
        fhs_stable = df_pop[df_pop['segment'] == 'stable_salaried']['fhs'].dropna()
        
        t_stat, p_val2 = stats.ttest_ind(fhs_stretched, fhs_stable)
        print(f"\nTest 2 (T-test) - Are stretched_salaried and stable_salaried different in DES?")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_val2:.4e}")
        print("Interpretation: The DES is significantly different between the two segments." if p_val2 < 0.05 else "Interpretation: No significant difference in DES between these two segments.")

if not df_pop_meta.empty and 'risk_tier' in df_pop_meta.columns and 'segment' in df_pop_meta.columns:
    # Test 3 — Chi-square
    contingency = pd.crosstab(df_pop_meta['segment'], df_pop_meta['risk_tier'])
    chi2, p_val3, dof, _ = stats.chi2_contingency(contingency)
    print(f"\nTest 3 (Chi-square) - Is risk_tier distribution independent of segment?")
    print(f"Chi2: {chi2:.4f}, p-value: {p_val3:.4e}, DoF: {dof}")
    print("Interpretation: Risk tier is dependent on segment (not independent)." if p_val3 < 0.05 else "Interpretation: Risk tier is independent of segment.")

if not df_pop.empty and 'fhs' in df_pop.columns:
    # Test 4 — Normality
    # Taking a sample up to 5000 as Shapiro-Wilk may not be accurate for very large N
    sample_fhs = df_pop['fhs'].dropna().sample(n=min(5000, len(df_pop['fhs'].dropna())), random_state=42)
    w_stat, p_val4 = stats.shapiro(sample_fhs)
    print(f"\nTest 4 (Normality) - Is DES normally distributed?")
    print(f"W-statistic: {w_stat:.4f}, p-value: {p_val4:.4e}")
    print("Interpretation: The distribution is NOT normally distributed." if p_val4 < 0.05 else "Interpretation: The distribution appears normally distributed.")

# %% [markdown]
# ### Section 6: Risk Tier Analysis
# Visualizing and quantifying the proportion of High, Medium, and Low risk customers
# within each segment based on metadata.

# %%
if not df_pop_meta.empty and 'risk_tier' in df_pop_meta.columns:
    print("\n--- Section 6: Risk Tier Analysis ---")
    
    # Calculate proportions
    risk_props = pd.crosstab(df_pop_meta['segment'], df_pop_meta['risk_tier'], normalize='index') * 100
    
    # Bar chart (stacked)
    risk_props.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='RdYlGn_r')
    plt.title('Risk Tier Distribution by Segment (%)')
    plt.ylabel('Percentage')
    plt.legend(title='Risk Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_09_risk_tier_dist.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Percentage of Risk Tiers per Segment:")
    print(risk_props.round(2))
    
    # Identify high risk segments
    if 'High' in risk_props.columns:
        high_risk_segs = risk_props[risk_props['High'] > 30].index.tolist()
        print(f"\nSegments with >30% High risk customers: {high_risk_segs}")

# %% [markdown]
# ### Section 7: Anomaly Analysis
# Evaluating anomalies across segments, analyzing severity, and plotting actual vs expected balances.

# %%
if not df_anomalies.empty:
    print("\n--- Section 7: Anomaly Analysis ---")
    
    # Bar chart: anomaly count by segment and severity
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_anomalies, x='segment', hue='anomaly_severity', palette='Reds')
    plt.title('Anomaly Count by Segment and Severity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_10_anomaly_counts.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Scatter plot: actual_balance vs yhat
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_anomalies, x='yhat', y='actual_balance', hue='anomaly_severity', palette='Reds', alpha=0.7)
    # Add y=x reference line
    max_val = max(df_anomalies['actual_balance'].max(), df_anomalies['yhat'].max())
    min_val = min(df_anomalies['actual_balance'].min(), df_anomalies['yhat'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Actual = Expected')
    plt.title('Actual Balance vs Expected (yhat) During Anomalies')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_11_actual_vs_yhat.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Breach severity statistics
    if 'actual_balance' in df_anomalies.columns and 'yhat_lower' in df_anomalies.columns:
        df_anomalies['breach_amount'] = df_anomalies['yhat_lower'] - df_anomalies['actual_balance']
        breach_stats = df_anomalies.groupby('segment')['breach_amount'].describe()
        print("Breach Severity (Amount below lower bound) Statistics by Segment:")
        print(breach_stats[['count', 'mean', 'max']])

# %% [markdown]
# ### Section 8: Feature Engineering & Model Preparation
# Aggregating daily customer data into customer-level features for modeling and clustering.
# Creating trend metrics and evaluating basic feature importance.

# %%
if not df_pop.empty:
    print("\n--- Section 8: Feature Engineering & Model Preparation ---")
    
    def calc_trend(series):
        if len(series) < 2: return 0
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]

    # Customer level aggregations
    df_features = df_pop.groupby('customer_id').agg(
        bal_mean=('balance', 'mean'),
        bal_std=('balance', 'std'),
        bal_min=('balance', 'min'),
        bal_max=('balance', 'max'),
        fhs_mean=('fhs', 'mean'),
        fhs_std=('fhs', 'std'),
        runway_mean=('liquidity_runway', 'mean'),
        vel_mean=('spend_velocity_ratio', 'mean'),
        vel_max=('spend_velocity_ratio', 'max'),
        bal_trend=('balance', calc_trend)
    ).reset_index()
    
    df_features['bal_cv'] = df_features['bal_std'] / (df_features['bal_mean'].abs() + 1e-5)
    
    # Merge with meta to get risk_tier and segment
    if not df_pop_meta.empty:
        df_features = df_features.merge(df_pop_meta[['customer_id', 'segment', 'risk_tier']], on='customer_id', how='left')
    
    print("Engineered Features (First 5 rows):")
    display(df_features.head()) if 'display' in globals() else print(df_features.head())
    
    print("\nFeature Summary Statistics:")
    print(df_features.describe())
    
    # Feature importance via correlation with risk tier
    if 'risk_tier' in df_features.columns:
        risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
        df_features['risk_encoded'] = df_features['risk_tier'].map(risk_map)
        
        numeric_feats = ['bal_mean', 'bal_std', 'bal_min', 'bal_max', 'bal_trend', 'bal_cv', 
                         'fhs_mean', 'fhs_std', 'runway_mean', 'vel_mean', 'vel_max']
        
        # Valid features that exist
        valid_feats = [f for f in numeric_feats if f in df_features.columns]
        
        if valid_feats and not df_features['risk_encoded'].isnull().all():
            feat_corr = df_features[valid_feats].corrwith(df_features['risk_encoded']).abs().sort_values(ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feat_corr.values, y=feat_corr.index, palette='viridis')
            plt.title('Feature Importance via Correlation with Risk Tier')
            plt.xlabel('Absolute Correlation')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig_12_feature_corr.png'), dpi=150, bbox_inches='tight')
            plt.show()

# %% [markdown]
# ### Section 9: Classification Model Training & Evaluation
# Training multi-class classification models (Logistic Regression & Random Forest) 
# to predict risk tiers and evaluating their performance.

# %%
if 'df_features' in locals() and not df_features.empty:
    print("\n--- Section 9: Classification Model Training & Evaluation ---")
    
    # Target creation (if risk_tier is missing, we create it from fhs_mean)
    if 'risk_tier' not in df_features.columns or df_features['risk_tier'].isnull().any():
        df_features['target'] = pd.cut(df_features['fhs_mean'], bins=[-np.inf, 30, 70, np.inf], labels=['High', 'Medium', 'Low'])
    else:
        df_features['target'] = df_features['risk_tier']
        
    df_features = df_features.dropna(subset=['target'])
    
    features_to_use = ['bal_mean', 'bal_std', 'bal_min', 'bal_max', 'bal_trend', 'bal_cv', 
                       'fhs_mean', 'fhs_std', 'runway_mean', 'vel_mean', 'vel_max']
    features_to_use = [f for f in features_to_use if f in df_features.columns]
    
    if len(features_to_use) > 0:
        X = df_features[features_to_use].fillna(0)
        y = df_features['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_preds = lr.predict(X_test_scaled)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        rf_preds = rf.predict(X_test_scaled)
        
        print("\nLogistic Regression Classification Report:")
        print(classification_report(y_test, lr_preds))
        
        print("\nRandom Forest Classification Report:")
        print(classification_report(y_test, rf_preds))
        
        # Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(confusion_matrix(y_test, lr_preds), annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                    xticklabels=lr.classes_, yticklabels=lr.classes_)
        axes[0].set_title('Logistic Regression Confusion Matrix')
        
        sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=rf.classes_, yticklabels=rf.classes_)
        axes[1].set_title('Random Forest Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_13_confusion_matrices.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # ROC Curves (One-vs-Rest)
        y_test_bin = label_binarize(y_test, classes=rf.classes_)
        n_classes = y_test_bin.shape[1]
        
        if n_classes > 1:
            rf_probs = rf.predict_proba(X_test_scaled)
            lr_probs = lr.predict_proba(X_test_scaled)
            
            plt.figure(figsize=(10, 8))
            colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
            
            for i, color in zip(range(n_classes), colors):
                fpr_rf, tpr_rf, _ = roc_curve(y_test_bin[:, i], rf_probs[:, i])
                plt.plot(fpr_rf, tpr_rf, color=color, lw=2, label=f'RF Class {rf.classes_[i]} (AUC = {auc(fpr_rf, tpr_rf):.2f})')
                
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.title('ROC Curves (Random Forest - One vs Rest)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig_14_roc_curves.png'), dpi=150, bbox_inches='tight')
            plt.show()
            
        # RF Feature Importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[features_to_use[i] for i in indices], palette='viridis')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_15_rf_feature_importance.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # Best model
        acc_lr = (lr_preds == y_test).mean() * 100
        acc_rf = (rf_preds == y_test).mean() * 100
        best_name = "Random Forest" if acc_rf >= acc_lr else "Logistic Regression"
        best_acc = max(acc_rf, acc_lr)
        print(f"\nBest model: {best_name} with accuracy {best_acc:.2f}%")

# %% [markdown]
# ### Section 10: K-Means Clustering Validation
# Using unsupervised K-Means to find natural groups in the data, evaluating 
# with Silhouette scores, and cross-tabulating against existing segments.

# %%
if 'df_features' in locals() and not df_features.empty:
    print("\n--- Section 10: K-Means Clustering Validation ---")
    
    clust_feats = ['bal_mean', 'bal_std', 'bal_cv', 'fhs_mean', 'runway_mean', 'vel_mean']
    clust_feats = [f for f in clust_feats if f in df_features.columns]
    
    if len(clust_feats) > 0:
        X_clust = df_features[clust_feats].fillna(0)
        X_clust_scaled = StandardScaler().fit_transform(X_clust)
        
        inertias = []
        silhouettes = []
        k_values = range(2, 13)
        
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            preds = km.fit_predict(X_clust_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_clust_scaled, preds))
            
        # Plot Elbow & Silhouette
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(k_values, inertias, marker='o')
        axes[0].set_title('Elbow Method (Inertia vs k)')
        axes[0].set_xlabel('k')
        axes[0].set_ylabel('Inertia')
        
        axes[1].plot(k_values, silhouettes, marker='o', color='orange')
        axes[1].set_title('Silhouette Scores vs k')
        axes[1].set_xlabel('k')
        axes[1].set_ylabel('Silhouette Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_16_clustering_metrics.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # Train final k=8 model
        km_final = KMeans(n_clusters=8, random_state=42, n_init=10)
        cluster_labels = km_final.fit_predict(X_clust_scaled)
        final_sil = silhouette_score(X_clust_scaled, cluster_labels)
        
        print(f"Final model (k=8) Silhouette Score: {final_sil:.4f}")
        
        df_features['cluster'] = cluster_labels
        
        # Cluster vs Segment Alignment
        if 'segment' in df_features.columns:
            alignment = pd.crosstab(df_features['cluster'], df_features['segment'])
            print("\nCluster-Segment Alignment Table:")
            print(alignment)
            
            # PCA 2D Scatter
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_clust_scaled)
            
            plt.figure(figsize=(12, 8))
            scatter = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_features['cluster'], palette='tab10', alpha=0.6)
            plt.title('PCA 2D Scatter: K-Means Clusters')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            
            # Annotate with segment majority
            for c in range(8):
                majority_seg = alignment.loc[c].idxmax() if c in alignment.index else "Unknown"
                c_mask = df_features['cluster'] == c
                if c_mask.sum() > 0:
                    c_center = X_pca[c_mask].mean(axis=0)
                    plt.annotate(f"C{c}:\n{majority_seg}", (c_center[0], c_center[1]), 
                                 weight='bold', color='black',
                                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
                    
            plt.legend(title="Cluster")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig_17_pca_clusters.png'), dpi=150, bbox_inches='tight')
            plt.show()

# %% [markdown]
# ### Section 11: Forecasting Analysis
# Comparing forecast models against naive baselines and assessing stress exposure growth.

# %%
if not df_seg_for.empty:
    print("\n--- Section 11: Forecasting Analysis ---")
    
    segments = df_seg_for['segment'].unique()
    target_segs = segments[:2] if len(segments) >= 2 else segments
    
    for seg in target_segs:
        seg_data = df_seg_for[df_seg_for['segment'] == seg].sort_values('forecast_week')
        
        plt.figure(figsize=(10, 5))
        plt.plot(seg_data['forecast_week'], seg_data['median_balance'], marker='o', label='Forecast Median')
        plt.plot(seg_data['forecast_week'], seg_data['median_naive'], marker='x', linestyle='--', label='Naive Median')
        plt.fill_between(seg_data['forecast_week'], seg_data['p25_balance'], seg_data['p75_balance'], alpha=0.2, label='Forecast P25-P75')
        
        plt.title(f'Forecast vs Naive Baseline: {seg}')
        plt.xlabel('Forecast Week')
        plt.ylabel('Balance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fig_18_forecast_{seg}.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # RMSE-like metric
        if 'median_balance' in seg_data.columns and 'median_naive' in seg_data.columns:
            rmse_like = np.sqrt(((seg_data['median_balance'] - seg_data['median_naive']) ** 2).mean())
            print(f"[{seg}] RMSE-like Metric (Forecast vs Naive): {rmse_like:.2f}")
            
    # Stress exposure growth
    if 'stress_exposure_pct' in df_seg_for.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_seg_for, x='forecast_week', y='stress_exposure_pct', hue='segment', marker='o')
        plt.title('Stress Exposure Growth Over Forecast Weeks')
        plt.xlabel('Forecast Week')
        plt.ylabel('Stress Exposure (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_19_stress_exposure.png'), dpi=150, bbox_inches='tight')
        plt.show()

# %% [markdown]
# ### Section 12: Summary & Key Findings
# Structured final recap of the EDA highlighting key insights, risks, and modeling success.

# %%
print("\n--- Section 12: Summary & Key Findings ---")

n_customers = df_pop['customer_id'].nunique() if not df_pop.empty else 0
n_segments = df_pop['segment'].nunique() if not df_pop.empty else 0
n_days = df_pop['date'].nunique() if not df_pop.empty and 'date' in df_pop.columns else 0

print(f"Dataset Overview:")
print(f" - Customers: {n_customers}")
print(f" - Segments: {n_segments}")
print(f" - History (Days): {n_days}")

if 'best_acc' in locals() and 'best_name' in locals():
    print(f"\nModeling:")
    print(f" - Best Classifier: {best_name} with Accuracy = {best_acc:.2f}%")

print("\nKey Statistical Findings:")
if 'p_val' in locals() and p_val < 0.05:
    print(" - ANOVA: Significant difference in DES across segments.")
if 'p_val2' in locals() and p_val2 < 0.05:
    print(" - T-test: Stretched vs Stable salaried DES are significantly different.")
if 'p_val3' in locals() and p_val3 < 0.05:
    print(" - Chi-square: Risk tier distribution strongly depends on segment.")
if 'p_val4' in locals():
    print(f" - Normality: DES distribution is {'normal' if p_val4 > 0.05 else 'non-normal'}.")

if 'final_sil' in locals():
    print(f"\nClustering:")
    print(f" - Silhouette Score (k=8): {final_sil:.4f}")

if 'high_risk_segs' in locals():
    print(f"\nTop Riskiest Segments (>30% High risk):")
    print(f" - {', '.join(high_risk_segs) if high_risk_segs else 'None'}")

if 'indices' in locals() and 'features_to_use' in locals():
    top_n = min(3, len(indices))
    top_feats = [features_to_use[i] for i in indices[:top_n]]
    print(f"\nTop {top_n} Features for Risk Prediction:")
    for i, feat in enumerate(top_feats, 1):
        print(f" {i}. {feat}")

print("\nEDA Completed Successfully. All figures saved in '../model_results/'.")
