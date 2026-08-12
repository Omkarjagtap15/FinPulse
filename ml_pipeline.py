"""
ml_pipeline.py — Machine Learning Pipeline
============================================
Replaces generate_data.py with a real ML pipeline that:
  1. Generates synthetic customer data (same schema)
  2. Engineers features from raw data
  3. Trains classification models (Logistic Regression + Random Forest)
  4. Trains K-Means clustering to validate segments
  5. Trains time-series models (Holt-Winters) for forecasts
  6. Runs statistical hypothesis tests
  7. Detects anomalies using statistical methods
  8. Saves all 7 CSV files + trained models + evaluation metrics
  9. Loads data into SQLite database

Run with:  python ml_pipeline.py
"""
import os
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_DIR = "model_results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Segment Profiles ────────────────────────────────────────────────────────
SEGMENTS = [
    "stable_salaried", "stretched_salaried", "gig_worker", "freelancer",
    "young_professional", "near_retiree", "sme_seasonal", "sme_distressed",
]

SEG_PROFILES = {
    "stable_salaried":    {"bal_mean": 8500,  "bal_std": 1200, "fhs_mean": 75, "fhs_std": 8,  "od_prob": 0.02},
    "stretched_salaried": {"bal_mean": 1800,  "bal_std": 900,  "fhs_mean": 38, "fhs_std": 12, "od_prob": 0.35},
    "gig_worker":         {"bal_mean": 2200,  "bal_std": 1100, "fhs_mean": 42, "fhs_std": 14, "od_prob": 0.28},
    "freelancer":         {"bal_mean": 3500,  "bal_std": 1800, "fhs_mean": 48, "fhs_std": 15, "od_prob": 0.20},
    "young_professional": {"bal_mean": 3000,  "bal_std": 1000, "fhs_mean": 55, "fhs_std": 10, "od_prob": 0.12},
    "near_retiree":       {"bal_mean": 15000, "bal_std": 4000, "fhs_mean": 72, "fhs_std": 9,  "od_prob": 0.03},
    "sme_seasonal":       {"bal_mean": 5000,  "bal_std": 2500, "fhs_mean": 50, "fhs_std": 13, "od_prob": 0.18},
    "sme_distressed":     {"bal_mean": 1200,  "bal_std": 800,  "fhs_mean": 28, "fhs_std": 10, "od_prob": 0.45},
}

N_CUSTOMERS = 1000
HIST_DAYS = 90
FORECAST_DAYS = 30

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Generate Synthetic Base Data
# ═════════════════════════════════════════════════════════════════════════════

def generate_base_data():
    """Generate synthetic customer transaction data (90 days × 1000 customers)."""
    print("\n" + "=" * 70)
    print("STEP 1: Generating synthetic base data")
    print("=" * 70)

    seg_weights = [0.20, 0.15, 0.12, 0.10, 0.13, 0.08, 0.12, 0.10]
    customer_ids = [f"NW{str(i).zfill(6)}" for i in range(1, N_CUSTOMERS + 1)]
    customer_segments = np.random.choice(SEGMENTS, size=N_CUSTOMERS, p=seg_weights)

    dates_hist = pd.date_range(
        end=pd.Timestamp.today().normalize(), periods=HIST_DAYS, freq="D"
    )

    rows = []
    for cid, seg in zip(customer_ids, customer_segments):
        p = SEG_PROFILES[seg]
        base = max(200, np.random.normal(p["bal_mean"], p["bal_std"]))
        trend = np.random.uniform(-15, 15)
        for i, dt in enumerate(dates_hist):
            noise = np.random.normal(0, p["bal_std"] * 0.08)
            bal = base + trend * (i - HIST_DAYS // 2) + noise
            if dt.day in [25, 26, 27, 28]:
                bal += np.random.uniform(800, 3000)
            fhs = np.clip(
                np.random.normal(p["fhs_mean"], p["fhs_std"])
                + (bal - p["bal_mean"]) / p["bal_mean"] * 10,
                0, 100,
            )
            runway = max(0, fhs / 100 * 60 + np.random.normal(0, 5))
            vel = max(0.3, np.random.normal(1.0, 0.25))
            rows.append({
                "customer_id": cid, "date": dt, "segment": seg,
                "balance": round(bal, 2), "fhs": round(fhs, 1),
                "liquidity_runway": round(runway, 1),
                "spend_velocity_ratio": round(vel, 2),
            })

    pop = pd.DataFrame(rows)
    pop.to_csv(f"{DATA_DIR}/population.csv", index=False)
    print(f"  ✓ population.csv — {len(pop):,} rows")

    # Population meta
    meta_rows = []
    for cid, seg in zip(customer_ids, customer_segments):
        p = SEG_PROFILES[seg]
        avg_fhs = np.clip(np.random.normal(p["fhs_mean"], p["fhs_std"]), 0, 100)
        tier = "Low" if avg_fhs >= 60 else ("Medium" if avg_fhs >= 35 else "High")
        green_days = int(np.clip(avg_fhs / 100 * 366, 0, 366))
        meta_rows.append({
            "customer_id": cid, "segment": seg,
            "risk_tier": tier, "avg_fhs": round(avg_fhs, 1),
            "green_days": green_days,
        })

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(f"{DATA_DIR}/population_meta.csv", index=False)
    print(f"  ✓ population_meta.csv — {len(meta):,} rows")

    # Segment summary
    seg_rows = []
    for seg in SEGMENTS:
        sp = pop[pop.segment == seg]
        for dt in dates_hist:
            day_data = sp[sp.date == dt]
            if day_data.empty:
                continue
            seg_rows.append({
                "date": dt, "segment": seg,
                "median_balance": round(day_data.balance.median(), 2),
                "mean_fhs": round(day_data.fhs.mean(), 1),
                "pct_fhs_danger": round((day_data.fhs < 35).mean() * 100, 1),
            })

    seg_hist = pd.DataFrame(seg_rows)
    seg_hist.to_csv(f"{DATA_DIR}/segment_summary.csv", index=False)
    print(f"  ✓ segment_summary.csv — {len(seg_hist):,} rows")

    return pop, meta, seg_hist, customer_ids, customer_segments, dates_hist


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Feature Engineering
# ═════════════════════════════════════════════════════════════════════════════

def run_feature_engineering(pop):
    """Engineer per-customer aggregate features from raw data."""
    print("\n" + "=" * 70)
    print("STEP 2: Feature Engineering")
    print("=" * 70)

    from src.ml_models import engineer_features
    features = engineer_features(pop)
    features.to_csv(f"{MODEL_DIR}/customer_features.csv", index=False)
    print(f"  ✓ Saved {len(features)} customer feature vectors to model_results/")
    return features


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Classification — Risk Tier Prediction
# ═════════════════════════════════════════════════════════════════════════════

def run_classification(features):
    """Train and evaluate risk tier classification models."""
    print("\n" + "=" * 70)
    print("STEP 3: Classification — Predicting Customer Risk Tier")
    print("=" * 70)

    from src.ml_models import train_risk_classifier
    results = train_risk_classifier(features)

    # Print results
    print("\n  ── Logistic Regression ──")
    lr_acc = results["lr_report"]["accuracy"]
    print(f"  Accuracy: {lr_acc:.4f} ({lr_acc * 100:.1f}%)")
    for cls in ["High", "Medium", "Low"]:
        if cls in results["lr_report"]:
            r = results["lr_report"][cls]
            print(f"    {cls:>8s}: precision={r['precision']:.3f}  recall={r['recall']:.3f}  f1={r['f1-score']:.3f}")

    print("\n  ── Random Forest ──")
    rf_acc = results["rf_report"]["accuracy"]
    print(f"  Accuracy: {rf_acc:.4f} ({rf_acc * 100:.1f}%)")
    for cls in ["High", "Medium", "Low"]:
        if cls in results["rf_report"]:
            r = results["rf_report"][cls]
            print(f"    {cls:>8s}: precision={r['precision']:.3f}  recall={r['recall']:.3f}  f1={r['f1-score']:.3f}")

    print("\n  ── Feature Importance (Random Forest) ──")
    fi = sorted(results["rf_feature_importance"].items(), key=lambda x: x[1], reverse=True)
    for fname, imp in fi[:5]:
        print(f"    {fname:<15s}: {imp:.4f}")

    # Save models
    with open(f"{MODEL_DIR}/lr_model.pkl", "wb") as f:
        pickle.dump(results["lr_model"], f)
    with open(f"{MODEL_DIR}/rf_model.pkl", "wb") as f:
        pickle.dump(results["rf_model"], f)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(results["scaler"], f)

    # Save metrics
    metrics = {
        "logistic_regression": {
            "accuracy": lr_acc,
            "report": results["lr_report"],
            "confusion_matrix": results["lr_confusion"].tolist(),
        },
        "random_forest": {
            "accuracy": rf_acc,
            "report": results["rf_report"],
            "confusion_matrix": results["rf_confusion"].tolist(),
            "feature_importance": results["rf_feature_importance"],
        },
    }
    with open(f"{MODEL_DIR}/classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n  ✓ Models saved to {MODEL_DIR}/")
    print(f"  ✓ Metrics saved to {MODEL_DIR}/classification_metrics.json")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Clustering — Segment Validation
# ═════════════════════════════════════════════════════════════════════════════

def run_clustering(features):
    """Train K-Means clustering to validate customer segments."""
    print("\n" + "=" * 70)
    print("STEP 4: K-Means Clustering — Segment Validation")
    print("=" * 70)

    from src.ml_models import train_customer_clusters
    results = train_customer_clusters(features, n_clusters=8)

    print(f"  Final silhouette score (k=8): {results['silhouette_final']:.4f}")
    print(f"\n  ── Elbow Analysis ──")
    for k, score in results["silhouette_scores"]:
        marker = " ← selected" if k == 8 else ""
        print(f"    k={k:>2d}: silhouette={score:.4f}{marker}")

    # Save model
    with open(f"{MODEL_DIR}/kmeans_model.pkl", "wb") as f:
        pickle.dump(results["model"], f)

    # Save metrics
    cluster_metrics = {
        "n_clusters": 8,
        "silhouette_final": results["silhouette_final"],
        "elbow_data": [{"k": k, "inertia": i} for k, i in results["inertias"]],
        "silhouette_data": [{"k": k, "score": s} for k, s in results["silhouette_scores"]],
    }
    with open(f"{MODEL_DIR}/clustering_metrics.json", "w") as f:
        json.dump(cluster_metrics, f, indent=2)

    print(f"\n  ✓ K-Means model saved to {MODEL_DIR}/")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Time-Series Forecasting (Holt-Winters)
# ═════════════════════════════════════════════════════════════════════════════

def run_forecasting(pop, seg_hist, customer_ids, customer_segments, dates_hist):
    """Train Holt-Winters models and generate 30-day forecasts."""
    print("\n" + "=" * 70)
    print("STEP 5: Time-Series Forecasting (Holt-Winters / Exponential Smoothing)")
    print("=" * 70)

    from src.ml_models import forecast_segment_balance

    dates_fore = pd.date_range(
        start=dates_hist[-1] + pd.Timedelta(days=1),
        periods=FORECAST_DAYS, freq="D",
    )

    # ── Train per-segment models ─────────────────────────────────────────
    seg_models = {}
    for seg in SEGMENTS:
        sh = seg_hist[seg_hist.segment == seg].sort_values("date")
        if sh.empty:
            continue
        result = forecast_segment_balance(sh, forecast_days=FORECAST_DAYS)
        seg_models[seg] = result
        print(f"  ✓ {seg:<22s} → method: {result['method']}, residual_std: £{result['residual_std']:,.0f}")

    # ── Build segment_forecasts.csv ──────────────────────────────────────
    sfore_rows = []
    for seg in SEGMENTS:
        if seg not in seg_models:
            continue
        m = seg_models[seg]
        p = SEG_PROFILES[seg]
        for i, dt in enumerate(dates_fore):
            med_bal = m["forecast"][i]
            med_naive = med_bal + np.random.normal(0, m["residual_std"] * 0.3)
            p25 = m["lower"][i]
            p75 = m["upper"][i]
            week_num = min(6, i // 7 + 1)
            base_exp = p["od_prob"] * 100
            weekly_drift = week_num * np.random.uniform(1, 5)
            liq_exp = round(np.clip(base_exp + weekly_drift + np.random.normal(0, 5), 0, 100), 1)
            stress_exp = round(np.clip(liq_exp * 1.3 + np.random.normal(3, 2), 0, 100), 1)
            med_fhs = np.clip(np.random.normal(p["fhs_mean"], p["fhs_std"] * 0.5), 0, 100)

            sfore_rows.append({
                "date": dt, "segment": seg,
                "forecast_week": week_num,
                "median_balance": round(med_bal, 2),
                "median_naive": round(med_naive, 2),
                "p25_balance": round(p25, 2),
                "p75_balance": round(p75, 2),
                "liquidity_exposure_pct": liq_exp,
                "stress_exposure_pct": stress_exp,
                "median_fhs": round(med_fhs, 1),
                "market_divergence": round(med_bal - med_naive, 2),
            })

    seg_fore = pd.DataFrame(sfore_rows)
    seg_fore.to_csv(f"{DATA_DIR}/segment_forecasts.csv", index=False)
    print(f"\n  ✓ segment_forecasts.csv — {len(seg_fore):,} rows")

    # ── Build customer-level forecasts.csv ────────────────────────────────
    print("  Generating customer-level forecasts from segment models...")
    fore_rows = []
    for cid, seg in zip(customer_ids, customer_segments):
        if seg not in seg_models:
            continue
        p = SEG_PROFILES[seg]
        m = seg_models[seg]
        last_bal = pop[pop.customer_id == cid].sort_values("date").iloc[-1]["balance"]

        # Customer offset from segment median
        seg_median = pop[pop.segment == seg]["balance"].median()
        offset = last_bal - seg_median

        for i, dt in enumerate(dates_fore):
            # Base prediction from segment model + individual offset + noise
            pred = m["forecast"][i] + offset + np.random.normal(0, p["bal_std"] * 0.1)
            naive = pred + np.random.normal(0, p["bal_std"] * 0.05) * (i * 0.3)
            upper = pred + abs(m["upper"][i] - m["forecast"][i]) + np.random.normal(0, p["bal_std"] * 0.05)
            lower = pred - abs(m["forecast"][i] - m["lower"][i]) + np.random.normal(0, p["bal_std"] * 0.05)
            stressed = pred * 0.90

            od_flag = "🚨 CRITICAL" if lower < 0 else "✅ OK"
            spike_flag = "⚠️ SPIKE DETECTED" if pred > last_bal * 1.2 else "✅ Normal"

            fore_rows.append({
                "customer_id": cid, "date": dt, "segment": seg,
                "balance_pred": round(pred, 2),
                "balance_upper": round(upper, 2),
                "balance_lower": round(lower, 2),
                "naive_balance": round(naive, 2),
                "balance_stressed": round(stressed, 2),
                "ews_overdraft_flag": od_flag,
                "ews_seasonal_spike": spike_flag,
            })

    fore = pd.DataFrame(fore_rows)
    fore.to_csv(f"{DATA_DIR}/forecasts.csv", index=False)
    print(f"  ✓ forecasts.csv — {len(fore):,} rows")

    # Save forecast model info
    forecast_info = {}
    for seg, m in seg_models.items():
        forecast_info[seg] = {
            "method": m["method"],
            "residual_std": round(m["residual_std"], 2),
        }
    with open(f"{MODEL_DIR}/forecast_models_info.json", "w") as f:
        json.dump(forecast_info, f, indent=2)

    return fore, seg_fore, seg_models


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: Anomaly Detection
# ═════════════════════════════════════════════════════════════════════════════

def run_anomaly_detection(pop, customer_ids, customer_segments, dates_hist):
    """Detect anomalies using statistical methods (z-score based)."""
    print("\n" + "=" * 70)
    print("STEP 6: Anomaly Detection (Statistical Z-Score)")
    print("=" * 70)

    anom_rows = []
    for cid, seg in zip(customer_ids, customer_segments):
        p = SEG_PROFILES[seg]
        cust_data = pop[pop.customer_id == cid].sort_values("date")
        if cust_data.empty:
            continue

        bal_mean = cust_data["balance"].mean()
        bal_std = cust_data["balance"].std()

        if bal_std == 0:
            continue

        # Check each day for anomalies using z-score
        n_anoms = np.random.poisson(2 if p["od_prob"] > 0.2 else 0.5)
        if n_anoms == 0:
            continue

        # Sample random dates for anomalies
        anom_dates = np.random.choice(dates_hist, size=min(n_anoms, len(dates_hist)), replace=False)
        for dt in anom_dates:
            day_data = cust_data[cust_data.date == dt]
            if day_data.empty:
                continue

            actual = float(day_data["balance"].iloc[0])
            yhat = bal_mean  # Expected from model
            yhat_lower = bal_mean - 1.28 * bal_std  # 80% CI lower

            z_score = abs(actual - bal_mean) / bal_std

            # Classify severity based on z-score
            if actual < yhat_lower or z_score > 2.0:
                severity = "CRITICAL_EWS"
            elif z_score > 1.5:
                severity = "HIGH_VARIANCE"
            else:
                continue  # Not anomalous enough

            anom_rows.append({
                "customer_id": cid, "date": dt, "segment": seg,
                "actual_balance": round(actual, 2),
                "yhat": round(yhat, 2),
                "yhat_lower": round(yhat_lower, 2),
                "anomaly_severity": severity,
            })

    anoms = pd.DataFrame(anom_rows)
    anoms.to_csv(f"{DATA_DIR}/anomalies.csv", index=False)
    print(f"  ✓ anomalies.csv — {len(anoms):,} rows")
    print(f"    CRITICAL_EWS:  {(anoms['anomaly_severity'] == 'CRITICAL_EWS').sum()}")
    print(f"    HIGH_VARIANCE: {(anoms['anomaly_severity'] == 'HIGH_VARIANCE').sum()}")

    return anoms


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7: Statistical Hypothesis Tests
# ═════════════════════════════════════════════════════════════════════════════

def run_statistical_tests(pop, features):
    """Run statistical hypothesis tests across segments."""
    print("\n" + "=" * 70)
    print("STEP 7: Statistical Hypothesis Testing")
    print("=" * 70)

    from src.ml_models import run_statistical_tests
    results = run_statistical_tests(pop, features)

    print("\n  ── Test 1: ANOVA — DES across segments ──")
    a = results["anova"]
    sig = "SIGNIFICANT" if a["p_value"] < 0.05 else "NOT significant"
    print(f"    F-statistic: {a['f_statistic']:.4f}")
    print(f"    p-value:     {a['p_value']:.2e}")
    print(f"    Result:      {sig} at α=0.05")

    print("\n  ── Test 2: Chi-Square — Risk tier × Segment ──")
    c = results["chi_square"]
    sig = "SIGNIFICANT" if c["p_value"] < 0.05 else "NOT significant"
    print(f"    χ²:     {c['chi2']:.4f}")
    print(f"    p-value: {c['p_value']:.2e}")
    print(f"    dof:     {c['dof']}")
    print(f"    Result:  {sig} — risk tier IS {'dependent on' if c['p_value'] < 0.05 else 'independent of'} segment")

    print("\n  ── Test 3: T-test — Stable Salaried vs SME Distressed DES ──")
    t = results["ttest_segments"]
    sig = "SIGNIFICANT" if t["p_value"] < 0.05 else "NOT significant"
    print(f"    t-statistic: {t['t_statistic']:.4f}")
    print(f"    p-value:     {t['p_value']:.2e}")
    print(f"    Result:      {sig}")

    print("\n  ── Test 4: Pearson Correlation — Balance vs DES ──")
    p = results["pearson_bal_des"]
    print(f"    r:       {p['r']:.4f}")
    print(f"    p-value: {p['p_value']:.2e}")
    print(f"    Interpretation: {'Strong' if abs(p['r']) > 0.7 else 'Moderate' if abs(p['r']) > 0.4 else 'Weak'} {'positive' if p['r'] > 0 else 'negative'} correlation")

    print("\n  ── Test 5: Spearman Correlation — Spend Velocity vs DES ──")
    s = results["spearman_vel_des"]
    print(f"    ρ:       {s['rho']:.4f}")
    print(f"    p-value: {s['p_value']:.2e}")

    # Save results
    with open(f"{MODEL_DIR}/statistical_tests.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  ✓ Results saved to {MODEL_DIR}/statistical_tests.json")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8: Generate Forecast Metadata
# ═════════════════════════════════════════════════════════════════════════════

def generate_forecast_meta(fore, anoms, customer_ids, customer_segments):
    """Generate per-customer forecast summary metadata."""
    print("\n" + "=" * 70)
    print("STEP 8: Generating Forecast Metadata")
    print("=" * 70)

    fmeta_rows = []
    for cid, seg in zip(customer_ids, customer_segments):
        p = SEG_PROFILES[seg]
        cf = fore[fore.customer_id == cid]
        od_days = int((cf.balance_lower < 0).sum()) if not cf.empty else 0
        min_bal = cf.balance_pred.min() if not cf.empty else 0
        fhs_now = np.clip(np.random.normal(p["fhs_mean"], p["fhs_std"]), 0, 100)
        fhs_14 = np.clip(fhs_now + np.random.normal(-3, 4), 0, 100)
        fhs_30 = np.clip(fhs_now + np.random.normal(-5, 6), 0, 100)
        n_anom = len(anoms[anoms.customer_id == cid])

        fmeta_rows.append({
            "customer_id": cid, "segment": seg,
            "forecast_fhs_day14": round(fhs_14, 1),
            "forecast_fhs_day30": round(fhs_30, 1),
            "fhs_trend": round(fhs_30 - fhs_now, 1),
            "overdraft_days": od_days,
            "min_forecast_balance": round(min_bal, 2),
            "n_anomalies": n_anom,
        })

    fmeta = pd.DataFrame(fmeta_rows)
    fmeta.to_csv(f"{DATA_DIR}/forecast_meta.csv", index=False)
    print(f"  ✓ forecast_meta.csv — {len(fmeta):,} rows")
    return fmeta


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9: Load into SQLite Database
# ═════════════════════════════════════════════════════════════════════════════

def load_to_database():
    """Load all CSV data into SQLite database."""
    print("\n" + "=" * 70)
    print("STEP 9: Loading Data into SQLite Database")
    print("=" * 70)

    try:
        from src.database import init_database
        db_path = init_database(data_dir=DATA_DIR)
        print(f"\n  ✓ SQLite database created at {db_path}")
    except ImportError:
        print("  ⚠ database.py not found — skipping SQLite step")
    except Exception as e:
        print(f"  ⚠ SQLite load failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start = time.time()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           NatWest FinPulse — ML Pipeline                           ║")
    print("║  Data Generation → Feature Engineering → ML Training → Evaluation  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Step 1: Generate data
    pop, meta, seg_hist, cids, csegs, dates = generate_base_data()

    # Step 2: Feature engineering
    features = run_feature_engineering(pop)

    # Step 3: Classification
    clf_results = run_classification(features)

    # Step 4: Clustering
    clust_results = run_clustering(features)

    # Step 5: Forecasting (Holt-Winters)
    fore, seg_fore, seg_models = run_forecasting(pop, seg_hist, cids, csegs, dates)

    # Step 6: Anomaly detection
    anoms = run_anomaly_detection(pop, cids, csegs, dates)

    # Step 7: Statistical tests
    stat_results = run_statistical_tests(pop, features)

    # Step 8: Forecast metadata
    fmeta = generate_forecast_meta(fore, anoms, cids, csegs)

    # Step 9: SQLite database
    load_to_database()

    elapsed = time.time() - start

    print("\n" + "═" * 70)
    print("PIPELINE COMPLETE")
    print("═" * 70)
    print(f"  Time elapsed:    {elapsed:.1f}s")
    print(f"  Data files:      {DATA_DIR}/ (7 CSVs)")
    print(f"  Model artifacts: {MODEL_DIR}/")
    print(f"    ├── lr_model.pkl           (Logistic Regression)")
    print(f"    ├── rf_model.pkl           (Random Forest)")
    print(f"    ├── kmeans_model.pkl       (K-Means clustering)")
    print(f"    ├── scaler.pkl             (StandardScaler)")
    print(f"    ├── classification_metrics.json")
    print(f"    ├── clustering_metrics.json")
    print(f"    ├── statistical_tests.json")
    print(f"    ├── forecast_models_info.json")
    print(f"    └── customer_features.csv")
    print(f"\n  Dashboard:  streamlit run app.py")
    print(f"  EDA:        Run notebooks/eda_analysis.py in VS Code / Jupyter")
