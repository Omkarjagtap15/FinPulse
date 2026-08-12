"""
Generate realistic sample data for NatWest FinPulse Dashboard
=============================================================
Creates all 7 CSV files the app expects in the data/ directory.
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)
os.makedirs("data", exist_ok=True)

SEGMENTS = [
    "stable_salaried", "stretched_salaried", "gig_worker", "freelancer",
    "young_professional", "near_retiree", "sme_seasonal", "sme_distressed"
]

SEG_PROFILES = {
    "stable_salaried":      {"bal_mean": 8500,  "bal_std": 1200, "fhs_mean": 75, "fhs_std": 8,  "od_prob": 0.02},
    "stretched_salaried":   {"bal_mean": 1800,  "bal_std": 900,  "fhs_mean": 38, "fhs_std": 12, "od_prob": 0.35},
    "gig_worker":           {"bal_mean": 2200,  "bal_std": 1100, "fhs_mean": 42, "fhs_std": 14, "od_prob": 0.28},
    "freelancer":           {"bal_mean": 3500,  "bal_std": 1800, "fhs_mean": 48, "fhs_std": 15, "od_prob": 0.20},
    "young_professional":   {"bal_mean": 3000,  "bal_std": 1000, "fhs_mean": 55, "fhs_std": 10, "od_prob": 0.12},
    "near_retiree":         {"bal_mean": 15000, "bal_std": 4000, "fhs_mean": 72, "fhs_std": 9,  "od_prob": 0.03},
    "sme_seasonal":         {"bal_mean": 5000,  "bal_std": 2500, "fhs_mean": 50, "fhs_std": 13, "od_prob": 0.18},
    "sme_distressed":       {"bal_mean": 1200,  "bal_std": 800,  "fhs_mean": 28, "fhs_std": 10, "od_prob": 0.45},
}

N_CUSTOMERS = 1000
HIST_DAYS = 90
FORECAST_DAYS = 30

# ── Assign customers to segments ─────────────────────────────────────────────
seg_weights = [0.20, 0.15, 0.12, 0.10, 0.13, 0.08, 0.12, 0.10]
customer_ids = [f"NW{str(i).zfill(6)}" for i in range(1, N_CUSTOMERS + 1)]
customer_segments = np.random.choice(SEGMENTS, size=N_CUSTOMERS, p=seg_weights)

# ── 1. population.csv ────────────────────────────────────────────────────────
print("Generating population.csv ...")
rows = []
dates_hist = pd.date_range(end=pd.Timestamp.today().normalize(), periods=HIST_DAYS, freq="D")

for cid, seg in zip(customer_ids, customer_segments):
    p = SEG_PROFILES[seg]
    base = max(200, np.random.normal(p["bal_mean"], p["bal_std"]))
    trend = np.random.uniform(-15, 15)  # daily drift
    for i, dt in enumerate(dates_hist):
        noise = np.random.normal(0, p["bal_std"] * 0.08)
        bal = base + trend * (i - HIST_DAYS // 2) + noise
        # Add salary bumps (monthly)
        if dt.day in [25, 26, 27, 28]:
            bal += np.random.uniform(800, 3000)
        fhs = np.clip(np.random.normal(p["fhs_mean"], p["fhs_std"]) + (bal - p["bal_mean"]) / p["bal_mean"] * 10, 0, 100)
        runway = max(0, fhs / 100 * 60 + np.random.normal(0, 5))
        vel = max(0.3, np.random.normal(1.0, 0.25))
        rows.append({
            "customer_id": cid, "date": dt, "segment": seg,
            "balance": round(bal, 2),
            "fhs": round(fhs, 1),
            "liquidity_runway": round(runway, 1),
            "spend_velocity_ratio": round(vel, 2),
        })

pop = pd.DataFrame(rows)
pop.to_csv("data/population.csv", index=False)
print(f"  → {len(pop):,} rows")

# ── 2. population_meta.csv ───────────────────────────────────────────────────
print("Generating population_meta.csv ...")
meta_rows = []
for cid, seg in zip(customer_ids, customer_segments):
    p = SEG_PROFILES[seg]
    avg_fhs = np.clip(np.random.normal(p["fhs_mean"], p["fhs_std"]), 0, 100)
    tier = "Low" if avg_fhs >= 60 else ("Medium" if avg_fhs >= 35 else "High")
    green_days = int(np.clip(avg_fhs / 100 * 366, 0, 366))
    meta_rows.append({
        "customer_id": cid, "segment": seg,
        "risk_tier": tier,
        "avg_fhs": round(avg_fhs, 1),
        "green_days": green_days,
    })

meta = pd.DataFrame(meta_rows)
meta.to_csv("data/population_meta.csv", index=False)
print(f"  → {len(meta):,} rows")

# ── 3. segment_summary.csv ───────────────────────────────────────────────────
print("Generating segment_summary.csv ...")
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
seg_hist.to_csv("data/segment_summary.csv", index=False)
print(f"  → {len(seg_hist):,} rows")

# ── 4. forecasts.csv ─────────────────────────────────────────────────────────
print("Generating forecasts.csv ...")
dates_fore = pd.date_range(start=dates_hist[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")
fore_rows = []

for cid, seg in zip(customer_ids, customer_segments):
    p = SEG_PROFILES[seg]
    last_bal = pop[(pop.customer_id == cid)].sort_values("date").iloc[-1]["balance"]
    trend = np.random.uniform(-20, 10)
    for i, dt in enumerate(dates_fore):
        pred = last_bal + trend * i + np.random.normal(0, p["bal_std"] * 0.1)
        naive = last_bal + np.random.normal(0, p["bal_std"] * 0.05) * i * 0.3
        upper = pred + abs(np.random.normal(p["bal_std"] * 0.5, p["bal_std"] * 0.1))
        lower = pred - abs(np.random.normal(p["bal_std"] * 0.5, p["bal_std"] * 0.1))
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
fore.to_csv("data/forecasts.csv", index=False)
print(f"  → {len(fore):,} rows")

# ── 5. segment_forecasts.csv ──────────────────────────────────────────────────
print("Generating segment_forecasts.csv ...")
sfore_rows = []
for seg in SEGMENTS:
    sf = fore[fore.segment == seg]
    p = SEG_PROFILES[seg]
    for i, dt in enumerate(dates_fore):
        day_data = sf[sf.date == dt]
        if day_data.empty:
            continue
        med_bal = day_data.balance_pred.median()
        med_naive = day_data.naive_balance.median()
        p25 = day_data.balance_pred.quantile(0.25)
        p75 = day_data.balance_pred.quantile(0.75)
        # Use segment risk profile for realistic exposure values
        week_num = min(6, i // 7 + 1)
        base_exp = p["od_prob"] * 100  # base exposure from segment profile
        weekly_drift = week_num * np.random.uniform(1, 5)  # exposure grows over forecast
        liq_exp = round(np.clip(base_exp + weekly_drift + np.random.normal(0, 5), 0, 100), 1)
        stress_exp = round(np.clip(liq_exp * 1.3 + np.random.normal(3, 2), 0, 100), 1)
        med_fhs = np.clip(np.random.normal(p["fhs_mean"], p["fhs_std"] * 0.5), 0, 100)

        sfore_rows.append({
            "date": dt, "segment": seg,
            "forecast_week": min(6, i // 7 + 1),
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
seg_fore.to_csv("data/segment_forecasts.csv", index=False)
print(f"  → {len(seg_fore):,} rows")

# ── 6. anomalies.csv ─────────────────────────────────────────────────────────
print("Generating anomalies.csv ...")
anom_rows = []
for cid, seg in zip(customer_ids, customer_segments):
    p = SEG_PROFILES[seg]
    n_anoms = np.random.poisson(2 if p["od_prob"] > 0.2 else 0.5)
    for _ in range(n_anoms):
        dt = np.random.choice(dates_hist)
        actual = np.random.normal(p["bal_mean"] * 0.4, p["bal_std"])
        yhat = np.random.normal(p["bal_mean"], p["bal_std"] * 0.3)
        yhat_lower = yhat - abs(np.random.normal(p["bal_std"] * 0.6, 100))
        severity = "CRITICAL_EWS" if actual < yhat_lower else "HIGH_VARIANCE"
        anom_rows.append({
            "customer_id": cid, "date": dt, "segment": seg,
            "actual_balance": round(actual, 2),
            "yhat": round(yhat, 2),
            "yhat_lower": round(yhat_lower, 2),
            "anomaly_severity": severity,
        })

anoms = pd.DataFrame(anom_rows)
anoms.to_csv("data/anomalies.csv", index=False)
print(f"  → {len(anoms):,} rows")

# ── 7. forecast_meta.csv ─────────────────────────────────────────────────────
print("Generating forecast_meta.csv ...")
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
fmeta.to_csv("data/forecast_meta.csv", index=False)
print(f"  → {len(fmeta):,} rows")

print("\n✅ All 7 data files created in data/ directory!")
