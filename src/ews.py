"""
ews.py — Early Warning Signal Logic
=====================================
Builds the EWS alert console and provides the pill badge utility.
"""
import pandas as pd
from src.config import SEG_LABEL, EWS_ACTIONS


def pill(band: str) -> str:
    """
    Generate an HTML pill badge for a risk band.

    Parameters
    ----------
    band : "RED", "AMBER", or "GREEN"

    Returns
    -------
    str : HTML span element with styled pill
    """
    css_class = {
        "RED":   "pill-red",
        "AMBER": "pill-amber",
        "GREEN": "pill-green",
    }.get(band, "pill-green")
    return f'<span class="pill {css_class}">{band}</span>'


def build_ews_console(anoms_df: pd.DataFrame, seg_filter: list) -> pd.DataFrame:
    """
    Build the Priority 1 EWS alert table from anomaly data.

    Filters for CRITICAL_EWS severity, sorts by breach amount,
    and returns the top 10 most critical alerts with recommended actions.

    Parameters
    ----------
    anoms_df   : DataFrame from anomalies.csv
    seg_filter : List of segment keys to include

    Returns
    -------
    pd.DataFrame with columns:
        Priority, Customer, Segment, Date, Breach £, Recommended action
    """
    if anoms_df.empty:
        return pd.DataFrame()

    df = anoms_df[anoms_df.segment.isin(seg_filter)].copy() if seg_filter else anoms_df.copy()
    if df.empty:
        return pd.DataFrame()

    # Map severity to priority labels
    sev_col = "anomaly_severity" if "anomaly_severity" in df.columns else None
    df["Priority"] = (
        df[sev_col].map({"CRITICAL_EWS": "P1 — CRITICAL", "HIGH_VARIANCE": "P2 — HIGH"})
        if sev_col else "P2 — HIGH"
    )

    # Calculate how far below threshold the balance fell
    df["Breach £"] = (df["yhat_lower"] - df["actual_balance"]).round(0).clip(lower=0)

    # Map segment to recommended action
    df["Recommended action"] = df["segment"].map(EWS_ACTIONS).fillna("Monitor")
    df["Segment"] = df["segment"].map(SEG_LABEL)
    df["Date"] = pd.to_datetime(df["date"]).dt.strftime("%d %b %Y")

    # Return top 10 critical alerts sorted by breach amount
    out = (
        df[df["Priority"] == "P1 — CRITICAL"]
        .sort_values("Breach £", ascending=False)
        .head(10)
        [["Priority", "customer_id", "Segment", "Date", "Breach £", "Recommended action"]]
        .rename(columns={"customer_id": "Customer"})
    )
    return out
