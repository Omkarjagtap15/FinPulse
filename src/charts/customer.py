"""
customer.py — Customer-Level Charts
=====================================
Two chart functions:
  - chart_customer() : 30-day balance forecast with EWS markers
  - chart_des()      : DES history sparkline with GREEN/RED thresholds
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.config import PRIMARY, NEUTRAL, BAND, RED, AMBER, GREEN


def chart_customer(cid, pop, fore, anoms, scenario=0.0):
    """
    Build the customer-level 30-day balance forecast chart.

    Parameters
    ----------
    cid      : str — customer ID
    pop      : DataFrame — population (historical balances)
    fore     : DataFrame — forecasts (30-day predictions)
    anoms    : DataFrame — anomalies
    scenario : float — stress test amount to subtract from forecast

    Returns
    -------
    go.Figure with traces:
        - Actual balance (solid blue)
        - Confidence band (light blue fill)
        - Forecast (dashed blue)
        - Naive baseline (dotted gray)
        - Stressed scenario (red long-dash)
        - Capital adequacy line (£0)
        - Overdraft risk zone (red shaded)
        - EWS markers (red dots = critical, amber circles = high variance)
    """
    fig = go.Figure()

    # ── Historical balance ───────────────────────────────────────────────
    h = pop[pop.customer_id == cid].sort_values("date")
    if h.empty:
        return fig
    fig.add_trace(go.Scatter(
        x=h.date, y=h.balance.round(0),
        name="Actual balance",
        line=dict(color=PRIMARY, width=2),
        hovertemplate="£%{y:,.0f}<extra>Actual</extra>",
    ))

    # ── Forecast traces ──────────────────────────────────────────────────
    f = fore[fore.customer_id == cid].sort_values("date")
    if not f.empty:
        adj = scenario

        # 80% confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([f.date, f.date[::-1]]),
            y=pd.concat([(f.balance_upper - adj).round(0),
                         (f.balance_lower - adj).round(0)[::-1]]),
            fill="toself", fillcolor=BAND,
            line=dict(color="rgba(0,0,0,0)"),
            name="Risk tolerance threshold (80% CI)",
            hoverinfo="skip",
        ))

        # Prophet forecast
        fig.add_trace(go.Scatter(
            x=f.date, y=(f.balance_pred - adj).round(0),
            name="Liquidity risk forecast",
            line=dict(color=PRIMARY, width=2, dash="dash"),
            hovertemplate="£%{y:,.0f}<extra>Forecast</extra>",
        ))

        # Naive baseline
        fig.add_trace(go.Scatter(
            x=f.date, y=(f.naive_balance - adj).round(0),
            name="Market volatility baseline",
            line=dict(color=NEUTRAL, width=1.5, dash="dot"),
            hovertemplate="£%{y:,.0f}<extra>Baseline</extra>",
        ))

        # 10% stress scenario
        if "balance_stressed" in f.columns:
            fig.add_trace(go.Scatter(
                x=f.date, y=(f.balance_stressed - adj).round(0),
                name="10% income stress scenario",
                line=dict(color=RED, width=1.2, dash="longdash"),
                hovertemplate="£%{y:,.0f}<extra>Stressed</extra>",
            ))

        # Capital adequacy threshold (£0)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="rgba(226,75,74,0.45)", line_width=1.5,
            annotation_text="Capital adequacy threshold (£0)",
            annotation_font_size=9, annotation_font_color=RED,
        )

        # Overdraft risk zone (where lower bound < £0)
        below = f[(f.balance_lower - adj) < 0]
        if not below.empty:
            fig.add_trace(go.Scatter(
                x=pd.concat([below.date, below.date[::-1]]),
                y=pd.concat([
                    pd.Series(np.zeros(len(below))),
                    (below.balance_lower - adj).clip(upper=0).round(0)[::-1],
                ]),
                fill="toself", fillcolor="rgba(226,75,74,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Overdraft risk zone",
                hoverinfo="skip",
            ))

        # Forecast start vertical line
        _vx = f.date.min().isoformat()
        fig.add_shape(
            type="line", x0=_vx, x1=_vx, y0=0, y1=1, yref="paper",
            line=dict(dash="dot", color="gray", width=1),
        )
        fig.add_annotation(
            x=_vx, y=1, yref="paper",
            text="forecast →",
            font=dict(size=10), showarrow=False, yshift=5,
        )

    # ── Anomaly markers ──────────────────────────────────────────────────
    if not anoms.empty:
        a = anoms[anoms.customer_id == cid]
        sev_col = "anomaly_severity" if "anomaly_severity" in a.columns else None
        crit = a[a[sev_col] == "CRITICAL_EWS"] if sev_col else pd.DataFrame()
        high = a[a[sev_col] == "HIGH_VARIANCE"] if sev_col else a

        if not crit.empty:
            fig.add_trace(go.Scatter(
                x=crit.date, y=crit.actual_balance.round(0),
                mode="markers", name="EWS — CRITICAL",
                marker=dict(color=RED, size=9, symbol="circle",
                            line=dict(width=2, color=RED)),
                hovertemplate="CRITICAL EWS: £%{y:,.0f}<extra></extra>",
            ))
        if not high.empty:
            fig.add_trace(go.Scatter(
                x=high.date, y=high.actual_balance.round(0),
                mode="markers", name="EWS — HIGH VARIANCE",
                marker=dict(color=AMBER, size=9, symbol="circle-open",
                            line=dict(width=2, color=AMBER)),
                hovertemplate="HIGH VARIANCE: £%{y:,.0f}<extra></extra>",
            ))

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=10)),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="rgba(0,0,0,0.05)", tickformat="£,.0f", tickfont=dict(size=10)),
    )
    return fig


def chart_des(cid, pop):
    """
    Build the DES (Dynamic Exposure Score) history sparkline.

    Parameters
    ----------
    cid : str — customer ID
    pop : DataFrame — population data with 'fhs' column

    Returns
    -------
    go.Figure — area chart with GREEN (60) and RED (35) threshold lines
    """
    h = pop[pop.customer_id == cid].sort_values("date")
    if h.empty:
        return go.Figure()

    fig = go.Figure(go.Scatter(
        x=h.date, y=h.fhs.round(1),
        fill="tozeroy",
        line=dict(color=PRIMARY, width=1.5),
        fillcolor=BAND,
        hovertemplate="DES: %{y:.1f}<extra></extra>",
        name="DES",
    ))

    # Threshold lines
    fig.add_hline(y=60, line_dash="dot", line_color=GREEN, line_width=1,
                  annotation_text="GREEN", annotation_font_size=9)
    fig.add_hline(y=35, line_dash="dot", line_color=RED, line_width=1,
                  annotation_text="RED", annotation_font_size=9)

    fig.update_layout(
        height=155,
        margin=dict(l=0, r=0, t=4, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(range=[0, 100], tickfont=dict(size=9),
                   gridcolor="rgba(0,0,0,0.05)"),
    )
    return fig
