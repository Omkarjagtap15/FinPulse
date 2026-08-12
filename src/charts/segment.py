"""
segment.py — Segment Forecast Time-Series Chart
=================================================
Displays historical median balance vs. 30-day forecast for a segment.
Includes Prophet forecast, naive baseline, confidence band,
stress scenario, and market divergence annotation.
"""
import pandas as pd
import plotly.graph_objects as go
from src.config import PRIMARY, NEUTRAL, BAND, RED


def chart_seg(seg, sf, sh):
    """
    Build segment-level forecast chart with multiple overlaid traces.

    Parameters
    ----------
    seg : str — segment key (e.g. "stretched_salaried")
    sf  : DataFrame — segment_forecasts data
    sh  : DataFrame — segment_summary (historical) data

    Returns
    -------
    go.Figure with traces:
        - Historical median (solid blue)
        - Confidence band p25–p75 (light blue fill)
        - Forecast median (dashed blue)
        - Naive baseline (dotted gray)
        - 10% stress scenario (red long-dash)
        - Forecast start marker (vertical dotted line)
        - Market divergence annotation (if |divergence| > £100)
    """
    fig = go.Figure()

    # ── Historical line ──────────────────────────────────────────────────
    if not sh.empty:
        h = sh[sh.segment == seg].sort_values("date")
        fig.add_trace(go.Scatter(
            x=h.date, y=h.median_balance.round(0),
            name="Historical median",
            line=dict(color=PRIMARY, width=2),
            hovertemplate="£%{y:,.0f}<extra>Historical</extra>",
        ))

    # ── Forecast traces ──────────────────────────────────────────────────
    if not sf.empty:
        f = sf[sf.segment == seg].sort_values("date")

        # Confidence band (p25–p75)
        fig.add_trace(go.Scatter(
            x=pd.concat([f.date, f.date[::-1]]),
            y=pd.concat([f.p75_balance, f.p25_balance[::-1]]),
            fill="toself", fillcolor=BAND,
            line=dict(color="rgba(0,0,0,0)"),
            name="Risk tolerance threshold (p25–p75)",
            hoverinfo="skip",
        ))

        # Prophet forecast
        fig.add_trace(go.Scatter(
            x=f.date, y=f.median_balance.round(0),
            name="Liquidity risk forecast",
            line=dict(color=PRIMARY, width=2, dash="dash"),
            hovertemplate="£%{y:,.0f}<extra>Forecast</extra>",
        ))

        # Naive baseline
        fig.add_trace(go.Scatter(
            x=f.date, y=f.median_naive.round(0),
            name="Market volatility baseline (7d rolling)",
            line=dict(color=NEUTRAL, width=1.5, dash="dot"),
            hovertemplate="£%{y:,.0f}<extra>Baseline</extra>",
        ))

        # 10% stress scenario
        if "stress_exposure_pct" in f.columns:
            fig.add_trace(go.Scatter(
                x=f.date, y=(f.median_balance * 0.90).round(0),
                name="10% market stress scenario",
                line=dict(color=RED, width=1.2, dash="longdash"),
                hovertemplate="£%{y:,.0f}<extra>Stressed</extra>",
            ))

        # Forecast start vertical line
        _vx = f.date.min().isoformat()
        fig.add_shape(
            type="line", x0=_vx, x1=_vx, y0=0, y1=1, yref="paper",
            line=dict(dash="dot", color="gray", width=1),
        )
        fig.add_annotation(
            x=_vx, y=1, yref="paper",
            text="forecast start →",
            font=dict(size=10), showarrow=False, yshift=5,
        )

        # Market divergence annotation
        if "market_divergence" in f.columns:
            div = f.market_divergence.mean()
            if abs(div) > 100 and len(f) > 14:
                fig.add_annotation(
                    x=f.date.iloc[14].isoformat(),
                    y=f.median_balance.iloc[14],
                    text=f"{'⬆' if div > 0 else '⬇'} Model vs baseline divergence: £{abs(div):,.0f}",
                    showarrow=True, arrowhead=2,
                    font=dict(size=10, color="#185FA5"),
                    bgcolor="#E6F1FB", bordercolor="#B5D4F4",
                )

    fig.update_layout(
        height=270,
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=10)),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="rgba(0,0,0,0.05)", tickformat="£,.0f", tickfont=dict(size=10)),
    )
    return fig
