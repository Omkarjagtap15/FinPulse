"""
heatmap.py — Segment Liquidity Exposure Heatmap
=================================================
Visualizes the % of customers breaching their risk tolerance
threshold per segment per forecast week.

Color scale: Green (0%) → Yellow (30%) → Pink (60%) → Red (100%)
"""
import plotly.graph_objects as go
from src.config import SEGMENTS, SEG_LABEL


def chart_heatmap(sf, segs):
    """
    Build an interactive heatmap showing liquidity exposure by segment × week.

    Parameters
    ----------
    sf   : DataFrame — filtered segment_forecasts data
    segs : list — segment keys to include

    Returns
    -------
    go.Figure — Plotly heatmap figure
    """
    if sf.empty:
        return go.Figure()

    df = sf[sf.segment.isin(segs)].copy()
    val_col = "liquidity_exposure_pct" if "liquidity_exposure_pct" in df.columns else "pct_fhs_danger"
    df["week"] = df["forecast_week"].astype(int)

    # Aggregate: average exposure and DES per segment per week
    wk = (
        df.groupby(["segment", "week"])
        .agg(danger=(val_col, "mean"), fhs=("median_fhs", "mean"))
        .reset_index()
    )
    weeks = sorted(wk["week"].unique())
    sl = [s for s in SEGMENTS[::-1] if s in segs]  # reverse order for y-axis

    # Build Z (values) and T (hover text) matrices
    Z, T = [], []
    for s in sl:
        row_z, row_t = [], []
        for w in weeks:
            row = wk[(wk.segment == s) & (wk.week == w)]
            v = row["danger"].values[0] if len(row) else 0
            f = row["fhs"].values[0] if len(row) else 50
            row_z.append(round(v, 1))
            row_t.append(
                f"{SEG_LABEL.get(s, s)}<br>Week {w}<br>"
                f"Liquidity exposure: {v:.0f}%<br>Median DES: {f:.1f}"
            )
        Z.append(row_z)
        T.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=Z,
        x=[f"Wk {w}" for w in weeks],
        y=[SEG_LABEL.get(s, s) for s in sl],
        text=T,
        hoverinfo="text",
        texttemplate="%{z:.0f}%",
        textfont=dict(size=11, color="#333"),
        colorscale=[
            [0,   "#EAF3DE"],   # Green  — 0% exposure
            [0.3, "#FAEEDA"],   # Yellow — 30%
            [0.6, "#F7C1C1"],   # Pink   — 60%
            [1.0, "#E24B4A"],   # Red    — 100%
        ],
        zmin=0, zmax=100,
        showscale=True,
        xgap=3, ygap=3,       # Visible grid lines between cells
        colorbar=dict(
            title="% exposure",
            tickfont=dict(size=10),
            thickness=12, len=0.8,
        ),
    ))

    fig.update_layout(
        height=290,
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(side="top", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
