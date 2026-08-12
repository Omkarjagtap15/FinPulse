"""
customer.py — View 2: Customer Exposure Detail
=================================================
Individual customer drill-down for advisors and relationship managers.

Sections:
  A) Customer KPI Cards (balance, DES, runway, velocity)
  B) Active EWS Banners (overdraft risk, seasonal spike)
  C) 30-Day Balance Forecast Chart
  D) DES History Sparkline
  E) 30-Day Forecast Summary (sidebar metrics)
  F) Recent EWS Events Table
  G) AI Personal Insight (Gemini)
"""
import pandas as pd
import streamlit as st
from src.config import SEG_LABEL, SEG_OPP, EWS_ACTIONS, RED, GREEN
from src.charts import chart_customer, chart_des
from src.ews import pill
from src.ai_engine import gemini_call, cust_ai_prompt


def render_customer_view(data, seg_choice, cid, scenario):
    """
    Render the full Customer Exposure Detail dashboard.

    Parameters
    ----------
    data       : dict — all loaded DataFrames
    seg_choice : str — selected segment key
    cid        : str — selected customer ID
    scenario   : float — stress test amount (£)
    """
    pop   = data["population"]
    meta  = data["meta"]
    fore  = data["forecasts"]
    anoms = data["anomalies"]
    fmeta = data["forecast_meta"]

    # ── Load customer metadata ───────────────────────────────────────────
    cmeta = (
        meta[meta.customer_id == cid].iloc[0]
        if not meta.empty and cid in meta.customer_id.values
        else {}
    )
    cfm = (
        fmeta[fmeta.customer_id == cid].iloc[0]
        if not fmeta.empty and cid in fmeta.customer_id.values
        else {}
    )

    st.markdown(f"## Customer exposure detail — {cid}")
    st.caption(
        f"Segment: {SEG_LABEL.get(seg_choice, seg_choice)} · "
        f"Risk tier: {cmeta.get('risk_tier', '—')} · "
        f"Product opportunity: {SEG_OPP.get(seg_choice, '—')}"
    )

    # ── Extract latest values ────────────────────────────────────────────
    ch = pop[pop.customer_id == cid].sort_values("date")
    last_bal = float(ch["balance"].iloc[-1]) if not ch.empty else 0
    last_fhs = float(ch["fhs"].iloc[-1]) if not ch.empty else 0
    last_run = float(ch["liquidity_runway"].iloc[-1]) if not ch.empty else 0
    last_vel = float(ch["spend_velocity_ratio"].iloc[-1]) if not ch.empty else 1
    od_days = int(cfm.get("overdraft_days", 0)) if cfm is not None and len(cfm) else 0
    fhs_trend = float(cfm.get("fhs_trend", 0)) if cfm is not None and len(cfm) else 0
    band = "RED" if last_fhs < 35 else ("AMBER" if last_fhs < 60 else "GREEN")

    # ═══════════════════════════════════════════════════════════════════════
    # A) KPI SUMMARY CARDS
    # ═══════════════════════════════════════════════════════════════════════
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current balance", f"£{last_bal:,.0f}")
    k2.metric("Dynamic Exposure Score (DES)", f"{last_fhs:.1f} / 100",
              delta=f"{fhs_trend:+.1f} 30d trend")
    k3.metric("Liquidity runway", f"{last_run:.0f} days")
    k4.metric("Spend velocity", f"{last_vel:.2f}×",
              delta="above norm — watch" if last_vel > 1.3 else "within norm",
              delta_color="inverse" if last_vel > 1.3 else "normal")

    # ═══════════════════════════════════════════════════════════════════════
    # B) ACTIVE EWS BANNERS
    # ═══════════════════════════════════════════════════════════════════════
    cust_fore = fore[fore.customer_id == cid].sort_values("date") if not fore.empty else pd.DataFrame()
    has_od = (not cust_fore.empty and "ews_overdraft_flag" in cust_fore.columns
              and (cust_fore["ews_overdraft_flag"] == "🚨 CRITICAL").any())
    has_sp = (not cust_fore.empty and "ews_seasonal_spike" in cust_fore.columns
              and (cust_fore["ews_seasonal_spike"] == "⚠️ SPIKE DETECTED").any())

    if has_od or has_sp:
        st.divider()
        st.markdown('<p class="sec-head">Active Early Warning Signals</p>', unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)

        with fc1:
            if has_od:
                od_c = (cust_fore["ews_overdraft_flag"] == "🚨 CRITICAL").sum()
                action = EWS_ACTIONS.get(seg_choice, "Contact customer")
                st.markdown(f"""<div class="ews-critical">
                  <div class="ews-label ews-crit-label">Priority 1 — Overdraft Risk / Capital Adequacy Breach</div>
                  <div class="ews-text">Capital adequacy threshold breached on
                  <strong>{od_c} of the next 30 forecast days</strong>.
                  Lower bound of risk tolerance threshold crosses £0.<br>
                  <strong>Recommended action:</strong> {action}</div>
                </div>""", unsafe_allow_html=True)

        with fc2:
            if has_sp:
                sp_c = (cust_fore["ews_seasonal_spike"] == "⚠️ SPIKE DETECTED").sum()
                st.markdown(f"""<div class="ews-high">
                  <div class="ews-label ews-high-label">Seasonal Spike Expected</div>
                  <div class="ews-text">Forecast balance exceeds
                  <strong>120% of historical median on {sp_c} days</strong>.
                  Likely seasonal income surge — monitor for lifestyle inflation
                  or investment product opportunity.</div>
                </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # C) FORECAST CHART + D) DES HISTORY + E–G) SIDEBAR
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    col_main, col_side = st.columns([1.6, 1], gap="large")

    with col_main:
        # ── C) Balance forecast ──────────────────────────────────────────
        st.markdown("#### 30-day liquidity risk forecast")
        if scenario > 0:
            st.info(f"Stress scenario active: forecast shifted -£{scenario:,} (immediate capital outflow)")
        st.plotly_chart(
            chart_customer(cid, pop, fore, anoms, scenario),
            use_container_width=True,
        )
        st.caption(
            "Solid = actual balance · Dashed = liquidity risk forecast · "
            "Dotted = market volatility baseline · Red long-dash = 10% stress scenario · "
            "Red filled dots = CRITICAL EWS · Amber open circles = HIGH VARIANCE EWS · "
            "Red shaded zone = predicted capital adequacy breach"
        )

        # ── D) DES history ───────────────────────────────────────────────
        st.markdown("#### Dynamic Exposure Score (DES) history")
        st.plotly_chart(chart_des(cid, pop), use_container_width=True)
        st.caption(
            "DES = 40% balance buffer + 25% liquidity runway + "
            "20% spend velocity + 15% income gap · Green ≥60 · Amber 35–60 · Red <35"
        )

    with col_side:
        # ── Exposure band pill ───────────────────────────────────────────
        st.markdown(f"**Exposure band:** {pill(band)}", unsafe_allow_html=True)
        st.divider()

        # ── E) 30-day forecast summary ───────────────────────────────────
        if cfm is not None and len(cfm):
            st.markdown('<p class="sec-head">30-day forecast summary</p>', unsafe_allow_html=True)
            st.markdown(f"DES in 14 days: **{cfm.get('forecast_fhs_day14', '—'):.1f}**")
            st.markdown(f"DES in 30 days: **{cfm.get('forecast_fhs_day30', '—'):.1f}**")
            st.markdown(f"Min forecast balance: **£{cfm.get('min_forecast_balance', 0):,.0f}**")
            st.markdown(f"Overdraft days predicted: **{od_days}**")
            st.markdown(f"Historical EWS events: **{int(cfm.get('n_anomalies', 0))}**")

        if not cust_fore.empty and "balance_stressed" in cust_fore.columns:
            min_s = (cust_fore["balance_stressed"] - scenario).min()
            col = RED if min_s < 0 else GREEN
            lbl = "BREACH under stress" if min_s < 0 else "Within stress tolerance"
            st.markdown(
                f"Min stressed balance: <strong style='color:{col}'>£{min_s:,.0f}</strong> — {lbl}",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── F) Recent EWS events ─────────────────────────────────────────
        if not anoms.empty:
            ca = anoms[anoms.customer_id == cid].sort_values("date", ascending=False).head(5)
            if not ca.empty:
                st.markdown('<p class="sec-head">Recent EWS events</p>', unsafe_allow_html=True)
                ca2 = ca.copy()
                ca2["Date"] = ca2["date"].dt.strftime("%d %b")
                ca2["Actual £"] = ca2["actual_balance"].round(0)
                ca2["Exp £"] = ca2["yhat"].round(0)
                disp = ["Date", "Actual £", "Exp £"]
                if "anomaly_severity" in ca2.columns:
                    ca2["Severity"] = ca2["anomaly_severity"]
                    disp.append("Severity")
                st.dataframe(ca2[disp], hide_index=True, use_container_width=True, height=170)

        st.divider()

        # ── G) AI personal insight ───────────────────────────────────────
        st.markdown('<p class="sec-head">AI personal insight</p>', unsafe_allow_html=True)
        if st.button("Generate ↗", key="cust_ai"):
            bal14 = (
                float(cust_fore["balance_pred"].iloc[13])
                if not cust_fore.empty and len(cust_fore) >= 14 else 0
            )
            fb = (
                f"Your DES is {last_fhs:.0f}/100 with {last_run:.0f} days of liquidity runway. "
                f"{'Consider reducing spend this week.' if last_fhs < 50 else 'Keep up the good habits.'}"
            )
            st.session_state[f"cai_{cid}"] = gemini_call(
                cust_ai_prompt(seg_choice, last_fhs, last_run, last_vel, od_days, bal14), fb
            )

        if st.session_state.get(f"cai_{cid}"):
            st.markdown(
                f"""<div class="ai-box"><div class="ai-label">Gemini · personal insight</div>
                <div class="ai-text">{st.session_state[f"cai_{cid}"]}</div></div>""",
                unsafe_allow_html=True,
            )
