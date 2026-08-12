"""
population.py — View 1: Population Risk Intelligence
======================================================
Bank-wide risk overview for risk managers.

Sections:
  A) KPI Summary Cards (customers at risk, P1 signals, worst segment, upsell)
  B) Segment Liquidity Exposure Heatmap
  C) EWS Console — Priority 1 alerts
  D) Revenue Opportunity Signals
  E) Segment Deep-Dive with forecast chart
  F) AI Intervention Recommendation (Gemini)
  G) Population EWS Flags (overdraft risk + seasonal spikes)
"""
import pandas as pd
import streamlit as st
from src.config import SEGMENTS, SEG_LABEL, SEG_OPP, EWS_ACTIONS
from src.charts import chart_heatmap, chart_seg
from src.ews import build_ews_console
from src.ai_engine import gemini_call, seg_ai_prompt


def render_population_view(data, seg_filter):
    """
    Render the full Population Risk Intelligence dashboard.

    Parameters
    ----------
    data       : dict — all loaded DataFrames from data_loader
    seg_filter : list — selected segment keys from sidebar
    """
    pop        = data["population"]
    meta       = data["meta"]
    seg_hist   = data["segment_hist"]
    fore       = data["forecasts"]
    seg_fore   = data["segment_fore"]
    anoms      = data["anomalies"]
    fmeta      = data["forecast_meta"]

    st.markdown("## Dynamic Exposure Monitor")
    st.caption(
        "30-day liquidity risk forecast · 80% risk tolerance threshold · "
        "Prophet time-series + market volatility baseline comparison"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # A) KPI SUMMARY CARDS
    # ═══════════════════════════════════════════════════════════════════════
    if not fmeta.empty:
        at_risk = int((fmeta["overdraft_days"] > 0).sum())
        p1_count = (
            int((anoms["anomaly_severity"] == "CRITICAL_EWS").sum())
            if not anoms.empty and "anomaly_severity" in anoms.columns
            else len(anoms)
        )
        worst_seg = (
            fmeta.groupby("segment")["overdraft_days"].mean().idxmax()
            if "segment" in fmeta.columns else "—"
        )
        opp_count = (
            int((meta["risk_tier"] == "Low").sum())
            if "risk_tier" in meta.columns else 0
        )
    else:
        at_risk = p1_count = opp_count = 0
        worst_seg = "—"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers at liquidity risk", f"{at_risk:,}", "overdraft predicted ≥1 day")
    k2.metric("Priority 1 EWS signals", f"{p1_count:,}", "CRITICAL threshold breaches")
    k3.metric("Highest exposure segment", SEG_LABEL.get(worst_seg, worst_seg), "most overdraft days")
    k4.metric("Upsell opportunities", f"{opp_count:,}", "customers in GREEN tier")
    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # B) HEATMAP + C) EWS CONSOLE
    # ═══════════════════════════════════════════════════════════════════════
    col_heat, col_ews = st.columns([1.5, 1], gap="large")

    with col_heat:
        st.markdown(
            '<p class="sec-head">Segment liquidity exposure heatmap — 6-week forecast</p>',
            unsafe_allow_html=True,
        )
        st.caption("Cell = % of customers breaching risk tolerance threshold that week.")
        sf_filt = seg_fore[seg_fore.segment.isin(seg_filter)] if seg_filter else seg_fore
        st.plotly_chart(
            chart_heatmap(sf_filt, seg_filter or SEGMENTS),
            use_container_width=True,
        )

    with col_ews:
        st.markdown(
            '<p class="sec-head">Early Warning Signal (EWS) console — Priority 1</p>',
            unsafe_allow_html=True,
        )
        st.caption("CRITICAL = balance breached risk tolerance threshold. Immediate action required.")
        ews_df = build_ews_console(anoms, seg_filter or SEGMENTS)
        if not ews_df.empty:
            st.dataframe(ews_df, hide_index=True, use_container_width=True, height=245)
        else:
            st.info("No CRITICAL EWS signals in current segment filter.")

        # ── D) Revenue opportunity signals ───────────────────────────────
        st.markdown(
            '<p class="sec-head" style="margin-top:10px">Revenue opportunity signals</p>',
            unsafe_allow_html=True,
        )
        opp_rows = []
        for s in (seg_filter or SEGMENTS):
            sm = meta[meta.segment == s]
            if sm.empty:
                continue
            gp = round((sm["green_days"] / 366 * 100).mean(), 0) if "green_days" in sm.columns else 0
            opp_rows.append({
                "Segment": SEG_LABEL.get(s, s),
                "Green %": f"{gp:.0f}%",
                "Opportunity": SEG_OPP.get(s, ""),
            })
        st.dataframe(pd.DataFrame(opp_rows), hide_index=True, use_container_width=True, height=150)

    # ═══════════════════════════════════════════════════════════════════════
    # E) SEGMENT DEEP-DIVE
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("#### Segment liquidity risk deep-dive")
    seg_sel = st.selectbox(
        "Select segment",
        options=seg_filter or SEGMENTS,
        format_func=lambda s: SEG_LABEL.get(s, s),
        key="deep",
    )

    col_ch, col_st = st.columns([1.6, 1], gap="large")

    with col_ch:
        st.plotly_chart(chart_seg(seg_sel, seg_fore, seg_hist), use_container_width=True)
        st.caption(
            "Solid = historical · Dashed = liquidity risk forecast · "
            "Dotted = market volatility baseline · Red long-dash = 10% stress scenario · "
            "Band = risk tolerance threshold"
        )

    with col_st:
        if not fmeta.empty and "segment" in fmeta.columns:
            sm = fmeta[fmeta.segment == seg_sel]
            sf2 = seg_fore[seg_fore.segment == seg_sel]

            if not sm.empty:
                st.metric("DES at day 14", f"{sm['forecast_fhs_day14'].mean():.1f}",
                          delta=f"{sm['fhs_trend'].mean():+.1f} vs today")
                st.metric("DES at day 30", f"{sm['forecast_fhs_day30'].mean():.1f}")
                st.metric("Avg overdraft days (30d)", f"{sm['overdraft_days'].mean():.1f}")

            if not sf2.empty and "stress_exposure_pct" in sf2.columns:
                st.metric("Stress-test exposure (10% shock)",
                          f"{sf2['stress_exposure_pct'].mean():.1f}%", delta_color="inverse")

            if not sf2.empty and "market_divergence" in sf2.columns:
                div = sf2["market_divergence"].mean()
                if abs(div) > 50:
                    st.markdown(f"""<div class="ews-high">
                      <div class="ews-label ews-high-label">Market divergence signal</div>
                      <div class="ews-text">Prophet diverges from market baseline by
                      £{abs(div):,.0f} — possible structural shift (interest rate
                      impact or regional economic stress).</div></div>""", unsafe_allow_html=True)

        # ── F) AI Intervention Recommendation ────────────────────────────
        st.markdown(
            '<p class="sec-head" style="margin-top:10px">AI intervention recommendation</p>',
            unsafe_allow_html=True,
        )
        if st.button("Generate ↗", key="seg_ai"):
            sf3 = seg_fore[seg_fore.segment == seg_sel]
            sm3 = fmeta[fmeta.segment == seg_sel] if not fmeta.empty else pd.DataFrame()
            pct_r = sf3["liquidity_exposure_pct"].mean() if not sf3.empty and "liquidity_exposure_pct" in sf3.columns else 0
            stress = sf3["stress_exposure_pct"].mean() if not sf3.empty and "stress_exposure_pct" in sf3.columns else 0
            mfhs = sf3["median_fhs"].mean() if not sf3.empty else 50
            od = sm3["overdraft_days"].mean() if not sm3.empty else 0
            nc = len(meta[meta.segment == seg_sel])
            fb = (f"{SEG_LABEL.get(seg_sel, seg_sel)}: {pct_r:.0f}% of {nc} customers in liquidity "
                  f"exposure zone. Recommend: {SEG_OPP.get(seg_sel, 'targeted intervention')}.")
            st.session_state[f"sai_{seg_sel}"] = gemini_call(
                seg_ai_prompt(seg_sel, pct_r, mfhs, od, stress, nc), fb
            )

        if st.session_state.get(f"sai_{seg_sel}"):
            st.markdown(f"""<div class="ai-box"><div class="ai-label">Gemini · risk intervention</div>
              <div class="ai-text">{st.session_state[f"sai_{seg_sel}"]}</div></div>""",
              unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # G) POPULATION EWS FLAGS
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("#### Early Warning Signal flags — population view")
    st.caption("Active OVERDRAFT RISK and SEASONAL SPIKE flags from the Prophet forecast engine.")

    col_od, col_sp = st.columns(2, gap="large")

    with col_od:
        st.markdown('<p class="sec-head">Overdraft risk flags</p>', unsafe_allow_html=True)
        if not fore.empty and "ews_overdraft_flag" in fore.columns:
            od_flags = (
                fore[fore["ews_overdraft_flag"] == "🚨 CRITICAL"]
                .groupby(["customer_id", "segment"]).size()
                .reset_index(name="critical_days")
                .sort_values("critical_days", ascending=False).head(8)
            )
            od_flags["Segment"] = od_flags["segment"].map(SEG_LABEL)
            od_flags["Action"] = od_flags["segment"].map(EWS_ACTIONS)
            st.dataframe(
                od_flags[["customer_id", "Segment", "critical_days", "Action"]]
                .rename(columns={"customer_id": "Customer", "critical_days": "Critical days"}),
                hide_index=True, use_container_width=True, height=230,
            )
        else:
            st.info("Run forecast_model.py to populate EWS flags.")

    with col_sp:
        st.markdown('<p class="sec-head">Seasonal spike detected</p>', unsafe_allow_html=True)
        if not fore.empty and "ews_seasonal_spike" in fore.columns:
            sp_flags = (
                fore[fore["ews_seasonal_spike"] == "⚠️ SPIKE DETECTED"]
                .groupby(["customer_id", "segment"]).size()
                .reset_index(name="spike_days")
                .sort_values("spike_days", ascending=False).head(8)
            )
            sp_flags["Segment"] = sp_flags["segment"].map(SEG_LABEL)
            sp_flags["Note"] = "Balance >120% of historical median"
            st.dataframe(
                sp_flags[["customer_id", "Segment", "spike_days", "Note"]]
                .rename(columns={"customer_id": "Customer", "spike_days": "Spike days"}),
                hide_index=True, use_container_width=True, height=230,
            )
        else:
            st.info("Run forecast_model.py to populate seasonal spike flags.")
