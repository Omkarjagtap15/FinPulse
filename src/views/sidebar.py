"""
sidebar.py — Dashboard Sidebar
================================
Renders the navigation sidebar with view selector, segment filters,
customer picker, stress test input, and Gemini status indicator.
"""
import streamlit as st
from src.config import SEGMENTS, SEG_LABEL
from src.ai_engine import GEMINI


def render_sidebar(meta):
    """
    Render the sidebar and return a dict of user-selected state.

    Parameters
    ----------
    meta : DataFrame — population_meta (for customer ID lookup)

    Returns
    -------
    dict with keys depending on the selected view:
        Always:
            - view : str ("Population risk intelligence" or "Customer exposure detail")
        If Population view:
            - seg_filter : list of segment keys
        If Customer view:
            - seg_choice  : str — selected segment
            - cid_choice  : str — selected customer ID
            - scenario    : float — stress test amount (£)
    """
    state = {}

    with st.sidebar:
        st.markdown("### 🏦 FinPulse")
        st.caption("Dynamic Exposure Monitor · 1,000 customers · 8 segments")
        st.divider()

        # ── View selector ────────────────────────────────────────────────
        state["view"] = st.radio(
            "Dashboard view",
            ["Population risk intelligence", "Customer exposure detail"],
        )
        st.divider()

        # ── View-specific controls ───────────────────────────────────────
        if "Population" in state["view"]:
            st.markdown("**Filters**")
            state["seg_filter"] = st.multiselect(
                "Segments",
                options=SEGMENTS, default=SEGMENTS,
                format_func=lambda s: SEG_LABEL.get(s, s),
            )
            st.slider("Forecast weeks", 1, 6, 4)
        else:
            state["seg_choice"] = st.selectbox(
                "Segment", SEGMENTS,
                format_func=lambda s: SEG_LABEL.get(s, s),
            )
            cids = (
                meta[meta.segment == state["seg_choice"]]["customer_id"].tolist()
                if not meta.empty else []
            )
            state["cid_choice"] = st.selectbox("Customer ID", cids)
            st.divider()
            st.markdown("**Scenario — liquidity stress test**")
            state["scenario"] = st.number_input(
                "Immediate spend (£)",
                min_value=0, max_value=20000, value=0, step=100,
            )
            if state["scenario"] > 0:
                st.caption(f"Forecast shifted down £{state['scenario']:,} — capital outflow simulation.")

        # ── Footer ───────────────────────────────────────────────────────
        st.divider()
        if GEMINI:
            st.success("Gemini AI insights active")
        else:
            st.warning("AI Insights offline (API Key required).")
        st.caption("Apache 2.0 · DCO signed")

    return state
