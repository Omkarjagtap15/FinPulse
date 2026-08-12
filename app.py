"""
NatWest FinPulse — Dynamic Exposure Monitor
============================================
Main entry point. Run with:  streamlit run app.py
FinPulse — Dynamic Exposure Monitor
=============================================
Orchestrator script that initializes Streamlit and routes to views.
"""
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

from src.styles import inject_styles
from src.data_loader import load_all_data
from src.views import render_sidebar, render_population_view, render_customer_view

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinPulse · Dynamic Exposure Monitor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Styles ────────────────────────────────────────────────────────────
inject_styles()

# ── Load Data ────────────────────────────────────────────────────────────────
data = load_all_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
state = render_sidebar(data["meta"])

# ── Route to View ────────────────────────────────────────────────────────────
if "Population" in state["view"]:
    render_population_view(data, state.get("seg_filter", []))
else:
    render_customer_view(
        data,
        seg_choice=state["seg_choice"],
        cid=state["cid_choice"],
        scenario=state.get("scenario", 0),
    )
