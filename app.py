"""
NatWest FinPulse — Dynamic Exposure Monitor
============================================
Main entry point. Run with:  streamlit run app.py

This file is intentionally slim — it only:
  1. Configures the Streamlit page
  2. Injects custom CSS styles
  3. Loads data (cached)
  4. Renders the sidebar
  5. Routes to the correct view
"""
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

from src.styles import inject_styles
from src.data_loader import load_all_data
from src.views import render_sidebar, render_population_view, render_customer_view

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="NatWest FinPulse · Dynamic Exposure Monitor",
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
