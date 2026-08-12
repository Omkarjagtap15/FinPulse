"""
styles.py — Custom CSS Styling
================================
All custom CSS injected into the Streamlit dashboard.
Includes pill badges, EWS banners, AI insight boxes, and section headers.
"""
import streamlit as st


def inject_styles():
    """Inject all custom CSS into the Streamlit page."""
    st.markdown("""
    <style>
    /* ── Layout ─────────────────────────────────────────────── */
    .block-container { padding-top: 1rem; }
    #MainMenu, footer { visibility: hidden; }

    /* ── Metric Cards ───────────────────────────────────────── */
    [data-testid="stMetricLabel"] { font-size: .75rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }

    /* ── Pill Badges (RED / AMBER / GREEN) ───────────────────── */
    .pill {
        display: inline-block;
        padding: 2px 11px;
        border-radius: 999px;
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .03em;
    }
    .pill-red   { background: #FCEBEB; color: #A32D2D; }
    .pill-amber { background: #FAEEDA; color: #854F0B; }
    .pill-green { background: #EAF3DE; color: #3B6D11; }

    /* ── EWS Banners ────────────────────────────────────────── */
    .ews-critical {
        background: #FCEBEB;
        border-left: 4px solid #E24B4A;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 4px 0;
    }
    .ews-high {
        background: #FAEEDA;
        border-left: 4px solid #EF9F27;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 4px 0;
    }
    .ews-label {
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .06em;
        text-transform: uppercase;
        margin-bottom: 3px;
    }
    .ews-crit-label { color: #A32D2D; }
    .ews-high-label { color: #854F0B; }
    .ews-text { font-size: .82rem; line-height: 1.5; }

    /* ── AI Insight Box ─────────────────────────────────────── */
    .ai-box {
        background: #E6F1FB;
        border: 1px solid #B5D4F4;
        border-radius: 10px;
        padding: 14px 16px;
        margin-top: 8px;
    }
    .ai-label {
        font-size: .68rem;
        font-weight: 700;
        color: #185FA5;
        letter-spacing: .05em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .ai-text { font-size: .84rem; color: #042C53; line-height: 1.6; }

    /* ── Section Headers ────────────────────────────────────── */
    .sec-head {
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #888;
        margin: 0 0 6px 0;
    }
    </style>
    """, unsafe_allow_html=True)
