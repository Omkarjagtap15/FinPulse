"""
config.py — Constants & Segment Definitions
=============================================
All segment names, labels, product opportunities, EWS actions,
and color constants used across the dashboard.
"""

# ── 8 Customer Segments ──────────────────────────────────────────────────────
SEGMENTS = [
    "stable_salaried", "stretched_salaried", "gig_worker", "freelancer",
    "young_professional", "near_retiree", "sme_seasonal", "sme_distressed",
]

# ── Human-Readable Labels ────────────────────────────────────────────────────
SEG_LABEL = {
    "stable_salaried":    "Stable Salaried",
    "stretched_salaried": "Stretched Salaried",
    "gig_worker":         "Gig Worker",
    "freelancer":         "Freelancer",
    "young_professional": "Young Professional",
    "near_retiree":       "Near Retiree",
    "sme_seasonal":       "SME Seasonal",
    "sme_distressed":     "SME Distressed",
}

# ── Product Opportunity per Segment ──────────────────────────────────────────
SEG_OPP = {
    "stable_salaried":    "Savings & investment products",
    "stretched_salaried": "Micro-buffer / overdraft intervention",
    "gig_worker":         "Flexible credit, income smoothing",
    "freelancer":         "Invoice financing, cash flow loan",
    "young_professional": "Wealth onboarding, ISA products",
    "near_retiree":       "Wealth management, pension advice",
    "sme_seasonal":       "Working capital loan (trough months)",
    "sme_distressed":     "Early intervention, loan restructuring",
}

# ── Recommended EWS Actions per Segment ──────────────────────────────────────
EWS_ACTIONS = {
    "stretched_salaried": "Offer £200 interest-free micro-buffer before payday gap",
    "gig_worker":         "Trigger income-smoothing product outreach",
    "freelancer":         "Flag for invoice financing conversation",
    "sme_distressed":     "Escalate to relationship manager — Priority 1",
    "young_professional": "Send ISA / savings prompt — churn prevention",
    "near_retiree":       "Monitor only — low risk",
    "stable_salaried":    "Monitor only — low risk",
    "sme_seasonal":       "Pre-approve working capital facility for trough period",
}

# ── Color Palette ────────────────────────────────────────────────────────────
PRIMARY   = "#378ADD"   # NatWest blue — forecasts, main lines
NEUTRAL   = "#B4B2A9"   # Gray — naive baseline
BAND      = "rgba(55,138,221,0.10)"  # Light blue — confidence bands
RED       = "#E24B4A"   # Danger — overdraft, critical EWS
AMBER     = "#EF9F27"   # Warning — high variance
GREEN     = "#639922"   # Healthy — good DES scores
