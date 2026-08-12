"""
ai_engine.py — Gemini AI Integration
======================================
Handles Gemini 1.5 Flash setup, prompt engineering, and API calls.
Provides two prompt builders:
  - seg_ai()  : For bank risk analysts (segment-level insights)
  - cust_ai() : For customer advisors (personal financial guidance)
"""
import os
from dotenv import load_dotenv
from src.config import SEG_LABEL, SEG_OPP

load_dotenv()

# ── Initialize Gemini ────────────────────────────────────────────────────────
GEMINI = False
_MODEL = None

try:
    import google.generativeai as genai
    _key = os.getenv("GEMINI_API_KEY", "")
    if _key:
        genai.configure(api_key=_key)
        _MODEL = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI = True
except Exception:
    pass


# ── API Call with Fallback ───────────────────────────────────────────────────

def gemini_call(prompt: str, fallback: str) -> str:
    """
    Call Gemini API with a prompt. If unavailable or fails, return fallback.

    Parameters
    ----------
    prompt   : The structured prompt to send to Gemini
    fallback : Text to return if Gemini is unavailable

    Returns
    -------
    str : AI-generated text or fallback
    """
    if not GEMINI:
        return fallback
    try:
        return _MODEL.generate_content(prompt).text.strip()
    except Exception:
        return fallback


# ── Prompt Builders ──────────────────────────────────────────────────────────

def seg_ai_prompt(seg: str, pct_r: float, mfhs: float,
                  od: float, stress: float, n: int) -> str:
    """
    Build a prompt for segment-level risk analysis.
    Target audience: Head of Retail Risk (internal bank use).

    Parameters
    ----------
    seg     : Segment key (e.g. "stretched_salaried")
    pct_r   : Liquidity exposure percentage
    mfhs    : Median DES score
    od      : Average overdraft days
    stress  : Stress exposure percentage (10% shock)
    n       : Number of customers in segment
    """
    return f"""You are a senior NatWest risk analyst. Write exactly 2 sentences for the Head of Retail Risk.
Be specific with numbers. No bullet points. No jargon.
Segment: {SEG_LABEL.get(seg, seg)} ({n} customers)
Liquidity exposure: {pct_r:.0f}%  Stress exposure (10% shock): {stress:.0f}%
Median DES: {mfhs:.1f}/100  Avg overdraft days: {od:.1f}
Recommended product: {SEG_OPP.get(seg, '')}
Sentence 1: describe the risk with numbers.
Sentence 2: recommend a specific NatWest action."""


def cust_ai_prompt(seg: str, fhs: float, runway: float,
                   vel: float, od: int, bal14: float) -> str:
    """
    Build a prompt for customer-facing personal financial insight.
    Target audience: The customer themselves (plain English).

    Parameters
    ----------
    seg     : Segment key
    fhs     : Current DES score
    runway  : Liquidity runway in days
    vel     : Spend velocity ratio
    od      : Overdraft days predicted
    bal14   : Predicted balance in 14 days
    """
    return f"""You are a friendly NatWest advisor. Write 2 plain-English sentences for a customer.
Segment: {SEG_LABEL.get(seg, seg)}  DES: {fhs:.0f}/100  Runway: {runway:.0f}d
Spend velocity: {vel:.2f}x  Overdraft days: {od}  Balance in 14d: £{bal14:,.0f}
Sentence 1: what their money situation looks like.
Sentence 2: one practical action this week."""
