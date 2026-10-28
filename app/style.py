"""Design tokens, global CSS and small HTML helpers for the dashboard."""
from __future__ import annotations

import streamlit as st

# Semantic colour tokens (one per policy / signal, reused by every chart)
C = {
    "naive": "#8A94A6", "rule": "#3B6FB6", "rl": "#0E8F7E",
    "price": "#2F5D9E", "carbon": "#6B7280", "pv": "#D9A404", "base": "#B8C0CC",
    "battery": "#7A5AC8", "grid": "#1B2430", "bad": "#C2410C", "good": "#0E7C6B",
    "ink": "#1B2430", "muted": "#5B6675", "line": "#E3E6EB",
}
POLICY_COLORS = {"Naive schedule": C["naive"], "Rule-based": C["rule"], "RL agent (PPO)": C["rl"]}
JOB_COLORS = ["#0E8F7E", "#3B6FB6", "#C46B1B", "#7A5AC8", "#B5446E", "#5E8C31", "#2A9DB5", "#8C6D1F"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; color: #1B2430; }
h1 { font-size: 1.65rem !important; font-weight: 650 !important; margin-bottom: .1rem; }
h2 { font-size: 1.2rem !important; font-weight: 620 !important; margin-top: 1.4rem; }
h3 { font-size: 1.02rem !important; font-weight: 600 !important; }
/* sidebar */
section[data-testid="stSidebar"] { background: #101A2B; }
section[data-testid="stSidebar"] * { color: #D5DCE8; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #FFFFFF; }
section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #1B2430 !important; }
section[data-testid="stSidebar"] label { font-size: .8rem; }
.brand { display:flex; gap:.6rem; align-items:center; margin: .2rem 0 1.1rem 0; }
.brand .mark { width: 30px; height: 30px; border-radius: 7px; background:#0E8F7E; display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; }
.brand .t { font-weight: 650; font-size: .95rem; line-height: 1.15; color:#fff; }
.brand .s { font-size: .72rem; color:#93A0B5; }
/* metric cards */
.kpi { background:#fff; border:1px solid #E3E6EB; border-radius:10px; padding: .95rem 1.05rem; height:100%; }
.kpi .l { font-size:.74rem; color:#5B6675; text-transform:uppercase; letter-spacing:.05em; font-weight:600; }
.kpi .v { font-size:1.65rem; font-weight:650; color:#1B2430; font-variant-numeric: tabular-nums; margin-top:.15rem; }
.kpi .u { font-size:.85rem; font-weight:500; color:#5B6675; margin-left:.2rem; }
.kpi .d { font-size:.8rem; margin-top:.2rem; font-variant-numeric: tabular-nums; }
.kpi .d.good { color:#0E7C6B; } .kpi .d.bad { color:#C2410C; } .kpi .d.flat { color:#5B6675; }
.pill { display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.72rem; font-weight:600; border:1px solid #E3E6EB; background:#fff; color:#3A4556; margin-right:.3rem; }
.pill.syn { background:#FFF4E5; border-color:#F3D4A6; color:#8A4B08; }
.pill.pub { background:#E8F5F2; border-color:#BFE3DB; color:#0B5E51; }
.pill.der { background:#EAF0FA; border-color:#C5D5EE; color:#264C85; }
.note { background:#fff; border:1px solid #E3E6EB; border-left:3px solid #0E8F7E; border-radius:6px; padding:.7rem .95rem; font-size:.88rem; color:#2B3544; }
.note.warn { border-left-color:#C2410C; }
.sub { color:#5B6675; font-size:.92rem; margin-bottom:.8rem; }
div[data-testid="stDataFrame"] { border:1px solid #E3E6EB; border-radius:8px; }
.stButton > button { border-radius:8px; font-weight:600; }
:focus-visible { outline: 2px solid #0E8F7E !important; outline-offset: 2px; }
@media (max-width: 760px) { .kpi .v { font-size:1.35rem; } .block-container { padding-left:1rem; padding-right:1rem; } }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(label: str, value: str, unit: str = "", delta: str = "", tone: str = "flat") -> str:
    d = f'<div class="d {tone}">{delta}</div>' if delta else ""
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}<span class="u">{unit}</span></div>{d}</div>'


def pct_tone(pct: float, good_if_positive: bool = True) -> str:
    if abs(pct) < 0.5:
        return "flat"
    return "good" if (pct > 0) == good_if_positive else "bad"


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="note {"warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.markdown(f"# {title}")
    st.markdown(f'<div class="sub">{subtitle}</div>', unsafe_allow_html=True)
