"""Energy optimisation dashboard.  Launch: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Energy RL Console", page_icon="⚡", layout="wide", initial_sidebar_state="auto")

from app import view_comparison, view_explorer, view_overview, view_research, view_robustness  # noqa: E402
from app.context import RESULTS_DIR, get_context  # noqa: E402
from app.style import inject_css  # noqa: E402

PAGES = {
    "Overview": view_overview,
    "Simulation explorer": view_explorer,
    "Policy comparison": view_comparison,
    "Robustness lab": view_robustness,
    "Research & methodology": view_research,
}


def main() -> None:
    inject_css()
    ctx = get_context()
    with st.sidebar:
        st.markdown('<div class="brand"><div class="mark">E</div><div><div class="t">Energy RL Console</div>'
                    '<div class="s">Load shifting for an industrial site</div></div></div>', unsafe_allow_html=True)
        page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
        st.markdown("---")
        seeds = sorted(ctx["rl"])
        if seeds:
            st.selectbox("RL model (training seed)", seeds, key="rl_seed",
                         help="Live views use this seed. Benchmark tables average all seeds.")
        else:
            st.session_state["rl_seed"] = None
            st.caption("No trained RL model loaded.")
        ready = (RESULTS_DIR / "episodes.csv").exists()
        st.caption(f"Models loaded: {len(seeds)} · Results: {'ready' if ready else 'not generated'}")
        st.caption("Simulation prototype on real 2019 French market, PV and load-shape data plus synthetic jobs and tariff structure. Not a certified controller.")
    PAGES[page].render()


main()
