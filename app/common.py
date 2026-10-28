"""Helpers shared by dashboard views."""
from __future__ import annotations

import streamlit as st

from app.context import get_context, policies_for
from app.style import note
from energyrl.evaluate import rollout
from energyrl.metrics import RL_LABEL
from energyrl.ood import ood_families


def family_titles() -> dict[str, str]:
    ctx = get_context()
    from energyrl.scenarios import DistSpec
    t = {"in_distribution": "In-distribution (held-out test days)"}
    for f in ood_families(DistSpec.from_config(ctx["cfg"])):
        t[f.key] = f"OOD: {f.title}"
    return t


def run_all(scenario, seed: int | None) -> dict[str, dict]:
    """Roll out every available policy on one scenario (real simulation, real trained model)."""
    return {name: rollout(pol, scenario, get_context()["cfg"]) for name, pol in policies_for(seed).items()}


def rl_seed() -> int | None:
    return st.session_state.get("rl_seed")


def require_ready() -> bool:
    """Show an actionable empty/error state and return False if the app cannot run."""
    ctx = get_context()
    if ctx["error"]:
        note(f"<b>Data not prepared.</b> {ctx['error']}<br>Run <code>python -m energyrl.prepare_data</code> and reload.", warn=True)
        return False
    if not ctx["rl"]:
        extra = "<br>".join(ctx.get("model_errors", []))
        note("<b>No trained RL model found</b> - baselines are shown, RL columns are hidden. "
             "Train with <code>python -m energyrl.train</code> (or the control panel under Research &amp; methodology)."
             + (f"<br>{extra}" if extra else ""), warn=True)
    return True


def has_rl(results: dict) -> bool:
    return RL_LABEL in results


def pct(a: float, b: float) -> float:
    """Percent reduction of a relative to b (positive = a is lower)."""
    return 100.0 * (1.0 - a / b) if b else 0.0


def vs(p: float, ref: str) -> str:
    """Human wording for a percent reduction: 'x% lower than naive' / 'x% higher than naive'."""
    return f"{abs(p):.1f}% {'lower' if p >= 0 else 'higher'} than {ref}"
