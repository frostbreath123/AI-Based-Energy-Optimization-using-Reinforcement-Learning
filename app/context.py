"""Cached resources shared by all dashboard pages (config, data, policies, results)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from energyrl.baselines import NaivePolicy, RuleBasedPolicy  # noqa: E402
from energyrl.config import MODELS_DIR, RESULTS_DIR, load_config  # noqa: E402
from energyrl.data.preprocess import load_processed  # noqa: E402
from energyrl.evaluate import RLPolicy, build_eval_scenarios  # noqa: E402
from energyrl.metrics import NAIVE, RL_LABEL, RULE  # noqa: E402
from energyrl.scenarios import ScenarioGenerator  # noqa: E402


@st.cache_resource(show_spinner="Loading data and models...")
def get_context() -> dict:
    cfg = load_config()
    ctx: dict = {"cfg": cfg, "error": None, "df": None, "gen": None, "rl": {}}
    try:
        ctx["df"] = load_processed()
        ctx["gen"] = ScenarioGenerator(ctx["df"], cfg)
    except FileNotFoundError as exc:
        ctx["error"] = str(exc)
        return ctx
    for seed in cfg["training"]["seeds"]:
        p = MODELS_DIR / f"ppo_seed{seed}" / "best_model.zip"
        if p.exists():
            try:
                ctx["rl"][seed] = RLPolicy(p, cfg, f"RL agent (seed {seed})")
            except RuntimeError as exc:
                ctx.setdefault("model_errors", []).append(str(exc))
    return ctx


@st.cache_data(show_spinner=False)
def eval_scenarios() -> dict:
    ctx = get_context()
    return build_eval_scenarios(ctx["cfg"], ctx["df"])


@st.cache_data(show_spinner=False)
def load_episodes(mtime: float) -> pd.DataFrame | None:
    p = RESULTS_DIR / "episodes.csv"
    return pd.read_csv(p) if p.exists() else None


def episodes() -> pd.DataFrame | None:
    p = RESULTS_DIR / "episodes.csv"
    return load_episodes(p.stat().st_mtime if p.exists() else 0.0)


def policies_for(seed: int | None) -> dict:
    """Policies used by the live views, keyed by the display labels."""
    ctx = get_context()
    pols = {NAIVE: NaivePolicy(), RULE: RuleBasedPolicy()}
    if ctx["rl"]:
        s = seed if seed in ctx["rl"] else sorted(ctx["rl"])[0]
        pols[RL_LABEL] = ctx["rl"][s]
    return pols
