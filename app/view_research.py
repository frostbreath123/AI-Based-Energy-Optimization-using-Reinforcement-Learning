"""E. Research and methodology (model, reward, data provenance, experiments, limitations, job control)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import charts
from app.common import require_ready
from app.context import MODELS_DIR, RESULTS_DIR, ROOT, get_context
from app.style import C, header, note

DOCS = ROOT / "docs"
LOG_DIR = RESULTS_DIR / "logs"


def read_doc(name: str) -> str:
    p = DOCS / name
    return p.read_text(encoding="utf-8") if p.exists() else f"*`docs/{name}` not found.*"


def _tab_model() -> None:
    st.markdown(read_doc("SIMULATION_DESIGN.md"))


def _tab_reward(cfg: dict) -> None:
    r = cfg["reward"]
    st.markdown("### Reward equation")
    st.latex(r"r_t = -S\left[\,w_c\frac{\mathrm{cost}_t}{C_{ref}} + w_e\frac{\mathrm{CO_2}_t}{E_{ref}} "
             r"+ w_l\,\mathrm{overdue}_t + w_s\,\mathrm{shed}_t + w_i\,\mathrm{invalid}_t\right]")
    st.latex(r"r_T \mathrel{-}= S\left[\,w_m\Big(N_{missed} + \sum_{j\in missed}\tfrac{\mathrm{remaining}_j}{\mathrm{duration}_j}\Big) "
             r"+ w_c\,\frac{\mathrm{battery\ settlement}}{C_{ref}}\right]")
    comp = pd.DataFrame([
        ("Electricity cost", "w_cost", r["w_cost"], "EUR / naive daily bill", "Import x tariff + demand-charge increment - export revenue"),
        ("Carbon", "w_co2", r["w_co2"], "kg / naive daily emissions", "Import x grid intensity; half the weight of cost by default"),
        ("Late job-hours", "w_late", r["w_late"], "per job-hour overdue", "Soft, dense signal that a deadline has been missed"),
        ("Missed jobs", "w_miss", r["w_miss"], "per unfinished job (+ remaining fraction)", "Larger than any saving from skipping the job"),
        ("Capacity shed", "w_shed", r["w_shed"], "per event", "Grid limit hit: physical violation"),
        ("Invalid request", "w_invalid", r["w_invalid"], "per event", "Premature job start, empty/full battery"),
    ], columns=["Component", "Weight", "Default", "Unit", "Role"])
    st.dataframe(comp, hide_index=True, width="stretch")
    st.markdown(read_doc("REWARD_DESIGN.md"))


def _tab_data() -> None:
    mp = ROOT / "data" / "manifest.json"
    if not mp.exists():
        note("Manifest missing. Run <code>python -m energyrl.data.manifest</code>.", warn=True)
        return
    m = json.loads(mp.read_text())
    pill = {"processed public data": "pub", "SYNTHETIC (generated)": "syn"}
    for d in m["datasets"]:
        cls = pill.get(d["class"], "der")
        with st.container(border=True):
            st.markdown(f"<span class='pill {cls}'>{d['class']}</span> **{d['name']}**", unsafe_allow_html=True)
            st.markdown(f"{d['role']}")
            st.caption(f"Publisher: {d['publisher']} · Resolution: {d['resolution']} · Geography: {d['geography']} · Units: {d['units']} · "
                       f"Licence: {d['license']} · Retrieved: {d['retrieved']}")
            if d["url"] != "-":
                st.caption(d["url"])
            st.caption(f"Processing: {d['preprocessing']}")
    ctx = get_context()
    if ctx["df"] is not None:
        mon = ctx["df"].groupby(ctx["df"].index.month)[["spot_eur_kwh", "pv_cf", "carbon_g_kwh"]].mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=mon.index, y=mon.spot_eur_kwh * 1000, name="Spot price (EUR/MWh)", marker_color=C["price"]))
        fig.add_trace(go.Scatter(x=mon.index, y=mon.carbon_g_kwh, name="Carbon (g/kWh)", yaxis="y2", line=dict(color=C["carbon"])))
        fig.add_trace(go.Scatter(x=mon.index, y=mon.pv_cf * 24 * 1000 / 24 * 10, name="PV (x10 W/kWp mean)", yaxis="y2", line=dict(color=C["pv"])))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False), xaxis_title="Month 2019")
        st.plotly_chart(charts._base(fig, 300, "Real 2019 French data: monthly means"), width="stretch")


def _tab_experiments() -> None:
    logs = sorted(MODELS_DIR.glob("ppo_seed*/train_log.csv"))
    if logs:
        fig = go.Figure()
        for lg in logs:
            d = pd.read_csv(lg)
            fig.add_trace(go.Scatter(x=d.timesteps, y=d.val_cost_saving_pct, name=f"{lg.parent.name} cost saving", mode="lines"))
        fig.update_yaxes(title_text="Validation cost saving vs naive (%)")
        fig.update_xaxes(title_text="Training timesteps")
        st.plotly_chart(charts._base(fig, 300, "Training progress (validation days, not OOD)"), width="stretch")
    else:
        note("No training logs yet.", warn=True)
    sp = RESULTS_DIR / "sensitivity.csv"
    st.markdown("### Reward-weight sensitivity (in-distribution test days)")
    if sp.exists():
        t = pd.read_csv(sp, index_col=0)
        st.dataframe(t.style.format("{:.2f}"), width="stretch")
    else:
        st.caption("Run `python -m energyrl.sensitivity` after training the sensitivity variants (see README).")
    st.markdown(read_doc("RESEARCH_LOGBOOK.md"))


def _job_panel() -> None:
    st.markdown("### Retrain or re-evaluate (runs in the background)")
    note("Jobs run as separate processes so the dashboard stays responsive. A retrained model is saved under a separate "
         "tag and does <b>not</b> overwrite the shipped models.")
    c1, c2, c3 = st.columns(3)
    steps = int(c1.number_input("Timesteps", 10_000, 2_000_000, 60_000, 10_000))
    seed = int(c2.number_input("Seed", 0, 99, 5))
    job = st.session_state.get("job")
    running = job is not None and job["proc"].poll() is None
    b1, b2 = st.columns(2)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2"}

    def launch(name: str, args: list[str]) -> None:
        log = open(LOG_DIR / f"ui_{name}.log", "w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-m", *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        st.session_state["job"] = {"name": name, "proc": proc, "log": LOG_DIR / f"ui_{name}.log"}

    if b1.button("Retrain (tag ui_retrain)", disabled=running):
        launch("retrain", ["energyrl.train", "--tag", "ui_retrain", "--seeds", str(seed), "--timesteps", str(steps)])
        st.rerun()
    if b2.button("Re-run evaluation + report", disabled=running):
        launch("evaluate", ["energyrl.evaluate"])
        st.rerun()
    job = st.session_state.get("job")
    if job:
        code = job["proc"].poll()
        state = "running" if code is None else ("finished" if code == 0 else f"failed (exit {code})")
        st.write(f"**{job['name']}**: {state}")
        if job["log"].exists():
            st.code("\n".join(job["log"].read_text(errors="replace").splitlines()[-12:]) or "(no output yet)", language="text")
        if code is None:
            st.button("Refresh status")
        elif code == 0 and job["name"] == "evaluate":
            if not job.get("post_done"):               # build tables/figures once, not on every rerun
                subprocess.run([sys.executable, "-m", "energyrl.report"], cwd=ROOT, env=env, capture_output=True)
                st.cache_data.clear()
                job["post_done"] = True
            st.success("Evaluation and report finished; results reloaded.")
        elif code == 0 and job["name"] == "retrain":
            st.success(f"Model saved to models/ui_retrain_seed{seed}/best_model.zip. It is not loaded automatically: copy it to "
                       "models/ppo_seed<N>/best_model.zip (keeping a backup) and restart the app to evaluate it.")


def render() -> None:
    header("Research and methodology", "How the simulation works, how the reward is built, where the data come from, and what was learned.")
    if not require_ready():
        return
    cfg = get_context()["cfg"]
    tabs = st.tabs(["Simulation model", "Reward function", "Data provenance", "Experiments", "Limitations", "Run jobs"])
    with tabs[0]:
        _tab_model()
    with tabs[1]:
        _tab_reward(cfg)
    with tabs[2]:
        _tab_data()
    with tabs[3]:
        _tab_experiments()
    with tabs[4]:
        st.markdown(read_doc("LIMITATIONS.md"))
    with tabs[5]:
        _job_panel()
