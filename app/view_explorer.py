"""B. Simulation explorer: build any scenario and see what each policy does."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import charts
from app.common import has_rl, pct, require_ready, rl_seed, run_all
from app.context import get_context
from app.style import header, kpi, note
from energyrl.metrics import NAIVE, RL_LABEL, RULE


def render() -> None:
    header("Simulation explorer",
           "Choose a real 2019 day and perturb tariff, solar and production. Every policy is re-simulated on the spot.")
    if not require_ready():
        return
    ctx = get_context()
    gen = ctx["gen"]
    days = gen.available_days()
    summer = [d for d in days if d.month == 6 and d.dayofweek == 1]
    default_day = summer[1].date() if len(summer) > 1 else days[0].date()

    with st.container(border=True):
        a, b, c, d = st.columns(4)
        day = a.date_input("Day", value=default_day, min_value=days[0].date(), max_value=days[-1].date(),
                           help="Real French 2019 prices, carbon intensity, Paris solar and an industrial load shape for that day.")
        tariff = b.radio("Off-peak placement", ["A", "B"], horizontal=True,
                         help="TURPE 7 HTA prices. A: off-peak 22-06 h (training). B: low-season off-peak 02-06 h and 11-15 h (CRE solar-following proposal, unseen).")
        price_scale = c.slider("Spot price multiplier", 0.5, 3.0, 1.0, 0.1, help="Training range 0.7-1.4x.")
        pv_scale = d.slider("PV availability", 0.1, 1.5, 1.0, 0.05, help="Training range 0.5-1.3x of the installed 900 kWp.")
        e, f, g, h, o = st.columns([1, 1.2, 1, 1, 1.2])
        option = o.radio("TURPE option", ["CU", "LU"], horizontal=True,
                         help="CU = short utilisation (big peak/off-peak spread), LU = long utilisation (flatter energy price).")
        n_jobs = e.slider("Production jobs", 3, 8, 5, help="Training: 4-6 jobs.")
        slack = f.slider("Deadline slack (h)", 0.5, 7.0, 4.0, 0.5, help="Spare hours between planned finish and deadline. Training 2-6 h.")
        drop = g.toggle("PV collapse", value=False, help="Unforecast: PV falls 90 % for 4 midday hours and the forecast does not know.")
        seed = int(h.number_input("Random seed", 0, 9999, 7, help="Controls job sizes/timing and forecast error."))

    spec = gen.spec.with_(tariff=tariff, tariff_options=(option,))
    rng = np.random.default_rng(seed)
    params = gen.draw_params(rng, spec)
    params.update(price_scale=price_scale, pv_scale=pv_scale, n_jobs=n_jobs, slack_mean=slack, spike=1.0,
                  has_drop=drop, drop_len=4, drop_depth=0.9)
    sc = gen.build(pd.Timestamp(day), params, spec, seed, "explorer")
    res = run_all(sc, rl_seed())
    m = {k: v["metrics"] for k, v in res.items()}

    cols = st.columns(len(m))
    for col, (name, mm) in zip(cols, m.items()):
        d = pct(mm["cost_eur"], m[NAIVE]["cost_eur"])
        col.markdown(kpi(name, f"{mm['cost_eur']:,.0f}", "EUR",
                         f"{mm['co2_kg']:,.0f} kg CO2e · {mm['completion_rate'] * 100:.0f}% done" + ("" if name == NAIVE else f" · {d:+.1f}% cost"),
                         "flat"), unsafe_allow_html=True)

    st.markdown("## Inputs")
    l, r = st.columns(2)
    l.plotly_chart(charts.fig_signals(sc), width="stretch")
    r.plotly_chart(charts.fig_renewables(sc), width="stretch")
    note("<b>Flexible vs inflexible load.</b> The grey base load (lighting, HVAC, compressors) must always be served. "
         "Production jobs are flexible: they can start any time after release and must finish before their deadline. "
         "The battery shifts energy in time but cannot export to the grid.")

    st.markdown("## Policy decisions")
    options = list(res)
    pol = st.radio("Policy to inspect", options, index=len(options) - 1, horizontal=True)
    st.plotly_chart(charts.fig_dispatch(res[pol]["trace"], len(sc.jobs), f"Power dispatch: {pol}"), width="stretch")
    st.plotly_chart(charts.fig_gantt(res[pol]["sim"], f"Job schedule: {pol} (outlined bar = original plan)"),
                    width="stretch")
    st.plotly_chart(charts.fig_soc({k: v["trace"] for k, v in res.items()}, ctx["cfg"]["site"]["battery"]["capacity_kwh"]),
                    width="stretch")
    tab = pd.DataFrame(m).T[["cost_eur", "co2_kg", "consumption_kwh", "jobs_completed", "jobs_late", "jobs_missed",
                             "load_shifted_kwh", "peak_kw", "renewable_share", "constraint_violations"]]
    tab.columns = ["Cost (EUR)", "CO2 (kg)", "Energy (kWh)", "Jobs done", "Late", "Missed", "Shifted (kWh)", "Peak (kW)",
                   "PV share", "Violations"]
    st.dataframe(tab.style.format({"Cost (EUR)": "{:,.0f}", "CO2 (kg)": "{:,.0f}", "Energy (kWh)": "{:,.0f}",
                                   "Shifted (kWh)": "{:,.0f}", "Peak (kW)": "{:,.0f}", "PV share": "{:.0%}"}),
                 width="stretch")
