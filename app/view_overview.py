"""A. Overview dashboard."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts
from app.common import family_titles, has_rl, pct, require_ready, rl_seed, run_all, vs
from app.context import episodes, eval_scenarios
from app.style import header, kpi, note, pct_tone
from energyrl.metrics import NAIVE, RL_LABEL, RULE, improvement, summary


def benchmark_table(df: pd.DataFrame) -> pd.DataFrame:
    titles = family_titles()
    s = summary(df, ["cost_eur", "co2_kg", "completion_rate"])
    rows = []
    for fam in s.family.unique():
        g = s[s.family == fam].set_index("policy")
        if RL_LABEL not in g.index:
            continue
        rl, nv, ru = g.loc[RL_LABEL], g.loc[NAIVE], g.loc[RULE]
        rows.append({"Scenario set": titles.get(fam, fam), "RL cost vs naive": pct(rl.cost_eur, nv.cost_eur),
                     "RL cost vs rule-based": pct(rl.cost_eur, ru.cost_eur), "RL CO2 vs naive": pct(rl.co2_kg, nv.co2_kg),
                     "RL CO2 vs rule-based": pct(rl.co2_kg, ru.co2_kg),
                     "Jobs completed (RL)": 100 * rl.completion_rate, "Jobs completed (rule)": 100 * ru.completion_rate})
    return pd.DataFrame(rows)


def render() -> None:
    header("Energy optimisation overview",
           "A reinforcement-learning agent schedules production jobs and a battery against real 2019 French prices, "
           "solar, carbon-intensity and industrial-load data. Numbers below come from live simulation of the selected day.")
    if not require_ready():
        return
    scs = eval_scenarios()
    titles = family_titles()
    c1, c2, c3 = st.columns([2.2, 1, 1.4])
    fam = c1.selectbox("Scenario set", list(scs), format_func=lambda k: titles[k], key="ov_fam",
                       help="In-distribution days resemble the training data. OOD sets deliberately do not.")
    ep = int(c2.number_input("Episode", 0, len(scs[fam]) - 1, 0, key="ov_ep", help="Index of the held-out day in the set."))
    sc = scs[fam][ep]
    res = run_all(sc, rl_seed())
    m = {k: v["metrics"] for k, v in res.items()}
    c3.markdown(f"<div style='padding-top:1.7rem' class='sub'>Day <b>{sc.date}</b>{' (weekend)' if sc.weekend else ''} · "
                f"{len(sc.jobs)} jobs · tariff {sc.meta['tariff']}</div>", unsafe_allow_html=True)

    focus = RL_LABEL if has_rl(res) else RULE
    f, nv, ru = m[focus], m[NAIVE], m[RULE]
    cols = st.columns(5)
    cost_d = pct(f["cost_eur"], nv["cost_eur"])
    cols[0].markdown(kpi("Electricity cost", f"{f['cost_eur']:,.0f}", "EUR",
                         f"{vs(cost_d, 'naive')} ({nv['cost_eur'] - f['cost_eur']:+,.0f} EUR/day)", pct_tone(cost_d)),
                     unsafe_allow_html=True)
    co2_d = pct(f["co2_kg"], nv["co2_kg"])
    cols[1].markdown(kpi("Carbon emissions", f"{f['co2_kg']:,.0f}", "kg CO2e", vs(co2_d, "naive"), pct_tone(co2_d)),
                     unsafe_allow_html=True)
    comp = f["completion_rate"] * 100
    cols[2].markdown(kpi("Production completed", f"{comp:.0f}", "%",
                         f"{f['jobs_completed']} of {f['jobs_total']} jobs · {f['jobs_late']} late",
                         "good" if f["jobs_missed"] == 0 else "bad"), unsafe_allow_html=True)
    cols[3].markdown(kpi("Renewable contribution", f"{100 * f['renewable_share']:.0f}", "% of load",
                         f"{f['pv_used_kwh']:,.0f} of {f['pv_generation_kwh']:,.0f} kWh PV used on site", "flat"),
                     unsafe_allow_html=True)
    vs_rule = pct(f["cost_eur"], ru["cost_eur"])
    cols[4].markdown(kpi("Cost vs rule-based", f"{abs(vs_rule):.1f}", "% " + ("lower" if vs_rule >= 0 else "higher"),
                         f"rule-based: {ru['cost_eur']:,.0f} EUR", pct_tone(vs_rule)), unsafe_allow_html=True)
    if not has_rl(res):
        st.caption("Showing the rule-based policy as the focus because no RL model is loaded.")

    st.markdown("## This day, policy by policy")
    keys = [("cost_eur", "Cost", "EUR"), ("co2_kg", "Emissions", "kg CO2e"), ("load_shifted_kwh", "Load shifted", "kWh")]
    st.plotly_chart(charts.fig_policy_bars(m, keys), width="stretch")
    st.plotly_chart(charts.fig_dispatch(res[focus]["trace"], len(sc.jobs), f"How the {focus} dispatched power over the day"),
                    width="stretch")

    st.markdown("## Benchmark across all held-out scenarios")
    df = episodes()
    if df is None:
        note("No evaluation results yet. Run <code>python -m energyrl.evaluate</code> then <code>python -m energyrl.report</code>.", warn=True)
        return
    tab = benchmark_table(df)
    if tab.empty:
        note("Evaluation results exist but contain no RL rows. Train a model and re-run the evaluation.", warn=True)
        return
    idr = tab.iloc[0]
    ood = tab.iloc[1:]
    a, b, c = st.columns(3)
    a.markdown(kpi("In-distribution: RL cost vs naive", f"{idr['RL cost vs naive']:+.1f}", "%",
                   f"vs rule-based {idr['RL cost vs rule-based']:+.1f}%", pct_tone(idr["RL cost vs naive"])), unsafe_allow_html=True)
    b.markdown(kpi("OOD average: RL cost vs naive", f"{ood['RL cost vs naive'].mean():+.1f}", "%",
                   f"vs rule-based {ood['RL cost vs rule-based'].mean():+.1f}%", pct_tone(ood["RL cost vs naive"].mean())),
               unsafe_allow_html=True)
    c.markdown(kpi("OOD worst case: RL vs rule-based", f"{ood['RL cost vs rule-based'].min():+.1f}", "%",
                   ood.loc[ood["RL cost vs rule-based"].idxmin(), "Scenario set"], pct_tone(ood["RL cost vs rule-based"].min())),
               unsafe_allow_html=True)
    st.dataframe(tab.style.format({c: "{:+.1f}%" for c in tab.columns[1:5]} | {c: "{:.1f}%" for c in tab.columns[5:]})
                 .background_gradient(subset=list(tab.columns[1:5]), cmap="RdYlGn", vmin=-15, vmax=25),
                 width="stretch", hide_index=True)
    st.caption("Positive = lower than the comparison policy. Averages over episodes and RL training seeds; "
               "see Policy comparison for spread and Robustness lab for failure cases.")
