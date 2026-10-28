"""C. Policy comparison on the pre-computed evaluation episodes."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts
from app.common import family_titles, require_ready, vs
from app.context import episodes
from app.style import header, kpi, note, pct_tone
from energyrl.metrics import NAIVE, POLICY_ORDER, RL_LABEL, RULE, episode_table, improvement, summary

COLS = [("cost_eur", "Cost (EUR/day)", "{:,.0f}"), ("co2_kg", "CO2 (kg/day)", "{:,.0f}"),
        ("consumption_kwh", "Energy (kWh)", "{:,.0f}"), ("completion_rate", "Jobs completed", "{:.1%}"),
        ("jobs_missed", "Jobs missed", "{:.2f}"), ("tardiness_hours", "Tardiness (job-h)", "{:.2f}"),
        ("renewable_share", "PV share of load", "{:.1%}"), ("load_shifted_kwh", "Load shifted (kWh)", "{:,.0f}"),
        ("shed_events", "Capacity sheds", "{:.2f}"), ("invalid_requests", "Ignored invalid requests", "{:.2f}"), ("inference_ms_per_step", "Decision time (ms)", "{:.2f}")]


def render() -> None:
    header("Policy comparison",
           "Naive production schedule vs transparent rule-based energy management vs the trained RL agent, "
           "on identical held-out scenarios.")
    if not require_ready():
        return
    df = episodes()
    if df is None:
        note("No evaluation results yet. Run <code>python -m energyrl.evaluate</code> (see README).", warn=True)
        return
    titles = family_titles()
    fam = st.selectbox("Scenario set", list(df.family.unique()), format_func=lambda k: titles.get(k, k), key="cmp_fam")
    d = df[df.family == fam]
    s = summary(d)
    ep = episode_table(d)
    n_ep = int(s.episodes.iloc[0])
    n_seeds = int(s[s.policy == RL_LABEL].seeds.iloc[0]) if (s.policy == RL_LABEL).any() else 0
    st.caption(f"{n_ep} episodes per policy; RL values average {n_seeds} independently trained seeds.")

    pols = [p for p in POLICY_ORDER if (s.policy == p).any()]
    base = s.set_index("policy")
    cards = st.columns(len(pols))
    for col, p in zip(cards, pols):
        r = base.loc[p]
        dn = 100 * (1 - r.cost_eur / base.loc[NAIVE].cost_eur)
        dc = 100 * (1 - r.co2_kg / base.loc[NAIVE].co2_kg)
        sub = "reference" if p == NAIVE else f"cost {vs(dn, 'naive')} · CO2 {vs(dc, 'naive')}"
        col.markdown(kpi(p, f"{r.cost_eur:,.0f}", f"EUR/day ± {r.cost_eur_std_ep:,.0f}", sub,
                         "flat" if p == NAIVE else pct_tone(dn)), unsafe_allow_html=True)

    st.markdown("## KPI table")
    rows = []
    for p in pols:
        r = base.loc[p]
        row = {"Policy": p}
        for k, label, fmt in COLS:
            row[label] = r[k]
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("Policy")
    st.dataframe(tab.style.format({label: fmt for _, label, fmt in COLS}), width="stretch")

    st.markdown("### Percentage improvement")
    imp_rows = []
    for baseline in (NAIVE, RULE):
        for metric, label in (("cost_eur", "Cost"), ("co2_kg", "CO2")):
            i = improvement(d, baseline, metric)
            for _, r in i.iterrows():
                if r.policy != NAIVE and not (baseline == RULE and r.policy == RULE):
                    imp_rows.append({"Policy": r.policy, "vs": baseline, "Metric": label, "Improvement": r.pct,
                                     "± per-episode sd": r.pct_std, "Episodes better": r.wins})
    if imp_rows:
        it = pd.DataFrame(imp_rows)
        st.dataframe(it.style.format({"Improvement": "{:+.1f}%", "± per-episode sd": "{:.1f} pp", "Episodes better": "{:.0%}"}),
                     width="stretch", hide_index=True)
        st.caption("Improvement = reduction of the total relative to the baseline (positive is better). "
                   "'Episodes better' = share of individual days on which the policy beats the baseline.")

    st.markdown("## Cost vs carbon trade-off")
    a, b = st.columns(2)
    a.plotly_chart(charts.fig_scatter(ep, "Each dot is one simulated day"), width="stretch")
    ep2 = ep.copy()
    ep2["Policy"] = ep2["group"]
    nv = ep.pivot(index="episode", columns="group", values="cost_eur")[NAIVE]
    ep2["cost_saving_pct"] = 100 * (1 - ep2["cost_eur"] / ep2["episode"].map(nv))
    b.plotly_chart(charts.fig_box(ep2[ep2.group != NAIVE], "family", "cost_saving_pct", "group",
                                  "Cost saving vs naive, per day", "% of naive cost"), width="stretch")

    if (s.policy == RL_LABEL).any():
        rl, ru, nv_ = base.loc[RL_LABEL], base.loc[RULE], base.loc[NAIVE]
        txt = (f"<b>Reading the trade-off.</b> Against the naive plan the RL agent changes cost by "
               f"{100 * (1 - rl.cost_eur / nv_.cost_eur):+.1f}% and emissions by {100 * (1 - rl.co2_kg / nv_.co2_kg):+.1f}%, "
               f"while completing {rl.completion_rate:.1%} of jobs (naive {nv_.completion_rate:.1%}). "
               f"The rule-based policy reaches {100 * (1 - ru.cost_eur / nv_.cost_eur):+.1f}% cost and "
               f"{100 * (1 - ru.co2_kg / nv_.co2_kg):+.1f}% CO2. "
               "Cheap hours and low-carbon hours do not always coincide, so a cost-focused schedule can raise emissions and "
               "vice-versa; the reward weights (w_cost=1, w_co2=0.5) set where the agent lands on that frontier. "
               "Missing production is never an acceptable way to save money: every unfinished job carries a penalty larger than the energy it saves.")
        note(txt)
