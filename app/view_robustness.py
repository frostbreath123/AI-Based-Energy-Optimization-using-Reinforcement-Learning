"""D. Robustness laboratory: held-out OOD scenarios, degradation, failure cases and live sweeps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import charts
from app.common import family_titles, pct, require_ready, rl_seed, run_all
from app.context import episodes, eval_scenarios, get_context, policies_for
from app.style import header, kpi, note
from energyrl.evaluate import evaluate_suite, make_scenarios
from energyrl.metrics import NAIVE, RL_LABEL, RULE, episode_table, failure_cases, improvement, summary
from energyrl.ood import ood_families, ood_spec


def _saving_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    titles = family_titles()
    imp = improvement(df, NAIVE, metric)
    piv = imp.pivot(index="family", columns="policy", values="pct").reindex(list(df.family.unique()))
    piv.index = [titles.get(i, i).replace("OOD: ", "") for i in piv.index]
    return piv[[c for c in [RULE, RL_LABEL] if c in piv.columns]]


def render() -> None:
    header("Robustness laboratory",
           "Ten held-out scenario families that differ from the training distribution. None was used for training, "
           "checkpoint selection or tuning. All policies see exactly the same days.")
    if not require_ready():
        return
    ctx = get_context()
    from energyrl.scenarios import DistSpec
    fams = ood_families(DistSpec.from_config(ctx["cfg"]))
    with st.expander("What makes each scenario out-of-distribution?", expanded=False):
        st.dataframe(pd.DataFrame([{"Scenario": f.title, "How it differs from training": f.what_changes} for f in fams]),
                     hide_index=True, width="stretch")

    df = episodes()
    if df is None:
        note("No evaluation results yet. Run <code>python -m energyrl.evaluate</code> (see README).", warn=True)
        return
    has_rl = df.policy.str.startswith("RL").any()
    cost = _saving_matrix(df, "cost_eur")
    co2 = _saving_matrix(df, "co2_kg")

    st.markdown("## Performance degradation")
    a, b = st.columns(2)
    a.plotly_chart(charts.fig_heatmap(cost.to_numpy(), list(cost.columns), list(cost.index), "Cost saving vs naive (%)", "%", 0.0),
                   width="stretch")
    b.plotly_chart(charts.fig_heatmap(co2.to_numpy(), list(co2.columns), list(co2.index), "CO2 saving vs naive (%)", "%", 0.0),
                   width="stretch")
    if has_rl:
        idv = cost.loc[cost.index[0], RL_LABEL]
        ood = cost.iloc[1:][RL_LABEL]
        vs_rule = improvement(df, RULE, "cost_eur").query("policy == @RL_LABEL").set_index("family")["pct"]
        vs_rule.index = [family_titles().get(i, i).replace("OOD: ", "") for i in vs_rule.index]
        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi("RL cost saving vs naive", f"{idv:+.1f}", "%", f"in-distribution · OOD mean {ood.mean():+.1f}%", "flat"),
                    unsafe_allow_html=True)
        c2.markdown(kpi("RL cost vs rule-based, in-distribution", f"{vs_rule.iloc[0]:+.1f}", "%",
                        "negative = RL costs more", "good" if vs_rule.iloc[0] >= 0 else "bad"), unsafe_allow_html=True)
        c3.markdown(kpi("RL vs rule-based, worst OOD family", f"{vs_rule.iloc[1:].min():+.1f}", "%", vs_rule.iloc[1:].idxmin(), "bad"),
                    unsafe_allow_html=True)
        note("<b>How to read this.</b> Relative savings versus the naive plan often <i>rise</i> out-of-distribution because the naive plan "
             "itself becomes more expensive (price spikes, tariff changes give more room to shift load). Robustness is therefore judged by "
             "(1) RL versus the rule-based policy on the same days, and (2) production continuity, shown below.")

    st.markdown("## Spread across individual days")
    ep = episode_table(df)
    nv = ep.pivot_table(index=["family", "episode"], columns="group", values="cost_eur")[NAIVE]
    ep["saving"] = 100 * (1 - ep["cost_eur"] / ep.set_index(["family", "episode"]).index.map(nv).to_numpy())
    ep["family"] = ep["family"].map(lambda f: family_titles().get(f, f).replace("OOD: ", "").replace("In-distribution (held-out test days)", "In-distribution"))
    st.plotly_chart(charts.fig_box(ep[ep.group != NAIVE], "family", "saving", "group",
                                   "Daily cost saving vs naive (box = interquartile range)", "% of naive cost"),
                    width="stretch")

    sm = summary(df, ["completion_rate", "constraint_violations", "jobs_missed"])
    cp = sm.pivot(index="family", columns="policy", values="completion_rate").reindex(list(df.family.unique())) * 100
    cp.index = [family_titles().get(i, i).replace("OOD: ", "") for i in cp.index]
    st.markdown("### Production continuity (jobs completed by end of day, %)")
    st.dataframe(cp.style.format("{:.1f}").background_gradient(cmap="RdYlGn", vmin=60, vmax=100, axis=None),
                 width="stretch")

    st.markdown("## Failure cases")
    if has_rl:
        fc = failure_cases(df, 8)
        fc["Scenario"] = fc.family.map(lambda f: family_titles().get(f, f).replace("OOD: ", ""))
        show = fc[["Scenario", "episode", "cost_naive", "cost_rule", "cost_rl", "rl_vs_rule_pct", "missed_rl", "missed_rule", "violations_rl"]]
        show.columns = ["Scenario", "Episode", "Naive EUR", "Rule EUR", "RL EUR", "RL vs rule (%)", "RL missed jobs", "Rule missed jobs", "RL violations"]
        st.dataframe(show.style.format({"Naive EUR": "{:,.0f}", "Rule EUR": "{:,.0f}", "RL EUR": "{:,.0f}",
                                        "RL vs rule (%)": "{:+.1f}", "RL missed jobs": "{:.2f}", "Rule missed jobs": "{:.2f}",
                                        "RL violations": "{:.2f}"}), hide_index=True, width="stretch")
        st.caption("Worst days for the RL agent, ranked by missed jobs then by cost relative to the rule-based policy "
                   "(RL values are seed averages). 'RL vs rule' > 0 means RL was more expensive.")
        pick = st.selectbox("Inspect a failure case", range(len(fc)), format_func=lambda i: f"{fc.Scenario[i]} - episode {int(fc.episode[i])}")
        sc = eval_scenarios()[fc.family[pick]][int(fc.episode[pick])]
        res = run_all(sc, rl_seed())
        st.caption(f"Day {sc.date}; scenario parameters: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                                                         for k, v in sc.meta.items() if k in ("price_scale", "pv_scale", "slack_mean", "n_jobs", "has_drop", "tariff")))
        l, r = st.columns(2)
        for col, name in zip((l, r), (RULE, RL_LABEL)):
            if name in res:
                col.plotly_chart(charts.fig_gantt(res[name]["sim"], f"{name}: {res[name]['metrics']['cost_eur']:,.0f} EUR, "
                                                                    f"{res[name]['metrics']['jobs_missed']} missed"), width="stretch")
    else:
        note("Train an RL model to analyse its failure cases.")

    st.markdown("## Run a family live")
    c1, c2, c3 = st.columns([2, 1, 1])
    fam = c1.selectbox("Family", [f.key for f in fams], format_func=lambda k: family_titles()[k], key="rob_fam")
    n = int(c2.number_input("Fresh episodes", 3, 40, 10, help="New random days, different seed from the stored evaluation."))
    seed = int(c3.number_input("Seed", 0, 99999, 2024))
    if st.button("Run live evaluation", type="primary"):
        spec = ood_spec(ctx["gen"].spec, next(f for f in fams if f.key == fam))
        scs = make_scenarios(ctx["gen"], spec, n, seed, fam)
        with st.spinner(f"Simulating {n} days x {len(policies_for(rl_seed()))} policies..."):
            live = evaluate_suite(policies_for(rl_seed()), scs, ctx["cfg"], "OOD", fam)
        t = live.groupby("policy")[["cost_eur", "co2_kg", "completion_rate", "jobs_missed", "constraint_violations"]].mean()
        t["cost vs naive (%)"] = 100 * (1 - t.cost_eur / t.loc[NAIVE].cost_eur)
        t["CO2 vs naive (%)"] = 100 * (1 - t.co2_kg / t.loc[NAIVE].co2_kg)
        st.dataframe(t.style.format("{:.2f}"), width="stretch")

    st.markdown("## Sensitivity sweep: price level x solar availability")
    st.caption("Grid of unseen price and PV multipliers (in-distribution tariff A, test-pool days). "
               "Shows where the learned policy's advantage over the rule-based policy appears or disappears.")
    g1, g2 = st.columns([1, 1])
    per_cell = int(g1.number_input("Days per cell", 3, 20, 6))
    if g2.button("Run sweep"):
        prices, pvs = [0.7, 1.4, 2.0, 3.0], [0.15, 0.5, 1.0, 1.3]
        pols = policies_for(rl_seed())
        z_rl, z_rule = np.zeros((len(pvs), len(prices))), np.zeros((len(pvs), len(prices)))
        bar = st.progress(0.0)
        gen = ctx["gen"]
        for i, pv in enumerate(pvs):
            for j, pr in enumerate(prices):
                spec = gen.spec.with_(season="test", price_scale=(pr, pr), pv_scale=(pv, pv), exclude_pv_low=0.0,
                                      exclude_slack_tight=0.0, drop_prob=0.0)
                scs = make_scenarios(gen, spec, per_cell, 99, "sweep")
                r = evaluate_suite(pols, scs, ctx["cfg"], "sweep").groupby("policy").cost_eur.sum()
                z_rule[i, j] = 100 * (1 - r[RULE] / r[NAIVE])
                z_rl[i, j] = 100 * (1 - r[RL_LABEL] / r[NAIVE]) if RL_LABEL in r else np.nan
                bar.progress((i * len(prices) + j + 1) / (len(pvs) * len(prices)))
        xl = [f"price x{p}" for p in prices]
        yl = [f"PV x{p}" for p in pvs]
        u, v = st.columns(2)
        u.plotly_chart(charts.fig_heatmap(z_rule, xl, yl, "Rule-based: cost saving vs naive (%)", "%"), width="stretch")
        if RL_LABEL in pols:
            v.plotly_chart(charts.fig_heatmap(z_rl, xl, yl, "RL: cost saving vs naive (%)", "%"), width="stretch")
