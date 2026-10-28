"""Aggregation of episode-level results into the tables used by the report and the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd

RL_LABEL = "RL agent (PPO)"
NAIVE, RULE = "Naive schedule", "Rule-based"
POLICY_ORDER = [NAIVE, RULE, RL_LABEL]
KPIS = ["cost_eur", "co2_kg", "consumption_kwh", "completion_rate", "on_time_rate", "jobs_missed", "jobs_late",
        "tardiness_hours", "renewable_share", "pv_utilization", "load_shifted_kwh", "peak_kw",
        "constraint_violations", "shed_events", "invalid_requests", "inference_ms_per_step"]
LOWER_IS_BETTER = {"cost_eur", "co2_kg", "consumption_kwh", "jobs_missed", "jobs_late", "tardiness_hours",
                   "peak_kw", "constraint_violations", "shed_events", "invalid_requests", "inference_ms_per_step"}


def group_policies(df: pd.DataFrame) -> pd.DataFrame:
    """Add `group`: all RL seeds map to one label; baselines keep their name."""
    out = df.copy()
    out["group"] = np.where(out["policy"].str.startswith("RL agent"), RL_LABEL, out["policy"])
    return out


def episode_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (family, episode, group). RL metrics are averaged over training seeds."""
    d = group_policies(df)
    num = d.select_dtypes("number").columns.difference(["episode"])
    return d.groupby(["suite", "family", "episode", "group"], as_index=False, sort=False)[list(num)].mean()


def summary(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Mean and std over episodes (and, for RL, std across training seeds of the per-seed mean)."""
    metrics = metrics or KPIS
    d = group_policies(df)
    per_seed = d.groupby(["suite", "family", "group", "policy"], as_index=False, sort=False)[metrics].mean()
    ep = episode_table(df)
    rows = []
    for (suite, fam, grp), g in ep.groupby(["suite", "family", "group"], sort=False):
        s = per_seed[(per_seed.family == fam) & (per_seed.group == grp)]
        row = {"suite": suite, "family": fam, "policy": grp, "episodes": len(g), "seeds": len(s)}
        for m in metrics:
            row[m] = g[m].mean()
            row[m + "_std_ep"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
            row[m + "_std_seed"] = s[m].std(ddof=1) if len(s) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out["policy"] = pd.Categorical(out["policy"], POLICY_ORDER, ordered=True)
    return out.sort_values(["suite", "family", "policy"], kind="stable").reset_index(drop=True)


def improvement(df: pd.DataFrame, baseline: str, metric: str = "cost_eur") -> pd.DataFrame:
    """Percent improvement of each policy vs `baseline` (positive = better), per family.
    `pct` uses the ratio of mean totals; `pct_std` is the std of per-episode paired improvements."""
    ep = episode_table(df)
    rows = []
    for fam, g in ep.groupby("family", sort=False):
        piv = g.pivot(index="episode", columns="group", values=metric)
        if baseline not in piv:
            continue
        for pol in piv.columns:
            if pol == baseline:
                continue
            base = piv[baseline]
            ratio = piv[pol].sum() / base.sum() if base.sum() else np.nan
            paired = 100 * (1 - piv[pol] / base.replace(0, np.nan))
            sign = 1 if metric in LOWER_IS_BETTER else -1
            rows.append({"family": fam, "policy": pol, "baseline": baseline, "metric": metric,
                         "pct": sign * 100 * (1 - ratio), "pct_std": paired.std(ddof=1),
                         "wins": float((piv[pol] < base).mean()) if sign == 1 else float((piv[pol] > base).mean())})
    return pd.DataFrame(rows)


def failure_cases(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Episodes where the RL agent does worst relative to the rule-based baseline
    (cost penalty, with unfinished jobs counted as failures)."""
    ep = episode_table(df)
    piv = ep.pivot_table(index=["family", "episode"], columns="group", values=["cost_eur", "co2_kg", "jobs_missed",
                                                                          "constraint_violations"])
    out = pd.DataFrame({
        "cost_naive": piv[("cost_eur", NAIVE)], "cost_rule": piv[("cost_eur", RULE)], "cost_rl": piv[("cost_eur", RL_LABEL)],
        "missed_rl": piv[("jobs_missed", RL_LABEL)], "missed_rule": piv[("jobs_missed", RULE)],
        "violations_rl": piv[("constraint_violations", RL_LABEL)],
    })
    out["rl_vs_rule_pct"] = 100 * (out.cost_rl / out.cost_rule - 1)
    out["rl_vs_naive_pct"] = 100 * (out.cost_rl / out.cost_naive - 1)
    return out.sort_values(["missed_rl", "rl_vs_rule_pct"], ascending=False).head(n).reset_index()


def paired_cost_ci(df: pd.DataFrame, a: str = RL_LABEL, b: str = RULE, metric: str = "cost_eur", n_boot: int = 4000,
                   seed: int = 0) -> pd.DataFrame:
    """Per-day percent reduction of `a` relative to `b` with a bootstrap 95 % CI, per family and pooled over all OOD families."""
    ep = episode_table(df)
    rng = np.random.default_rng(seed)

    def row(name: str, d: np.ndarray) -> dict:
        boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
        return {"family": name, "days": len(d), "mean_pct": d.mean(), "ci_low": np.percentile(boots, 2.5),
                "ci_high": np.percentile(boots, 97.5)}

    rows, pooled = [], []
    for fam, g in ep.groupby("family", sort=False):
        p = g.pivot(index="episode", columns="group", values=metric)
        d = (100 * (1 - p[a] / p[b])).to_numpy()
        rows.append(row(fam, d))
        if fam != "in_distribution":
            pooled.append(d)
    rows.append(row("ALL_OOD_pooled", np.concatenate(pooled)))
    return pd.DataFrame(rows)
