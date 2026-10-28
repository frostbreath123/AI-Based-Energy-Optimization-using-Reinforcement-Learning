"""Generate summary tables (CSV/Markdown) and static figures from results/episodes.csv.

    python -m energyrl.report
"""
from __future__ import annotations

import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from energyrl.config import MODELS_DIR, RESULTS_DIR  # noqa: E402
from energyrl.metrics import (NAIVE, POLICY_ORDER, RL_LABEL, RULE, episode_table, failure_cases, improvement,  # noqa: E402
                              paired_cost_ci, summary)

COLORS = {NAIVE: "#8A94A6", RULE: "#3B6FB6", RL_LABEL: "#0E8F7E"}
FIG = RESULTS_DIR / "figures"


def _style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 140})


def fig_savings_by_family(df: pd.DataFrame) -> None:
    fams = list(df.family.unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=False)
    for ax, metric, title in zip(axes, ["cost_eur", "co2_kg"], ["Electricity cost saving vs naive (%)",
                                                                   "Carbon saving vs naive (%)"]):
        imp = improvement(df, NAIVE, metric)
        w = 0.38
        for i, pol in enumerate([RULE, RL_LABEL]):
            s = imp[imp.policy == pol].set_index("family").reindex(fams)
            ax.bar(np.arange(len(fams)) + (i - 0.5) * w, s.pct, w, label=pol, color=COLORS[pol],
                   yerr=s.pct_std / np.sqrt(len(df[df.family == fams[0]].episode.unique())), capsize=2)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([f.replace("_", "\n") for f in fams], fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "savings_by_family.png")
    plt.close(fig)


def fig_tradeoff(df: pd.DataFrame, family: str, fname: str) -> None:
    ep = episode_table(df)
    g = ep[ep.family == family]
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    for pol in POLICY_ORDER:
        s = g[g.group == pol]
        ax.scatter(s.cost_eur, s.co2_kg, s=18, alpha=0.7, color=COLORS[pol], label=pol)
    ax.set_xlabel("Daily electricity cost (EUR)")
    ax.set_ylabel("Daily emissions (kg CO2e)")
    ax.set_title(f"Cost / carbon trade-off: {family}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def fig_completion(df: pd.DataFrame) -> None:
    s = summary(df, ["completion_rate", "constraint_violations"])
    fams = list(s.family.unique())
    fig, ax = plt.subplots(figsize=(11, 3.6))
    w = 0.27
    for i, pol in enumerate(POLICY_ORDER):
        v = s[s.policy == pol].set_index("family").reindex(fams)
        ax.bar(np.arange(len(fams)) + (i - 1) * w, 100 * v.completion_rate, w, color=COLORS[pol], label=pol)
    ax.set_ylim(60, 101)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f.replace("_", "\n") for f in fams], fontsize=7)
    ax.set_ylabel("Jobs completed by end of day (%)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "completion_by_family.png")
    plt.close(fig)


def fig_training_curves() -> None:
    logs = sorted(glob.glob(str(MODELS_DIR / "ppo_seed*" / "train_log.csv")))
    if not logs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for lg in logs:
        d = pd.read_csv(lg)
        name = lg.split("ppo_seed")[1].split("\\")[0].split("/")[0]
        axes[0].plot(d.timesteps, d.val_reward, label=f"seed {name}")
        axes[1].plot(d.timesteps, d.val_cost_saving_pct)
        axes[2].plot(d.timesteps, d.val_co2_saving_pct)
    for ax, t in zip(axes, ["Validation reward", "Validation cost saving vs naive (%)",
                            "Validation CO2 saving vs naive (%)"]):
        ax.set_title(t)
        ax.set_xlabel("Training timesteps")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png")
    plt.close(fig)


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r) + " |")
    return "\n".join(out)


def build_report() -> None:
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_DIR / "episodes.csv")
    s = summary(df)
    s.to_csv(RESULTS_DIR / "summary_by_family.csv", index=False)
    for baseline in (NAIVE, RULE):
        for metric in ("cost_eur", "co2_kg"):
            imp = improvement(df, baseline, metric)
            imp.to_csv(RESULTS_DIR / f"improvement_{metric}_vs_{'naive' if baseline == NAIVE else 'rule'}.csv", index=False)
    key = s[["suite", "family", "policy", "episodes", "seeds", "cost_eur", "cost_eur_std_ep", "co2_kg",
             "completion_rate", "jobs_missed", "renewable_share", "load_shifted_kwh", "shed_events", "invalid_requests"]].copy()
    key.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    (RESULTS_DIR / "comparison_table.md").write_text(to_markdown(key), encoding="utf-8")
    paired_cost_ci(df).to_csv(RESULTS_DIR / "rl_vs_rule_cost_ci.csv", index=False)
    paired_cost_ci(df, metric="co2_kg").to_csv(RESULTS_DIR / "rl_vs_rule_co2_ci.csv", index=False)
    fc = failure_cases(df)
    fc.to_csv(RESULTS_DIR / "failure_cases.csv", index=False)
    fig_savings_by_family(df)
    fig_tradeoff(df, "in_distribution", "tradeoff_id.png")
    fig_tradeoff(df, "compound_shock", "tradeoff_compound_shock.png")
    fig_completion(df)
    fig_training_curves()
    ci = improvement(df, NAIVE, "cost_eur")
    meta = {"families": list(df.family.unique()), "policies": sorted(df.policy.unique())}
    (RESULTS_DIR / "report_meta.json").write_text(json.dumps(meta, indent=2))
    print("report written to", RESULTS_DIR)
    print(ci[ci.policy == RL_LABEL][["family", "pct", "pct_std", "wins"]].round(2).to_string(index=False))


if __name__ == "__main__":
    build_report()
