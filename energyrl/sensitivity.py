"""Reward-weight sensitivity: evaluate agents trained with different reward priorities on the same
in-distribution test scenarios and report the raw KPIs (not rewards, so runs are comparable).

    python -m energyrl.sensitivity
"""
from __future__ import annotations

from energyrl.baselines import NaivePolicy, RuleBasedPolicy
from energyrl.config import MODELS_DIR, RESULTS_DIR, load_config
from energyrl.data.preprocess import load_processed
from energyrl.evaluate import RLPolicy, evaluate_suite, make_scenarios
from energyrl.scenarios import ScenarioGenerator

VARIANTS = {
    "ppo_seed0": "Default (w_cost=1, w_co2=0.5)",
    "sens_cost_only_seed0": "Cost only (w_co2=0)",
    "sens_carbon_heavy_seed0": "Carbon-heavy (w_co2=2)",
    "sens_no_production_penalty_seed0": "No production penalty (w_miss=w_late=0)",
}


def main() -> None:
    cfg = load_config()
    df = load_processed()
    gen = ScenarioGenerator(df, cfg)
    scs = make_scenarios(gen, gen.spec.with_(season="test"), cfg["evaluation"]["episodes_id"],
                         cfg["evaluation"]["eval_seed"], "id")
    pols = {"Naive schedule": NaivePolicy(), "Rule-based": RuleBasedPolicy()}
    for folder, label in VARIANTS.items():
        p = MODELS_DIR / folder / "best_model.zip"
        if p.exists():
            pols[label] = RLPolicy(p, cfg, label)
        else:
            print(f"[sensitivity] missing {p}, skipped")
    res = evaluate_suite(pols, scs, cfg, "ID", "in_distribution")
    cols = ["cost_eur", "co2_kg", "completion_rate", "jobs_missed", "tardiness_hours", "constraint_violations",
            "load_shifted_kwh", "peak_kw"]
    tab = res.groupby("policy", sort=False)[cols].mean()
    base = tab.loc["Naive schedule"]
    tab["cost_saving_pct"] = 100 * (1 - tab.cost_eur / base.cost_eur)
    tab["co2_saving_pct"] = 100 * (1 - tab.co2_kg / base.co2_kg)
    RESULTS_DIR.mkdir(exist_ok=True)
    tab.to_csv(RESULTS_DIR / "sensitivity.csv")
    print(tab.round(2).to_string())


if __name__ == "__main__":
    main()
