"""Compare training variants on VALIDATION days only (never test or OOD scenarios).

    python scripts/select_reward_weights.py
Evaluates models/exp_*/best_model.zip against the naive and rule-based baselines on 60 validation days and
writes results/weight_selection_val.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from energyrl.baselines import NaivePolicy, RuleBasedPolicy  # noqa: E402
from energyrl.config import MODELS_DIR, RESULTS_DIR, load_config  # noqa: E402
from energyrl.data.preprocess import load_processed  # noqa: E402
from energyrl.evaluate import RLPolicy, evaluate_suite, make_scenarios  # noqa: E402
from energyrl.scenarios import ScenarioGenerator  # noqa: E402


def main() -> None:
    cfg, df = load_config(), load_processed()
    gen = ScenarioGenerator(df, cfg)
    val = make_scenarios(gen, gen.spec.with_(season="val"), 60, 31337, "val")
    pols = {"Naive": NaivePolicy(), "Rule-based": RuleBasedPolicy()}
    for p in sorted(MODELS_DIR.glob("exp_*/best_model.zip")):
        pols[p.parent.name] = RLPolicy(p, cfg)
    r = evaluate_suite(pols, val, cfg, "val")
    t = r.groupby("policy", sort=False)[["cost_eur", "co2_kg", "completion_rate", "tardiness_hours", "shed_events"]].mean()
    t["cost_saving_pct"] = 100 * (1 - t.cost_eur / t.loc["Naive"].cost_eur)
    t["co2_saving_pct"] = 100 * (1 - t.co2_kg / t.loc["Naive"].co2_kg)
    t["cost_vs_rule_pct"] = 100 * (1 - t.cost_eur / t.loc["Rule-based"].cost_eur)
    RESULTS_DIR.mkdir(exist_ok=True)
    t.to_csv(RESULTS_DIR / "weight_selection_val.csv")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
