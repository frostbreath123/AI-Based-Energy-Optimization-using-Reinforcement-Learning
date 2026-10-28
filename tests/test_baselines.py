import numpy as np

from energyrl.baselines import NaivePolicy, RuleBasedPolicy
from energyrl.evaluate import rollout
from energyrl.scenarios import ScenarioGenerator
from energyrl.sim import FacilitySim


def test_naive_follows_plan_exactly(cfg, df):
    gen = ScenarioGenerator(df, cfg)
    for i in range(10):
        sc = gen.sample(np.random.default_rng(i))
        m = rollout(NaivePolicy(), sc, cfg)["metrics"]
        assert m["load_shifted_kwh"] == 0 and m["completion_rate"] == 1.0
        assert m["on_time_rate"] == 1.0 and m["constraint_violations"] == 0


def test_rule_based_saves_cost_and_keeps_production(cfg, df):
    gen = ScenarioGenerator(df, cfg)
    cost_n = cost_r = 0.0
    for i in range(25):
        sc = gen.sample(np.random.default_rng(100 + i))
        n = rollout(NaivePolicy(), sc, cfg)["metrics"]
        r = rollout(RuleBasedPolicy(), sc, cfg)["metrics"]
        cost_n += n["cost_eur"]
        cost_r += r["cost_eur"]
        assert r["jobs_missed"] == 0 and r["shed_events"] == 0
    assert cost_r < cost_n


def test_rule_based_uses_forecast_not_actual_pv(cfg, df):
    """Changing the ACTUAL pv (forecast untouched) must not change the first-hour decision."""
    gen = ScenarioGenerator(df, cfg)
    sc = gen.sample(np.random.default_rng(4))
    a1 = RuleBasedPolicy().act(FacilitySim(sc, cfg))
    sc.pv = sc.pv * 0.0
    a2 = RuleBasedPolicy().act(FacilitySim(sc, cfg))
    assert (a1 == a2).all()
