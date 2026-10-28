import json

import numpy as np
import pandas as pd
import pytest

from energyrl.data.validate import validate_processed
from energyrl.evaluate import build_eval_scenarios
from energyrl.metrics import NAIVE, RL_LABEL, RULE, improvement, summary
from energyrl.ood import ood_families, ood_spec
from energyrl.scenarios import DistSpec, ScenarioGenerator, split_dates


def test_processed_data_schema_and_units(df):
    rep = validate_processed(df)
    assert rep["ok"] and len(df) % 24 == 0
    assert df.spot_eur_kwh.between(-0.6, 1.2).all() and df.pv_cf.between(0, 1).all()
    assert 10 < df.carbon_g_kwh.mean() < 300 and df.load_c1.between(0.2, 4).all()


def test_validation_rejects_bad_units(df):
    bad = df.copy()
    bad["spot_eur_kwh"] = bad["spot_eur_kwh"] * 1000      # forgot EUR/MWh -> EUR/kWh
    with pytest.raises(ValueError):
        validate_processed(bad)


def test_carbon_intensity_computation(tmp_path, monkeypatch):
    import energyrl.data.preprocess as P
    monkeypatch.setattr(P, "RAW_DIR", tmp_path)
    for m in range(1, 13):
        d = {"unix_seconds": [1672531200 + 3600 * i for i in range(4)],
             "production_types": [{"name": "Nuclear", "data": [100, 100, 0, None]},
                                  {"name": "Fossil hard coal", "data": [100, 300, 100, 100]}]}
        (tmp_path / f"public_power_fr_2023_{m:02d}.json").write_text(json.dumps(d))
    ci = P.load_carbon(2023, "fr", {"Nuclear": 10, "Fossil hard coal": 800})
    assert ci.iloc[0] == pytest.approx((100 * 10 + 100 * 800) / 200)
    assert ci.iloc[2] == pytest.approx(800)


def test_date_pools_disjoint_and_winter_excluded(df, cfg):
    p = split_dates(df, cfg)
    sets = {k: set(v) for k, v in p.items()}
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not sets[a] & sets[b], (a, b)
    assert all(d.month in cfg["data"]["train_months"] for k in ("train", "val", "test") for d in p[k])
    assert all(d.month in cfg["data"]["ood_season_months"] for d in p["ood_winter"])


def test_training_sampler_excludes_joint_corner(df, cfg):
    gen = ScenarioGenerator(df, cfg)
    rng = np.random.default_rng(0)
    spec = gen.spec
    for _ in range(300):
        sc = gen.sample(rng, spec=spec)
        assert not (sc.meta["pv_scale"] < spec.exclude_pv_low and sc.meta["slack_mean"] < spec.exclude_slack_tight)


def test_ood_families_leave_training_ranges(df, cfg):
    gen = ScenarioGenerator(df, cfg)
    train = gen.spec
    rng = np.random.default_rng(1)
    checks = {"price_spike": lambda p: p["price_scale"] > train.price_scale[1],
              "low_renewables": lambda p: p["pv_scale"] < train.pv_scale[0],
              "tight_deadlines": lambda p: p["slack_mean"] < train.slack_mean[0],
              "production_surge": lambda p: p["n_jobs"] > train.n_jobs[1],
              "renewable_drop": lambda p: p["drop_len"] > train.drop_hours[1] and p["has_drop"]}
    fams = {f.key: f for f in ood_families(train)}
    for key, fn in checks.items():
        spec = ood_spec(train, fams[key])
        assert all(fn(gen.draw_params(rng, spec)) for _ in range(50)), key
    s = ood_spec(train, fams["combo_familiar"])
    assert train.pv_scale[0] <= s.pv_scale[0] and s.pv_scale[1] < train.exclude_pv_low
    assert train.slack_mean[0] <= s.slack_mean[0] and s.slack_mean[1] < train.exclude_slack_tight
    assert fams["tariff_restructure"].spec_overrides["tariff"] == "B" and train.tariff == "A"


def test_eval_scenarios_are_reproducible_and_held_out(df, cfg):
    a = build_eval_scenarios(cfg, df)
    b = build_eval_scenarios(cfg, df)
    assert set(a) == {"in_distribution"} | {f.key for f in ood_families(DistSpec.from_config(cfg))}
    for k in a:
        assert [s.date for s in a[k]] == [s.date for s in b[k]]
        assert all((x.price == y.price).all() for x, y in zip(a[k], b[k]))
    pools = split_dates(df, cfg)
    train_days = {str(d.date()) for d in pools["train"] + pools["val"]}
    for k, scs in a.items():
        assert not ({s.date for s in scs} & train_days), k


def test_metrics_aggregation_and_improvement():
    rows = []
    for ep in range(4):
        for pol, c in [(NAIVE, 100), (RULE, 90), ("RL agent (seed 0)", 80), ("RL agent (seed 1)", 70)]:
            rows.append({"suite": "ID", "family": "f", "episode": ep, "policy": pol, "cost_eur": c, "co2_kg": c})
    df = pd.DataFrame(rows)
    imp = improvement(df, NAIVE, "cost_eur")
    rl = imp[imp.policy == RL_LABEL].iloc[0]
    assert rl.pct == pytest.approx(25.0) and rl.wins == 1.0      # mean of seeds = 75 vs 100
    s = summary(df, ["cost_eur"])
    assert s[s.policy == RL_LABEL].cost_eur.iloc[0] == pytest.approx(75.0)
    assert s[s.policy == RL_LABEL].cost_eur_std_seed.iloc[0] == pytest.approx(np.std([80, 70], ddof=1))
