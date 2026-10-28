import numpy as np
import pytest

from energyrl.sim import FacilitySim

K = 8


def act(cfg, runs=(), level=0.0):
    a = np.zeros(K + 1, dtype=int)
    for k in runs:
        a[k] = 1
    a[K] = cfg["site"]["battery_levels"].index(level)
    return a


def test_idle_cost_and_carbon_are_exact(cfg, manual):
    sim = FacilitySim(manual(price=0.2, carbon=500, base=100), cfg)
    while not sim.finished:
        sim.step(act(cfg))
    m = sim.metrics()
    assert m["energy_cost_eur"] == pytest.approx(24 * 100 * 0.2)
    assert m["co2_kg"] == pytest.approx(24 * 100 * 0.5)
    assert m["demand_charge_eur"] == pytest.approx(100 * cfg["site"]["demand_charge_eur_kw_day"])


def test_job_completes_and_adds_energy(cfg, manual, job):
    sc = manual(jobs=[job(planned_start=2, duration=3, power_kw=400, release=0, deadline=10)])
    sim = FacilitySim(sc, cfg)
    for t in range(24):
        sim.step(act(cfg, runs=[0] if 2 <= t < 5 else []))
    m = sim.metrics()
    assert m["jobs_completed"] == 1 and m["on_time_rate"] == 1.0 and m["jobs_missed"] == 0
    assert m["consumption_kwh"] == pytest.approx(24 * 100 + 3 * 400)


def test_premature_job_request_is_ignored_and_counted(cfg, manual, job):
    sc = manual(jobs=[job(planned_start=8, duration=2, power_kw=400, release=6, deadline=14)])
    sim = FacilitySim(sc, cfg)
    r = sim.step(act(cfg, runs=[0]))            # t=0 < release
    assert r.invalid_requests == 1
    assert sim.remaining[0] == 2 and r.import_kw == pytest.approx(100)


def test_grid_limit_never_exceeded_and_sheds(cfg, manual, job):
    jobs = [job(0, 2, 900, 0, 10), job(0, 2, 900, 0, 10), job(0, 2, 900, 0, 10)]
    sim = FacilitySim(manual(jobs=jobs), cfg)
    r = sim.step(act(cfg, runs=[0, 1, 2]))
    assert r.import_kw <= cfg["site"]["grid_import_limit_kw"] + 1e-6
    assert r.shed_events >= 1


def test_battery_limits_and_no_export(cfg, manual):
    b = cfg["site"]["battery"]
    sim = FacilitySim(manual(base=100, pv=0.0), cfg)
    for _ in range(12):
        sim.step(act(cfg, level=1.0))
        assert sim.soc <= b["soc_max"] * b["capacity_kwh"] + 1e-6
    soc_full = sim.soc
    r = sim.step(act(cfg, level=-1.0))          # discharge limited by the 100 kW load: no export
    assert r.export_kw == pytest.approx(0.0, abs=1e-9)
    assert r.import_kw >= -1e-9
    assert sim.soc < soc_full


def test_energy_balance_every_step(cfg, manual, job):
    sc = manual(jobs=[job(3, 3, 300, 0, 12)], pv=250.0)
    sim = FacilitySim(sc, cfg)
    rng = np.random.default_rng(0)
    while not sim.finished:
        a = np.concatenate([rng.integers(0, 2, K), [rng.integers(0, 5)]])
        sim.step(a)
    for r in sim.trace:
        lhs = r["import"] - r["export"]
        rhs = r["load"] + max(r["battery_kw"], 0) - r["pv"] - max(-r["battery_kw"], 0)
        assert lhs == pytest.approx(rhs, abs=1e-6)


def test_missed_job_and_tardiness(cfg, manual, job):
    sc = manual(jobs=[job(0, 3, 300, 0, 5)])
    sim = FacilitySim(sc, cfg)
    for t in range(24):
        sim.step(act(cfg, runs=[0] if t in (10, 11, 12) else []))   # finishes at hour 13 -> 8 h late
    m = sim.metrics()
    assert m["jobs_completed"] == 1 and m["jobs_late"] == 1 and m["tardiness_hours"] == 8
    sim2 = FacilitySim(sc, cfg)
    for _ in range(24):
        sim2.step(act(cfg))
    assert sim2.metrics()["jobs_missed"] == 1


def test_step_after_end_raises(cfg, manual):
    sim = FacilitySim(manual(), cfg)
    for _ in range(24):
        sim.step(act(cfg))
    with pytest.raises(RuntimeError):
        sim.step(act(cfg))


def test_stepwise_battery_settlement_equals_end_of_day_settlement(cfg, manual):
    """Per-step settlement of stored energy must sum to -(soc_end - soc_start) x value (same episode total as before)."""
    sim = FacilitySim(manual(price=0.2, base=100), cfg)
    total = 0.0
    rng = np.random.default_rng(3)
    while not sim.finished:
        a = np.zeros(K + 1, dtype=int)
        a[K] = rng.integers(0, 5)
        total += sim.step(a).settlement
    assert total == pytest.approx(-(sim.soc - sim.soc0) * 0.2 * cfg["site"]["battery"]["eff_discharge"])
    assert sim.metrics()["battery_settlement_eur"] == pytest.approx(total)
