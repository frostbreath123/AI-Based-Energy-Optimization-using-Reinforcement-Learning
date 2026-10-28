"""Baseline energy-management policies. Each exposes `act(sim) -> action` (see sim.py)."""
from __future__ import annotations

import numpy as np

from energyrl.sim import FacilitySim


class NaivePolicy:
    """Follows the original production schedule: each job runs in its planned window. Battery idle."""
    name = "Naive schedule"

    def act(self, sim: FacilitySim) -> np.ndarray:
        a = np.zeros(sim.K + 1, dtype=int)
        for k, j in enumerate(sim.jobs):
            if j.planned_start <= sim.t < j.planned_start + j.duration:
                a[k] = 1
        a[sim.K] = int(np.argmin(np.abs(sim.levels)))     # idle level
        return a


class RuleBasedPolicy:
    """Transparent energy-aware rules (uses the day-ahead tariff and PV forecast, never the actual PV).

    Jobs:    a job runs now if (a) its slack is exhausted (must run to meet the deadline), or
             (b) now is one of its `remaining` cheapest hours (effective price) before the deadline.
             Effective price = tariff x (1 - 0.8 x forecast PV coverage of the load).
    Battery: charge in the cheapest 30 % of day-ahead hours or on PV surplus; discharge in the most
             expensive 30 % when the price spread pays for the round-trip losses.
    Capacity: mandatory jobs are committed first; optional jobs are added only while the grid limit holds.
    """
    name = "Rule-based"

    def __init__(self, low_q: float = 0.30, high_q: float = 0.70, pv_credit: float = 0.8):
        self.low_q, self.high_q, self.pv_credit = low_q, high_q, pv_credit

    def _eff_price(self, sim: FacilitySim) -> np.ndarray:
        sc = sim.sc
        cover = np.clip(sc.pv_forecast / np.maximum(sc.base_kw + 400.0, 1.0), 0.0, 1.0)
        return sc.price * (1.0 - self.pv_credit * cover)

    def act(self, sim: FacilitySim) -> np.ndarray:
        t, sc, H = sim.t, sim.sc, sim.H
        eff = self._eff_price(sim)
        a = np.zeros(sim.K + 1, dtype=int)
        load = float(sc.base_kw[t])
        mandatory, optional = [], []
        for k in range(sim.n_jobs):
            if not sim.eligible(k):
                continue
            if sim.slack(k) <= 0:
                mandatory.append(k)
                continue
            window = np.arange(t, min(sim.jobs[k].deadline, H))
            order = window[np.argsort(eff[window], kind="stable")]
            if t in order[: int(sim.remaining[k])]:
                optional.append(k)
        for k in mandatory:
            a[k] = 1
            load += sim.jobs[k].power_kw
        for k in optional:
            if load + sim.jobs[k].power_kw <= sim.limit:
                a[k] = 1
                load += sim.jobs[k].power_kw
        # battery
        day = sc.price[:H]
        lo, hi = np.quantile(day, self.low_q), np.quantile(day, self.high_q)
        pays = hi * sim.eff_c * sim.eff_d > lo + 0.01
        pv_surplus = float(sc.pv_forecast[t]) - load > 200.0
        level = 0.0
        if (pays and sc.price[t] <= lo and sim.soc < sim.soc_max - 1) or (pv_surplus and sim.soc < sim.soc_max - 1):
            level = 1.0 if sc.price[t] <= lo else 0.5
            if load + level * sim.p_max > sim.limit:
                level = 0.0
        elif pays and sc.price[t] >= hi and sim.soc > sim.soc_min + 1:
            level = -1.0
        a[sim.K] = int(np.argmin(np.abs(sim.levels - level)))
        return a


class RandomPolicy:
    """Uniform random actions; only used for smoke tests / as a sanity floor."""
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, sim: FacilitySim) -> np.ndarray:
        a = self.rng.integers(0, 2, sim.K + 1)
        a[sim.K] = self.rng.integers(0, len(sim.levels))
        return a
