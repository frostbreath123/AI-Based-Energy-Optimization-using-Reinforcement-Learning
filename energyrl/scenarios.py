"""Scenario construction: real public series + synthetic tariff / load / jobs.

A `Scenario` is one day (24 h) plus `lookahead` extra hours of forecast data, with all the
exogenous inputs the simulator needs. `ScenarioGenerator` draws scenarios from a
`DistSpec` (a set of parameter ranges). Training uses the in-distribution spec from the config;
the OOD suite (energyrl/ood.py) overrides ranges to leave that spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Job:
    """A preemptible production batch. `duration` hours of work at `power_kw`."""
    planned_start: int
    duration: int
    power_kw: float
    release: int
    deadline: int          # job must have completed `duration` run-hours by the END of hour deadline-1

    @property
    def energy_kwh(self) -> float:
        return self.duration * self.power_kw


@dataclass
class Scenario:
    name: str
    date: str
    weekend: bool
    spot: np.ndarray           # EUR/kWh, length n
    price: np.ndarray          # EUR/kWh total tariff (spot * scale + network adder + fixed), length n
    carbon: np.ndarray         # gCO2/kWh, length n
    pv: np.ndarray             # kW actual, length n
    pv_forecast: np.ndarray    # kW forecast issued day-ahead, length n
    base_kw: np.ndarray        # kW inflexible load, length n
    jobs: list[Job]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistSpec:
    """Parameter ranges ([lo, hi]) describing a scenario distribution."""
    price_scale: tuple[float, float] = (0.7, 1.4)
    pv_scale: tuple[float, float] = (0.5, 1.3)
    forecast_sd: tuple[float, float] = (0.10, 0.25)
    drop_prob: float = 0.10
    drop_hours: tuple[int, int] = (1, 2)
    drop_depth: tuple[float, float] = (0.2, 0.5)
    n_jobs: tuple[int, int] = (4, 6)
    power_scale: tuple[float, float] = (0.8, 1.2)
    slack_mean: tuple[float, float] = (2.0, 6.0)
    base_scale: tuple[float, float] = (0.9, 1.1)
    tariff: str = "A"                               # off-peak placement: A (training) or B (OOD)
    tariff_options: tuple[str, ...] = ("CU", "LU")  # TURPE 7 HTA utilisation options drawn per scenario
    price_spike: tuple[float, float] = (1.0, 1.0)   # multiplier on the spot price in the evening window
    late_start_bias: float = 0.0                    # 0..1 shift of planned starts toward the afternoon
    exclude_pv_low: float = 0.0                     # rejection region (training only), see config
    exclude_slack_tight: float = 0.0
    season: str = "train"                           # which date pool to use: train|val|test|ood_winter
    load_clusters: tuple[int, ...] = (1, 10)        # ELMAS industrial load-shape clusters to draw from
    ood_load_clusters: tuple[int, ...] = (14, 11)   # held-out clusters (used only by the OOD suite)

    @classmethod
    def from_config(cls, cfg: dict) -> "DistSpec":
        t = cfg["train_dist"]
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in t.items()
              if k in cls.__dataclass_fields__}
        d = cfg["data"]
        return cls(load_clusters=tuple(d["load_clusters_train"]), ood_load_clusters=tuple(d["load_clusters_ood"]), **kw)

    def with_(self, **kw) -> "DistSpec":
        return replace(self, **kw)


# --------------------------------------------------------------------------- date pools
def split_dates(df: pd.DataFrame, cfg: dict) -> dict[str, list[pd.Timestamp]]:
    """Deterministic disjoint train/val/test pools within the in-distribution season,
    plus a winter pool that is never used for training or validation."""
    d = cfg["data"]
    days = sorted(df.index.normalize().unique())
    in_season = [x for x in days if x.month in d["train_months"]]
    winter = [x for x in days if x.month in d["ood_season_months"]]
    rng = np.random.default_rng(cfg["project"]["seed"])
    perm = rng.permutation(len(in_season))
    n = len(in_season)
    n_tr = int(round(d["split_fractions"]["train"] * n))
    n_va = int(round(d["split_fractions"]["val"] * n))
    pick = lambda idx: sorted(in_season[i] for i in idx)  # noqa: E731
    return {"train": pick(perm[:n_tr]), "val": pick(perm[n_tr:n_tr + n_va]),
            "test": pick(perm[n_tr + n_va:]), "ood_winter": winter}


# --------------------------------------------------------------------------- tariff
def _in_ranges(hours: np.ndarray, ranges: list[list[int]]) -> np.ndarray:
    m = np.zeros(hours.shape, dtype=bool)
    for lo, hi in ranges:
        m |= (hours >= lo) & (hours < hi)
    return m


def turpe_ranges(index: pd.DatetimeIndex, cfg_t: dict, placement: str) -> np.ndarray:
    """TURPE 7 HTA time range of every local timestamp: one of PTE, HPH, HCH, HPE, HCE."""
    hours, dow, month = index.hour.to_numpy(), index.dayofweek.to_numpy(), index.month.to_numpy()
    high = np.isin(month, cfg_t["high_season_months"])
    pl = cfg_t[f"placement_{placement}"]
    if "offpeak" in pl:
        off = _in_ranges(hours, pl["offpeak"])
    else:
        off = np.where(high, _in_ranges(hours, pl["offpeak_high_season"]), _in_ranges(hours, pl["offpeak_low_season"]))
    off = off | ((dow == 6) & pl["sunday_offpeak"])
    peak = np.isin(month, cfg_t["peak_months"]) & (dow <= 5) & _in_ranges(hours, cfg_t["peak_hours"])
    rng = np.where(high, np.where(off, "HCH", "HPH"), np.where(off, "HCE", "HPE"))
    return np.where(peak, "PTE", rng)


def network_charge(index: pd.DatetimeIndex, cfg_t: dict, placement: str, option: str) -> np.ndarray:
    """Variable TURPE 7 HTA charge in EUR/kWh for every timestamp (c EUR/kWh in the grid / 100)."""
    grid = cfg_t["turpe7_hta_c_eur_kwh"][option]
    return np.array([grid[r] for r in turpe_ranges(index, cfg_t, placement)]) / 100.0


# --------------------------------------------------------------------------- generator
class ScenarioGenerator:
    def __init__(self, df: pd.DataFrame, cfg: dict, spec: DistSpec | None = None):
        self.df, self.cfg = df, cfg
        self.spec = spec or DistSpec.from_config(cfg)
        self.pools = split_dates(df, cfg)
        self.lookahead = max(cfg["observation"]["price_lookahead"], cfg["observation"]["carbon_lookahead"],
                             cfg["observation"]["pv_lookahead"]) + 1
        self.H = cfg["site"]["horizon_hours"]
        # Last day cannot provide look-ahead rows; it is simply excluded from sampling.
        self._last = df.index.normalize().max()

    # -- public
    def sample(self, rng: np.random.Generator, name: str = "scenario", spec: DistSpec | None = None) -> Scenario:
        spec = spec or self.spec
        for _ in range(200):
            p = self._draw_params(rng, spec)
            if spec.exclude_pv_low and p["pv_scale"] < spec.exclude_pv_low and p["slack_mean"] < spec.exclude_slack_tight:
                continue
            break
        pool = [d for d in self.pools[spec.season] if d < self._last]
        day = pool[int(rng.integers(len(pool)))]
        return self._build(day, p, spec, rng, name)

    def draw_params(self, rng: np.random.Generator, spec: DistSpec | None = None) -> dict:
        return self._draw_params(rng, spec or self.spec)

    def available_days(self) -> list[pd.Timestamp]:
        return [d for d in sorted(self.df.index.normalize().unique()) if d < self._last]

    def build(self, day, params: dict, spec: DistSpec | None = None, seed: int = 0, name: str = "custom") -> Scenario:
        """Build a scenario for an explicit day and parameter dict (used by the dashboard explorer)."""
        ts = pd.Timestamp(day)
        ts = ts.tz_localize(self.df.index.tz) if ts.tzinfo is None else ts
        return self._build(ts, params, spec or self.spec, np.random.default_rng(seed), name)

    # -- internals
    def _draw_params(self, rng: np.random.Generator, s: DistSpec) -> dict[str, Any]:
        u = lambda r: float(rng.uniform(*r))  # noqa: E731
        return {
            "price_scale": u(s.price_scale), "pv_scale": u(s.pv_scale), "forecast_sd": u(s.forecast_sd),
            "n_jobs": int(rng.integers(s.n_jobs[0], s.n_jobs[1] + 1)), "power_scale": u(s.power_scale),
            "slack_mean": u(s.slack_mean), "base_scale": u(s.base_scale), "spike": u(s.price_spike),
            "has_drop": bool(rng.random() < s.drop_prob),
            "drop_len": int(rng.integers(s.drop_hours[0], s.drop_hours[1] + 1)),
            "drop_depth": u(s.drop_depth),
        }

    def _window(self, day: pd.Timestamp) -> pd.DataFrame:
        end = day + pd.Timedelta(hours=self.H + self.lookahead - 1)
        w = self.df.loc[day:end]
        if len(w) < self.H + self.lookahead:                       # tail of the year: wrap the last rows
            pad = self.df.iloc[-(self.H + self.lookahead - len(w)):]
            w = pd.concat([w, pad])
        return w.iloc[: self.H + self.lookahead]

    def _build(self, day: pd.Timestamp, p: dict, spec: DistSpec, rng: np.random.Generator, name: str) -> Scenario:
        cfg = self.cfg
        w = self._window(day)
        n = len(w)
        hours = np.array([(i % 24) for i in range(n)])
        weekend = np.array([(day + pd.Timedelta(hours=i)).dayofweek >= 5 for i in range(n)])
        option = str(spec.tariff_options[int(rng.integers(len(spec.tariff_options)))])
        # tariff: real spot scaled + real TURPE 7 HTA network charge + assumed fixed adder
        spot = w["spot_eur_kwh"].to_numpy(float).copy()
        spot_scaled = spot * p["price_scale"]
        evening = (hours >= 17) & (hours < 21)
        spot_scaled = np.where(evening & (spot_scaled > -1), spot_scaled * p["spike"], spot_scaled)
        adder = network_charge(w.index, cfg["tariff"], spec.tariff, option)
        price = spot_scaled + adder + cfg["tariff"]["fixed_adder_eur_kwh"]
        price = np.maximum(price, 0.0)       # customer never gets paid to consume
        carbon = w["carbon_g_kwh"].to_numpy(float)
        # PV: real Renewables.ninja profile x installed kWp x scenario scale; forecast = noisy (issued day-ahead)
        kwp = cfg["site"]["pv_kwp"]
        pv_fc = w["pv_cf"].to_numpy(float) * kwp * p["pv_scale"]
        err = np.exp(rng.normal(0.0, p["forecast_sd"], n))
        pv_forecast = pv_fc * err
        pv = pv_fc.copy()
        if p["has_drop"]:                  # unexpected drop (cloud front/inverter trip): NOT in the forecast
            start = int(rng.integers(9, 15))
            pv[start:start + p["drop_len"]] *= (1.0 - p["drop_depth"])
        cluster = int(spec.load_clusters[int(rng.integers(len(spec.load_clusters)))])
        base = w[f"load_c{cluster}"].to_numpy(float) * cfg["site"]["base_kw_mean"] * p["base_scale"]
        jobs = self._make_jobs(rng, p, spec, base)
        meta = {**{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in p.items()},
                "tariff": spec.tariff, "season": spec.season, "load_cluster": cluster, "tariff_option": option}
        return Scenario(name=name, date=str(day.date()), weekend=bool(weekend[0]), spot=spot_scaled, price=price,
                        carbon=carbon, pv=pv, pv_forecast=pv_forecast, base_kw=base, jobs=jobs, meta=meta)

    def _make_jobs(self, rng: np.random.Generator, p: dict, spec: DistSpec, base: np.ndarray) -> list[Job]:
        """Planned schedule is capacity-feasible by construction (the naive plan never violates the grid limit)."""
        limit = self.cfg["site"]["grid_import_limit_kw"]
        H = self.H
        n_jobs = min(p["n_jobs"], self.cfg["site"]["max_jobs"])
        load = base[:H].copy()
        jobs: list[Job] = []
        for _ in range(n_jobs):
            for _try in range(60):
                dur = int(rng.choice([2, 3, 4]))
                power = float(rng.uniform(250, 550) * p["power_scale"])
                lo, hi = 5, H - dur
                if spec.late_start_bias > 0:
                    lo = int(5 + spec.late_start_bias * 8)
                start = int(rng.integers(lo, hi + 1))
                if np.all(load[start:start + dur] + power <= limit):
                    break
            else:
                continue                      # could not place another job without breaking capacity
            slack = int(np.clip(round(rng.normal(p["slack_mean"], 1.0)), 1, 12))
            lead = int(rng.integers(2, 7))
            deadline = int(min(H, start + dur + slack))
            deadline = max(deadline, start + dur)
            jobs.append(Job(start, dur, power, max(0, start - lead), deadline))
            load[start:start + dur] += power
        jobs.sort(key=lambda j: (j.deadline, j.planned_start))
        return jobs
