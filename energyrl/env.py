"""Gymnasium environment wrapping FacilitySim with the normalized multi-objective reward.

Observation (float32, roughly in [-1, 1]; fixed normalisation constants from the config):
  time: t/H, sin/cos(2 pi hour/24), weekend flag
  tariff now + next P hours (absolute, plus the same window rescaled to 0..1 and the share of the window cheaper than now), grid carbon now + next C hours, PV now (actual) + forecast next F hours
  inflexible load now + next 2 hours, battery SoC fraction, peak-so-far / grid limit
  per job slot (K slots, sorted by deadline): [active, released, remaining/4, slack/8, power/600]
Action: MultiDiscrete([2]*K + [len(battery_levels)])  -> run flag per job, battery level index.
Reward: see energyrl/reward.py and docs/REWARD_DESIGN.md.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from energyrl.baselines import NaivePolicy
from energyrl.reward import RewardWeights, step_reward
from energyrl.scenarios import DistSpec, Scenario, ScenarioGenerator
from energyrl.sim import FacilitySim


def obs_size(cfg: dict) -> int:
    o = cfg["observation"]
    return 4 + 2 * (1 + o["price_lookahead"]) + 1 + (1 + o["carbon_lookahead"]) + (1 + o["pv_lookahead"]) + 3 + 2 \
        + 5 * cfg["site"]["max_jobs"]


def observe(sim: FacilitySim, cfg: dict) -> np.ndarray:
    o, s = cfg["observation"], cfg["site"]
    sc, t, H = sim.sc, min(sim.t, sim.H - 1), sim.H
    hour = t % 24
    n_total = len(sc.price)
    idx = lambda n: np.minimum(np.arange(t, t + n + 1), n_total - 1)  # noqa: E731
    price = sc.price[idx(o["price_lookahead"])]
    rel = (price - price.min()) / max(price.max() - price.min(), 1e-6)   # scale-free view: 0 = cheapest hour ahead, 1 = dearest
    parts: list[np.ndarray] = [
        np.array([t / H, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), float(sc.weekend)]),
        price / o["price_norm_eur_kwh"],
        rel,
        np.array([np.mean(price < price[0])]),                             # share of the look-ahead window cheaper than now
        sc.carbon[idx(o["carbon_lookahead"])] / o["carbon_norm_g_kwh"],
        np.concatenate([[sc.pv[t]], sc.pv_forecast[idx(o["pv_lookahead"])[1:]]]) / s["pv_kwp"],
        sc.base_kw[idx(2)] / sim.limit,
        np.array([sim.soc / sim.cap, sim.peak / sim.limit]),
    ]
    jobs = np.zeros((s["max_jobs"], 5))
    for k in range(sim.n_jobs):
        j = sim.jobs[k]
        active = sim.remaining[k] > 0
        jobs[k] = [float(active), float(active and j.release <= sim.t), sim.remaining[k] / 4.0,
                   np.clip(sim.slack(k), -8, 8) / 8.0 if active else 0.0, j.power_kw / 600.0]
    parts.append(jobs.ravel())
    return np.concatenate(parts).astype(np.float32)


class EnergyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, generator: ScenarioGenerator, spec: DistSpec | None = None,
                 weights: RewardWeights | None = None):
        super().__init__()
        self.cfg, self.gen = cfg, generator
        self.dist = spec or generator.spec
        self.weights = weights or RewardWeights.from_config(cfg)
        K = cfg["site"]["max_jobs"]
        self.action_space = spaces.MultiDiscrete([2] * K + [len(cfg["site"]["battery_levels"])])
        self.observation_space = spaces.Box(-10.0, 10.0, (obs_size(cfg),), dtype=np.float32)
        self.sim: FacilitySim | None = None
        self.cost_ref = self.co2_ref = 1.0

    @staticmethod
    def reference(scenario: Scenario, cfg: dict) -> tuple[float, float]:
        """Cost (EUR) and emissions (kg) of the naive schedule on this scenario (reward normalisers)."""
        sim = FacilitySim(scenario, cfg)
        pol = NaivePolicy()
        while not sim.finished:
            sim.step(pol.act(sim))
        m = sim.metrics()
        return max(m["cost_eur"], 50.0), max(m["co2_kg"], 50.0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        scenario = (options or {}).get("scenario")
        if scenario is None:
            scenario = self.gen.sample(self.np_random, "train", self.dist)
        self.sim = FacilitySim(scenario, self.cfg)
        self.cost_ref, self.co2_ref = self.reference(scenario, self.cfg)
        return observe(self.sim, self.cfg), {"scenario": scenario.name}

    def step(self, action):
        assert self.sim is not None, "call reset() first"
        res = self.sim.step(np.asarray(action))
        reward = step_reward(res, self.weights, self.cost_ref, self.co2_ref)
        obs = observe(self.sim, self.cfg)
        info: dict[str, Any] = {}
        if res.done:
            info["episode_metrics"] = self.sim.metrics()
            info["cost_ref"], info["co2_ref"] = self.cost_ref, self.co2_ref
        return obs, float(reward), res.done, False, info
