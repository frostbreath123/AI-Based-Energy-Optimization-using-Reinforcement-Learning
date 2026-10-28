"""Rollouts, policy wrappers and the in-distribution / OOD evaluation suites."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from energyrl.baselines import NaivePolicy, RuleBasedPolicy
from energyrl.config import MODELS_DIR, RESULTS_DIR, load_config
from energyrl.data.preprocess import load_processed
from energyrl.env import EnergyEnv, observe
from energyrl.ood import ood_families, ood_spec
from energyrl.reward import RewardWeights, step_reward
from energyrl.scenarios import DistSpec, Scenario, ScenarioGenerator
from energyrl.sim import FacilitySim


class Policy(Protocol):
    name: str

    def act(self, sim: FacilitySim) -> np.ndarray: ...


class RLPolicy:
    """Deterministic inference with a trained Stable-Baselines3 model."""

    def __init__(self, model: str | Path | BaseAlgorithm, cfg: dict, name: str = "RL agent (PPO)"):
        """`model` is a saved-model path or an already-instantiated SB3 model (used during training)."""
        if isinstance(model, (str, Path)):
            path = Path(model)
            if not path.exists() and not path.with_suffix(".zip").exists():
                raise FileNotFoundError(f"Trained model not found: {path}. Train with: python -m energyrl.train")
            try:
                model = PPO.load(str(path), device="cpu")
            except Exception as exc:                                     # corrupted / incompatible file
                raise RuntimeError(f"Could not load model {path}: {exc}") from exc
        self.model, self.cfg, self.name = model, cfg, name

    def act(self, sim: FacilitySim) -> np.ndarray:
        action, _ = self.model.predict(observe(sim, self.cfg), deterministic=True)
        return np.asarray(action, dtype=int)


def rollout(policy: Policy, scenario: Scenario, cfg: dict, weights: RewardWeights | None = None) -> dict:
    """Run one episode. Returns metrics + trace + episode reward + per-step inference time."""
    weights = weights or RewardWeights.from_config(cfg)
    sim = FacilitySim(scenario, cfg)
    cost_ref, co2_ref = EnergyEnv.reference(scenario, cfg)
    ret, t_inf = 0.0, 0.0
    while not sim.finished:
        t0 = time.perf_counter()
        a = policy.act(sim)
        t_inf += time.perf_counter() - t0
        res = sim.step(a)
        ret += step_reward(res, weights, cost_ref, co2_ref)
    m = sim.metrics()
    m.update(reward=ret, inference_ms_per_step=1000 * t_inf / sim.H, cost_ref=cost_ref, co2_ref=co2_ref)
    return {"metrics": m, "trace": sim.trace, "sim": sim}


def make_scenarios(gen: ScenarioGenerator, spec: DistSpec, n: int, seed: int, tag: str) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    return [gen.sample(rng, f"{tag}-{i}", spec) for i in range(n)]


def evaluate_suite(policies: dict[str, Policy], scenarios: list[Scenario], cfg: dict, suite: str,
                   family: str = "in_distribution") -> pd.DataFrame:
    rows = []
    for i, sc in enumerate(scenarios):
        for pname, pol in policies.items():
            m = rollout(pol, sc, cfg)["metrics"]
            rows.append({"suite": suite, "family": family, "episode": i, "scenario": sc.name, "date": sc.date,
                         "policy": pname, **m})
    return pd.DataFrame(rows)


def build_eval_scenarios(cfg: dict, df: pd.DataFrame | None = None) -> dict[str, list[Scenario]]:
    """Fixed, seeded evaluation scenarios: one in-distribution set + one set per OOD family."""
    df = df if df is not None else load_processed()
    gen = ScenarioGenerator(df, cfg)
    train = gen.spec
    ev = cfg["evaluation"]
    out = {"in_distribution": make_scenarios(gen, train.with_(season="test"), ev["episodes_id"],
                                             ev["eval_seed"], "id")}
    for k, fam in enumerate(ood_families(train)):
        out[fam.key] = make_scenarios(gen, ood_spec(train, fam), ev["episodes_ood"], ev["eval_seed"] + 100 + k, fam.key)
    return out


def load_policies(cfg: dict, include_rl: bool = True) -> dict[str, Policy]:
    pols: dict[str, Policy] = {"Naive schedule": NaivePolicy(), "Rule-based": RuleBasedPolicy()}
    if include_rl:
        for seed in cfg["training"]["seeds"]:
            p = MODELS_DIR / f"ppo_seed{seed}" / "best_model.zip"
            if p.exists():
                pols[f"RL agent (seed {seed})"] = RLPolicy(p, cfg, f"RL agent (seed {seed})")
    return pols


def run_full_evaluation(cfg: dict | None = None, out_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    cfg = cfg or load_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_eval_scenarios(cfg)
    pols = load_policies(cfg)
    frames = []
    for family, scs in scenarios.items():
        suite = "ID" if family == "in_distribution" else "OOD"
        print(f"[eval] {family}: {len(scs)} scenarios x {len(pols)} policies", flush=True)
        frames.append(evaluate_suite(pols, scs, cfg, suite, family))
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(out_dir / "episodes.csv", index=False)
    return res


if __name__ == "__main__":
    r = run_full_evaluation()
    print(r.groupby(["family", "policy"])[["cost_eur", "co2_kg", "completion_rate"]].mean().round(2))
