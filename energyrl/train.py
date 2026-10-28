"""Train PPO agents on the custom environment.

    python -m energyrl.train                       # all seeds in config, default reward
    python -m energyrl.train --timesteps 20000 --seeds 0 --tag smoke
    python -m energyrl.train --tag co2x4 --w-co2 2.0 --seeds 0   # reward-weight sensitivity run

Artifacts per run (models/<tag>_seed<S>/): checkpoints/, best_model.zip (best validation reward),
final_model.zip, train_log.csv (validation curve), config_snapshot.json.
Only the TRAIN date pool is used for training; the VAL pool selects the best checkpoint.
The OOD scenarios are never touched here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from energyrl.baselines import NaivePolicy
from energyrl.config import MODELS_DIR, load_config
from energyrl.data.preprocess import load_processed
from energyrl.env import EnergyEnv
from energyrl.evaluate import RLPolicy, evaluate_suite, make_scenarios
from energyrl.reward import RewardWeights
from energyrl.scenarios import DistSpec, ScenarioGenerator


class ValidationCallback(BaseCallback):
    """Every `every` steps: roll out the current policy on fixed validation scenarios, log, keep the best."""

    def __init__(self, cfg: dict, val_scenarios, weights: RewardWeights, out_dir: Path, every: int):
        super().__init__()
        self.cfg, self.val, self.weights, self.out, self.every = cfg, val_scenarios, weights, out_dir, every
        self.best = -np.inf
        self.rows: list[dict] = []
        self._next = every
        self._naive = evaluate_suite({"naive": NaivePolicy()}, val_scenarios, cfg, "val")
        self._t0 = time.time()

    def _validate(self) -> None:
        df = evaluate_suite({"rl": RLPolicy(self.model, self.cfg)}, self.val, self.cfg, "val")
        naive = self._naive
        row = {"timesteps": self.num_timesteps, "val_reward": df.reward.mean(),
               "val_cost_eur": df.cost_eur.mean(), "val_cost_saving_pct": 100 * (1 - df.cost_eur.sum() / naive.cost_eur.sum()),
               "val_co2_saving_pct": 100 * (1 - df.co2_kg.sum() / naive.co2_kg.sum()),
               "val_completion_rate": df.completion_rate.mean(), "val_jobs_missed": df.jobs_missed.mean(),
               "val_violations": df.constraint_violations.mean(), "elapsed_s": time.time() - self._t0}
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.out / "train_log.csv", index=False)
        if row["val_reward"] > self.best:
            self.best = row["val_reward"]
            self.model.save(str(self.out / "best_model"))
        print(f"  [val @ {self.num_timesteps}] reward {row['val_reward']:.2f} cost saving {row['val_cost_saving_pct']:.1f}% "
              f"co2 {row['val_co2_saving_pct']:.1f}% completion {row['val_completion_rate']:.3f}", flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            self._validate()
        return True


def train_one(cfg: dict, df: pd.DataFrame, seed: int, tag: str, total_timesteps: int,
              weights: RewardWeights) -> Path:
    tc = cfg["training"]
    out = MODELS_DIR / f"{tag}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    set_random_seed(seed)
    gen = ScenarioGenerator(df, cfg)
    spec = DistSpec.from_config(cfg)

    def make(i: int):
        def _f():
            env = EnergyEnv(cfg, gen, spec.with_(season="train"), weights)
            env.reset(seed=seed * 1000 + i)
            return Monitor(env)
        return _f

    venv = DummyVecEnv([make(i) for i in range(tc["n_envs"])])
    venv.seed(seed)
    val = make_scenarios(gen, spec.with_(season="val"), tc["eval_episodes"], 777 + seed, "val")
    lr = tc["learning_rate"]
    if tc.get("lr_schedule") == "linear":
        lr = (lambda base: (lambda progress_remaining: base * progress_remaining))(lr)
    model = PPO("MlpPolicy", venv, learning_rate=lr, n_steps=tc["n_steps"],
                batch_size=tc["batch_size"], n_epochs=tc["n_epochs"], gamma=tc["gamma"],
                gae_lambda=tc["gae_lambda"], ent_coef=tc["ent_coef"], clip_range=tc["clip_range"],
                policy_kwargs=dict(net_arch=list(tc["net_arch"])), seed=seed, device="cpu", verbose=0)
    callbacks = [CheckpointCallback(max(tc["checkpoint_every"] // tc["n_envs"], 1), str(out / "checkpoints"),
                                    name_prefix="ckpt"),
                 ValidationCallback(cfg, val, weights, out, tc["eval_every"])]
    (out / "config_snapshot.json").write_text(json.dumps(
        {"seed": seed, "tag": tag, "timesteps": total_timesteps, "weights": weights.__dict__,
         "training": tc}, indent=2))
    t0 = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("Interrupted: saving current model as final_model.zip", flush=True)
    model.save(str(out / "final_model"))
    if not (out / "best_model.zip").exists():
        model.save(str(out / "best_model"))
    print(f"[train] {tag} seed {seed}: {time.time() - t0:.0f}s -> {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--tag", default="ppo")
    ap.add_argument("--w-cost", type=float)
    ap.add_argument("--w-co2", type=float)
    ap.add_argument("--w-miss", type=float)
    ap.add_argument("--w-late", type=float)
    ap.add_argument("--n-steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr-schedule", choices=["constant", "linear"])
    args = ap.parse_args()
    cfg = load_config()
    df = load_processed()
    weights = RewardWeights.from_config(cfg)
    over = {k: v for k, v in dict(w_cost=args.w_cost, w_co2=args.w_co2, w_miss=args.w_miss,
                                  w_late=args.w_late).items() if v is not None}
    weights = weights.with_(**over)
    for key, val in dict(n_steps=args.n_steps, batch_size=args.batch_size, lr_schedule=args.lr_schedule).items():
        if val is not None:
            cfg["training"][key] = val
    steps = args.timesteps or cfg["training"]["total_timesteps"]
    for seed in (args.seeds if args.seeds else cfg["training"]["seeds"]):
        train_one(cfg, df, seed, args.tag, steps, weights)


if __name__ == "__main__":
    main()
