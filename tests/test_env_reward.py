import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check
from stable_baselines3.common.env_checker import check_env as sb3_check

from energyrl.baselines import NaivePolicy
from energyrl.env import EnergyEnv, obs_size
from energyrl.evaluate import rollout
from energyrl.reward import RewardWeights
from energyrl.scenarios import ScenarioGenerator


class DoNothing:
    name = "idle"

    def act(self, sim):
        a = np.zeros(sim.K + 1, dtype=int)
        a[sim.K] = 2
        return a


@pytest.fixture
def env(cfg, df):
    return EnergyEnv(cfg, ScenarioGenerator(df, cfg))


def test_gymnasium_and_sb3_compatibility(env):
    gym_check(env, skip_render_check=True)
    sb3_check(env, warn=True)


def test_observation_shape_and_finite(env, cfg):
    obs, _ = env.reset(seed=1)
    assert obs.shape == (obs_size(cfg),) and np.isfinite(obs).all()
    for _ in range(24):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
    assert term and "episode_metrics" in info


def test_determinism_same_seed(cfg, df):
    def episode(seed):
        e = EnergyEnv(cfg, ScenarioGenerator(df, cfg))
        e.reset(seed=seed)
        e.action_space.seed(seed)
        rs = [e.step(e.action_space.sample())[1] for _ in range(24)]
        return rs, e.sim.metrics()["cost_eur"]
    assert episode(5) == episode(5)
    assert episode(5) != episode(6)


def test_naive_reward_is_minus_weighted_sum(cfg, df):
    """Normalisation check: naive return = -S*(w_cost + w_co2) (battery idle, nothing late)."""
    gen = ScenarioGenerator(df, cfg)
    sc = gen.sample(np.random.default_rng(3))
    w = RewardWeights.from_config(cfg)
    r = rollout(NaivePolicy(), sc, cfg, w)["metrics"]
    assert r["reward"] == pytest.approx(-w.scale * (w.w_cost + w.w_co2), rel=1e-6)


def test_abandoning_production_is_never_rewarded(cfg, df):
    gen = ScenarioGenerator(df, cfg)
    rng = np.random.default_rng(11)
    for _ in range(20):
        sc = gen.sample(rng)
        naive = rollout(NaivePolicy(), sc, cfg)["metrics"]["reward"]
        idle = rollout(DoNothing(), sc, cfg)["metrics"]["reward"]
        assert idle < naive


def test_reward_weights_change_reward(cfg, df):
    gen = ScenarioGenerator(df, cfg)
    sc = gen.sample(np.random.default_rng(2))
    a = rollout(DoNothing(), sc, cfg, RewardWeights.from_config(cfg))["metrics"]["reward"]
    b = rollout(DoNothing(), sc, cfg, RewardWeights.from_config(cfg).with_(w_miss=0.0, w_late=0.0))["metrics"]["reward"]
    assert b > a
