"""Reward function (documented in docs/REWARD_DESIGN.md).

For every hour t the agent receives

    r_t = - S * ( w_c * cost_t / C_ref  +  w_e * co2_t / E_ref
                  + w_l * overdue_t + w_s * shed_t + w_i * invalid_t )

and at the final hour additionally

    r_T -= S * w_m * (missed_jobs + missed_work_fraction)

with  cost_t  = import_kWh * tariff + demand-charge increment - export revenue
                + settlement_t (change in stored battery energy x mean tariff x discharge efficiency, negative when charging) [EUR]
      co2_t   = import_kWh * grid_intensity / 1000                               [kg CO2e]
      C_ref, E_ref = cost / emissions of the NAIVE schedule on the same scenario [EUR, kg]
      overdue_t = jobs unfinished and past deadline at the end of the hour       [job-hours]
      shed_t  = jobs forcibly shed because of the grid import limit              [events]
      invalid_t = ineligible requests (job before release, battery full/empty)   [events]
S = reward scale (only conditions the optimiser).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from energyrl.sim import StepResult


@dataclass(frozen=True)
class RewardWeights:
    scale: float = 10.0
    w_cost: float = 1.0
    w_co2: float = 0.5
    w_late: float = 0.03
    w_miss: float = 0.30
    w_shed: float = 0.10
    w_invalid: float = 0.01

    @classmethod
    def from_config(cls, cfg: dict) -> "RewardWeights":
        r = cfg["reward"]
        return cls(scale=r["scale"], w_cost=r["w_cost"], w_co2=r["w_co2"], w_late=r["w_late"],
                   w_miss=r["w_miss"], w_shed=r["w_shed"], w_invalid=r["w_invalid"])

    def with_(self, **kw) -> "RewardWeights":
        return replace(self, **kw)


def step_reward(res: StepResult, w: RewardWeights, cost_ref: float, co2_ref: float) -> float:
    pen = (w.w_cost * (res.cost + res.settlement) / cost_ref + w.w_co2 * res.co2_kg / co2_ref
           + w.w_late * res.overdue_jobs + w.w_shed * res.shed_events + w.w_invalid * res.invalid_requests)
    if res.done:
        pen += w.w_miss * (res.missed_jobs + res.missed_work_frac)
    return -w.scale * pen
