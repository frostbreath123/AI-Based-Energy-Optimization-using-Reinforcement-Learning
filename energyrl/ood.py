"""Held-out out-of-distribution (OOD) scenario suite.

Every family leaves the training distribution (config `train_dist`) in a documented way.
Dates for the synthetic-perturbation families come from the held-out TEST pool; the winter family
uses real winter days that are never part of the train/val pools. No OOD scenario is used for
training, model selection or hyper-parameter tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

from energyrl.scenarios import DistSpec


@dataclass(frozen=True)
class OODFamily:
    key: str
    title: str
    what_changes: str          # how it differs from the training distribution
    spec_overrides: dict


def ood_families(train: DistSpec) -> list[OODFamily]:
    t = train
    return [
        OODFamily("price_spike", "Price spikes",
                  f"Spot price scale 3-5x and an extra 2-3x evening (17-21 h) spike. Training: scale {t.price_scale[0]}-{t.price_scale[1]}x, evening spike up to {t.price_spike[1]}x.",
                  dict(price_scale=(3.0, 5.0), price_spike=(2.0, 3.0), season="test")),
        OODFamily("tariff_restructure", "Unseen off-peak placement (TURPE 7 solar-following hours)",
                  "Same real TURPE 7 HTA prices, but the low-season off-peak hours move from 22-06 h to 02-06 h and 11-15 h (the placement the CRE wants to favour). Training only saw placement A (off-peak 22-06 h).",
                  dict(tariff="B", season="test")),
        OODFamily("low_renewables", "Low, noisy renewables",
                  f"PV scale 0.15-0.35x with forecast error sd 30-50 %. Training: PV scale {t.pv_scale[0]}-{t.pv_scale[1]}x, sd {t.forecast_sd[0]:.0%}-{t.forecast_sd[1]:.0%}.",
                  dict(pv_scale=(0.15, 0.35), forecast_sd=(0.30, 0.50), season="test")),
        OODFamily("renewable_drop", "Unforecast PV collapse",
                  "Sunny day, but PV falls 85-98 % for 3-5 consecutive midday hours without being in the forecast. Training: 10 % of days, 1-2 h, 20-50 % depth.",
                  dict(pv_scale=(0.9, 1.3), drop_prob=1.0, drop_hours=(3, 5), drop_depth=(0.85, 0.98), season="test")),
        OODFamily("production_surge", "Production surge",
                  f"7-8 jobs, 1.1-1.4x job power, jobs planned later in the day. Training: {t.n_jobs[0]}-{t.n_jobs[1]} jobs, 0.8-1.2x power, any start.",
                  dict(n_jobs=(7, 8), power_scale=(1.1, 1.4), late_start_bias=0.7, season="test")),
        OODFamily("tight_deadlines", "Tighter deadlines",
                  f"Mean slack between completion and deadline 0.5-1.5 h. Training: {t.slack_mean[0]}-{t.slack_mean[1]} h.",
                  dict(slack_mean=(0.5, 1.5), season="test")),
        OODFamily("combo_familiar", "Familiar conditions, new combination",
                  "Low PV (0.5-0.7x) together with tight deadlines (slack 2-3 h). Each range is inside the training range, but training excludes the joint corner (PV<0.75x and slack<3.5 h).",
                  dict(pv_scale=(0.5, 0.7), slack_mean=(2.0, 3.0), season="test")),
        OODFamily("compound_shock", "Compound disturbance",
                  "Simultaneously: price 3-5x with 2-3x evening spike, PV 0.15-0.35x plus 3-5 h collapse, tariff placement B, slack 1-2 h, 6-8 jobs at 1.1-1.4x power.",
                  dict(price_scale=(3.0, 5.0), price_spike=(2.0, 3.0), pv_scale=(0.15, 0.35), drop_prob=1.0,
                       drop_hours=(3, 5), drop_depth=(0.85, 0.98), tariff="B", slack_mean=(1.0, 2.0),
                       n_jobs=(6, 8), power_scale=(1.1, 1.4), season="test")),
        OODFamily("unseen_industry", "Unseen industrial sector load shape",
                  f"Inflexible load follows ELMAS clusters {list(t.ood_load_clusters)} (other industrial sectors with a different daily/weekly rhythm) "
                  f"never used in training (clusters {list(t.load_clusters)}).",
                  dict(load_clusters=t.ood_load_clusters, season="test")),
        OODFamily("winter_season", "Unseen season (real winter data)",
                  "Real December-February days (low solar, different price and carbon dynamics). Training/validation used March-November only.",
                  dict(season="ood_winter")),
    ]


def ood_spec(train: DistSpec, family: OODFamily) -> DistSpec:
    """Training spec with the family's overrides applied; rejection regions are switched off."""
    return train.with_(exclude_pv_low=0.0, exclude_slack_tight=0.0, **family.spec_overrides)
