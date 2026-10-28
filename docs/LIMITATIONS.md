# Limitations

**Simulation realism**
* The industrial plant, job schedules, base load and network tariff are **synthetic**. No measured factory load data were used,
  so absolute savings (EUR, kg) describe this stylised site, not a real plant.
* Hourly resolution: jobs cannot be shorter than one hour, intra-hour price/PV variability is invisible, and 15-minute demand-charge
  billing is approximated by a daily peak charge.
* Jobs are fully preemptible with no restart cost; no thermal/comfort model, ramp limits, equipment wear or battery degradation.
* PV forecast error is a single day-ahead log-normal draw; real forecasts are rolling and correlated with weather regimes.
* The electricity tariff combines a 2019 wholesale price with the real 2025-26 TURPE 7 HTA network energy charge and an assumed 0.03 EUR/kWh supply/tax adder. Power (EUR/kW/yr) components are reduced to one demand-charge approximation, the 2026 indexation is not applied, the fixed-peak hours are assumed, and a real contract (retail margin, capacity mechanism, accise) would differ.

**Data**
* One year (2019 market/PV/carbon, 2018 load shape), one country (France). PV is a single Paris point (proxy for the national fleet).
  Carbon intensity is production-based (no import/export accounting) and uses lifecycle emission factors, five of which are analyst assumptions;
  France is nuclear-dominated (about 54 g/kWh), so carbon varies little and is only weakly in conflict with cost.
* ELMAS profiles are averages over many customers, hence smoother than a single plant; only their shape is used, aligned to 2019 by weekday (364-day shift), not matched to 2019 weather.
* Renewables.ninja data are licensed CC BY-NC 4.0 (non-commercial).
* The network tariff values are real (CRE, TURPE 7 HTA, transcribed by hand) but their year (2025-26) differs from the market data year (2019); jobs, plant size and the supply/tax adder are synthetic. No DISCOM or NREL tariff was used (different countries from the French data).
* Five days (DST transitions and data gaps) are excluded (360 of 365 days kept).
* The "unseen season" OOD family uses real winter days of the same year, so the market regime is shared with training.

**Learning and evaluation**
* PPO was trained for a limited budget (1 000 000 steps per seed, 3 seeds); training curves are flattening but convergence is **not** claimed.
  Three seeds give only a rough idea of seed variance.
* Evaluation uses 60 in-distribution and 30 episodes per OOD family. Per-family standard deviations are reported; small differences
  (about 2 points or less) between RL and the rule-based policy are not statistically strong.
* The rule-based baseline is a simple heuristic; a day-ahead optimiser (linear programming) was not built and might beat the agent. Both now see the full day-ahead tariff.
* OOD families vary parameters the designers thought of. They do not prove robustness to arbitrary distribution shift.
* The agent has no action masking: it sometimes requests impossible actions (premature job starts). The simulator ignores and counts them, but a
  deployed controller would need a safety layer; the shed events (hard-limit hits) are the operationally relevant violations.
* The reward weights express one preference; other preferences yield different policies (see the sensitivity analysis).

**Not provided**
No hardware-in-the-loop, no safety certification, no real-time deployment, no uncertainty quantification of the learned policy.
