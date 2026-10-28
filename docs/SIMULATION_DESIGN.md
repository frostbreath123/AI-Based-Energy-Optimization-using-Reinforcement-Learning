# Simulation design

The environment models one **industrial site over one day (24 hourly steps)**. Code: `energyrl/sim.py` (physics and
accounting, no Gym dependency), `energyrl/env.py` (Gymnasium wrapper, observation, reward), `energyrl/scenarios.py`
(scenario generation). All constants live in `config/default.yaml`.

## Site model
| Element | Model | Default |
|---|---|---|
| Inflexible base load | Real hourly load SHAPE of a French industrial customer cluster (ELMAS, normalised to mean 1) x mean level; training draws clusters 1 and 10, held-out clusters 14 and 11 are an OOD family | mean 250 kW |
| Flexible production jobs | Preemptible batches: `duration` run-hours at `power_kW`; released at `release`, must complete `duration` run-hours before `deadline`; planned (original) start defined | 4-6 jobs/day, 2-4 h, 250-550 kW |
| On-site PV | 900 kWp x real Renewables.ninja Paris capacity factor x scenario scale; surplus exported at a feed-in price | 0.05 EUR/kWh |
| Battery | 1200 kWh, 400 kW, 95 % one-way efficiency, SoC 10-95 %; 5 power levels (-100 %..+100 %); cannot export to the grid | |
| Grid connection | Hard import limit | 1800 kW |
| Demand charge | Each new daily peak (kW) is charged when it is set | 0.0403 EUR/kW-day (TURPE 7 HTA CU power term 14.70 EUR/kW/yr / 365; approximation) |

**Flexible vs inflexible load:** the base load must always be served. Jobs are the flexible load: the agent decides in
which hours they run. The original production plan (used by the naive baseline) is capacity-feasible by construction.

## Exogenous signals
* **Tariff** = real day-ahead spot price (France, 2019) x scenario scale + **real TURPE 7 HTA network charge** (CRE deliberation 2025-40; five time ranges
  PTE / HPH / HCH / HPE / HCE, options CU and LU drawn per scenario) + an assumed fixed adder (0.03 EUR/kWh for supply margin and taxes); floored at 0.
  Off-peak **placement A** (22-06 h and Sundays; training) and **placement B** (low-season off-peak 02-06 h and 11-15 h, the CRE's solar-following proposal; held out as an OOD family).
  The tariff year (2025-26) differs from the market data year (2019): the wholesale price is only the variable part of the signal.
* **Carbon intensity**: real French 2019 generation mix x emission factors (see `docs/DATA_SOURCES.md`).
* **PV actual**: real Renewables.ninja (MERRA-2) Paris profile x installed kWp x scenario scale. **PV forecast**: actual x log-normal error (sd 10-25 %
  in training) issued day-ahead. Optional **unforecast drop**: actual PV reduced for 1-2 h (training, 10 % of days) or 3-5 h (OOD).
* All prices, carbon and the forecast are known to the agent for the look-ahead window; the *actual* PV is observed only for the current hour.

## Observation (122 values, float32)
Time (t/24, sin/cos hour, weekend); tariff now + 24 h ahead (tariffs are known day-ahead, the same information the rule-based baseline uses), the same 25 values rescaled to 0..1 within the window plus the share of the window cheaper than now (scale-free view that transfers across price levels and tariff structures); carbon now + 8 h; PV now + 12 h forecast; inflexible load now + 2 h (ELMAS shape);
battery SoC; peak-so-far; per job slot (8 slots sorted by deadline): `[active, released, remaining/4, slack/8, power/600]`.
Slack = deadline - now - remaining hours (<= 0 means "must run every remaining hour"). No other signals are exposed.

## Action space
`MultiDiscrete([2]*8 + [5])`: a run flag for each of up to 8 job slots, and a battery level index.

## Transition order within an hour
1. Job requests are filtered: requests for jobs not yet released are **ignored and counted** as invalid; completed jobs are no-ops.
2. Battery request is clipped by power, SoC window and (for discharge) by the net load, so the battery never exports.
3. If grid import would exceed the limit: battery **charging is cancelled first**, then the running job with the **most slack is shed**
   (counted as a constraint violation) until the limit holds.
4. Cost = import x tariff + demand-charge increment - export x feed-in. Emissions = import x intensity.
5. Job progress, completion times, overdue jobs, PV utilisation and shifted energy are updated.
6. The change in stored battery energy is settled at the day's mean tariff x discharge efficiency in the hour it happens (charging = negative cost; same episode total as an end-of-day settlement, but a denser learning signal). At the last hour unfinished jobs are *missed*.

Invalid or physically impossible requests therefore never create impossible outcomes: they are corrected by the simulator and
reported in the metrics (`invalid_requests`, `shed_events`).

## Episode, determinism and reproducibility
Episodes have exactly 24 steps (`terminated=True`, never truncated). `reset(seed=s)` seeds the environment RNG that selects the day
and all random parameters; fixed scenario lists (`build_eval_scenarios`) use fixed seeds so every policy sees identical inputs.
Tests cover determinism (`tests/test_env_reward.py`) and Gymnasium/SB3 compatibility (`check_env`).

## Key assumptions
Hourly resolution (limits job granularity to 1 h); jobs are preemptible without restart cost; no ramp limits, no thermal comfort
model; battery degradation ignored; PV forecast error is a single day-ahead draw, not a rolling forecast; grid carbon intensity is
production-based (no import accounting); the plant and its schedules are synthetic. This is a simulation, **not a safety-certified controller**.
