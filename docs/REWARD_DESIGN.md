## Design rationale and justification

### Quantities, units and normalisation
| Symbol | Meaning | Unit |
|---|---|---|
| `cost_t` | import x tariff + demand-charge increment - export revenue | EUR |
| `CO2_t` | import x grid carbon intensity / 1000 | kg CO2e |
| `C_ref`, `E_ref` | cost / emissions of the **naive plan on the same scenario** (computed at `reset`) | EUR, kg |
| `overdue_t` | jobs unfinished and past deadline at the end of hour t | job-hours |
| `shed_t` | jobs forcibly stopped because the 1800 kW grid limit would be exceeded | events |
| `invalid_t` | ignored requests: premature job start, empty/full battery | events |
| `N_missed`, remaining/duration | unfinished jobs at the end of the day, and how unfinished each is | jobs, fraction |
| `S = 10` | reward scale (conditioning of the optimiser only) | - |

Dividing by the naive plan's own bill and emissions makes the reward **dimensionless and comparable across days** (a 5 000 EUR
price-spike day and a 2 000 EUR normal day contribute equally per percent saved). It also gives every weight a plain meaning:
"1 unit = 100 % of the naive daily bill". The naive return is exactly `-S (w_c + w_e)` (checked in `tests/test_env_reward.py`).

### Coefficients (config `reward:`)
| Weight | Value | Rationale |
|---|---|---|
| `w_cost` | 1.0 | Reference unit. |
| `w_co2` | 0.5 | Carbon counts half as much as cost: an assumed preference (not derived from a carbon price): large enough to move the policy, small enough not to dominate cost. Its effect is measured in the sensitivity section. |
| `w_late` | 0.15 per late job-hour | A late hour should cost clearly more than the 1-2 % savings a typical shift gains. an earlier iteration (v1, German data) used 0.03 and the agent accepted occasional 1-2 h lateness (0.17 job-h/day); 0.15 reduced this to 0.01 job-h/day at similar savings. |
| `w_miss` | 0.30 per unfinished job (+ remaining fraction) | Must exceed the most that skipping a job can ever save. Skipping one job saves only that job's share of the daily bill and emissions (a minority of C_ref in all generated scenarios); 0.3 per job plus the proportional term is meant to exceed that, and the test below checks it on 20 random days ( `test_abandoning_production_is_never_rewarded`). This is empirical, not a proof. |
| `w_shed` | 0.10 | Hitting the physical grid limit is a hard violation: ~3x the profit of a typical shift. |
| `w_invalid` | 0.01 | Small, only to discourage useless requests; the simulator already neutralises them. |

### Infeasible actions and production failures
* Infeasible requests never change physics: premature job starts are ignored; battery requests are clipped to power, SoC window and no-export;
  grid-limit excess is resolved by cancelling battery charging, then shedding the job with the most slack. Each correction is **counted** and penalised.
* A job that is not finished by the end of the day is a *missed job* (large terminal penalty); a job finished after its deadline or still
  running late accumulates `overdue` penalty each hour it is late.
* Changes in battery energy are settled at the day's mean tariff x discharge efficiency in the hour they happen (credit when charging, debit when discharging).
  The episode total equals an end-of-day settlement, so the agent cannot profit by draining the battery or hoarding it for free; spreading the credit over the
  day gives the optimiser a dense arbitrage signal (it was added after plain end-of-day settlement learned battery arbitrage poorly).

### Reward hacking: what could go wrong, what prevents it
| Exploit | Mitigation | Evidence |
|---|---|---|
| Skip production to save energy | `w_miss`, `w_late` exceed achievable savings | Sensitivity run with both set to 0: the agent completes only **29 %** of jobs for a "49 % cost saving" (a fake saving); the default agent completes 99.8 % |
| Drain the battery at the end for free energy | Settlement of stored energy at the mean tariff | `test_stepwise_battery_settlement_equals_end_of_day_settlement` |
| Violate the grid limit | Simulator enforces limit; sheds counted and penalised | `shed_events` metric; tests |
| Raise the peak to exploit nothing | Demand charge increment inside `cost_t` | Peak kW reported |
| Time-shift cost by lying about PV | Agent sees forecast + current PV only; the simulator computes actual flows | - |
| Reward normaliser gaming (`C_ref`) | `C_ref` depends only on the scenario and the naive plan, never on the agent | - |

### Choosing weights and training settings (validation days only)
* Carbon weight: on the real TURPE tariff, four agents (`w_co2` = 0, 0.1, 0.25, 0.5; 600 000 steps; seed 0) were compared on 60 **validation** days: cost saving vs naive 5.4 / 4.6 / 4.7 / 5.1 %, CO2 saving 9.7 / 9.3 / 9.1 / 9.9 %; `w_co2` = 0.5 had the best CO2 and near-best cost, so the default was kept (differences are within seed noise).
* Training settings (`results/training_variants_val.csv`, 800 000 steps): constant vs linearly decayed learning rate, 192 vs 384 steps per env and batch 256 vs 512, with and without step-wise battery settlement. All variants were 0.3-1.1 % cheaper than the rule-based policy on validation; bigger batch + linear decay was adopted.

### Effect of changing the weights (measured, 1 seed per variant, 60 in-distribution test days)
Source: `results/sensitivity.csv` (`python -m energyrl.sensitivity`).

| Variant | Cost (EUR/day) | CO2 (kg/day) | Cost saving vs naive | CO2 saving vs naive | Jobs completed | Late job-h |
|---|---|---|---|---|---|---|
| Naive plan | 865 | 438.2 | - | - | 100 % | 0 |
| Rule-based | 797 | 422.5 | 7.8 % | 3.6 % | 100 % | 0 |
| **Default** (`w_co2` = 0.5) | 793 | 394.4 | 8.4 % | 10.0 % | 99.8 % | 0.02 |
| Cost only (`w_co2` = 0) | 794 | 401.3 | 8.2 % | 8.4 % | 100 % | 0.00 |
| Carbon-heavy (`w_co2` = 2) | 803 | 395.5 | 7.1 % | 9.8 % | 100 % | 0.00 |
| No production penalty | 437 | 223.8 | 49.4 % (**invalid**) | 48.9 % (**invalid**) | 29 % | 11.72 |

Raising `w_co2` from 0 to 2 costs about 1.1 points of cost saving while adding 1.4 points of CO2 saving: the trade-off is mild because France's grid is
nuclear-dominated (mean 54 g/kWh) so cost and carbon signals are only weakly in conflict. Differences of about 1 point between single-seed variants are within seed noise
(seed-to-seed spread of the default agent is about 0.3 points of cost saving). With no production penalty the agent skips 71 % of the work for a fake 49 % saving.
