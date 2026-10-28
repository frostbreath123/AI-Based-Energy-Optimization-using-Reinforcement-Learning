# Final report: AI-based energy optimisation with reinforcement learning

All numbers below were produced by the code in this repository (`results/episodes.csv`, `results/comparison_table.csv`, `results/rl_vs_rule_*_ci.csv`,
`results/sensitivity.csv`) from the models in `models/ppo_seed{0,1,2}`.

## 1. Problem and approach
A synthetic industrial site (mean 250 kW inflexible load, 4-8 preemptible production jobs per day with deadlines, 900 kWp PV, 1.2 MWh battery,
1.8 MW grid limit) buys power at a time-varying tariff and carbon intensity. A fixed production plan (naive) ignores prices; a rule-based
controller shifts jobs and the battery using the day-ahead tariff; a PPO agent (Stable-Baselines3) learns a policy by interacting with the simulator.

**Real data (France):** 2019 day-ahead prices and generation mix (carbon intensity) from Energy-Charts, Paris PV from Renewables.ninja, industrial load shapes
from the ELMAS dataset (Enedis / MINES Paris), and the **TURPE 7 HTA network tariff** (CRE deliberation 2025-40: five time ranges, options CU/LU).
**Synthetic:** production jobs, plant size, forecast errors, the supply margin/tax adder, and the mapping of the 2025-26 network tariff onto 2019 wholesale prices
(`docs/DATA_SOURCES.md`, `docs/SIMULATION_DESIGN.md`).

## 2. Method summary
* **Environment:** `energyrl/env.py` on `energyrl/sim.py`; 24 hourly steps; 122-dim observation (tariff 24 h ahead in absolute and scale-free form, carbon 8 h, PV forecast 12 h, load, battery, job slots); `MultiDiscrete([2]*8+[5])` action. The simulator enforces release times, the grid limit and battery limits.
* **Reward:** normalised multi-objective (cost, carbon, lateness, missed jobs, sheds, invalid requests); battery energy settled step by step at the day's mean tariff: `docs/REWARD_DESIGN.md`.
* **Agent:** PPO, MLP 2x256, 8 parallel envs, 1 000 000 steps per seed, 3 seeds, learning rate decayed linearly, 384 steps x 8 envs per update, batch 512; best checkpoint by validation reward. About 19 min per seed on CPU (6 runs in parallel).
* **Training distribution:** spot price 0.5-2.0x with an evening spike up to 1.6x, PV 0.5-1.3x, 4-6 jobs, deadline slack 2-6 h, TURPE off-peak placement A, options CU/LU, ELMAS clusters 1 and 10.
* **Splits (no leakage):** in-season days (Mar-Nov) split 70/15/15 into train / validation / test; Dec-Feb only for the winter OOD family; ELMAS clusters 14 and 11 and off-peak placement B only for OOD. Tests assert the pools are disjoint.
* **Evaluation:** 60 in-distribution and 30 episodes per OOD family (10 families), identical scenarios for every policy; RL = mean over 3 seeds; bootstrap 95 % confidence intervals on per-day differences.

## 3. Results

### 3.1 In-distribution (60 held-out days, mean per day)
| Policy | Cost (EUR) | CO2 (kg) | Jobs completed | Late job-h | Decision time |
|---|---|---|---|---|---|
| Naive plan | 865 | 438.2 | 100 % | 0 | 0.005 ms/step |
| Rule-based | 797 (-7.8 %) | 422.5 (-3.6 %) | 100 % | 0 | 0.15 ms/step |
| **RL agent (3 seeds)** | **791 (-8.5 %)** | **394.2 (-10.0 %)** | 99.9 % | 0.01 | 1.0 ms/step |

* **Cost:** RL is **0.7 % cheaper** than the rule-based policy (seeds: 0.6 %, 1.1 %, 0.6 %) and cheaper on 68 % of days. The per-day mean advantage is 0.81 % with a 95 % CI of [-0.31 %, 1.89 %], so **the in-distribution cost advantage is small and not statistically established** on 60 days.
* **Carbon:** RL emits **6.7 % less CO2** than the rule-based policy (per-day mean 8.1 %, CI [6.7 %, 9.7 %]): clearly significant.

### 3.2 Out-of-distribution (30 episodes per family)
| Family | RL cost vs naive | RL cost vs rule (total) | RL vs rule per-day mean [95 % CI] | RL CO2 vs rule | Jobs done RL / rule | RL sheds per day |
|---|---|---|---|---|---|---|
| In-distribution (reference) | 8.5 % | +0.7 % | +0.8 % [-0.3, 1.9] | 6.7 % | 99.9 / 100 % | 0.02 |
| Price spikes (3-5x) | 26.8 % | +2.4 % | +0.9 % [-2.8, 4.1] | 4.3 % | 99.4 / 100 | 0.00 |
| Unseen off-peak placement | 8.6 % | +1.5 % | **+1.5 % [0.4, 2.6]** | 6.0 % | 100 / 100 | 0.00 |
| Low, noisy PV | 5.9 % | +0.6 % | +0.7 % [-0.05, 1.5] | 3.8 % | 100 / 100 | 0.00 |
| Unforecast PV collapse | 7.8 % | +1.3 % | **+1.5 % [0.3, 2.6]** | 5.4 % | 100 / 99.4 | 0.04 |
| Production surge (7-8 jobs) | 9.6 % | -0.03 % | +0.2 % [-2.3, 2.3] | 3.9 % | **93.1 / 88.0** | 0.52 |
| Tight deadlines | 9.1 % | +1.1 % | **+1.3 % [0.1, 2.4]** | 5.9 % | 100 / 99.3 | 0.01 |
| Familiar conditions, new combination | 7.1 % | +1.3 % | **+1.4 % [0.6, 2.1]** | 5.4 % | 100 / 100 | 0.00 |
| Compound disturbance | 23.3 % | +5.1 % | **+4.1 % [0.9, 6.8]** | 6.2 % | **91.1** / 98.3 | 0.31 |
| Unseen industrial load shape | 7.5 % | +1.6 % | **+2.1 % [0.9, 3.2]** | 6.2 % | 100 / 99.4 | 0.00 |
| Unseen season (real winter) | 7.7 % | -0.4 % | -0.2 % [-1.5, 1.2] | 3.8 % | 99.8 / 100 | 0.00 |
| **All OOD pooled (300 days)** | **11.3 %** (rule 10.1 %) | +1.4 % | **+1.35 % [0.74, 1.92]** | 5.8 % [5.3, 6.4] | | |

Positive = RL cheaper / lower than the rule-based policy. Bold CIs exclude zero. "Total" is the ratio of summed costs; the per-day mean is the average of daily percentage differences. Full tables: `results/comparison_table.csv`; figures: `results/figures/`.

### 3.3 Reward-weight sensitivity (1 seed each, `docs/REWARD_DESIGN.md`)
Default (w_co2 = 0.5): 8.4 % cost / 10.0 % CO2 saving vs naive; cost-only: 8.2 % / 8.4 %; carbon-heavy: 7.1 % / 9.8 %. Removing the production penalties produces a fake 49 % "saving" with only 29 % of jobs completed.

## 4. Interpretation
**Does the RL agent beat the rule-based baseline on cost?** Yes, on average and with statistical support out-of-distribution (pooled 1.35 % cheaper, CI [0.74, 1.92]) and cheaper in 9 of the 11 scenario sets, but **the margin is small** (about 1 %), and in-distribution it is not statistically established. It is clearly better on carbon (4-7 % lower than the rule-based policy in every family) at equal or slightly lower cost, with production completion at or near 100 %. On this tariff the rule-based heuristic is already close to what is achievable on cost (the spread of the real TURPE energy charge is small relative to total cost), which limits how much any learned policy can save.

**Where the policy generalises.** Every one of the 10 OOD families has the RL agent cheaper than the naive plan (5.9 %-26.8 %). The advantage over the rule-based policy is significant for an unseen off-peak placement (+1.5 %), an unforecast PV collapse (+1.5 %), tight deadlines (+1.3 %), the new load/PV combination (+1.4 %), an unseen industrial load shape (+2.1 %) and the compound disturbance (+4.1 %).

**Where it does not.** (a) Production surge (7-8 jobs): the agent has no cost advantage (+0.2 %, CI [-2.3, 2.3]) and completes only 93.1 % of jobs with 0.5 capacity sheds/day (the rule-based policy: 88.0 %). (b) Compound disturbance: cheaper (+4.1 %) but completes only 91.1 % of jobs versus 98.3 % for the rule-based policy, i.e. part of its cost advantage there is bought with missed production (worst days: `results/failure_cases.csv`, agent misses about 1.3 jobs). (c) Real winter days (-0.2 %) and 3-5x price spikes (+0.9 %, CI includes zero) show no reliable cost advantage. (d) The agent still issues impossible requests (premature job starts, up to ~12/day in surge/compound); the simulator ignores and counts them, but a deployed controller would need a safety layer or action masking. (e) The rule-based policy is only a transparent heuristic; a day-ahead optimiser (linear programming) was not built and could beat the agent.

**Statistical strength.** 30 days per family, 3 seeds. Intervals above are bootstrap 95 % CIs of the per-day mean (no multiple-comparison correction). Convergence at 1 M steps is not claimed; validation curves had plateaued.

## 5. Limitations
See `docs/LIMITATIONS.md`.

## 6. Process notes affecting how to read the results
* **Iterations.** v1 (German 2023 data, 8 h look-ahead) and v2 (late-job weight 0.03 -> 0.15) were replaced by France/ELMAS/Renewables.ninja data (v3, user-requested), then by the real TURPE 7 tariff (v4, user-requested), then by agent improvements: scale-free price features, step-wise battery settlement, bigger batch with linear learning-rate decay. All earlier results and models are archived (`results/archive_*`, `models/archive_*`).
* **The training price range was widened after seeing OOD results.** With the first real-tariff models, the agent was 5.3 % *worse* than the rule-based policy on the OOD price-spike family (training range 0.7-1.4x spot). I then widened the training range to 0.5-2.0x with evening spikes up to 1.6x and moved the held-out family further out (3-5x, spike 2-3x). That family remains out of the training range, but the choice was informed by seeing its result, so its numbers are not a clean blind test. Other families were not tuned. No model was ever trained on test or OOD scenarios.
* **Selection on validation only:** training variants (`results/training_variants_val.csv`) and the carbon weight were compared on validation days; the chosen settings gave only 0.3-1.1 % differences between variants.
* **Tariff data:** the TURPE energy-charge values are copied from the CRE PDF by hand; the fixed-peak hours (9-11 h, 18-20 h), the supply/tax adder (0.03 EUR/kWh) and the use of a 2025-26 network tariff with 2019 spot prices are assumptions.
* DISCOM (India) and NREL (US) tariffs were not used (different countries from the French load/market data).

## 7. Reproduction
`python -m energyrl.prepare_data && python -m energyrl.train && python -m energyrl.evaluate && python -m energyrl.report && python -m pytest -q`, or `python scripts/run_all.py`. Dashboard: `streamlit run app/streamlit_app.py`.
