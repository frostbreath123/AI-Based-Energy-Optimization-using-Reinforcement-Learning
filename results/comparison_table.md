| suite | family | policy | episodes | seeds | cost_eur | cost_eur_std_ep | co2_kg | completion_rate | jobs_missed | renewable_share | load_shifted_kwh | shed_events | invalid_requests |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ID | in_distribution | Naive schedule | 60 | 1 | 865.06 | 423.71 | 438.17 | 1.00 | 0.00 | 0.24 | 0.00 | 0.00 | 0.00 |
| ID | in_distribution | Rule-based | 60 | 1 | 797.25 | 399.59 | 422.50 | 1.00 | 0.00 | 0.28 | 4121.40 | 0.00 | 0.03 |
| ID | in_distribution | RL agent (PPO) | 60 | 3 | 791.39 | 402.32 | 394.19 | 1.00 | 0.01 | 0.28 | 3769.85 | 0.02 | 0.57 |
| OOD | combo_familiar | Naive schedule | 30 | 1 | 922.16 | 309.21 | 483.70 | 1.00 | 0.00 | 0.18 | 0.00 | 0.00 | 0.00 |
| OOD | combo_familiar | Rule-based | 30 | 1 | 868.29 | 284.11 | 476.13 | 1.00 | 0.00 | 0.19 | 3509.85 | 0.00 | 0.00 |
| OOD | combo_familiar | RL agent (PPO) | 30 | 3 | 857.03 | 285.19 | 450.67 | 1.00 | 0.00 | 0.20 | 3625.67 | 0.00 | 0.32 |
| OOD | compound_shock | Naive schedule | 30 | 1 | 4003.90 | 1075.01 | 772.31 | 1.00 | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 |
| OOD | compound_shock | Rule-based | 30 | 1 | 3235.16 | 997.92 | 743.56 | 0.98 | 0.13 | 0.04 | 6349.69 | 0.23 | 0.00 |
| OOD | compound_shock | RL agent (PPO) | 30 | 3 | 3070.34 | 912.68 | 697.14 | 0.91 | 0.68 | 0.04 | 6863.53 | 0.31 | 8.40 |
| OOD | low_renewables | Naive schedule | 30 | 1 | 1062.82 | 265.08 | 550.81 | 1.00 | 0.00 | 0.09 | 0.00 | 0.00 | 0.00 |
| OOD | low_renewables | Rule-based | 30 | 1 | 1006.22 | 235.30 | 542.21 | 1.00 | 0.00 | 0.09 | 4204.81 | 0.00 | 0.00 |
| OOD | low_renewables | RL agent (PPO) | 30 | 3 | 1000.47 | 240.10 | 521.59 | 1.00 | 0.00 | 0.09 | 3677.39 | 0.00 | 0.39 |
| OOD | price_spike | Naive schedule | 30 | 1 | 2399.73 | 1102.30 | 452.52 | 1.00 | 0.00 | 0.23 | 0.00 | 0.00 | 0.00 |
| OOD | price_spike | Rule-based | 30 | 1 | 1799.79 | 879.13 | 427.07 | 1.00 | 0.00 | 0.25 | 4491.16 | 0.00 | 0.33 |
| OOD | price_spike | RL agent (PPO) | 30 | 3 | 1755.81 | 813.77 | 408.61 | 0.99 | 0.03 | 0.26 | 4164.97 | 0.00 | 1.09 |
| OOD | production_surge | Naive schedule | 30 | 1 | 1380.83 | 516.98 | 761.10 | 1.00 | 0.00 | 0.18 | 0.00 | 0.00 | 0.00 |
| OOD | production_surge | Rule-based | 30 | 1 | 1247.82 | 465.16 | 723.24 | 0.88 | 0.90 | 0.18 | 5764.23 | 1.07 | 0.00 |
| OOD | production_surge | RL agent (PPO) | 30 | 3 | 1248.13 | 467.39 | 694.78 | 0.93 | 0.52 | 0.19 | 5658.22 | 0.52 | 13.04 |
| OOD | renewable_drop | Naive schedule | 30 | 1 | 1079.88 | 365.97 | 571.27 | 1.00 | 0.00 | 0.16 | 0.00 | 0.00 | 0.00 |
| OOD | renewable_drop | Rule-based | 30 | 1 | 1008.63 | 334.19 | 553.88 | 0.99 | 0.03 | 0.19 | 4302.09 | 0.03 | 0.03 |
| OOD | renewable_drop | RL agent (PPO) | 30 | 3 | 996.04 | 343.42 | 524.20 | 1.00 | 0.00 | 0.19 | 3797.17 | 0.04 | 0.44 |
| OOD | tariff_restructure | Naive schedule | 30 | 1 | 855.30 | 258.32 | 449.32 | 1.00 | 0.00 | 0.22 | 0.00 | 0.00 | 0.00 |
| OOD | tariff_restructure | Rule-based | 30 | 1 | 793.68 | 243.19 | 437.39 | 1.00 | 0.00 | 0.25 | 4237.03 | 0.00 | 0.03 |
| OOD | tariff_restructure | RL agent (PPO) | 30 | 3 | 781.94 | 244.58 | 410.95 | 1.00 | 0.00 | 0.25 | 3657.70 | 0.00 | 0.18 |
| OOD | tight_deadlines | Naive schedule | 30 | 1 | 892.89 | 366.50 | 486.87 | 1.00 | 0.00 | 0.24 | 0.00 | 0.00 | 0.00 |
| OOD | tight_deadlines | Rule-based | 30 | 1 | 820.74 | 338.76 | 469.86 | 0.99 | 0.03 | 0.27 | 3863.42 | 0.03 | 0.07 |
| OOD | tight_deadlines | RL agent (PPO) | 30 | 3 | 811.98 | 340.18 | 442.24 | 1.00 | 0.00 | 0.27 | 3960.26 | 0.01 | 0.34 |
| OOD | unseen_industry | Naive schedule | 30 | 1 | 846.98 | 310.94 | 438.61 | 1.00 | 0.00 | 0.25 | 0.00 | 0.00 | 0.00 |
| OOD | unseen_industry | Rule-based | 30 | 1 | 796.48 | 296.71 | 429.77 | 0.99 | 0.03 | 0.28 | 3980.26 | 0.03 | 0.00 |
| OOD | unseen_industry | RL agent (PPO) | 30 | 3 | 783.92 | 307.10 | 402.96 | 1.00 | 0.00 | 0.28 | 3944.21 | 0.00 | 0.40 |
| OOD | winter_season | Naive schedule | 30 | 1 | 1320.70 | 565.47 | 713.15 | 1.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 |
| OOD | winter_season | Rule-based | 30 | 1 | 1215.02 | 520.27 | 695.12 | 1.00 | 0.00 | 0.13 | 4437.53 | 0.00 | 0.10 |
| OOD | winter_season | RL agent (PPO) | 30 | 3 | 1219.30 | 528.99 | 668.40 | 1.00 | 0.01 | 0.13 | 4055.41 | 0.00 | 0.38 |