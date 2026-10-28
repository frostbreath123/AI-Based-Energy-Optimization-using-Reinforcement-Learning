| suite | family | policy | episodes | seeds | cost_eur | cost_eur_std_ep | co2_kg | completion_rate | jobs_missed | renewable_share | load_shifted_kwh | shed_events | invalid_requests |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ID | in_distribution | Naive schedule | 60 | 1 | 754.02 | 342.40 | 438.17 | 1.00 | 0.00 | 0.24 | 0.00 | 0.00 | 0.00 |
| ID | in_distribution | Rule-based | 60 | 1 | 714.25 | 328.77 | 424.01 | 1.00 | 0.00 | 0.28 | 4152.38 | 0.00 | 0.00 |
| ID | in_distribution | RL agent (PPO) | 60 | 3 | 708.84 | 331.83 | 395.06 | 1.00 | 0.00 | 0.28 | 3605.16 | 0.01 | 0.56 |
| OOD | combo_familiar | Naive schedule | 30 | 1 | 804.64 | 239.08 | 483.70 | 1.00 | 0.00 | 0.18 | 0.00 | 0.00 | 0.00 |
| OOD | combo_familiar | Rule-based | 30 | 1 | 776.76 | 223.94 | 476.78 | 1.00 | 0.00 | 0.19 | 3607.39 | 0.00 | 0.00 |
| OOD | combo_familiar | RL agent (PPO) | 30 | 3 | 766.94 | 226.59 | 451.46 | 1.00 | 0.00 | 0.20 | 3574.42 | 0.00 | 0.46 |
| OOD | compound_shock | Naive schedule | 30 | 1 | 2584.09 | 607.49 | 772.31 | 1.00 | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 |
| OOD | compound_shock | Rule-based | 30 | 1 | 2227.81 | 583.16 | 748.73 | 0.98 | 0.13 | 0.04 | 6447.06 | 0.23 | 0.00 |
| OOD | compound_shock | RL agent (PPO) | 30 | 3 | 2202.32 | 574.74 | 700.30 | 0.91 | 0.69 | 0.04 | 6295.45 | 0.27 | 6.91 |
| OOD | low_renewables | Naive schedule | 30 | 1 | 953.26 | 188.25 | 550.81 | 1.00 | 0.00 | 0.09 | 0.00 | 0.00 | 0.00 |
| OOD | low_renewables | Rule-based | 30 | 1 | 920.29 | 177.40 | 542.99 | 1.00 | 0.00 | 0.09 | 4194.90 | 0.00 | 0.00 |
| OOD | low_renewables | RL agent (PPO) | 30 | 3 | 913.96 | 176.72 | 521.87 | 1.00 | 0.00 | 0.09 | 3534.75 | 0.00 | 0.16 |
| OOD | price_spike | Naive schedule | 30 | 1 | 1551.19 | 686.63 | 452.52 | 1.00 | 0.00 | 0.23 | 0.00 | 0.00 | 0.00 |
| OOD | price_spike | Rule-based | 30 | 1 | 1271.76 | 572.49 | 429.53 | 1.00 | 0.00 | 0.25 | 4455.51 | 0.00 | 0.23 |
| OOD | price_spike | RL agent (PPO) | 30 | 3 | 1339.53 | 604.82 | 411.09 | 1.00 | 0.01 | 0.26 | 3825.52 | 0.00 | 0.33 |
| OOD | production_surge | Naive schedule | 30 | 1 | 1214.42 | 388.20 | 761.10 | 1.00 | 0.00 | 0.18 | 0.00 | 0.00 | 0.00 |
| OOD | production_surge | Rule-based | 30 | 1 | 1139.77 | 343.97 | 728.75 | 0.91 | 0.70 | 0.18 | 5756.51 | 0.83 | 0.00 |
| OOD | production_surge | RL agent (PPO) | 30 | 3 | 1156.83 | 373.21 | 705.87 | 0.95 | 0.36 | 0.19 | 4795.14 | 0.52 | 10.23 |
| OOD | renewable_drop | Naive schedule | 30 | 1 | 931.69 | 310.31 | 571.27 | 1.00 | 0.00 | 0.16 | 0.00 | 0.00 | 0.00 |
| OOD | renewable_drop | Rule-based | 30 | 1 | 892.38 | 293.58 | 556.92 | 0.99 | 0.03 | 0.19 | 4328.89 | 0.03 | 0.00 |
| OOD | renewable_drop | RL agent (PPO) | 30 | 3 | 883.99 | 302.49 | 526.43 | 1.00 | 0.00 | 0.19 | 3604.79 | 0.00 | 0.34 |
| OOD | tariff_restructure | Naive schedule | 30 | 1 | 757.56 | 221.14 | 449.32 | 1.00 | 0.00 | 0.22 | 0.00 | 0.00 | 0.00 |
| OOD | tariff_restructure | Rule-based | 30 | 1 | 722.63 | 203.46 | 437.75 | 1.00 | 0.00 | 0.25 | 4237.71 | 0.00 | 0.03 |
| OOD | tariff_restructure | RL agent (PPO) | 30 | 3 | 712.69 | 209.96 | 411.37 | 1.00 | 0.00 | 0.25 | 3454.20 | 0.00 | 0.30 |
| OOD | tight_deadlines | Naive schedule | 30 | 1 | 793.88 | 293.89 | 486.87 | 1.00 | 0.00 | 0.24 | 0.00 | 0.00 | 0.00 |
| OOD | tight_deadlines | Rule-based | 30 | 1 | 745.49 | 274.98 | 470.06 | 0.99 | 0.03 | 0.27 | 4022.58 | 0.03 | 0.03 |
| OOD | tight_deadlines | RL agent (PPO) | 30 | 3 | 746.21 | 286.18 | 442.82 | 1.00 | 0.00 | 0.27 | 3979.83 | 0.00 | 0.43 |
| OOD | unseen_industry | Naive schedule | 30 | 1 | 751.15 | 259.05 | 438.61 | 1.00 | 0.00 | 0.25 | 0.00 | 0.00 | 0.00 |
| OOD | unseen_industry | Rule-based | 30 | 1 | 725.07 | 252.14 | 429.45 | 0.99 | 0.03 | 0.28 | 3993.86 | 0.03 | 0.00 |
| OOD | unseen_industry | RL agent (PPO) | 30 | 3 | 716.58 | 263.94 | 403.16 | 1.00 | 0.00 | 0.28 | 3563.66 | 0.00 | 0.26 |
| OOD | winter_season | Naive schedule | 30 | 1 | 1189.77 | 397.91 | 713.15 | 1.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 |
| OOD | winter_season | Rule-based | 30 | 1 | 1127.00 | 371.39 | 695.19 | 1.00 | 0.00 | 0.13 | 4438.92 | 0.00 | 0.10 |
| OOD | winter_season | RL agent (PPO) | 30 | 3 | 1129.95 | 383.93 | 671.94 | 1.00 | 0.00 | 0.13 | 3840.31 | 0.03 | 0.29 |