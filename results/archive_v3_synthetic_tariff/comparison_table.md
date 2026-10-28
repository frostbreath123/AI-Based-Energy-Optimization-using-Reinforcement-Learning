| suite | family | policy | episodes | seeds | cost_eur | cost_eur_std_ep | co2_kg | completion_rate | jobs_missed | renewable_share | load_shifted_kwh | shed_events | invalid_requests |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ID | in_distribution | Naive schedule | 60 | 1 | 1758.69 | 591.59 | 481.53 | 1.00 | 0.00 | 0.26 | 0.00 | 0.00 | 0.00 |
| ID | in_distribution | Rule-based | 60 | 1 | 1701.40 | 538.15 | 473.80 | 0.99 | 0.03 | 0.27 | 4030.85 | 0.03 | 0.47 |
| ID | in_distribution | RL agent (PPO) | 60 | 3 | 1599.11 | 550.33 | 446.23 | 1.00 | 0.01 | 0.28 | 3347.45 | 0.00 | 0.31 |
| OOD | combo_familiar | Naive schedule | 30 | 1 | 2039.25 | 583.11 | 553.64 | 1.00 | 0.00 | 0.17 | 0.00 | 0.00 | 0.00 |
| OOD | combo_familiar | Rule-based | 30 | 1 | 1894.13 | 526.23 | 539.67 | 1.00 | 0.00 | 0.18 | 4446.28 | 0.00 | 0.27 |
| OOD | combo_familiar | RL agent (PPO) | 30 | 3 | 1823.47 | 529.45 | 516.28 | 1.00 | 0.00 | 0.18 | 3715.25 | 0.00 | 0.28 |
| OOD | compound_shock | Naive schedule | 30 | 1 | 4180.10 | 1182.59 | 786.05 | 1.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 |
| OOD | compound_shock | Rule-based | 30 | 1 | 3300.52 | 923.58 | 758.84 | 0.99 | 0.07 | 0.03 | 6907.59 | 0.10 | 0.00 |
| OOD | compound_shock | RL agent (PPO) | 30 | 3 | 3446.83 | 789.26 | 721.11 | 0.91 | 0.67 | 0.03 | 6504.33 | 0.14 | 9.89 |
| OOD | low_renewables | Naive schedule | 30 | 1 | 2213.19 | 531.76 | 581.63 | 1.00 | 0.00 | 0.08 | 0.00 | 0.00 | 0.00 |
| OOD | low_renewables | Rule-based | 30 | 1 | 2082.99 | 389.28 | 575.15 | 1.00 | 0.00 | 0.08 | 4154.45 | 0.00 | 0.00 |
| OOD | low_renewables | RL agent (PPO) | 30 | 3 | 2036.39 | 459.17 | 554.73 | 1.00 | 0.00 | 0.08 | 3472.95 | 0.00 | 0.19 |
| OOD | price_spike | Naive schedule | 30 | 1 | 2341.33 | 805.86 | 491.44 | 1.00 | 0.00 | 0.28 | 0.00 | 0.00 | 0.00 |
| OOD | price_spike | Rule-based | 30 | 1 | 2078.07 | 659.07 | 481.76 | 0.99 | 0.07 | 0.30 | 4096.44 | 0.07 | 0.60 |
| OOD | price_spike | RL agent (PPO) | 30 | 3 | 2030.10 | 676.96 | 445.75 | 0.99 | 0.06 | 0.30 | 3519.09 | 0.00 | 1.33 |
| OOD | production_surge | Naive schedule | 30 | 1 | 2551.73 | 739.44 | 625.37 | 1.00 | 0.00 | 0.19 | 0.00 | 0.00 | 0.00 |
| OOD | production_surge | Rule-based | 30 | 1 | 2292.91 | 647.89 | 582.33 | 0.83 | 1.30 | 0.20 | 5802.18 | 1.37 | 0.67 |
| OOD | production_surge | RL agent (PPO) | 30 | 3 | 2350.28 | 754.13 | 573.28 | 0.92 | 0.60 | 0.21 | 5181.79 | 0.06 | 14.16 |
| OOD | renewable_drop | Naive schedule | 30 | 1 | 1955.81 | 628.23 | 493.57 | 1.00 | 0.00 | 0.19 | 0.00 | 0.00 | 0.00 |
| OOD | renewable_drop | Rule-based | 30 | 1 | 1968.72 | 587.82 | 490.63 | 1.00 | 0.00 | 0.21 | 4046.66 | 0.00 | 0.50 |
| OOD | renewable_drop | RL agent (PPO) | 30 | 3 | 1785.34 | 566.46 | 457.94 | 1.00 | 0.00 | 0.21 | 3328.68 | 0.00 | 0.50 |
| OOD | tariff_restructure | Naive schedule | 30 | 1 | 1683.93 | 673.96 | 407.55 | 1.00 | 0.00 | 0.24 | 0.00 | 0.00 | 0.00 |
| OOD | tariff_restructure | Rule-based | 30 | 1 | 1285.65 | 492.32 | 381.38 | 1.00 | 0.00 | 0.27 | 4208.15 | 0.00 | 0.70 |
| OOD | tariff_restructure | RL agent (PPO) | 30 | 3 | 1372.88 | 589.47 | 362.24 | 0.99 | 0.03 | 0.28 | 3713.04 | 0.00 | 2.06 |
| OOD | tight_deadlines | Naive schedule | 30 | 1 | 1863.26 | 543.86 | 484.48 | 1.00 | 0.00 | 0.23 | 0.00 | 0.00 | 0.00 |
| OOD | tight_deadlines | Rule-based | 30 | 1 | 1732.43 | 496.96 | 473.19 | 1.00 | 0.00 | 0.25 | 3694.01 | 0.00 | 0.20 |
| OOD | tight_deadlines | RL agent (PPO) | 30 | 3 | 1665.53 | 506.55 | 444.17 | 1.00 | 0.00 | 0.26 | 3504.93 | 0.00 | 0.53 |
| OOD | unseen_industry | Naive schedule | 30 | 1 | 1735.70 | 694.24 | 494.03 | 1.00 | 0.00 | 0.26 | 0.00 | 0.00 | 0.00 |
| OOD | unseen_industry | Rule-based | 30 | 1 | 1664.94 | 613.74 | 476.12 | 1.00 | 0.00 | 0.28 | 4239.69 | 0.00 | 0.60 |
| OOD | unseen_industry | RL agent (PPO) | 30 | 3 | 1547.22 | 629.57 | 448.34 | 1.00 | 0.01 | 0.30 | 3355.60 | 0.00 | 0.38 |
| OOD | winter_season | Naive schedule | 30 | 1 | 2370.20 | 615.66 | 765.36 | 1.00 | 0.00 | 0.09 | 0.00 | 0.00 | 0.00 |
| OOD | winter_season | Rule-based | 30 | 1 | 2180.26 | 557.87 | 742.42 | 0.98 | 0.10 | 0.10 | 4511.30 | 0.10 | 0.07 |
| OOD | winter_season | RL agent (PPO) | 30 | 3 | 2143.23 | 581.99 | 723.28 | 1.00 | 0.00 | 0.10 | 3677.87 | 0.00 | 0.44 |