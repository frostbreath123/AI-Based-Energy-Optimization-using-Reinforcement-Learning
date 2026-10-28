| suite | family | policy | episodes | seeds | cost_eur | cost_eur_std_ep | co2_kg | completion_rate | jobs_missed | renewable_share | load_shifted_kwh | shed_events | invalid_requests |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ID | in_distribution | Naive schedule | 60 | 1 | 2662.50 | 814.55 | 3315.02 | 1.00 | 0.00 | 0.19 | 0.00 | 0.00 | 0.00 |
| ID | in_distribution | Rule-based | 60 | 1 | 2445.15 | 815.41 | 3176.11 | 1.00 | 0.00 | 0.21 | 4143.51 | 0.00 | 0.32 |
| ID | in_distribution | RL agent (PPO) | 60 | 3 | 2494.82 | 860.24 | 2939.57 | 1.00 | 0.00 | 0.21 | 3497.87 | 0.00 | 0.57 |
| OOD | combo_familiar | Naive schedule | 30 | 1 | 3052.14 | 1120.44 | 3822.28 | 1.00 | 0.00 | 0.13 | 0.00 | 0.00 | 0.00 |
| OOD | combo_familiar | Rule-based | 30 | 1 | 2823.31 | 1046.64 | 3762.59 | 1.00 | 0.00 | 0.14 | 4115.42 | 0.00 | 0.00 |
| OOD | combo_familiar | RL agent (PPO) | 30 | 3 | 2914.34 | 1111.50 | 3507.15 | 1.00 | 0.00 | 0.14 | 3856.45 | 0.02 | 0.53 |
| OOD | compound_shock | Naive schedule | 30 | 1 | 8019.77 | 2152.10 | 5432.63 | 1.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 |
| OOD | compound_shock | Rule-based | 30 | 1 | 5917.91 | 1788.81 | 5085.49 | 0.99 | 0.07 | 0.02 | 6325.80 | 0.07 | 0.00 |
| OOD | compound_shock | RL agent (PPO) | 30 | 3 | 6399.56 | 1886.33 | 4855.56 | 0.96 | 0.32 | 0.02 | 6619.73 | 1.41 | 11.67 |
| OOD | low_renewables | Naive schedule | 30 | 1 | 3076.46 | 807.59 | 3617.57 | 1.00 | 0.00 | 0.06 | 0.00 | 0.00 | 0.00 |
| OOD | low_renewables | Rule-based | 30 | 1 | 2796.22 | 757.40 | 3611.46 | 1.00 | 0.00 | 0.06 | 4476.07 | 0.00 | 0.00 |
| OOD | low_renewables | RL agent (PPO) | 30 | 3 | 2919.05 | 807.69 | 3338.67 | 1.00 | 0.00 | 0.06 | 3687.58 | 0.00 | 0.38 |
| OOD | price_spike | Naive schedule | 30 | 1 | 4703.49 | 1596.27 | 3119.29 | 1.00 | 0.00 | 0.16 | 0.00 | 0.00 | 0.00 |
| OOD | price_spike | Rule-based | 30 | 1 | 3656.48 | 1390.76 | 2849.92 | 1.00 | 0.00 | 0.16 | 4016.46 | 0.00 | 0.17 |
| OOD | price_spike | RL agent (PPO) | 30 | 3 | 3722.39 | 1387.04 | 2723.12 | 0.99 | 0.03 | 0.16 | 3918.73 | 0.04 | 5.72 |
| OOD | production_surge | Naive schedule | 30 | 1 | 4465.27 | 742.61 | 5628.72 | 1.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 |
| OOD | production_surge | Rule-based | 30 | 1 | 3923.17 | 581.76 | 5077.54 | 0.78 | 1.67 | 0.12 | 5894.91 | 1.93 | 0.17 |
| OOD | production_surge | RL agent (PPO) | 30 | 3 | 4223.91 | 721.67 | 4907.13 | 0.92 | 0.63 | 0.12 | 5790.00 | 0.69 | 11.60 |
| OOD | renewable_drop | Naive schedule | 30 | 1 | 2726.74 | 850.72 | 3273.04 | 1.00 | 0.00 | 0.11 | 0.00 | 0.00 | 0.00 |
| OOD | renewable_drop | Rule-based | 30 | 1 | 2495.77 | 783.44 | 3063.66 | 1.00 | 0.00 | 0.12 | 4728.19 | 0.00 | 0.27 |
| OOD | renewable_drop | RL agent (PPO) | 30 | 3 | 2502.46 | 817.73 | 2921.80 | 1.00 | 0.00 | 0.13 | 4100.13 | 0.01 | 0.58 |
| OOD | tariff_restructure | Naive schedule | 30 | 1 | 2741.13 | 1043.81 | 3463.40 | 1.00 | 0.00 | 0.17 | 0.00 | 0.00 | 0.00 |
| OOD | tariff_restructure | Rule-based | 30 | 1 | 2092.25 | 737.02 | 3017.18 | 1.00 | 0.00 | 0.17 | 4364.16 | 0.00 | 0.07 |
| OOD | tariff_restructure | RL agent (PPO) | 30 | 3 | 2273.38 | 784.82 | 2953.33 | 0.99 | 0.04 | 0.17 | 3823.15 | 0.03 | 2.53 |
| OOD | tight_deadlines | Naive schedule | 30 | 1 | 2553.26 | 688.30 | 3211.79 | 1.00 | 0.00 | 0.21 | 0.00 | 0.00 | 0.00 |
| OOD | tight_deadlines | Rule-based | 30 | 1 | 2373.89 | 639.14 | 3089.70 | 1.00 | 0.00 | 0.21 | 3604.24 | 0.00 | 0.30 |
| OOD | tight_deadlines | RL agent (PPO) | 30 | 3 | 2423.01 | 670.05 | 2814.22 | 1.00 | 0.00 | 0.21 | 3893.18 | 0.00 | 0.38 |
| OOD | winter_season | Naive schedule | 30 | 1 | 2911.77 | 882.92 | 3641.30 | 1.00 | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 |
| OOD | winter_season | Rule-based | 30 | 1 | 2641.21 | 793.74 | 3560.88 | 1.00 | 0.00 | 0.04 | 4115.03 | 0.00 | 0.00 |
| OOD | winter_season | RL agent (PPO) | 30 | 3 | 2775.11 | 835.54 | 3455.55 | 1.00 | 0.00 | 0.04 | 3836.13 | 0.00 | 0.62 |