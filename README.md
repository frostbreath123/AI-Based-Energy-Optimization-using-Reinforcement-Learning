# AI-Based Energy Optimization Using Reinforcement Learning

A complete, runnable prototype: a custom Gymnasium simulation of an industrial site (real French data (2019 day-ahead prices, Renewables.ninja PV, generation-mix carbon intensity, ELMAS industrial load shapes, TURPE 7 HTA network tariff) + synthetic production jobs), a PPO agent trained with Stable-Baselines3, two baselines, a held-out
out-of-distribution (OOD) test suite and a Streamlit dashboard.

> **Simulation only.** This is a research prototype, not a safety-certified industrial controller.
> Start with [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) for results and [docs/RESEARCH_LOGBOOK.md](docs/RESEARCH_LOGBOOK.md) for the work log.

## What is inside
| Path | Content |
|---|---|
| `energyrl/` | Python package: data pipeline, scenario generator, simulator (`sim.py`), Gym env (`env.py`), reward, baselines, training, evaluation, OOD suite, metrics, report |
| `app/` | Streamlit dashboard (5 views) |
| `config/default.yaml` | Every tunable parameter (site, tariff, reward weights, training, evaluation) |
| `data/raw`, `data/processed`, `data/manifest.json` | Downloaded public data, processed hourly table, provenance manifest |
| `models/` | Trained PPO models (`ppo_seed{0,1,2}/best_model.zip`) and reward-sensitivity variants; `archive_v1_*` = first training iteration |
| `results/` | Episode-level results, comparison tables, figures, training logs |
| `docs/` | Final report, logbook, reward design, simulation design, data sources, limitations |
| `tests/` | 37 automated tests |

## Setup (Python 3.10+; developed on 3.13, Windows, CPU only)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# optional smaller CPU-only PyTorch:  pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Run every command below **from the project root**.

## Quick start (everything is already prepared in this repository)
```bash
streamlit run app/streamlit_app.py
```
Open http://localhost:8501. The app loads the saved models and results; every number shown on the Overview and Simulation
explorer pages is produced by simulating the selected day live with the trained model.

## Reproduce the experiments
```bash
python -m energyrl.prepare_data          # download public data (Renewables.ninja, ELMAS, CRE TURPE 7 PDF, Energy-Charts) and build the hourly table
python -m energyrl.data.manifest         # write data/manifest.json and docs/DATA_SOURCES.md
python -m energyrl.train                 # train PPO for seeds 0,1,2 (1M steps each, ~19 min per seed with 3 in parallel on 8 cores)
python -m energyrl.evaluate              # in-distribution + 10 OOD families, all policies -> results/episodes.csv
python -m energyrl.report                # tables + figures in results/
python -m pytest -q                      # automated tests
```
Reward-weight sensitivity runs and the whole pipeline in one go:
```bash
python scripts/run_all.py --quick        # ~3-5 min pipeline smoke test (40k steps; overwrites models/ppo_seed*!)
python scripts/run_all.py                # full study: data, 3 seeds + 3 sensitivity variants, evaluation, sensitivity, report
```
Single custom run: `python -m energyrl.train --tag mytag --seeds 3 --timesteps 200000 --w-co2 1.0`.
Back up `models/` and `results/` first if you want to keep the shipped artifacts; `--quick` overwrites them.

Load and run a trained model yourself:
```python
from energyrl.config import load_config
from energyrl.data.preprocess import load_processed
from energyrl.scenarios import ScenarioGenerator
from energyrl.evaluate import RLPolicy, rollout
import numpy as np
cfg, df = load_config(), load_processed()
sc = ScenarioGenerator(df, cfg).sample(np.random.default_rng(0))
out = rollout(RLPolicy("models/ppo_seed0/best_model.zip", cfg), sc, cfg)
print(out["metrics"]["cost_eur"], out["metrics"]["co2_kg"])
```

## Dashboard guide
1. **Overview** - KPI cards (cost, carbon, production completion, PV contribution, saving vs rule-based) for a chosen held-out day, plus a benchmark table over all scenario sets.
2. **Simulation explorer** - pick any 2019 day, change tariff schedule, price/PV multipliers, job count, deadline slack and an unforecast PV collapse; all three policies are re-simulated; tariff, PV, load dispatch, job Gantt chart and battery SoC are shown.
3. **Policy comparison** - KPI table with spread, percentage improvements, cost/carbon scatter and per-day saving distribution.
4. **Robustness lab** - OOD definitions, degradation heatmaps, failure cases, live OOD run, price x PV sweep.
5. **Research & methodology** - simulation model, reward equation, data provenance, experiments, limitations, and background retrain/evaluate jobs.

## Troubleshooting
* *"Data not prepared"* in the app: run `python -m energyrl.prepare_data` (needs internet once; raw files are then cached in `data/raw`).
* *Download fails*: the script retries 3 times; re-run, or check that `re.jrc.ec.europa.eu` and `api.energy-charts.info` are reachable.
* *No RL model / failed to load model*: run `python -m energyrl.train`; models are loaded from `models/ppo_seed<N>/best_model.zip`.
* *`ModuleNotFoundError: energyrl`*: run from the project root (or `set PYTHONPATH=.`).
* *Training is slow*: reduce `training.total_timesteps` / `n_envs` in `config/default.yaml`, or pass `--timesteps`.
* *Streamlit shows stale results after re-training*: press `C` then "Clear cache" in the app menu, or restart Streamlit.
* Windows PowerShell: use `$env:PYTHONPATH="."` instead of `set`.

## Data licences and attribution
Renewables.ninja (CC BY-NC 4.0, non-commercial use); ELMAS (Bellinguer et al., Sci Data 2023, CC BY 4.0); Energy-Charts (Fraunhofer ISE) with data from Bundesnetzagentur | SMARD.de, CC BY 4.0. Full provenance in
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). Plant size, jobs, forecast errors and the supply/tax adder are **synthetic**; the TURPE 7 values are real (CRE) but transcribed by hand and labelled so everywhere.
