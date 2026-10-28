"""Reproduce the whole study end-to-end.

    python scripts/run_all.py              # full run (~25-40 min on 8 CPU cores)
    python scripts/run_all.py --quick      # 40k-step smoke run to verify the pipeline (~3 min)

Steps: data -> manifest -> main PPO seeds + reward-sensitivity variants -> evaluation (ID + OOD) -> sensitivity table -> report.
Training runs execute in parallel processes (one per seed/variant).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def run(args: list[str], wait: bool = True):
    print("$ python -m", " ".join(args), flush=True)
    p = subprocess.Popen([sys.executable, "-m", *args], cwd=ROOT, env=ENV)
    if wait and p.wait() != 0:
        sys.exit(f"step failed: {' '.join(args)}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    a = ap.parse_args()
    steps_main = ["--timesteps", "40000"] if a.quick else []
    steps_sens = ["--timesteps", "40000"] if a.quick else ["--timesteps", "400000"]
    if not a.skip_data:
        run(["energyrl.prepare_data"])
        run(["energyrl.data.manifest"])
    jobs = [run(["energyrl.train", "--seeds", str(s), "--tag", "ppo", *steps_main], wait=False) for s in (0, 1, 2)]
    jobs += [run(["energyrl.train", "--seeds", "0", "--tag", "sens_cost_only", "--w-co2", "0.0", *steps_sens], wait=False),
             run(["energyrl.train", "--seeds", "0", "--tag", "sens_carbon_heavy", "--w-co2", "2.0", *steps_sens], wait=False),
             run(["energyrl.train", "--seeds", "0", "--tag", "sens_no_production_penalty", "--w-miss", "0.0",
                  "--w-late", "0.0", *steps_sens], wait=False)]
    if any(j.wait() != 0 for j in jobs):
        sys.exit("a training job failed")
    run(["energyrl.evaluate"])
    run(["energyrl.sensitivity"])
    run(["energyrl.report"])
    print("Done. Launch the dashboard with: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
