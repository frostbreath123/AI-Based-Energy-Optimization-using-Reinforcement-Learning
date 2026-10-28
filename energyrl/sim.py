"""Physics/accounting core of the facility simulation (no Gym, no reward).

One step = one hour. The simulator enforces every physical and operational constraint:
  * jobs can only run once released and while work remains (premature requests are ignored + counted),
  * the grid import limit is never exceeded (battery charging is cut first, then jobs are shed + counted),
  * battery power / energy / efficiency limits are enforced (battery cannot export to the grid),
  * production deadlines are tracked (overdue hours, completion times, missed jobs).
Action vector: [run_job_0 .. run_job_{K-1}, battery_level_index].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from energyrl.scenarios import Scenario


@dataclass
class StepResult:
    cost: float = 0.0              # EUR this hour (energy + demand-charge increment - export revenue)
    co2_kg: float = 0.0
    overdue_jobs: int = 0          # jobs past their deadline and unfinished at the end of this hour
    shed_events: int = 0
    invalid_requests: int = 0
    import_kw: float = 0.0
    export_kw: float = 0.0
    settlement: float = 0.0        # EUR: stored battery energy valued at the day's mean tariff, booked in the hour it is stored
    missed_jobs: int = 0
    missed_work_frac: float = 0.0  # sum over missed jobs of remaining/duration
    done: bool = False


class FacilitySim:
    def __init__(self, scenario: Scenario, cfg: dict):
        self.sc, self.cfg = scenario, cfg
        s = cfg["site"]
        self.H = s["horizon_hours"]
        self.K = s["max_jobs"]
        self.limit = float(s["grid_import_limit_kw"])
        self.feed_in = float(s["feed_in_eur_kwh"])
        self.dc_rate = float(s["demand_charge_eur_kw_day"])
        b = s["battery"]
        self.cap, self.p_max = float(b["capacity_kwh"]), float(b["power_kw"])
        self.eff_c, self.eff_d = float(b["eff_charge"]), float(b["eff_discharge"])
        self.soc_min, self.soc_max = b["soc_min"] * self.cap, b["soc_max"] * self.cap
        self.levels = np.asarray(s["battery_levels"], dtype=float)
        self.jobs = scenario.jobs[: self.K]
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.t = 0
        self.soc0 = self.cfg["site"]["battery"]["soc_init"] * self.cap
        self.soc = self.soc0
        self.remaining = np.array([j.duration for j in self.jobs], dtype=int)
        self.done_time = np.full(len(self.jobs), -1, dtype=int)
        self.peak = 0.0
        self.storage_value = float(np.mean(self.sc.price[: self.H])) * self.eff_d    # EUR per kWh of stored energy
        self.acc = dict(energy_cost=0.0, demand_charge=0.0, export_rev=0.0, import_kwh=0.0, export_kwh=0.0,
                        consumption_kwh=0.0, co2_kg=0.0, pv_gen=0.0, pv_used=0.0, shifted_kwh=0.0,
                        job_kwh=0.0, sheds=0, invalid=0, overdue_hours=0, terminal_cost=0.0,
                        batt_charge_kwh=0.0, batt_discharge_kwh=0.0)
        self.trace: list[dict] = []

    @property
    def n_jobs(self) -> int:
        return len(self.jobs)

    @property
    def finished(self) -> bool:
        return self.t >= self.H

    def eligible(self, k: int) -> bool:
        return k < self.n_jobs and self.remaining[k] > 0 and self.jobs[k].release <= self.t

    def slack(self, k: int) -> int:
        """Spare hours: deadline - now - remaining work (<=0 means it must run every hour now)."""
        return self.jobs[k].deadline - self.t - int(self.remaining[k])

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray) -> StepResult:
        if self.finished:
            raise RuntimeError("step() called after the episode ended")
        t, sc = self.t, self.sc
        res = StepResult()
        action = np.asarray(action, dtype=int)
        req = [bool(action[k] > 0) if k < self.n_jobs else False for k in range(self.K)]
        run: list[int] = []
        for k in range(self.n_jobs):
            if not req[k] or self.remaining[k] <= 0:
                continue                               # not requested, or already complete (harmless)
            if self.jobs[k].release > t:
                res.invalid_requests += 1              # premature start: ignored
                continue
            run.append(k)
        level = float(self.levels[int(np.clip(action[self.K], 0, len(self.levels) - 1))])

        pv = float(sc.pv[t])
        price, carbon = float(sc.price[t]), float(sc.carbon[t])
        base = float(sc.base_kw[t])

        def dispatch(run_jobs: list[int], lvl: float):
            load = base + sum(self.jobs[k].power_kw for k in run_jobs)
            charge = discharge = 0.0
            if lvl > 0:
                charge = min(lvl * self.p_max, (self.soc_max - self.soc) / self.eff_c)
                charge = max(charge, 0.0)
            elif lvl < 0:
                discharge = min(-lvl * self.p_max, max(self.soc - self.soc_min, 0.0) * self.eff_d,
                                max(load - pv, 0.0))          # battery never exports
            net = load + charge - pv - discharge
            return load, charge, discharge, net

        load, charge, discharge, net = dispatch(run, level)
        if level != 0 and charge == 0.0 and discharge == 0.0:
            res.invalid_requests += 1                 # battery full / empty / nothing to offset
        shed_n = 0
        if net > self.limit + 1e-9 and charge > 0:    # first sacrifice battery charging
            level = 0.0
            load, charge, discharge, net = dispatch(run, level)
        while net > self.limit + 1e-9 and run:        # then shed the job with the most slack
            victim = max(run, key=lambda k: (self.slack(k), self.jobs[k].deadline))
            run.remove(victim)
            shed_n += 1
            load, charge, discharge, net = dispatch(run, level)
        res.shed_events = shed_n

        imp, exp = max(net, 0.0), max(-net, 0.0)
        new_peak = max(self.peak, imp)
        dc = (new_peak - self.peak) * self.dc_rate
        self.peak = new_peak
        energy_cost = imp * price
        export_rev = exp * self.feed_in
        res.cost = energy_cost + dc - export_rev
        res.co2_kg = imp * carbon / 1000.0
        res.import_kw, res.export_kw = imp, exp

        # battery state; the change in stored energy is settled at the day's mean tariff (charging = negative cost) so
        # that the end-of-day battery inventory is credited in the hours that create it (same episode total, denser signal)
        d_soc = charge * self.eff_c - discharge / self.eff_d
        self.soc += d_soc
        res.settlement = -d_soc * self.storage_value

        # jobs
        job_load = np.zeros(self.K)
        for k in run:
            job_load[k] = self.jobs[k].power_kw
            self.remaining[k] -= 1
            if self.remaining[k] == 0:
                self.done_time[k] = t + 1
            j = self.jobs[k]
            if not (j.planned_start <= t < j.planned_start + j.duration):
                self.acc["shifted_kwh"] += j.power_kw
            self.acc["job_kwh"] += j.power_kw
        overdue = sum(1 for k in range(self.n_jobs) if self.remaining[k] > 0 and t + 1 >= self.jobs[k].deadline)
        res.overdue_jobs = overdue

        a = self.acc
        a["energy_cost"] += energy_cost; a["demand_charge"] += dc; a["export_rev"] += export_rev
        a["import_kwh"] += imp; a["export_kwh"] += exp; a["consumption_kwh"] += load
        a["co2_kg"] += res.co2_kg; a["pv_gen"] += pv; a["pv_used"] += min(pv, load + charge)
        a["sheds"] += shed_n; a["invalid"] += res.invalid_requests; a["overdue_hours"] += overdue
        a["batt_charge_kwh"] += charge; a["batt_discharge_kwh"] += discharge; a["terminal_cost"] += res.settlement

        self.trace.append({"t": t, "price": price, "carbon": carbon, "pv": pv, "pv_forecast": float(sc.pv_forecast[t]),
                           "base": base, "jobs": job_load.tolist(), "load": load, "import": imp, "export": exp,
                           "battery_kw": charge - discharge, "soc": self.soc, "cost": res.cost,
                           "co2": res.co2_kg, "shed": shed_n})
        self.t += 1
        if self.t >= self.H:
            res.done = True
            miss = [k for k in range(self.n_jobs) if self.remaining[k] > 0]
            res.missed_jobs = len(miss)
            res.missed_work_frac = float(sum(self.remaining[k] / self.jobs[k].duration for k in miss))
        return res

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> dict:
        """KPIs of the (finished) episode, computed purely from accumulated simulation results."""
        a = self.acc
        n = self.n_jobs
        done = int(np.sum(self.remaining == 0)) if n else 0
        on_time = int(sum(1 for k in range(n) if self.remaining[k] == 0 and self.done_time[k] <= self.jobs[k].deadline))
        late_done = done - on_time
        tard = 0
        for k in range(n):
            end = self.done_time[k] if self.remaining[k] == 0 else self.H
            tard += max(0, int(end) - self.jobs[k].deadline)
        cost = a["energy_cost"] + a["demand_charge"] - a["export_rev"] + a["terminal_cost"]
        return {
            "cost_eur": cost, "energy_cost_eur": a["energy_cost"], "demand_charge_eur": a["demand_charge"],
            "export_revenue_eur": a["export_rev"], "battery_settlement_eur": a["terminal_cost"],
            "import_kwh": a["import_kwh"], "consumption_kwh": a["consumption_kwh"], "co2_kg": a["co2_kg"],
            "jobs_total": n, "jobs_completed": done, "jobs_on_time": on_time, "jobs_late": late_done,
            "jobs_missed": n - done, "completion_rate": done / n if n else 1.0,
            "on_time_rate": on_time / n if n else 1.0, "tardiness_hours": tard,
            "pv_generation_kwh": a["pv_gen"], "pv_used_kwh": a["pv_used"],
            "pv_utilization": a["pv_used"] / a["pv_gen"] if a["pv_gen"] > 0 else 0.0,
            "renewable_share": a["pv_used"] / a["consumption_kwh"] if a["consumption_kwh"] > 0 else 0.0,
            "load_shifted_kwh": a["shifted_kwh"], "job_kwh": a["job_kwh"], "peak_kw": self.peak,
            "constraint_violations": a["sheds"] + a["invalid"], "shed_events": a["sheds"],
            "invalid_requests": a["invalid"], "battery_throughput_kwh": a["batt_charge_kwh"],
        }
