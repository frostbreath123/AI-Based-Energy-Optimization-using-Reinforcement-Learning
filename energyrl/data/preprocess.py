"""Turn raw downloads into one clean hourly table (local Europe/Paris time).

Columns of the processed table:
  spot_eur_kwh      day-ahead spot price FR (EUR/kWh)                    [processed public data]
  pv_cf             PV capacity factor, kW per kWp (Renewables.ninja)    [processed public data]
  carbon_g_kwh      grid carbon intensity of French generation           [derived: public mix x emission factors]
  load_c<k>         ELMAS cluster-k industrial load shape, mean = 1      [processed public data, 2018 shape on 2019 calendar]
Job schedules, deadlines, forecast errors and the network tariff are SYNTHETIC (energyrl/scenarios.py).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from energyrl.config import PROCESSED_CSV, PROCESSED_DIR, RAW_DIR, load_config

ELMAS_TS = RAW_DIR / "elmas" / "ELMAS_dataset" / "Time_series_18_clusters.csv"
LOAD_SHIFT_DAYS = 364    # 52 weeks: maps a 2019 calendar day onto the same weekday and season of 2018


def load_pv(year: int) -> pd.Series:
    d = json.loads((RAW_DIR / f"pv_ninja_paris_{year}.json").read_text())["data"]
    s = pd.Series({pd.Timestamp(int(k), unit="ms", tz="UTC"): v["electricity"] for k, v in d.items()}, dtype=float)
    return s.sort_index().rename("pv_cf")  # kW per 1 kWp


def load_prices(year: int, country: str) -> pd.Series:
    d = json.loads((RAW_DIR / f"prices_{country}_{year}.json").read_text())
    s = pd.Series(d["price"], index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True), dtype=float)
    return (s.resample("1h").mean() / 1000.0).rename("spot_eur_kwh")  # EUR/MWh -> EUR/kWh


def load_carbon(year: int, country: str, factors: dict[str, float]) -> pd.Series:
    frames = []
    for month in range(1, 13):
        d = json.loads((RAW_DIR / f"public_power_{country}_{year}_{month:02d}.json").read_text())
        idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
        cols = {pt["name"]: pd.Series(pt["data"], index=idx, dtype=float)
                for pt in d["production_types"] if pt["name"] in factors}
        frames.append(pd.DataFrame(cols))
    mix = pd.concat(frames).sort_index()
    mix = mix[~mix.index.duplicated()]
    mix = mix.resample("1h").mean().clip(lower=0).fillna(0)  # MW; null/negative generation -> 0
    ef = pd.Series({k: factors[k] for k in mix.columns})
    total = mix.sum(axis=1)
    return ((mix * ef).sum(axis=1) / total.replace(0, np.nan)).rename("carbon_g_kwh")


def load_elmas(clusters: list[int], target_year: int) -> pd.DataFrame:
    """ELMAS hourly cluster profiles (local time, decimal comma), normalised to mean 1 and shifted to `target_year`."""
    raw = pd.read_csv(ELMAS_TS, sep=";", dtype=str)
    raw["Time"] = pd.to_datetime(raw["Time"])
    ts = raw.set_index("Time").apply(lambda s: pd.to_numeric(s.str.replace(",", "."), errors="coerce"))
    ts = ts[[str(c) for c in clusters]]
    ts = ts / ts.mean()
    ts.columns = [f"load_c{c}" for c in clusters]
    ts.index = ts.index + pd.Timedelta(days=LOAD_SHIFT_DAYS)        # naive local timestamps on the target calendar
    return ts[ts.index.year == target_year]


def build_processed() -> pd.DataFrame:
    cfg = load_config()["data"]
    year, country = cfg["year"], cfg["country"]
    clusters = sorted(set(cfg["load_clusters_train"]) | set(cfg["load_clusters_ood"]))
    market = pd.concat([load_prices(year, country), load_pv(year),
                        load_carbon(year, country, cfg["emission_factors"])], axis=1)
    market = market.loc[f"{year}-01-01":f"{year}-12-31 23:00"]
    n_raw = len(market)
    market = market.interpolate(limit=3, limit_area="inside")     # short gaps (<= 3 h) only
    market["pv_cf"] = market["pv_cf"].clip(0, 1)
    market.index = market.index.tz_convert(cfg["timezone"])
    market.index.name = "local_time"
    loads = load_elmas(clusters, year)
    loads = loads.interpolate(limit=3, limit_area="inside")
    # join on naive local wall-clock time so ELMAS (naive local) and market (tz-aware) line up per hour
    market["_wall"] = market.index.tz_localize(None)
    df = market.merge(loads, left_on="_wall", right_index=True, how="left").drop(columns="_wall")
    df = df.dropna()
    day = df.index.normalize()
    counts = df.groupby(day).size()
    good = counts[counts == 24].index                              # drops DST days and any incomplete day
    df = df[day.isin(good)]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV)
    stats = {"rows_raw_aligned": n_raw, "rows_final": len(df), "days_final": len(good),
             "price_eur_kwh_min": float(df.spot_eur_kwh.min()), "price_eur_kwh_max": float(df.spot_eur_kwh.max()),
             "carbon_g_kwh_mean": float(df.carbon_g_kwh.mean()), "pv_cf_mean": float(df.pv_cf.mean())}
    (PROCESSED_DIR / "processing_stats.json").write_text(json.dumps(stats, indent=2))
    return df


def load_processed() -> pd.DataFrame:
    """Load the processed table; raises a helpful error if the pipeline was not run."""
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(f"{PROCESSED_CSV} not found. Run: python -m energyrl.prepare_data")
    df = pd.read_csv(PROCESSED_CSV, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(load_config()["data"]["timezone"])
    return df
