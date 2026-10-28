"""Download public datasets (France) into data/raw with retrieval metadata.

Sources (see docs/DATA_SOURCES.md):
  * Renewables.ninja API (MERRA-2): hourly PV output for Paris   -> pv_ninja_paris_<year>.json
  * ELMAS dataset (Bellinguer et al., Sci Data 2023, Figshare)   -> elmas/ELMAS_dataset/*.csv
  * CRE deliberation 2025-40 (TURPE 7 HTA-BT tariff grid, PDF)                  -> tariff/CRE_2025-40_TURPE7_HTA-BT.pdf
  * Energy-Charts API (Fraunhofer ISE): day-ahead price FR, public generation FR (monthly files)
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import date, datetime, timezone

import requests

from energyrl.config import RAW_DIR, load_config

NINJA_URL = "https://www.renewables.ninja/api/data/pv"
ELMAS_URL = "https://ndownloader.figshare.com/files/41895786"
TURPE_URL = "https://www.cre.fr/fileadmin/Documents/Deliberations/2025/250204_2025-40_TURPE_7_HTA-BT.pdf"
EC_PRICE_URL = "https://api.energy-charts.info/price"
EC_POWER_URL = "https://api.energy-charts.info/public_power"


def _get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=180)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:  # network / HTTP failure
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Download failed for {url}: {last}")


def download_all(force: bool = False) -> dict:
    cfg = load_config()["data"]
    year = cfg["year"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RAW_DIR / "retrieval_log.json"
    prev = json.loads(log_path.read_text()) if log_path.exists() else {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    meta: dict[str, dict] = {}

    def fetched(key: str, fresh: bool, **extra) -> None:
        meta[key] = {"retrieved": today if fresh else prev.get(key, {}).get("retrieved", "unknown (cached)"), **extra}

    pv_path = RAW_DIR / f"pv_ninja_paris_{year}.json"
    fresh = force or not pv_path.exists()
    if fresh:
        pv = cfg["pv"]
        params = {"lat": cfg["location"]["lat"], "lon": cfg["location"]["lon"], "date_from": f"{year}-01-01",
                  "date_to": f"{year}-12-31", "dataset": pv["dataset"], "capacity": 1, "system_loss": pv["system_loss"],
                  "tracking": 0, "tilt": pv["tilt"], "azim": pv["azim"], "format": "json"}
        pv_path.write_bytes(_get(NINJA_URL, params).content)
    fetched("ninja_pv", fresh, file=pv_path.name)

    elmas_dir = RAW_DIR / "elmas"
    fresh = force or not (elmas_dir / "ELMAS_dataset" / "Time_series_18_clusters.csv").exists()
    if fresh:
        elmas_dir.mkdir(exist_ok=True)
        zipfile.ZipFile(io.BytesIO(_get(ELMAS_URL).content)).extractall(elmas_dir)
    fetched("elmas", fresh, file="elmas/ELMAS_dataset/Time_series_18_clusters.csv")

    tariff_path = RAW_DIR / "tariff" / "CRE_2025-40_TURPE7_HTA-BT.pdf"
    fresh = force or not tariff_path.exists()
    if fresh:
        tariff_path.parent.mkdir(exist_ok=True)
        tariff_path.write_bytes(_get(TURPE_URL).content)
    fetched("turpe", fresh, file="tariff/" + tariff_path.name)

    price_path = RAW_DIR / f"prices_{cfg['country']}_{year}.json"
    fresh = force or not price_path.exists()
    if fresh:
        r = _get(EC_PRICE_URL, {"bzn": cfg["bidding_zone"], "start": f"{year}-01-01", "end": f"{year}-12-31"})
        price_path.write_bytes(r.content)
    fetched("prices", fresh, file=price_path.name)

    files, any_fresh = [], False
    for month in range(1, 13):
        start = date(year, month, 1)
        nxt = date(year + (month == 12), month % 12 + 1, 1)
        end = date.fromordinal(nxt.toordinal() - 1)
        p = RAW_DIR / f"public_power_{cfg['country']}_{year}_{month:02d}.json"
        if force or not p.exists():
            p.write_bytes(_get(EC_POWER_URL, {"country": cfg["country"], "start": start.isoformat(),
                                              "end": end.isoformat()}).content)
            any_fresh = True
        files.append(p.name)
    fetched("public_power", any_fresh, files=files)
    log_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    print(json.dumps(download_all(), indent=2))
