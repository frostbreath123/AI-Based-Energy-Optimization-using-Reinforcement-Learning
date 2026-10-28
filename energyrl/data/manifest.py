"""Generate data/manifest.json and docs/DATA_SOURCES.md (provenance, licences, preprocessing).

Run: python -m energyrl.data.manifest   (after energyrl.prepare_data)
"""
from __future__ import annotations

import hashlib
import json

from energyrl.config import PROCESSED_CSV, RAW_DIR, ROOT, load_config

PUBLIC, DERIVED, SYNTHETIC = "processed public data", "derived from public data + literature factors", "SYNTHETIC (generated)"


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_manifest() -> dict:
    cfg = load_config()
    d = cfg["data"]
    log = json.loads((RAW_DIR / "retrieval_log.json").read_text())
    year, ctry = d["year"], d["country"]
    pv = RAW_DIR / f"pv_ninja_paris_{year}.json"
    pr = RAW_DIR / f"prices_{ctry}_{year}.json"
    pw = sorted(RAW_DIR.glob(f"public_power_{ctry}_{year}_*.json"))
    tp = RAW_DIR / "tariff" / "CRE_2025-40_TURPE7_HTA-BT.pdf"
    el = RAW_DIR / "elmas" / "ELMAS_dataset" / "Time_series_18_clusters.csv"
    datasets = [
        {"id": "ninja_pv", "name": "Renewables.ninja hourly PV output (MERRA-2), 1 kWp reference system", "class": PUBLIC,
         "publisher": "Renewables.ninja (Pfenninger & Staffell; Imperial College London / ETH Zurich)",
         "url": "https://www.renewables.ninja/api/data/pv", "retrieved": log["ninja_pv"]["retrieved"],
         "files": [pv.name], "sha256_16": [_sha(pv)], "units": "kW per kWp", "resolution": "hourly (UTC)",
         "geography": f"{d['location']['name']} (lat {d['location']['lat']}, lon {d['location']['lon']}), tilt {d['pv']['tilt']} deg, south, losses {d['pv']['system_loss']:.0%}",
         "license": "Renewables.ninja states CC BY-NC 4.0 (non-commercial, attribution); verify at https://www.renewables.ninja/about. Anonymous API access, no account created.",
         "preprocessing": "Parsed from the JSON API response, clipped to [0,1] kW/kWp, converted from UTC to Europe/Paris local time.",
         "role": "Actual on-site PV generation = capacity factor x installed kWp x scenario scale."},
        {"id": "elmas", "name": "ELMAS: hourly electrical load profiles of French industrial and tertiary sectors (18 clusters, 2018)",
         "class": PUBLIC, "publisher": "Bellinguer, Girard, Bocquet, Chevalier (MINES Paris - PSL / Enedis); Scientific Data 10, 2023",
         "url": "https://doi.org/10.1038/s41597-023-02542-z (data: https://doi.org/10.6084/m9.figshare.23889780.v1)",
         "retrieved": log["elmas"]["retrieved"], "files": [el.name], "sha256_16": [_sha(el)],
         "units": "average energy per hour per cluster (normalised to mean 1 here: only the SHAPE is used)",
         "resolution": "hourly, one year (2018), averaged over 55,730 customers", "geography": "France (Enedis network)",
         "license": "CC BY 4.0 (Figshare record)",
         "preprocessing": f"Decimal commas parsed; each cluster divided by its annual mean; timestamps shifted by 364 days so 2018 weekdays/seasons align with the 2019 calendar; clusters {d['load_clusters_train']} used for training scenarios, {d['load_clusters_ood']} held out for an OOD family.",
         "role": "Shape of the plant's inflexible load (base_kw_mean x shape). Absolute plant size is an assumption."},
        {"id": "ec_price", "name": f"Day-ahead spot price, bidding zone {d['bidding_zone']}", "class": PUBLIC,
         "publisher": "Fraunhofer ISE Energy-Charts API; data from Bundesnetzagentur | SMARD.de",
         "url": f"https://api.energy-charts.info/price?bzn={d['bidding_zone']}", "retrieved": log["prices"]["retrieved"],
         "files": [pr.name], "sha256_16": [_sha(pr)], "units": "EUR/MWh (converted to EUR/kWh)", "resolution": "hourly",
         "geography": "France bidding zone", "license": "CC BY 4.0 (stated in the API response: 'CC BY 4.0 from Bundesnetzagentur | SMARD.de')",
         "preprocessing": "Resampled to hourly mean, /1000, local time. Negative prices kept in the spot series; the customer tariff is floored at 0.",
         "role": "Wholesale component of the tariff (tariff = spot x scale + network adder + levies)."},
        {"id": "ec_power", "name": "Public net electricity generation by source, France", "class": DERIVED,
         "publisher": "Fraunhofer ISE Energy-Charts API (ENTSO-E based)", "url": f"https://api.energy-charts.info/public_power?country={ctry}",
         "retrieved": log["public_power"]["retrieved"], "files": [p.name for p in pw], "sha256_16": [_sha(p) for p in pw],
         "units": "MW per source type", "resolution": "hourly/15 min, resampled to hourly mean", "geography": "France",
         "license": "Energy-Charts states CC BY 4.0 for its data (the response carries no licence field; verify at https://www.energy-charts.info).",
         "preprocessing": "Per-source MW averaged hourly; null/negative -> 0; carbon intensity = sum(generation x emission factor) / sum(generation). Imports/exports and pumped-storage charging excluded (production-based).",
         "role": "Time-varying grid carbon intensity (gCO2e/kWh); France is nuclear-dominated so the level is low (about 54 g/kWh mean)."},
        {"id": "emission_factors", "name": "Emission factors per generation technology", "class": "literature constants",
         "publisher": "IPCC AR5 WGIII (2014) Annex III, Table A.III.2 (median lifecycle): coal 820, gas CCGT 490, biomass 230, geothermal 38, hydro 24, nuclear 12, solar PV utility 48, onshore wind 11, offshore wind 12 gCO2e/kWh",
         "url": "https://www.ipcc.ch/report/ar5/wg3/", "retrieved": "n/a", "files": [], "units": "gCO2e/kWh", "resolution": "constant",
         "geography": "global medians", "license": "IPCC reports: free for non-commercial use with citation",
         "preprocessing": "Lignite (1000), oil (650), coal-derived gas (600), waste (400), others (500) are ANALYST ASSUMPTIONS (not IPCC values) - see config/default.yaml.",
         "role": "Convert the generation mix into carbon intensity."},
        {"id": "turpe7", "name": "TURPE 7 HTA network tariff, variable energy component (grid of 1 Aug 2025)", "class": "published regulated tariff (transcribed by hand into config/default.yaml)",
         "publisher": "Commission de regulation de l'energie (CRE), deliberation no 2025-40 of 4 Feb 2025", "url": "https://www.cre.fr/fileadmin/Documents/Deliberations/2025/250204_2025-40_TURPE_7_HTA-BT.pdf",
         "retrieved": log.get("turpe", {}).get("retrieved", "unknown"), "files": [tp.name], "sha256_16": [_sha(tp)] if tp.exists() else [],
         "units": "c EUR/kWh for the five ranges PTE, HPH, HCH, HPE, HCE; options CU (short) and LU (long utilisation)",
         "resolution": "five time ranges (fixed peak Dec-Feb; high season Nov-Mar, low season Apr-Oct; full vs off-peak hours)",
         "geography": "France, medium-voltage (HTA) grid users", "license": "Public regulatory document (CRE); cite the deliberation",
         "preprocessing": "Values copied from the 'Grilles simplifiees de la composante de soutirage au 1er aout 2025' table (CU 5.85/4.31/2.03/1.03/0.71, LU 2.71/2.14/1.50/0.94/0.69 c/kWh). Off-peak placement A = 22-06 h and Sundays (CRE text); placement B = low-season off-peak 02-06 h + 11-15 h (derived from CRE Table 47 'hours to favour' 02-06 h and 11-17 h; 8 h/day). Exact fixed-peak hours 09-11 h and 18-20 h are an ASSUMPTION inside the CRE windows (08-12 h, 17-21 h). The 2026 +3.04 % indexation and the power (EUR/kW/yr) components are not applied, except that the CU power term 14.70 EUR/kW/yr/365 is used as the daily demand charge. Supply margin, accise and CTA (0.03 EUR/kWh) are assumed, not sourced.",
         "role": "Network part of the electricity price: gives the peak / intermediate / off-peak structure. NOTE: tariff year (2025-26) differs from the market data year (2019); the wholesale price is used as a level-shifting signal."},
        {"id": "synthetic_jobs", "name": "Production jobs, deadlines, forecast errors, PV drops, plant size", "class": SYNTHETIC,
         "publisher": "this project", "url": "-", "retrieved": "n/a", "files": [], "units": "kW, hours", "resolution": "hourly",
         "geography": "stylised manufacturing site", "license": "project code",
         "preprocessing": "Generated by energyrl/scenarios.py from seeded random draws; ELMAS gives only the base-load shape, not job schedules.",
         "role": "Flexible loads: job power, durations, release times, deadlines."},
    ]
    return {"generated_by": "energyrl.data.manifest", "year": year, "datasets": datasets,
            "processed_file": str(PROCESSED_CSV.relative_to(ROOT)), "processed_sha256_16": _sha(PROCESSED_CSV)}


def write_docs() -> None:
    m = build_manifest()
    (ROOT / "data" / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    lines = ["# Data sources and provenance", "",
             "Generated by `python -m energyrl.data.manifest`. All real data are for **France**: market, PV and carbon for "
             f"calendar year {m['year']}, industrial load shapes from 2018 (aligned by weekday). Processed public data, "
             "derived data and synthetic data are labelled separately.", ""]
    for d in m["datasets"]:
        lines += [f"## {d['name']}", f"- **Class:** {d['class']}", f"- **Publisher:** {d['publisher']}",
                  f"- **URL:** {d['url']}", f"- **Retrieved:** {d['retrieved']}", f"- **Units:** {d['units']}",
                  f"- **Resolution:** {d['resolution']}", f"- **Geography:** {d['geography']}",
                  f"- **License:** {d['license']}", f"- **Preprocessing / notes:** {d['preprocessing']}",
                  f"- **Role in simulation:** {d['role']}"]
        if d["files"]:
            lines.append(f"- **Local files (data/raw):** {', '.join(d['files'][:3])}{' ...' if len(d['files']) > 3 else ''}")
        lines.append("")
    lines += ["## Coherence and limitations of combining sources", "",
              "- Everything real is French. PV is a Paris point (proxy for the national fleet), price and carbon are national, load shapes are from the French distribution network.",
              "- ELMAS is 2018, market/PV/carbon are 2019. The load shape is shifted by 364 days so weekday and season align; year-specific weather effects on load are not matched to 2019 weather.",
              "- ELMAS profiles are averages over many customers (anonymisation), so they are smoother than a single plant's load; only their shape is used.",
              "- Renewables.ninja's anonymous API begins at 2019, which fixed the data year; its licence is non-commercial.",
              "- Timestamps are aligned in UTC then converted to Europe/Paris local time; DST-transition days and incomplete days are dropped (360 of 365 days kept). Gaps up to 3 h are interpolated.",
              "- The network tariff is the real TURPE 7 HTA grid (2025-26) combined with 2019 wholesale prices; the supply margin/taxes, jobs and plant size are synthetic.",
              "- An earlier iteration used German 2023 data (PVGIS, Energy-Charts DE-LU); those files and results are archived under `data/archive_germany_2023/`, `results/archive_v2_germany_lookahead8/`.", ""]
    (ROOT / "docs" / "DATA_SOURCES.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    write_docs()
    print("wrote data/manifest.json and docs/DATA_SOURCES.md")
