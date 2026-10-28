"""Schema, unit and range checks for the processed table."""
from __future__ import annotations

import pandas as pd

EXPECTED = {
    "spot_eur_kwh": (-0.6, 1.2),    # EUR/kWh
    "pv_cf": (0.0, 1.0),            # kW per kWp
    "carbon_g_kwh": (0.0, 1100.0),  # gCO2eq/kWh
}
LOAD_SHAPE_RANGE = (0.2, 4.0)       # ELMAS shapes are normalised to mean 1


def validate_processed(df: pd.DataFrame) -> dict:
    problems = []
    ranges = dict(EXPECTED)
    ranges.update({c: LOAD_SHAPE_RANGE for c in df.columns if c.startswith("load_c")})
    if not any(c.startswith("load_c") for c in df.columns):
        problems.append("no ELMAS load_c<k> columns")
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            problems.append(f"missing column {col}")
            continue
        if df[col].isna().any():
            problems.append(f"NaNs in {col}")
        if df[col].min() < lo or df[col].max() > hi:
            problems.append(f"{col} outside [{lo},{hi}]: {df[col].min():.3f}..{df[col].max():.3f}")
    if len(df) % 24 != 0:
        problems.append("row count not a multiple of 24")
    if not df.index.is_monotonic_increasing:
        problems.append("index not sorted")
    if problems:
        raise ValueError("Processed data validation failed: " + "; ".join(problems))
    return {"ok": True, "rows": len(df), "columns": list(df.columns)}
