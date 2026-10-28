"""CLI: download + preprocess + validate.  `python -m energyrl.prepare_data [--force]`"""
from __future__ import annotations

import argparse

from energyrl.data.download import download_all
from energyrl.data.preprocess import build_processed
from energyrl.data.validate import validate_processed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download raw files")
    args = ap.parse_args()
    download_all(force=args.force)
    df = build_processed()
    print("Processed rows:", len(df), "days:", len(df) // 24)
    print("Validation:", validate_processed(df))


if __name__ == "__main__":
    main()
