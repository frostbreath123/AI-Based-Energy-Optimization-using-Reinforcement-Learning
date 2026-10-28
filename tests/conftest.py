import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from energyrl.config import PROCESSED_CSV, load_config  # noqa: E402
from energyrl.scenarios import Job, Scenario  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def df():
    if not PROCESSED_CSV.exists():
        pytest.skip("processed data missing: run python -m energyrl.prepare_data")
    from energyrl.data.preprocess import load_processed
    return load_processed()


def make_scenario(jobs=None, price=0.2, pv=0.0, carbon=400.0, base=100.0, n=40) -> Scenario:
    return Scenario(name="manual", date="2023-06-01", weekend=False, spot=np.full(n, price), price=np.full(n, price),
                    carbon=np.full(n, carbon), pv=np.full(n, pv), pv_forecast=np.full(n, pv),
                    base_kw=np.full(n, base), jobs=jobs or [])


@pytest.fixture
def manual():
    return make_scenario


@pytest.fixture
def job():
    return Job
