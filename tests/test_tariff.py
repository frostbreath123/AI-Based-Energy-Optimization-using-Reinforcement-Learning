import numpy as np
import pandas as pd
import pytest

from energyrl.scenarios import network_charge, turpe_ranges

TZ = "Europe/Paris"


def idx(*stamps):
    return pd.DatetimeIndex([pd.Timestamp(s, tz=TZ) for s in stamps])


def test_turpe_grid_is_ordered_like_a_peak_signal(cfg):
    for opt, grid in cfg["tariff"]["turpe7_hta_c_eur_kwh"].items():
        assert grid["PTE"] > grid["HPH"] > grid["HCH"] > grid["HPE"] > grid["HCE"] > 0, opt


def test_range_classification_placement_a(cfg):
    t = cfg["tariff"]
    r = turpe_ranges(idx("2019-01-15 09:30",   # Tuesday, Jan, fixed peak window
                         "2019-01-15 13:00",   # Tuesday, high season, full hours
                         "2019-01-15 23:00",   # high season off-peak (22-06)
                         "2019-07-17 13:00",   # Wednesday, low season, full hours
                         "2019-07-17 03:00",   # low season off-peak night
                         "2019-07-21 13:00",   # Sunday: off-peak all day
                         "2019-01-20 09:30"),  # Sunday in peak window: no peak on Sunday
                     t, "A")
    assert list(r) == ["PTE", "HPH", "HCH", "HPE", "HCE", "HCE", "HCH"]


def test_placement_b_moves_low_season_offpeak_to_midday(cfg):
    t = cfg["tariff"]
    mid, night = idx("2019-07-17 12:00"), idx("2019-07-17 23:00")
    assert turpe_ranges(mid, t, "A")[0] == "HPE" and turpe_ranges(mid, t, "B")[0] == "HCE"
    assert turpe_ranges(night, t, "A")[0] == "HCE" and turpe_ranges(night, t, "B")[0] == "HPE"
    winter = idx("2019-01-16 23:00", "2019-01-16 12:00")
    assert list(turpe_ranges(winter, t, "A")) == list(turpe_ranges(winter, t, "B"))   # high season unchanged


def test_network_charge_units_and_options(cfg):
    t = cfg["tariff"]
    i = idx("2019-01-15 09:30")
    assert network_charge(i, t, "A", "CU")[0] == pytest.approx(0.0585)   # 5.85 c/kWh -> EUR/kWh
    assert network_charge(i, t, "A", "LU")[0] == pytest.approx(0.0271)
    assert np.all(network_charge(idx("2019-07-17 03:00"), t, "A", "CU") == pytest.approx(0.0071))
