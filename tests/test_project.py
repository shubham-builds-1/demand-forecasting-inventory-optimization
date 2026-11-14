import math

import numpy as np
import pandas as pd

from src.config import HORIZON
from src.features import LAGS, MIN_LAG, add_features
from src.inventory import eoq, recommend_order, safety_stock
from src.models import seasonal_naive, weekday_average
from src.simulate import simulate


def test_no_leakage_all_lags_beyond_horizon():
    assert MIN_LAG > HORIZON and all(l > HORIZON for l in LAGS)


def test_lag_feature_matches_history():
    dates = pd.date_range("2020-01-01", periods=500)
    df = pd.DataFrame({"date": dates, "store": 1, "item": 1, "sales": np.arange(500, dtype=float)})
    f = add_features(df)
    assert f.loc[300, f"lag_{MIN_LAG}"] == 300 - MIN_LAG


def test_eoq_formula():
    # D=1000, S=50, H=0.2*10=2 -> sqrt(2*1000*50/2) = 223.6
    assert math.isclose(float(eoq(1000, 10, order_cost=50, holding_rate=0.2)), math.sqrt(50000), rel_tol=1e-9)


def test_safety_stock_grows_with_service_level_and_variability():
    assert safety_stock(5, 14, 0.99) > safety_stock(5, 14, 0.90) > 0
    assert safety_stock(10, 14, 0.95) > safety_stock(5, 14, 0.95)
    assert math.isclose(float(safety_stock(5, 4, 0.5)), 0.0, abs_tol=1e-9)  # z(50%) = 0


def test_recommend_order():
    row = {"order_up_to": 200.0, "expected_demand": 150.0, "safety_stock": 50.0, "eoq": 300.0}
    assert recommend_order(row, on_hand=50, on_order=20)["order_qty"] == 130
    assert recommend_order(row, on_hand=250)["order_qty"] == 0


def test_baselines_continue_weekly_pattern():
    hist = np.tile(np.arange(7), 10)[None, :].astype(float)   # ... 5 6 | next should be 0 1 2
    assert list(seasonal_naive(hist, 3)[0]) == [0, 1, 2]
    assert list(weekday_average(hist, 3)[0]) == [0, 1, 2]


def test_simulation_constant_demand_never_stocks_out():
    S, T = 2, 60
    demand = np.full((S, T), 10.0)
    target = np.full((S, T), 10.0 * 14 + 20)
    res = simulate(demand, target, target[:, 0], lead_time=7, review_period=7)
    assert res["lost"].sum() == 0
