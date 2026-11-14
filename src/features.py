"""Feature engineering for a *direct* multi-step forecaster.

Key idea: every lag / rolling feature is shifted by at least HORIZON+1 days, so a forecast for any day
in the next HORIZON days only uses data that is actually known at forecast time (no leakage)."""
import numpy as np
import pandas as pd

from .config import HORIZON

MIN_LAG = HORIZON + 1
LAGS = [MIN_LAG, MIN_LAG + 7, MIN_LAG + 21, 364]
ROLL_WINDOWS = [7, 28, 91]

CALENDAR = ["dow", "month", "day", "dayofyear", "weekofyear", "is_weekend"]
FEATURES = (["store", "item"] + CALENDAR + [f"lag_{l}" for l in LAGS]
            + [f"roll_mean_{w}" for w in ROLL_WINDOWS] + ["roll_std_28"])


def extend_frame(df, horizon=HORIZON):
    """Append `horizon` future days (sales = NaN) for every store-item series."""
    future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=horizon)
    keys = df[["store", "item"]].drop_duplicates()
    fut = keys.merge(pd.DataFrame({"date": future_dates}), how="cross")
    fut["sales"] = np.nan
    out = pd.concat([df, fut], ignore_index=True)
    return out.sort_values(["store", "item", "date"]).reset_index(drop=True)


def add_features(df):
    df = df.copy()
    d = df["date"]
    df["dow"] = d.dt.dayofweek
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dayofyear"] = d.dt.dayofyear
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    g = df.groupby(["store", "item"])["sales"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)

    shifted = g.shift(MIN_LAG)
    gs = shifted.groupby([df["store"], df["item"]])
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = gs.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
    df["roll_std_28"] = gs.transform(lambda s: s.rolling(28, min_periods=28).std())
    return df
