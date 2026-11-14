"""Data loading. Uses the Kaggle file if present, otherwise generates realistic synthetic data
with the same schema (date, store, item, sales) so the project runs out of the box."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PATH, SEED


def generate_synthetic(n_stores=10, n_items=50, start="2013-01-01", end="2017-12-31", seed=SEED):
    """Daily sales with trend, yearly + weekly seasonality, item/store effects and noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    T = len(dates)
    t = np.arange(T)

    base = rng.lognormal(mean=3.0, sigma=0.5, size=n_items)        # item popularity
    store_f = rng.uniform(0.7, 1.4, size=n_stores)                 # store size
    dow_f = np.array([0.80, 0.85, 0.90, 0.95, 1.10, 1.30, 1.20])   # Mon..Sun
    amp = rng.uniform(0.10, 0.40, size=n_items)                    # seasonality strength per item
    phase = rng.uniform(0, 2 * np.pi, size=n_items)                # peak season per item
    seasonal = 1 + amp[:, None] * np.sin(2 * np.pi * t[None, :] / 365.25 + phase[:, None])  # (I, T)
    trend = 1 + 0.06 * t / 365.25
    weekly = dow_f[dates.dayofweek.to_numpy()]

    mean = (store_f[:, None, None] * base[None, :, None] * seasonal[None, :, :]
            * trend[None, None, :] * weekly[None, None, :])        # (S, I, T)
    mean = mean * rng.gamma(shape=25, scale=1 / 25, size=mean.shape)   # demand shocks
    sales = rng.poisson(mean)

    return pd.DataFrame({
        "date": np.tile(dates, n_stores * n_items),
        "store": np.repeat(np.arange(1, n_stores + 1), n_items * T),
        "item": np.tile(np.repeat(np.arange(1, n_items + 1), T), n_stores),
        "sales": sales.ravel(),
    })


def load_sales(path=None, quick=False):
    """Returns (dataframe sorted by store,item,date, source_name)."""
    path = Path(path) if path else DATA_PATH
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        source = "kaggle"
    else:
        print(f"[data] {path} not found -> generating synthetic data (see README to use real data)")
        df = generate_synthetic()
        source = "synthetic"
    if quick:  # small subset for fast testing
        df = df[(df["store"] <= 3) & (df["item"] <= 10)]
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df, source


def unit_cost(items):
    """Deterministic assumed unit cost (5.00 - 59.99) per item id. Replace with real costs if you have them."""
    items = np.asarray(items)
    return 5 + ((items * 7919) % 5500) / 100
