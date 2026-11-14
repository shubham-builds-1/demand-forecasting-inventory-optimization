"""Forecasting models: two classical baselines + a global LightGBM model, and evaluation metrics.

Arrays are shaped (n_series, n_days)."""
import numpy as np
import lightgbm as lgb

from .config import SEED


# ---------- Baselines (use only history up to the forecast origin) ----------
def seasonal_naive(hist, horizon):
    """Repeat the last observed week."""
    last_week = hist[:, -7:]
    return last_week[:, np.arange(horizon) % 7]


def weekday_average(hist, horizon, weeks=8):
    """Average of the same weekday over the last `weeks` weeks."""
    window = hist[:, -7 * weeks:].reshape(hist.shape[0], weeks, 7).mean(axis=1)
    return window[:, np.arange(horizon) % 7]


# ---------- Global LightGBM model (one model for all store-item series) ----------
def fit_lgbm(X, y):
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X, y)
    return model


# ---------- Metrics ----------
def evaluate(actual, pred):
    err = actual - pred
    pos = actual > 0
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE_%": float(np.mean(np.abs(err[pos]) / actual[pos]) * 100),
        "WAPE_%": float(np.abs(err).sum() / actual.sum() * 100),
    }
