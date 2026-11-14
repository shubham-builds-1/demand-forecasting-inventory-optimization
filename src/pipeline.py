"""End-to-end pipeline:  python -m src.pipeline [--quick] [--data path/to/train.csv]

1. Load data  2. Build features  3. Validation fit -> forecast-error sigma per series
4. Holdout fit -> compare models on last 90 days  5. Business simulation (baseline vs ML policy)
6. Final fit on all data -> next-90-day forecast + inventory plan  7. Save artifacts"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from .config import ARTIFACTS, HORIZON, LEAD_TIME, REVIEW_PERIOD, SERVICE_LEVEL
from .data import load_sales, unit_cost
from .features import FEATURES, MIN_LAG, add_features, extend_frame
from .inventory import build_plan, safety_stock
from .models import evaluate, fit_lgbm, seasonal_naive, weekday_average
from .simulate import simulate, summarize


def _window_sum(f, start_offset, length):
    """sum of f[:, t+start_offset : t+start_offset+length] for every t (pads the end with the recent mean)."""
    pad = np.repeat(f[:, -7:].mean(axis=1, keepdims=True), start_offset + length, axis=1)
    fp = np.hstack([f, pad])
    c = np.hstack([np.zeros((f.shape[0], 1)), fp.cumsum(axis=1)])
    T = f.shape[1]
    t = np.arange(T)
    return c[:, t + start_offset + length] - c[:, t + start_offset]


def run(data_path=None, quick=False):
    t0 = time.time()
    ARTIFACTS.mkdir(exist_ok=True)
    H, L, z_sl = HORIZON, LEAD_TIME, SERVICE_LEVEL
    R = REVIEW_PERIOD
    P = L + R   # protection period: an order placed now must cover demand until the next one can arrive

    # 1. data ----------------------------------------------------------------------------
    df, source = load_sales(data_path, quick)
    keys = df[["store", "item"]].drop_duplicates().reset_index(drop=True)
    S = len(keys)
    dates = np.sort(df["date"].unique())
    T = len(dates)
    assert len(df) == S * T, "Every store-item must have a sales value for every date"
    sales = df["sales"].to_numpy(float).reshape(S, T)
    costs = unit_cost(keys["item"].to_numpy())
    test_start, val_start = T - H, T - 2 * H
    print(f"[data] source={source} | series={S} | days={T} | rows={len(df):,}")

    # 2. features ------------------------------------------------------------------------
    feat = add_features(df)
    X, y = feat[FEATURES], feat["sales"]
    t_idx = np.tile(np.arange(T), S)
    usable = feat[f"lag_{MIN_LAG}"].notna().to_numpy()

    # 3. validation fit -> sigma (std of daily forecast error) per series ----------------
    m_val = fit_lgbm(X[(t_idx < val_start) & usable], y[(t_idx < val_start) & usable])
    val_mask = (t_idx >= val_start) & (t_idx < test_start)
    pred_val = m_val.predict(X[val_mask]).reshape(S, H)
    resid = sales[:, val_start:test_start] - pred_val
    # Forecast errors on neighbouring days are correlated, so sigma*sqrt(P) would understate risk.
    # Measure the std of the error summed over every P-day window instead, then express it as an
    # "effective daily sigma" (sigma_P / sqrt(P)) so that safety_stock() stays z * sigma * sqrt(P).
    cs = np.hstack([np.zeros((S, 1)), resid.cumsum(axis=1)])
    sigma_P = (cs[:, P:] - cs[:, :-P]).std(axis=1, ddof=1)
    sigma = sigma_P / np.sqrt(P)
    print(f"[val ] fitted | mean effective daily error sigma = {sigma.mean():.2f}")

    # 4. holdout fit (train+val) -> compare models on the last H days ---------------------
    train_mask = (t_idx < test_start) & usable
    m_hold = fit_lgbm(X[train_mask], y[train_mask])
    pred_lgbm = np.clip(m_hold.predict(X[t_idx >= test_start]).reshape(S, H), 0, None)
    hist = sales[:, :test_start]
    actual = sales[:, test_start:]
    preds = {
        "Seasonal naive (last week)": seasonal_naive(hist, H),
        "Weekday average (8 wks)": weekday_average(hist, H),
        "LightGBM": pred_lgbm,
    }
    model_metrics = {name: evaluate(actual, p) for name, p in preds.items()}
    base_wape = model_metrics["Seasonal naive (last week)"]["WAPE_%"]
    for name, m in model_metrics.items():
        m["WAPE_improvement_vs_naive_%"] = (base_wape - m["WAPE_%"]) / base_wape * 100
    print("\n[holdout] forecast accuracy on last", H, "days")
    print(pd.DataFrame(model_metrics).T.round(2).to_string())

    # 5. business simulation ---------------------------------------------------------------
    # Same weekly-review order-up-to policy for both; only the demand estimate / error estimate differs.
    # Run at several service levels so we compare the full inventory-vs-stockout trade-off
    # (one service level alone can hide that a policy simply holds more or less stock).
    sim = {"assumptions": {"lead_time_days": L, "review_period_days": R,
                           "target_service_level": z_sl, "holdout_days": H},
           "by_service_level": {}}
    for sl in [0.90, 0.95, 0.98, 0.99]:
        z = safety_stock(1.0, P, sl)  # = z * sqrt(P)
        up_ml = _window_sum(pred_lgbm, 1, P) + z * sigma[:, None]
        up_b = np.zeros((S, H))       # business-as-usual: trailing 28-day mean and std of demand
        for t in range(H):
            w = sales[:, test_start + t - 27: test_start + t + 1]
            up_b[:, t] = w.mean(axis=1) * P + z * w.std(axis=1, ddof=1)
        res_b = summarize(simulate(actual, up_b, up_b[:, 0], L, R), costs)
        res_ml = summarize(simulate(actual, up_ml, up_b[:, 0], L, R), costs)
        pct = {k: (res_ml[k] - res_b[k]) / res_b[k] * 100 for k in res_b if res_b[k]}
        sim["by_service_level"][str(sl)] = {"baseline_moving_average": res_b,
                                            "forecast_driven_policy": res_ml, "pct_change": pct}
    sim["headline"] = sim["by_service_level"][str(z_sl)]
    print("\n[simulation] inventory-policy comparison on the last", H, "days (baseline vs forecast-driven)")
    rows = []
    for sl, r in sim["by_service_level"].items():
        b, m = r["baseline_moving_average"], r["forecast_driven_policy"]
        rows.append({"target_SL": sl, "fill%_base": b["fill_rate_%"], "fill%_ml": m["fill_rate_%"],
                     "inv_base": b["avg_inventory_units"], "inv_ml": m["avg_inventory_units"],
                     "cost_base": b["total_cost"], "cost_ml": m["total_cost"]})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # 6. final fit on all data -> next-H-day forecast + inventory plan --------------------
    ext = add_features(extend_frame(df, H))
    tt = np.tile(np.arange(T + H), S)
    all_usable = ext[f"lag_{MIN_LAG}"].notna().to_numpy()
    fit_mask = (tt < T) & all_usable
    m_final = fit_lgbm(ext.loc[fit_mask, FEATURES], ext.loc[fit_mask, "sales"])
    fut_pred = np.clip(m_final.predict(ext.loc[tt >= T, FEATURES]).reshape(S, H), 0, None)
    plan = build_plan(keys, fut_pred, sigma, costs)

    # 7. save artifacts -------------------------------------------------------------------
    test_dates = pd.to_datetime(dates[test_start:])
    hold = pd.DataFrame({
        "date": np.tile(test_dates, S),
        "store": np.repeat(keys["store"].to_numpy(), H),
        "item": np.repeat(keys["item"].to_numpy(), H),
        "actual": actual.ravel(),
        "naive": preds["Seasonal naive (last week)"].ravel().round(2),
        "weekday_avg": preds["Weekday average (8 wks)"].ravel().round(2),
        "lightgbm": pred_lgbm.ravel().round(2),
    })
    hold.to_csv(ARTIFACTS / "holdout_forecasts.csv", index=False)

    recent_days = 3 * H
    recent = df[df["date"] >= pd.Timestamp(dates[T - recent_days])]
    recent.to_csv(ARTIFACTS / "recent_sales.csv", index=False)

    future_dates = pd.date_range(pd.Timestamp(dates[-1]) + pd.Timedelta(days=1), periods=H)
    pd.DataFrame({
        "date": np.tile(future_dates, S),
        "store": np.repeat(keys["store"].to_numpy(), H),
        "item": np.repeat(keys["item"].to_numpy(), H),
        "forecast": fut_pred.ravel().round(2),
    }).to_csv(ARTIFACTS / "future_forecast.csv", index=False)

    plan.to_csv(ARTIFACTS / "inventory_plan.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "importance": m_final.feature_importances_}) \
        .sort_values("importance", ascending=False).to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    meta = {"data_source": source, "n_series": S, "n_days": T, "n_rows": int(len(df)),
            "holdout_days": H, "models": model_metrics}
    (ARTIFACTS / "metrics.json").write_text(json.dumps(meta, indent=2))
    (ARTIFACTS / "simulation.json").write_text(json.dumps(sim, indent=2))
    print(f"\n[done] artifacts saved to {ARTIFACTS} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small subset (3 stores x 10 items) for a fast run")
    ap.add_argument("--data", default=None, help="path to Kaggle train.csv (default: data/train.csv)")
    args = ap.parse_args()
    run(args.data, args.quick)
