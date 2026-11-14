"""REST API.  Run:  uvicorn api.main:app --reload   then open http://127.0.0.1:8000/docs
Requires `python -m src.pipeline` to have been run first (it creates the artifacts)."""
import json
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import ARTIFACTS
from src.inventory import recommend_order

app = FastAPI(title="Demand Forecasting & Inventory Optimization API", version="1.0")


@lru_cache(maxsize=1)
def _data():
    try:
        return (pd.read_csv(ARTIFACTS / "future_forecast.csv", parse_dates=["date"]),
                pd.read_csv(ARTIFACTS / "inventory_plan.csv"))
    except FileNotFoundError:
        raise HTTPException(503, "Artifacts missing. Run: python -m src.pipeline")


def _plan_row(plan, store, item):
    row = plan[(plan.store == store) & (plan.item == item)]
    if row.empty:
        raise HTTPException(404, f"Unknown store/item: {store}/{item}")
    return row.iloc[0]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast/{store}/{item}")
def forecast(store: int, item: int):
    """Next-90-day daily demand forecast for one store-item."""
    fut, plan = _data()
    _plan_row(plan, store, item)
    f = fut[(fut.store == store) & (fut.item == item)]
    return {"store": store, "item": item,
            "forecast": [{"date": d.strftime("%Y-%m-%d"), "units": v} for d, v in zip(f.date, f.forecast)]}


@app.get("/recommend")
def recommend(store: int, item: int, on_hand: float, on_order: float = 0):
    """Given current stock, how many units should we order right now?"""
    _, plan = _data()
    row = _plan_row(plan, store, item)
    return {"store": store, "item": item, **recommend_order(row, on_hand, on_order)}


@app.get("/metrics")
def metrics():
    try:
        return json.loads((ARTIFACTS / "metrics.json").read_text())
    except FileNotFoundError:
        raise HTTPException(503, "Artifacts missing. Run: python -m src.pipeline")
