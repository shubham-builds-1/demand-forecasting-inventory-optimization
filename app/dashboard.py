"""Streamlit dashboard.  Run:  streamlit run app/dashboard.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.config import ARTIFACTS, LEAD_TIME, REVIEW_PERIOD, SERVICE_LEVEL
from src.inventory import recommend_order

st.set_page_config(page_title="Demand Forecasting & Inventory Optimization", layout="wide")


@st.cache_data
def load():
    return dict(
        hold=pd.read_csv(ARTIFACTS / "holdout_forecasts.csv", parse_dates=["date"]),
        recent=pd.read_csv(ARTIFACTS / "recent_sales.csv", parse_dates=["date"]),
        future=pd.read_csv(ARTIFACTS / "future_forecast.csv", parse_dates=["date"]),
        plan=pd.read_csv(ARTIFACTS / "inventory_plan.csv"),
        metrics=json.loads((ARTIFACTS / "metrics.json").read_text()),
        sim=json.loads((ARTIFACTS / "simulation.json").read_text()),
    )


if not (ARTIFACTS / "inventory_plan.csv").exists():
    st.error("No results found. Run `python -m src.pipeline` first, then refresh this page.")
    st.stop()

d = load()
st.title("📦 Demand Forecasting & Inventory Optimization")
if d["metrics"]["data_source"] == "synthetic":
    st.info("Showing results on **synthetic** data. Add the Kaggle `train.csv` to `data/` and re-run the pipeline for real-data results.")

plan = d["plan"]
store = st.sidebar.selectbox("Store", sorted(plan.store.unique()))
item = st.sidebar.selectbox("Item", sorted(plan[plan.store == store].item.unique()))
row = plan[(plan.store == store) & (plan.item == item)].iloc[0]
sel = lambda df: df[(df.store == store) & (df.item == item)].set_index("date")

tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🛒 Order recommendation", "💰 Business impact"])

with tab1:
    st.subheader(f"Store {store} · Item {item}")
    chart = sel(d["recent"])["sales"].rename("Actual").to_frame()
    h = sel(d["hold"])
    chart = chart.join(h[["naive", "weekday_avg", "lightgbm"]].rename(columns={
        "naive": "Seasonal naive", "weekday_avg": "Weekday average", "lightgbm": "LightGBM"}), how="left")
    st.caption("Last 90 days are a holdout: models never saw them when forecasting.")
    st.line_chart(chart)
    st.subheader("Forecast for the next 90 days")
    st.line_chart(sel(d["future"])["forecast"].rename("Forecast (units/day)"))
    st.subheader("Accuracy on holdout (all series)")
    st.dataframe(pd.DataFrame(d["metrics"]["models"]).T.round(2))

with tab2:
    st.subheader("How much should I order today?")
    c1, c2 = st.columns(2)
    on_hand = c1.number_input("Units on hand", min_value=0, value=int(row.expected_demand // 3), step=10)
    on_order = c2.number_input("Units already on order", min_value=0, value=0, step=10)
    rec = recommend_order(row, on_hand, on_order)
    (st.success if rec["order_qty"] > 0 else st.info)(f"**{rec['action']}**")
    m = st.columns(4)
    m[0].metric("Inventory position", f"{rec['inventory_position']:.0f}")
    m[1].metric("Expected demand (next orders cover)", f"{rec['expected_demand']:.0f}")
    m[2].metric("Safety stock", f"{rec['safety_stock']:.0f}")
    m[3].metric("Order-up-to level", f"{rec['order_up_to']:.0f}")
    st.caption(f"Policy: review every {REVIEW_PERIOD} days, lead time {LEAD_TIME} days, "
               f"target service level {SERVICE_LEVEL:.0%}. Reference EOQ = {row.eoq:.0f} units "
               f"(≈ every {row.eoq_cycle_days:.0f} days of demand).")

with tab3:
    sim = d["sim"]
    st.write("Replay of the last 90 days: **moving-average planning** (what many small businesses do) vs "
             "**forecast-driven planning** (this project). Same ordering rules and service-level targets; "
             "only the demand estimate differs.")
    rows = []
    for sl, r in sim["by_service_level"].items():
        b, m = r["baseline_moving_average"], r["forecast_driven_policy"]
        rows.append({"Target service level": f"{float(sl):.0%}",
                     "Fill rate – baseline %": b["fill_rate_%"], "Fill rate – forecast %": m["fill_rate_%"],
                     "Avg inventory – baseline": b["avg_inventory_units"],
                     "Avg inventory – forecast": m["avg_inventory_units"],
                     "Inventory change %": r["pct_change"]["avg_inventory_units"],
                     "Total cost change %": r["pct_change"]["total_cost"]})
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True)
    st.caption("Costs use assumed unit costs, 20% yearly holding rate, $50/order and 40% margin on lost sales "
               "(see src/config.py). Fill rate is measured in units, so it sits above the cycle-service-level target.")
