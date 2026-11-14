"""Inventory optimisation: safety stock, order-up-to level, EOQ and the order recommendation.

Policy: periodic-review base-stock (R, S). Every REVIEW_PERIOD days, look at the inventory position
(on-hand + on-order) and order enough to bring it up to the order-up-to level

    S = forecast demand over (review period + lead time) + safety stock

because an order placed today must cover demand until the *next* order can arrive."""
import math

import numpy as np
from scipy.stats import norm

from .config import HOLDING_RATE, LEAD_TIME, ORDER_COST, REVIEW_PERIOD, SERVICE_LEVEL


def safety_stock(sigma_daily, protection_days, service_level=SERVICE_LEVEL):
    """z * sigma_daily * sqrt(P): buffer so that P(no stockout during the protection period) = service_level.
    sigma_daily is an *effective* daily forecast-error std (see pipeline: measured on P-day windows)."""
    z = norm.ppf(service_level)
    return z * np.asarray(sigma_daily) * np.sqrt(protection_days)


def eoq(annual_demand, unit_cost, order_cost=ORDER_COST, holding_rate=HOLDING_RATE):
    """Economic Order Quantity = sqrt(2*D*S / H), H = holding_rate * unit_cost per unit per year."""
    h = np.asarray(unit_cost) * holding_rate
    return np.sqrt(2 * np.asarray(annual_demand) * order_cost / h)


def build_plan(keys, forecast, sigma_daily, unit_costs, lead_time=LEAD_TIME,
               review_period=REVIEW_PERIOD, service_level=SERVICE_LEVEL):
    """keys: DataFrame[store,item]; forecast: (n_series, horizon) future demand forecast."""
    protection = lead_time + review_period
    expected = forecast[:, :protection].sum(axis=1)
    ss = safety_stock(sigma_daily, protection, service_level)
    daily = forecast.mean(axis=1)
    q = eoq(daily * 365, unit_costs)
    plan = keys.copy()
    plan["unit_cost"] = unit_costs
    plan["expected_demand"] = expected
    plan["sigma_daily"] = sigma_daily
    plan["safety_stock"] = ss
    plan["order_up_to"] = expected + ss
    plan["eoq"] = q
    plan["eoq_cycle_days"] = q / daily
    return plan.round(2)


def recommend_order(plan_row, on_hand, on_order=0):
    """Turn a plan row + current stock into a concrete action (run at every review)."""
    position = on_hand + on_order
    target = float(plan_row["order_up_to"])
    qty = max(0, math.ceil(target - position))
    return {
        "inventory_position": position,
        "expected_demand": round(float(plan_row["expected_demand"]), 1),
        "safety_stock": round(float(plan_row["safety_stock"]), 1),
        "order_up_to": round(target, 1),
        "order_qty": qty,
        "action": f"ORDER {qty} units now" if qty > 0 else "No order needed",
    }
