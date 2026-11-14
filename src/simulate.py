"""Business simulation: replay the holdout period day by day under a periodic-review order-up-to policy and
count stockouts, inventory held and costs. Vectorised over all series."""
import numpy as np

from .config import HOLDING_RATE, MARGIN_RATE, ORDER_COST


def simulate(demand, order_up_to, init_inv, lead_time, review_period):
    """demand/order_up_to: (S, T). Lost-sales model, deterministic lead time, order every review_period days."""
    S, T = demand.shape
    on_hand = init_inv.astype(float).copy()
    arrivals = np.zeros((S, T + lead_time + 1))
    lost, inv_sum, n_orders = np.zeros(S), np.zeros(S), np.zeros(S)

    for t in range(T):
        on_hand += arrivals[:, t]                                  # receive deliveries
        served = np.minimum(on_hand, demand[:, t])                 # sell what we have
        lost += demand[:, t] - served
        on_hand -= served
        inv_sum += on_hand
        pipeline = arrivals[:, t + 1: t + lead_time + 1].sum(axis=1)
        position = on_hand + pipeline
        qty = np.maximum(np.ceil(order_up_to[:, t] - position), 0) if t % review_period == 0 else np.zeros(S)
        arrivals[:, t + lead_time] += qty                          # order lands after lead time
        n_orders += qty > 0

    return {"T": T, "demand_total": demand.sum(axis=1), "lost": lost,
            "avg_inv": inv_sum / T, "n_orders": n_orders}


def summarize(res, unit_cost):
    T = res["T"]
    holding = (res["avg_inv"] * unit_cost * HOLDING_RATE / 365 * T).sum()
    stockout = (res["lost"] * unit_cost * MARGIN_RATE).sum()
    ordering = res["n_orders"].sum() * ORDER_COST
    return {
        "fill_rate_%": float((1 - res["lost"].sum() / res["demand_total"].sum()) * 100),
        "units_lost": float(res["lost"].sum()),
        "avg_inventory_units": float(res["avg_inv"].sum()),
        "avg_inventory_value": float((res["avg_inv"] * unit_cost).sum()),
        "holding_cost": float(holding),
        "stockout_cost": float(stockout),
        "ordering_cost": float(ordering),
        "total_cost": float(holding + stockout + ordering),
    }
