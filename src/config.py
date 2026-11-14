"""Central configuration. Change business assumptions here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"   # Kaggle "Store Item Demand Forecasting" file
ARTIFACTS = ROOT / "artifacts"            # everything the dashboard / API reads

# --- Forecasting ---
HORIZON = 90          # days to forecast ahead (also the size of the holdout window)

# --- Inventory / business assumptions (the dataset has no costs, so these are assumed) ---
LEAD_TIME = 7         # days between placing an order and receiving it
REVIEW_PERIOD = 7     # we review stock and place orders once a week
SERVICE_LEVEL = 0.95  # target probability of NOT stocking out during lead time
ORDER_COST = 50.0     # fixed cost per purchase order
HOLDING_RATE = 0.20   # yearly holding cost as a fraction of unit cost
MARGIN_RATE = 0.40    # profit lost per unsold unit = MARGIN_RATE * unit cost

SEED = 42
