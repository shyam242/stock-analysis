import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

N_FC, N_SKU, N_DAYS = 5, 100, 36
START = pd.Timestamp("2025-01-01")
PLANNED_LT = 3

FC_EVENTS = {"FC1": 54, "FC2": 50, "FC3": 46, "FC4": 43, "FC5": 30}
FC_CAUSES = {
    "FC1": (26, 26, 2),
    "FC2": (26, 22, 2),
    "FC3": (26, 18, 2),
    "FC4": (24, 17, 2),
    "FC5": (16, 12, 2),
}
assert sum(FC_EVENTS.values()) == 223
for fc, c in FC_CAUSES.items():
    assert sum(c) == FC_EVENTS[fc]

base_demand = rng.integers(20, 61, size=(N_FC, N_SKU))

plan = {}
for f, fc in enumerate(FC_EVENTS):
    spike, delay, other = FC_CAUSES[fc]
    causes = ["spike"] * spike + ["delay"] * delay + ["other"] * other
    skus = rng.choice(N_SKU, size=len(causes), replace=False)
    days = rng.integers(8, N_DAYS - 4, size=len(causes))
    for s, d, c in zip(skus, days, causes):
        plan[(f, int(s), int(d))] = c

rows = []
for f in range(N_FC):
    fc = f"FC{f+1}"
    for s in range(N_SKU):
        b = base_demand[f, s]
        stock = 10 * b
        pending = {} 
        for d in range(N_DAYS):
            date = START + pd.Timedelta(days=d)
            cause = plan.get((f, s, d))
            promo = int(rng.random() < 0.05)
            demand = int(rng.poisson(b * (1.25 if promo else 1.0)))
            demand = min(demand, int(1.6 * b)) 
            po_due = 0
            delay_days = 0
            receipts = pending.pop(d, 0)
            if cause: 
                pending[d + 1] = pending.get(d + 1, 0) + receipts
                receipts = 0
            if cause and receipts and d + 1 < N_DAYS:
                pending[d + 1] = pending.get(d + 1, 0) + receipts
                receipts = 0

            if cause == "delay":
                stock = int(0.3 * b)
                po_due, delay_days = 1, int(rng.integers(2, 5))
                receipts = 0
                if d + delay_days < N_DAYS:
                    pending[d + delay_days] = 10 * b
            elif cause == "spike":
                promo = int(rng.random() < 0.6)
                demand = int(rng.uniform(3.0, 4.5) * b) + int(stock)
            elif cause == "other":
                stock = int(0.2 * b)
                demand = max(demand, int(1.2 * b))

            opening = stock
            available = opening + receipts
            sold = min(demand, available)
            closing = available - sold
            stockout = int(demand > available)
            lost = demand - sold

            if cause in ("spike", "other", "delay") and d + 1 < N_DAYS:
                pending[d + 1] = pending.get(d + 1, 0) + 10 * b
            if closing + sum(pending.values()) < 6 * b and not pending:
                pending[d + PLANNED_LT] = 10 * b - closing

            rows.append(dict(
                date=date, fc_id=fc, sku_id=f"SKU{s+1:03d}", opening_stock=opening,
                receipts=receipts, demand=demand, units_sold=sold, closing_stock=closing,
                promo_flag=promo, po_due_today=po_due, planned_lead_time=PLANNED_LT,
                replenishment_delay_days=delay_days, lost_sales=lost, stockout_flag=stockout,
            ))
            stock = closing

df = pd.DataFrame(rows)
df.to_csv("inventory_data.csv", index=False)
print(f"rows={len(df):,}  stockouts={df.stockout_flag.sum()}  rate={df.stockout_flag.mean():.2%}")
