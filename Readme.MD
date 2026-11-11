
**Stack:** Python · Pandas · NumPy · Matplotlib

Root cause analysis of stockouts across **5 fulfillment centers, 100 SKUs and 36 days (18,000 SKU-days)**, using Pareto analysis and a structured 5-Whys.

> **Data note:** The dataset is **synthetic** (`generate_data.py`, seed 42), built to mimic daily fulfillment-center inventory with demand spikes and late supplier deliveries. Root causes are **not labelled** in the data; `analysis.py` re-derives them from observable signals only.

## Key results
| Metric | Result |
|---|---|
| SKU-days analysed | 18,000 |
| Stockout events | **223 (1.24%)** |
| Demand spikes | 118 events (52.9%) |
| Supplier replenishment delays | 95 events (42.6%) |
| Top-2 causes combined | **95.5%** of stockouts |
| Worst vs best FC (FC1 vs FC5) | **1.8x** stockout-rate variance (1.50% vs 0.83%) |

## Method
1. **Data pipeline (Pandas):** load, sort and compute trailing 7-day demand per FC-SKU.
2. **Root-cause classification from raw signals:**
   - *Demand spike:* demand > 2x trailing 7-day mean
   - *Supplier delay:* a PO was due and arrived late
   - *Other:* neither signal
3. **Pareto analysis** of causes.
4. **5-Whys RCA** backed by data (promo overlap, average delay length).
5. **FC benchmarking** of stockout rate and lost units.
6. **Safety-stock recommendation** using the combined-variability formula  
   `SS = Z × √(LT·σd² + d̄²·σLT²)` at a 95% service level.

## Recommendations
- **Promo-aware safety stock:** raise buffers for SKUs/periods with promotions (62% of spikes coincide with a promo).
- **Dynamic lead-time buffering:** use observed supplier lead-time variability (σ ≈ 0.83 days, avg 3.0 days late) instead of a fixed 3-day assumption. Median safety-stock uplift on affected SKUs: ~16%.
- **Target FC1 and FC2 first:** they account for the highest stockout rates and lost units.

## Files
| File | Purpose |
|---|---|
| `generate_data.py` | Builds `inventory_data.csv` |
| `analysis.py` | Full RCA pipeline, charts, summary |
| `inventory_data.csv` | 18,000-row dataset |
| `safety_stock_recommendations.csv` | Per FC-SKU safety stock for affected SKUs |
| `findings_summary.txt` | Text summary of results |
| `chart_*.png` | Pareto, FC stockout rate, cause by FC, weekly trend |

## Run
```bash
pip install pandas numpy matplotlib
python generate_data.py
python analysis.py
```
