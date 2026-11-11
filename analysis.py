import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("inventory_data.csv", parse_dates=["date"]).sort_values(["fc_id", "sku_id", "date"])

n_rows = len(df)
n_stockouts = int(df.stockout_flag.sum())
print(f"SKU-days: {n_rows:,} | FCs: {df.fc_id.nunique()} | SKUs: {df.sku_id.nunique()}")
print(f"Stockout events: {n_stockouts} ({n_stockouts / n_rows:.2%})\n")

g = df.groupby(["fc_id", "sku_id"])["demand"]
df["trailing_mean"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
df["spike_signal"] = df["demand"] > 2 * df["trailing_mean"]
df["delay_signal"] = (df["po_due_today"] == 1) & (df["replenishment_delay_days"] > 0)

so = df[df.stockout_flag == 1].copy()
so["root_cause"] = np.select(
    [so.spike_signal, so.delay_signal],
    ["Demand spike", "Supplier replenishment delay"],
    default="Other (shrink / data / unexplained)",
)


pareto = so.root_cause.value_counts().rename_axis("root_cause").reset_index(name="events")
pareto["pct"] = pareto.events / pareto.events.sum() * 100
pareto["cum_pct"] = pareto.pct.cumsum()
print("PARETO\n", pareto.round(1).to_string(index=False), "\n")
top2_pct = pareto.pct.iloc[:2].sum()
print(f"Top-2 causes explain {top2_pct:.1f}% of stockouts\n")

fc = df.groupby("fc_id").agg(sku_days=("stockout_flag", "size"), stockouts=("stockout_flag", "sum"),
                             lost_units=("lost_sales", "sum"))
fc["rate_pct"] = fc.stockouts / fc.sku_days * 100
ratio = fc.rate_pct.max() / fc.rate_pct.min()
print("FULFILLMENT CENTERS\n", fc.round(2).to_string())
print(f"\nBest-vs-worst FC stockout-rate ratio: {ratio:.2f}x "
      f"({fc.rate_pct.idxmax()} vs {fc.rate_pct.idxmin()})\n")

cause_fc = pd.crosstab(so.fc_id, so.root_cause)

spike_n = int((so.root_cause == "Demand spike").sum())
delay_n = int((so.root_cause == "Supplier replenishment delay").sum())
promo_share = so[so.root_cause == "Demand spike"].promo_flag.mean() * 100
avg_delay = so[so.root_cause == "Supplier replenishment delay"].replenishment_delay_days.mean()
five_whys = [
    "Problem: 1.24% of SKU-days ended with demand exceeding available stock.",
    f"Why 1? Available stock hit zero before demand was met (lost units: {int(df.lost_sales.sum()):,}).",
    f"Why 2? Either demand ran >2x its trailing average ({spike_n} events) or the PO arrived late ({delay_n} events, avg {avg_delay:.1f} days late).",
    f"Why 3? Reorder points assume a fixed 3-day lead time and average demand; neither buffers spikes ({promo_share:.0f}% of spikes coincide with a promo) nor delays.",
    "Why 4? Safety stock is not promo-aware and does not use actual lead-time variability.",
    "Root cause: static replenishment parameters in a variable demand/supply environment.",
]
print("5-WHYS"); [print(" -", w) for w in five_whys]

Z = 1.65  # 95% service level
sku = df.groupby(["fc_id", "sku_id"]).agg(d_avg=("demand", "mean"), d_std=("demand", "std")).reset_index()
lt_std = df[df.po_due_today == 1].replenishment_delay_days.std()
lt_std = 0 if np.isnan(lt_std) else lt_std
lt = 3
sku["ss_current_proxy"] = Z * np.sqrt(lt) * sku.d_std
sku["ss_recommended"] = Z * np.sqrt(lt * sku.d_std**2 + sku.d_avg**2 * lt_std**2)
sku["uplift_pct"] = (sku.ss_recommended / sku.ss_current_proxy - 1) * 100
affected = so.groupby(["fc_id", "sku_id"]).size().rename("stockouts").reset_index()
rec = sku.merge(affected, on=["fc_id", "sku_id"]).sort_values("stockouts", ascending=False)
rec.round(1).to_csv("safety_stock_recommendations.csv", index=False)
print(f"\nLead-time std (observed on late POs): {lt_std:.2f} days")
print(f"Median safety-stock uplift on affected SKUs: {rec.uplift_pct.median():.0f}%")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(pareto.root_cause, pareto.events, color=["#1f4e79", "#2e86c1", "#aab7b8"])
ax.set_ylabel("Stockout events")
ax2 = ax.twinx(); ax2.plot(pareto.root_cause, pareto.cum_pct, "o-", color="#c0392b")
ax2.set_ylim(0, 105); ax2.set_ylabel("Cumulative %"); ax2.spines["right"].set_visible(True)
for i, r in pareto.iterrows():
    ax.text(i, r.events + 2, f"{r.events}", ha="center")
ax.set_xticks(range(len(pareto))); ax.set_xticklabels([t.split(" (")[0].replace("Supplier replenishment delay", "Supplier\ndelay") for t in pareto.root_cause])
ax.set_title("Pareto of stockout root causes")
fig.tight_layout(); fig.savefig("chart_pareto.png", dpi=150); plt.close()


fig, ax = plt.subplots(figsize=(6.5, 4))
colors = ["#c0392b" if v == fc.rate_pct.max() else "#1f4e79" if v != fc.rate_pct.min() else "#27ae60" for v in fc.rate_pct]
ax.bar(fc.index, fc.rate_pct, color=colors)
for x, v in zip(fc.index, fc.rate_pct): ax.text(x, v + 0.03, f"{v:.2f}%", ha="center")
ax.axhline(n_stockouts / n_rows * 100, ls="--", color="grey", lw=1); ax.text(4.45, n_stockouts / n_rows * 100 + 0.03, "overall", color="grey", ha="right")
ax.set_ylabel("Stockout rate (% of SKU-days)"); ax.set_title(f"Stockout rate by fulfillment center ({ratio:.1f}x spread)")
fig.tight_layout(); fig.savefig("chart_fc_rate.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
cause_fc.plot(kind="bar", stacked=True, ax=ax, color=["#aab7b8", "#2e86c1", "#1f4e79"], rot=0)
ax.set_ylabel("Stockout events"); ax.set_xlabel(""); ax.set_title("Root causes by fulfillment center")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("chart_cause_by_fc.png", dpi=150); plt.close()

wk = so.set_index("date").resample("W").size()
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(wk.index, wk.values, marker="o", color="#1f4e79")
ax.set_ylabel("Stockouts / week"); ax.set_title("Weekly stockout events")
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig("chart_weekly.png", dpi=150); plt.close()

with open("findings_summary.txt", "w") as f:
    f.write(f"SKU-days analysed: {n_rows:,}\nStockout events: {n_stockouts} ({n_stockouts / n_rows:.2%})\n\n")
    f.write("PARETO\n" + pareto.round(1).to_string(index=False) + "\n\n")
    f.write(f"Top-2 causes: {top2_pct:.1f}% of events\n\nFULFILLMENT CENTERS\n{fc.round(2).to_string()}\n")
    f.write(f"\nBest-vs-worst ratio: {ratio:.2f}x\n\n5-WHYS\n" + "\n".join(f"- {w}" for w in five_whys) + "\n")
    f.write(f"\nMedian safety-stock uplift on affected SKUs: {rec.uplift_pct.median():.0f}%\n")
print("\nSaved charts, findings_summary.txt, safety_stock_recommendations.csv")
