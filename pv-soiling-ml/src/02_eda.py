"""
Step 2 (Week 4): exploratory data analysis.
Saves a data dictionary, summary tables and plots to outputs/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import PROC, FIG, TAB, TARGET, FEATURES, NICE, region, style

style()
df = pd.read_csv(PROC / "site_month.csv")
sites = df.drop_duplicates("site_id")
all_sites = pd.read_csv(PROC.parent / "raw" / "sites.csv")
all_sites["censored"] = all_sites["iwsr"].astype(str).str.startswith(">")
df["region"] = df.apply(region, axis=1)
d = df[df.has_target].copy()

# ---------- data dictionary ----------
desc = {
    "site_id": "Site code given in this project (S001-S255)",
    "month": "Calendar month (1-12)",
    TARGET: "TARGET: soiling loss in % of energy per day (from NREL monthly soiling rate x -100)",
    "interval_count": "Number of soiling intervals NREL used to estimate that month",
    "tilt": "Panel tilt angle (degrees)", "mounting_tracking": "1 = single-axis tracker, 0 = fixed",
    "type": "PV system or soiling station", "state": "U.S. state",
}
for f in FEATURES:
    desc.setdefault(f, NICE.get(f, f))
rows = []
for col, text in desc.items():
    s = df[col]
    rows.append({"column": col, "description": text, "type": str(s.dtype),
                 "missing_%": round(100 * s.isna().mean(), 1),
                 "min": s.min() if s.dtype != object else "", "max": s.max() if s.dtype != object else ""})
pd.DataFrame(rows).to_csv(TAB / "data_dictionary.csv", index=False)

# ---------- summary numbers ----------
summary = {
    "sites in NREL file": len(all_sites),
    "sites with only '>0.99' annual IWSR (censored)": int(all_sites.censored.sum()),
    "sites with monthly soiling rates": df.site_id.nunique(),
    "site-month rows": len(df),
    "rows with 'Insufficient data'": int((~df.has_target).sum()),
    "rows used for modelling": len(d),
    "unique site locations": d[["lat", "lon"]].drop_duplicates().shape[0],
    "counties": d.county.nunique(),
    "target mean (%/day)": round(d[TARGET].mean(), 4),
    "target median (%/day)": round(d[TARGET].median(), 4),
    "target min (%/day)": round(d[TARGET].min(), 4),
    "target max (%/day)": round(d[TARGET].max(), 4),
}
pd.Series(summary).to_csv(TAB / "eda_summary.csv", header=["value"])
print(pd.Series(summary).to_string())

# sites per state: all vs modelled (shows geographic imbalance)
st = pd.DataFrame({"all_sites": all_sites.state.value_counts(),
                   "censored": all_sites[all_sites.censored].state.value_counts(),
                   "modelled": sites.state.value_counts()}).fillna(0).astype(int)
st.sort_values("all_sites", ascending=False).to_csv(TAB / "sites_by_state.csv")

# ---------- plots ----------
# 1. target distribution
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].hist(d[TARGET], bins=40, color="#2f6db5")
ax[0].set(title="Monthly soiling rate (all site-months)", xlabel="% energy lost per day", ylabel="count")
ax[1].hist(np.log(d[TARGET]), bins=40, color="#7a9cc6")
ax[1].set(title="Same, log scale", xlabel="log(% per day)")
fig.savefig(FIG / "01_target_distribution.png"); plt.close(fig)

# 2. seasonality
fig, ax = plt.subplots(figsize=(7, 3.2))
d.boxplot(column=TARGET, by="month", ax=ax, showfliers=False, grid=False)
ax.set(title="Soiling rate by month", xlabel="month", ylabel="% per day"); fig.suptitle("")
fig.savefig(FIG / "02_target_by_month.png"); plt.close(fig)

# 3. by region
order = d.groupby("region")[TARGET].median().sort_values().index
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.boxplot([d.loc[d.region == r, TARGET] for r in order], showfliers=False)
ax.set_xticks(range(1, len(order) + 1),
              [f"{r.replace(' California', '\nCalifornia')}\n(n={d[d.region==r].site_id.nunique()} sites)" for r in order],
              fontsize=8)
ax.set(title="Soiling rate by region", ylabel="% per day")
fig.savefig(FIG / "03_target_by_region.png"); plt.close(fig)

# 4. map of sites
sm = d.groupby(["site_id", "lat", "lon"])[TARGET].mean().reset_index()
fig, ax = plt.subplots(figsize=(7, 4))
c = all_sites[all_sites.censored]
ax.scatter(c.lon, c.lat, s=10, c="lightgrey", label="'>0.99' only (no monthly data)")
sc = ax.scatter(sm.lon, sm.lat, c=sm[TARGET], s=14, cmap="viridis")
plt.colorbar(sc, label="mean soiling (% per day)")
ax.set(title="NREL soiling sites", xlabel="longitude", ylabel="latitude"); ax.legend(loc="lower left")
fig.savefig(FIG / "04_site_map.png"); plt.close(fig)

# 5. censoring and state imbalance
fig, ax = plt.subplots(figsize=(7, 3.2))
top = st.sort_values("all_sites", ascending=False)
ax.bar(top.index, top.modelled, label="has monthly data", color="#2f6db5")
ax.bar(top.index, top.censored, bottom=top.modelled, label="'>0.99' only", color="lightgrey")
ax.set(title="Sites per state", ylabel="sites"); ax.legend()
fig.savefig(FIG / "05_sites_per_state.png"); plt.close(fig)

# 6. correlation of features with target
corr = d[FEATURES + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET).sort_values()
corr.to_csv(TAB / "feature_target_spearman.csv", header=["spearman_rho"])
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh([NICE[i] for i in corr.index], corr.values, color=np.where(corr.values > 0, "#c0504d", "#2f6db5"))
ax.set(title="Spearman correlation with soiling rate", xlabel="rho")
fig.savefig(FIG / "06_feature_correlation.png"); plt.close(fig)

# 7. feature-feature correlation (multicollinearity check)
cm = d[FEATURES].corr(method="spearman")
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES)), [NICE[f] for f in FEATURES], rotation=90, fontsize=7)
ax.set_yticks(range(len(FEATURES)), [NICE[f] for f in FEATURES], fontsize=7)
plt.colorbar(im); ax.set_title("Feature correlation (Spearman)"); ax.grid(False)
fig.savefig(FIG / "07_feature_heatmap.png"); plt.close(fig)

# 8. key scatter plots
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
for a, f in zip(ax, ["dslr_mean", "AOD_55", "PRECTOTCORR"]):
    a.scatter(d[f], d[TARGET], s=5, alpha=0.4)
    a.set(xlabel=NICE[f], ylabel="% per day")
fig.suptitle("Soiling rate vs key drivers")
fig.savefig(FIG / "08_key_scatter.png"); plt.close(fig)

# 9. Pakistan vs training range (are Pakistani conditions covered?)
pk = pd.read_csv(PROC / "pakistan_features.csv")
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
for a, f in zip(ax, ["AOD_55", "T2M", "dslr_mean"]):
    a.hist(d[f], bins=30, alpha=0.6, label="US training sites", density=True)
    a.hist(pk[f], bins=15, alpha=0.6, label="Pakistani cities", density=True)
    a.set(xlabel=NICE[f])
ax[0].legend(fontsize=7)
fig.suptitle("Do U.S. training conditions cover Pakistan?")
fig.savefig(FIG / "09_pakistan_vs_training.png"); plt.close(fig)

cover = []
for f in FEATURES:
    lo, hi = d[f].min(), d[f].max()
    out = ((pk[f] < lo) | (pk[f] > hi)).mean() * 100
    cover.append({"feature": f, "train_min": round(lo, 3), "train_max": round(hi, 3),
                  "pakistan_min": round(pk[f].min(), 3), "pakistan_max": round(pk[f].max(), 3),
                  "pakistan_rows_outside_%": round(out, 1)})
pd.DataFrame(cover).to_csv(TAB / "pakistan_coverage.csv", index=False)
print("EDA done")
