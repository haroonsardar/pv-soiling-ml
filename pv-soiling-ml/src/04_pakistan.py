"""
Step 5 (Week 7): apply the trained models to Pakistani cities and check the
Lahore estimate against the field measurement of Ullah et al. (2020):
about 0.8 % per day for a 30-degree panel, October to January.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from common import PROC, FIG, TAB, MOD, TARGET, FEATURES, style

style()
train = pd.read_csv(PROC / "site_month.csv"); train = train[train.has_target]
pk = pd.read_csv(PROC / "pakistan_features.csv")
best = json.load(open(TAB / "best_model.json"))["best_model"]
files = {"Ridge regression": "ridge_regression", "Random Forest": "random_forest", "XGBoost": "xgboost"}

lo, hi = train[FEATURES].min(), train[FEATURES].max()
outside = ((pk[FEATURES] < lo) | (pk[FEATURES] > hi))
pk["n_features_outside_training_range"] = outside.sum(1)
pk["features_outside"] = outside.apply(lambda r: ", ".join(r.index[r]), axis=1)
for name, f in files.items():
    pk[f"pred_{f}"] = joblib.load(MOD / f"{f}.joblib").predict(pk[FEATURES])
best_col = f"pred_{files[best]}"
pk["pred_best"] = pk[best_col]
pk["est_monthly_loss_pct_if_not_cleaned"] = pk["pred_best"] * 30.4  # rough, ignores rain cleaning

cols = ["city", "month", "pred_best", "pred_ridge_regression", "pred_random_forest", "pred_xgboost",
        "n_features_outside_training_range", "features_outside"]
pk[cols].round(4).to_csv(TAB / "pakistan_predictions.csv", index=False)

# Lahore check (Oct-Jan)
lah = pk[(pk.city == "Lahore") & (pk.month.isin([10, 11, 12, 1]))]
check = {
    "measured_Ullah_2020_pct_per_day": 0.8,
    **{f"predicted_{n}": round(float(lah[f"pred_{f}"].mean()), 4) for n, f in files.items()},
    "training_max_pct_per_day": round(float(train[TARGET].max()), 4),
    "training_99th_percentile": round(float(train[TARGET].quantile(0.99)), 4),
    "lahore_months_with_out_of_range_inputs": int((lah.n_features_outside_training_range > 0).sum()),
}
json.dump(check, open(TAB / "lahore_check.json", "w"), indent=1)
print(json.dumps(check, indent=1))

# heatmap of predictions
piv = pk.pivot(index="city", columns="month", values="pred_best")
fig, ax = plt.subplots(figsize=(8, 3.6))
im = ax.imshow(piv.values, cmap="YlOrBr", aspect="auto")
ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_yticks(range(len(piv)), piv.index); ax.grid(False)
for i in range(piv.shape[0]):
    for j in range(12):
        flag = pk[(pk.city == piv.index[i]) & (pk.month == j + 1)].n_features_outside_training_range.iloc[0]
        ax.text(j, i, f"{piv.values[i, j]:.2f}" + ("*" if flag else ""), ha="center", va="center", fontsize=6)
plt.colorbar(im, label="% per day")
ax.set_title(f"Estimated soiling rate for Pakistani cities ({best})\n* = at least one input outside U.S. training range")
fig.savefig(FIG / "14_pakistan_heatmap.png"); plt.close(fig)

# Lahore vs measured
fig, ax = plt.subplots(figsize=(6, 3.2))
names = list(files) + ["Measured (Ullah et al. 2020)"]
vals = [check[f"predicted_{n}"] for n in files] + [0.8]
ax.bar(names, vals, color=["#7a9cc6"] * 3 + ["#c0504d"])
ax.axhline(check["training_max_pct_per_day"], ls="--", color="k", lw=0.8, label="highest value in U.S. training data")
ax.set(title="Lahore, October-January, 30-degree fixed panel", ylabel="% per day"); ax.legend(fontsize=7)
plt.xticks(fontsize=7)
fig.savefig(FIG / "15_lahore_check.png"); plt.close(fig)
