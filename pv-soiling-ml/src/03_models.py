"""
Steps 3-4 (Weeks 5-7): leakage-safe modelling and evaluation.

Test design
  * Main test: GroupKFold by SITE (5 folds). A site is never in both train and test.
  * Stricter test: GroupKFold by CLIMATE GRID CELL. Sites in the same NASA POWER
    cell have identical climate inputs, so holding out whole cells checks
    generalisation to new climates, which is what Pakistan needs.
  * For comparison only: plain random KFold, to show how a random split (used in
    some past studies) can make results look better than they are.
Hyperparameters are tuned with an inner GroupKFold on the training folds only
(nested cross-validation), and scaling lives inside the pipeline, so nothing from
the test fold leaks into training.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from common import PROC, FIG, TAB, MOD, TARGET, FEATURES, NICE, SEED, region, style

style()
df = pd.read_csv(PROC / "site_month.csv")
df = df[df.has_target].reset_index(drop=True)
df["region"] = df.apply(region, axis=1)
X, y = df[FEATURES], df[TARGET].values
g_site, g_cell = df["site_id"].values, df["cell_id"].values


class MonthlyMeanBaseline(BaseEstimator, RegressorMixin):
    """Predicts the average training soiling rate for each calendar month."""
    def fit(self, X, y):
        m = np.round(np.degrees(np.arctan2(X["month_sin"], X["month_cos"])) / 30) % 12
        self.means_ = pd.Series(y).groupby(m.values).mean()
        self.overall_ = float(np.mean(y))
        return self

    def predict(self, X):
        m = np.round(np.degrees(np.arctan2(X["month_sin"], X["month_cos"])) / 30) % 12
        return m.map(self.means_).fillna(self.overall_).values


MODELS = {
    "Baseline (monthly mean)": (MonthlyMeanBaseline(), None),
    "Ridge regression": (make_pipeline(StandardScaler(), Ridge()),
                         {"ridge__alpha": [0.1, 1, 10, 30, 100, 300]}),
    "Random Forest": (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
                      {"max_depth": [4, 8, None], "min_samples_leaf": [3, 10, 25],
                       "max_features": [0.3, 0.6]}),
    "XGBoost": (XGBRegressor(n_estimators=400, random_state=SEED, n_jobs=4, verbosity=0),
                {"max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1],
                 "subsample": [0.8], "colsample_bytree": [0.6, 1.0], "min_child_weight": [5, 20]}),
}


def metrics(yt, yp):
    return {"MAE": mean_absolute_error(yt, yp), "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "R2": r2_score(yt, yp)}


def cross_validate(name, est, grid, outer, groups):
    oof = np.zeros(len(y)); folds = []; best = []
    for k, (tr, te) in enumerate(outer.split(X, y, groups)):
        if grid:
            inner = GroupKFold(n_splits=3, shuffle=True, random_state=SEED)
            gs = GridSearchCV(clone(est), grid, cv=inner, scoring="neg_mean_absolute_error", n_jobs=1)
            gs.fit(X.iloc[tr], y[tr], groups=groups[tr] if groups is not None else g_site[tr])
            model = gs.best_estimator_; best.append(gs.best_params_)
        else:
            model = clone(est).fit(X.iloc[tr], y[tr])
        oof[te] = model.predict(X.iloc[te])
        folds.append(metrics(y[te], oof[te]))
    f = pd.DataFrame(folds)
    out = {"model": name, **{f"{m}": metrics(y, oof)[m] for m in ["MAE", "RMSE", "R2"]},
           **{f"{m}_fold_sd": f[m].std() for m in ["MAE", "RMSE", "R2"]}}
    return out, oof, best


schemes = {
    "Site-held-out (main)": (GroupKFold(n_splits=5, shuffle=True, random_state=SEED), g_site),
    "Grid-cell-held-out (strict)": (GroupKFold(n_splits=5, shuffle=True, random_state=SEED), g_cell),
    "Random split (for comparison only)": (KFold(n_splits=5, shuffle=True, random_state=SEED), None),
}
results, oofs, params = [], {}, {}
for sname, (outer, groups) in schemes.items():
    for mname, (est, grid) in MODELS.items():
        r, oof, best = cross_validate(mname, est, grid, outer, groups)
        r["scheme"] = sname; results.append(r)
        oofs[(sname, mname)] = oof; params[f"{sname} | {mname}"] = best
        print(f"{sname:36s} {mname:24s} MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} R2={r['R2']:.3f}")

res = pd.DataFrame(results)[["scheme", "model", "MAE", "MAE_fold_sd", "RMSE", "RMSE_fold_sd", "R2", "R2_fold_sd"]]
res.round(4).to_csv(TAB / "model_comparison.csv", index=False)
json.dump(params, open(TAB / "tuned_params_per_fold.json", "w"), indent=1, default=str)

# ---------- comparison chart ----------
main = res[res.scheme == "Site-held-out (main)"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for i, sch in enumerate(schemes):
    s = res[res.scheme == sch]
    ax[0].bar(np.arange(len(s)) + i * 0.27, s.MAE, 0.27, yerr=s.MAE_fold_sd, label=sch, capsize=2)
    ax[1].bar(np.arange(len(s)) + i * 0.27, s.R2, 0.27, yerr=s.R2_fold_sd, label=sch, capsize=2)
for a, t in zip(ax, ["MAE (% per day, lower is better)", "R² (higher is better)"]):
    a.set_xticks(np.arange(len(main)) + 0.27, [m.replace(" (monthly mean)", "").replace(" ", "\n") for m in main.model], fontsize=8)
    a.set_title(t)
ax[1].axhline(0, color="k", lw=0.8); ax[0].legend(fontsize=7)
fig.suptitle("Model comparison under three test designs (error bars = SD across folds)", y=1.03)
fig.savefig(FIG / "10_model_comparison.png"); plt.close(fig)

# ---------- best model on main scheme ----------
best_name = main.sort_values("MAE").iloc[0]["model"]
oof = oofs[("Site-held-out (main)", best_name)]
df["pred"], df["err"] = oof, oof - y
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].scatter(y, oof, s=6, alpha=0.4)
lim = [0, max(y.max(), oof.max()) * 1.05]
ax[0].plot(lim, lim, "k--", lw=0.8); ax[0].set(xlim=lim, ylim=lim, xlabel="measured (% per day)",
                                               ylabel="predicted", title=f"{best_name}: unseen sites")
reg = df.groupby("region").apply(lambda t: pd.Series(metrics(t[TARGET], t.pred) | {"sites": t.site_id.nunique(),
                                                                                    "bias": t.err.mean()}))
reg.round(4).to_csv(TAB / "error_by_region.csv")
ax[1].barh(reg.index, reg.MAE); ax[1].set(title="MAE by region", xlabel="% per day")
fig.savefig(FIG / "11_best_model_errors.png"); plt.close(fig)
df.groupby("month").apply(lambda t: pd.Series(metrics(t[TARGET], t.pred))).round(4).to_csv(TAB / "error_by_month.csv")

# worst sites, for error analysis
ws = df.groupby(["site_id", "county", "state"]).agg(mean_measured=(TARGET, "mean"), mean_pred=("pred", "mean"),
                                                    MAE=("err", lambda e: np.abs(e).mean())).sort_values("MAE", ascending=False)
ws.head(15).round(4).to_csv(TAB / "worst_sites.csv")

# ---------- final models on all data (tuned with site-grouped CV) ----------
final = {}
for mname, (est, grid) in MODELS.items():
    if grid:
        gs = GridSearchCV(clone(est), grid, cv=GroupKFold(n_splits=5, shuffle=True, random_state=SEED),
                          scoring="neg_mean_absolute_error", n_jobs=1).fit(X, y, groups=g_site)
        final[mname] = gs.best_estimator_; params[f"FINAL | {mname}"] = gs.best_params_
    else:
        final[mname] = clone(est).fit(X, y)
    joblib.dump(final[mname], MOD / f"{mname.split(' (')[0].replace(' ', '_').lower()}.joblib")
json.dump({k: v for k, v in params.items() if k.startswith("FINAL")}, open(TAB / "final_hyperparameters.json", "w"),
          indent=1, default=str)
json.dump({"best_model": best_name}, open(TAB / "best_model.json", "w"))

# ---------- importance ----------
pi = permutation_importance(final[best_name], X, y, n_repeats=20, random_state=SEED,
                            scoring="neg_mean_absolute_error")
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
imp.to_csv(TAB / "permutation_importance.csv", header=["MAE_increase"])
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh([NICE[i] for i in imp.index], imp.values, color="#2f6db5")
ax.set(title=f"Permutation importance ({best_name})", xlabel="increase in MAE when shuffled")
fig.savefig(FIG / "12_permutation_importance.png"); plt.close(fig)

try:
    import shap
    xgb = final["XGBoost"]
    sv = shap.TreeExplainer(xgb).shap_values(X)
    shap.summary_plot(sv, X.rename(columns=NICE), show=False, max_display=12)
    plt.title("SHAP values (XGBoost)"); plt.savefig(FIG / "13_shap_summary.png"); plt.close()
    pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False).to_csv(
        TAB / "shap_mean_abs.csv", header=["mean_abs_shap"])
except Exception as e:
    print("SHAP skipped:", e)
print("best model:", best_name)
