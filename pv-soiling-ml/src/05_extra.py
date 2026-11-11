"""
Step 6 (review additions): extra checks asked for during paper review.

  1. A spatial baseline: inverse-distance-weighted average of the k nearest
     training sites for the same calendar month (uses only location, no climate).
  2. An ablation: XGBoost without the three aerosol (AOD) features.
  3. A paired Wilcoxon signed-rank test on per-site mean absolute errors, so the
     XGBoost vs Random Forest difference is tested and not just eyeballed.
  4. The no-AOD model applied to Pakistan, to see if the Lahore estimate moves.

The outer folds are the same as in 03_models.py (same seed), so numbers line up.
"""
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import wilcoxon
from xgboost import XGBRegressor
from common import PROC, TAB, TARGET, FEATURES, SEED

df = pd.read_csv(PROC / "site_month.csv")
df = df[df.has_target].reset_index(drop=True)
y = df[TARGET].values
g_site, g_cell = df["site_id"].values, df["cell_id"].values
AOD = ["AOD_55", "aod_annual", "aod_x_dslr"]
NOAOD = [f for f in FEATURES if f not in AOD]
XALL = df[FEATURES + ["lat", "lon", "month"]]


class MonthlyMeanBaseline(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.means_ = pd.Series(y).groupby(X["month"].values).mean(); self.overall_ = float(np.mean(y)); return self

    def predict(self, X):
        return X["month"].map(self.means_).fillna(self.overall_).values


class KNNSpatial(BaseEstimator, RegressorMixin):
    """IDW mean of the k nearest training sites (haversine) that have data for the same month."""
    def __init__(self, k=5, power=2.0):
        self.k, self.power = k, power

    def fit(self, X, y):
        self.tr_ = X[["lat", "lon", "month"]].assign(y=y).reset_index(drop=True)
        self.mm_ = self.tr_.groupby("month").y.mean(); return self

    def predict(self, X):
        out = np.empty(len(X))
        for i, (la, lo, m) in enumerate(X[["lat", "lon", "month"]].values):
            t = self.tr_[self.tr_.month == m]
            if len(t) == 0:
                out[i] = self.tr_.y.mean(); continue
            p1, p2 = np.radians(la), np.radians(t.lat.values)
            dl = np.radians(t.lon.values - lo)
            a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
            d = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) + 1.0  # km, +1 avoids divide by zero
            idx = np.argsort(d)[: self.k]
            w = 1 / d[idx] ** self.power
            out[i] = np.sum(w * t.y.values[idx]) / w.sum()
        return out


class Cols(BaseEstimator, RegressorMixin):
    """Wraps an estimator so it only sees a chosen list of columns."""
    def __init__(self, est=None, cols=None):
        self.est, self.cols = est, cols

    def fit(self, X, y):
        self.est_ = clone(self.est).fit(X[self.cols], y); return self

    def predict(self, X):
        return self.est_.predict(X[self.cols])


RF = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
XGB = XGBRegressor(n_estimators=400, random_state=SEED, n_jobs=4, verbosity=0)
RF_GRID = {"est__max_depth": [4, 8, None], "est__min_samples_leaf": [3, 10, 25], "est__max_features": [0.3, 0.6]}
XGB_GRID = {"est__max_depth": [2, 3, 5], "est__learning_rate": [0.03, 0.1], "est__subsample": [0.8],
            "est__colsample_bytree": [0.6, 1.0], "est__min_child_weight": [5, 20]}
MODELS = {
    "Baseline (monthly mean)": (MonthlyMeanBaseline(), None),
    "Spatial kNN (IDW)": (KNNSpatial(), {"k": [3, 5, 10, 20]}),
    "Random Forest": (Cols(RF, FEATURES), RF_GRID),
    "XGBoost": (Cols(XGB, FEATURES), XGB_GRID),
    "XGBoost without AOD": (Cols(XGB, NOAOD), XGB_GRID),
}


def metrics(yt, yp):
    return {"MAE": mean_absolute_error(yt, yp), "RMSE": float(np.sqrt(mean_squared_error(yt, yp))), "R2": r2_score(yt, yp)}


def run(est, grid, groups):
    outer = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof, folds = np.zeros(len(y)), []
    for tr, te in outer.split(XALL, y, groups):
        if grid:
            gs = GridSearchCV(clone(est), grid, cv=GroupKFold(n_splits=3, shuffle=True, random_state=SEED),
                              scoring="neg_mean_absolute_error", n_jobs=1)
            model = gs.fit(XALL.iloc[tr], y[tr], groups=groups[tr]).best_estimator_
        else:
            model = clone(est).fit(XALL.iloc[tr], y[tr])
        oof[te] = model.predict(XALL.iloc[te]); folds.append(metrics(y[te], oof[te]))
    return oof, pd.DataFrame(folds)


rows, oof_tab = [], df[["site_id", "cell_id", "month", TARGET]].copy()
for sname, groups in {"site": g_site, "cell": g_cell}.items():
    for mname, (est, grid) in MODELS.items():
        oof, f = run(est, grid, groups)
        oof_tab[f"{sname}|{mname}"] = oof
        r = {"scheme": sname, "model": mname, **metrics(y, oof), **{f"{m}_fold_sd": f[m].std() for m in ["MAE", "RMSE", "R2"]}}
        rows.append(r); print(r, flush=True)
res = pd.DataFrame(rows); res.round(4).to_csv(TAB / "extra_model_comparison.csv", index=False)
oof_tab.to_csv(TAB / "oof_predictions_extra.csv", index=False)

# paired Wilcoxon on per-site MAE
tests = []
for sname in ["site", "cell"]:
    e = oof_tab.assign(**{m: (oof_tab[f"{sname}|{m}"] - oof_tab[TARGET]).abs() for m in MODELS}).groupby("site_id")[list(MODELS)].mean()
    for a, b in [("XGBoost", "Random Forest"), ("XGBoost", "Baseline (monthly mean)"),
                 ("XGBoost", "Spatial kNN (IDW)"), ("Random Forest", "Baseline (monthly mean)"),
                 ("XGBoost", "XGBoost without AOD")]:
        st = wilcoxon(e[a], e[b])
        tests.append({"scheme": sname, "A": a, "B": b, "n_sites": len(e), "median_MAE_A": e[a].median(),
                      "median_MAE_B": e[b].median(), "A_better_sites": int((e[a] < e[b]).sum()),
                      "statistic": st.statistic, "p_value": st.pvalue})
tests = pd.DataFrame(tests); tests.to_csv(TAB / "wilcoxon_tests.csv", index=False); print(tests.round(4))

# no-AOD final model applied to Pakistan
pk = pd.read_csv(PROC / "pakistan_features.csv")
gs = GridSearchCV(Cols(XGB, NOAOD), XGB_GRID, cv=GroupKFold(n_splits=5, shuffle=True, random_state=SEED),
                  scoring="neg_mean_absolute_error", n_jobs=1).fit(XALL, y, groups=g_site)
pk["pred_xgb_noaod"] = gs.best_estimator_.predict(pk.assign(lat=0, lon=0))
lo, hi = df[NOAOD].min(), df[NOAOD].max()
pk["n_out_noaod"] = ((pk[NOAOD] < lo) | (pk[NOAOD] > hi)).sum(1)
lah = pk[(pk.city == "Lahore") & pk.month.isin([10, 11, 12, 1])]
extra = {"xgb_noaod_params": gs.best_params_,
         "lahore_oct_jan_xgb_noaod": round(float(lah.pred_xgb_noaod.mean()), 4),
         "city_means_xgb_noaod": pk.groupby("city").pred_xgb_noaod.mean().round(3).to_dict(),
         "pk_city_months_out_of_range_noaod": round(float((pk.n_out_noaod > 0).mean()), 3)}
pk[["city", "month", "pred_xgb_noaod", "n_out_noaod"]].round(4).to_csv(TAB / "pakistan_predictions_noaod.csv", index=False)
json.dump(extra, open(TAB / "extra_results.json", "w"), indent=1, default=str); print(extra)
