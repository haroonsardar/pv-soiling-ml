"""
Figures for the IEEE paper: no in-figure titles (the caption does that job),
Times-like fonts at 8 pt, sized to one column (3.5 in) or two columns (7.16 in).
Run after 03_models.py, 04_pakistan.py and 05_extra.py.
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import joblib
from common import ROOT, PROC, TAB, MOD, TARGET, FEATURES, NICE, region

OUT = ROOT / "outputs" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL, PAGE = 3.5, 7.16
GREY = "#bdbdbd"

d = pd.read_csv(PROC / "site_month.csv"); d = d[d.has_target].reset_index(drop=True)
d["region"] = d.apply(region, axis=1)
sites = pd.read_csv(ROOT / "data" / "raw" / "sites.csv")
pk = pd.read_csv(PROC / "pakistan_features.csv")

# Fig. 1: site map
sm = d.groupby(["site_id", "lat", "lon"])[TARGET].mean().reset_index()
cen = sites[~sites.site_id.isin(sm.site_id)]
fig, ax = plt.subplots(figsize=(COL, 2.1))
ax.scatter(cen.lon, cen.lat, s=6, c=GREY, lw=0, label="annual IWSR $>$0.99 only")
sc = ax.scatter(sm.lon, sm.lat, c=sm[TARGET], s=8, cmap="viridis", lw=0, label="monthly rates")
cb = plt.colorbar(sc, ax=ax, pad=0.01); cb.set_label("mean soiling rate (%/day)"); cb.ax.tick_params(labelsize=7)
ax.set(xlabel="longitude (°)", ylabel="latitude (°)"); ax.legend(loc="lower left", frameon=False, markerscale=1.5)
fig.savefig(OUT / "fig_map.png"); plt.close(fig)

# Fig. 2: Pakistan vs U.S. coverage
fig, ax = plt.subplots(1, 3, figsize=(COL, 1.45))
for a, f, lab in zip(ax, ["AOD_55", "T2M", "dslr_mean"], ["AOD (550 nm)", "temperature (°C)", "days since rain"]):
    bins = np.linspace(min(d[f].min(), pk[f].min()), max(d[f].max(), pk[f].max()), 25)
    a.hist(d[f], bins=bins, density=True, color="#4c72b0", alpha=0.6, label="U.S. sites")
    a.hist(pk[f], bins=bins, density=True, color="#dd8452", alpha=0.6, label="Pakistan")
    a.set_xlabel(lab); a.set_yticks([])
    a.spines["left"].set_visible(False)
ax[0].set_ylabel("density")
fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(w_pad=0.4); fig.savefig(OUT / "fig_coverage.png"); plt.close(fig)

# Fig. 4: model comparison (site and cell held out, plus random split)
mc = pd.read_csv(TAB / "model_comparison.csv"); ex = pd.read_csv(TAB / "extra_model_comparison.csv")
smap = {"site": "Site-held-out (main)", "cell": "Grid-cell-held-out (strict)"}
ex["scheme"] = ex.scheme.map(smap)
ridge = mc[mc.model == "Ridge regression"]
rows = pd.concat([ex, ridge[ridge.scheme.isin(smap.values())]], ignore_index=True)
rnd = mc[mc.scheme.str.startswith("Random")]
order = ["Baseline (monthly mean)", "Spatial kNN (IDW)", "Ridge regression", "Random Forest", "XGBoost", "XGBoost without AOD"]
short = ["Monthly\nmean", "Spatial\nkNN", "Ridge", "RF", "XGB", "XGB\nno AOD"]
schemes = [("Site-held-out (main)", "site held out", "#4c72b0", rows),
           ("Grid-cell-held-out (strict)", "grid cell held out", "#c44e52", rows),
           ("Random split (for comparison only)", "random split", GREY, rnd)]
fig, ax = plt.subplots(1, 2, figsize=(PAGE, 1.9))
x = np.arange(len(order)); w = 0.27
for k, (sch, lab, colr, src) in enumerate(schemes):
    s = src[src.scheme == sch].set_index("model").reindex(order)
    for a, m in zip(ax, ["MAE", "R2"]):
        a.bar(x + (k - 1) * w, s[m], w, yerr=s[f"{m}_fold_sd"], color=colr, label=lab,
              error_kw={"lw": 0.6, "capsize": 1.5}, edgecolor="none")
for a, yl in zip(ax, ["MAE (%/day)", "$R^2$"]):
    a.set_xticks(x, short); a.set_ylabel(yl)
ax[1].axhline(0, color="k", lw=0.6)
ax[0].set_ylim(0, 0.065)
ax[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.1, 1.18))
fig.tight_layout(); fig.savefig(OUT / "fig_comparison.png"); plt.close(fig)

# Fig. 5: XGBoost errors on unseen sites
oof = pd.read_csv(TAB / "oof_predictions_extra.csv")
pred = oof["site|XGBoost"].values; y = oof[TARGET].values
er = pd.read_csv(TAB / "error_by_region.csv").sort_values("MAE")
fig, ax = plt.subplots(1, 2, figsize=(COL, 1.7), gridspec_kw={"width_ratios": [1, 1.1]})
lim = [0, max(y.max(), pred.max()) * 1.05]
ax[0].scatter(y, pred, s=3, alpha=0.35, lw=0, color="#4c72b0")
ax[0].plot(lim, lim, "k--", lw=0.6); ax[0].set(xlim=lim, ylim=lim, xlabel="measured (%/day)", ylabel="predicted (%/day)")
ax[0].set_aspect("equal")
ax[1].barh([r.replace("Northern ", "N. ").replace("Central ", "C. ").replace("Southern ", "S. ").replace("California", "Calif.") for r in er.region], er.MAE, color="#4c72b0", height=0.6)
ax[1].set_xlabel("MAE (%/day)"); ax[1].tick_params(axis="y", labelsize=6.5)
fig.tight_layout(w_pad=0.6); fig.savefig(OUT / "fig_errors.png"); plt.close(fig)

# Fig. 6: SHAP summary (final XGBoost)
import shap
X = d[FEATURES]
xgb = joblib.load(MOD / "xgboost.joblib")
sv = shap.TreeExplainer(xgb).shap_values(X)
plt.figure()
shap.summary_plot(sv, X.rename(columns=NICE), show=False, max_display=12, plot_size=(COL, 2.3), color_bar_label="feature value")
f = plt.gcf()
for a in f.axes:
    a.tick_params(labelsize=7); a.xaxis.label.set_size(8); a.yaxis.label.set_size(8)
f.axes[0].set_xlabel("SHAP value (%/day)")
f.savefig(OUT / "fig_shap.png"); plt.close("all")

# Fig. 7: Pakistan heatmap
pp = pd.read_csv(TAB / "pakistan_predictions.csv")
piv = pp.pivot(index="city", columns="month", values="pred_xgboost")
flag = pp.pivot(index="city", columns="month", values="n_features_outside_training_range").reindex(piv.index)
fig, ax = plt.subplots(figsize=(COL, 2.3))
im = ax.imshow(piv.values, cmap="YlOrBr", aspect="auto", vmin=0)
ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_yticks(range(len(piv)), piv.index)
for sp in ax.spines.values(): sp.set_visible(False)
vmax = np.nanmax(piv.values)
for i in range(piv.shape[0]):
    for j in range(12):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:.2f}"[1:] + ("*" if flag.values[i, j] > 0 else ""), ha="center", va="center",
                fontsize=4.6, color="white" if v > 0.7 * vmax else "black")
cb = plt.colorbar(im, ax=ax, pad=0.01); cb.set_label("predicted soiling (%/day)"); cb.ax.tick_params(labelsize=6)
fig.savefig(OUT / "fig_pakistan.png"); plt.close(fig)
print("saved to", OUT)
