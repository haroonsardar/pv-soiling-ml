"""
Step 1 (Week 4): build the site-month dataset.

Inputs (data/raw):
  sites.csv         one row per NREL soiling-map site (255 sites)
  monthly_raw.txt   monthly soiling rates per site, exported from the NREL
                    Soiling Map JSON (soiling_data_2023_11_27.json)
  climate_raw.csv   NASA POWER monthly climatology + rainfall features for each
                    site location and for selected Pakistani cities

Output (data/processed):
  site_month.csv    one row per site and calendar month, with target + features
  pakistan_features.csv  same features for Pakistani cities (no target)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CLIMATE_COLS = ["PRECTOTCORR", "RH2M", "T2M", "T2M_MAX", "WS2M", "WS10M", "AOD_55",
                "ALLSKY_SFC_SW_DWN", "QV2M", "rain_days", "dslr_mean", "max_dry_spell"]


def load_sites():
    s = pd.read_csv(RAW / "sites.csv")
    # ">0.99" means the annual loss was below 1%; NREL did not give an exact value
    s["iwsr_censored"] = s["iwsr"].astype(str).str.startswith(">")
    s["iwsr_num"] = pd.to_numeric(s["iwsr"], errors="coerce")
    return s


def load_monthly():
    rows = []
    for rec in open(RAW / "monthly_raw.txt").read().strip().split(";"):
        sid, month, rate, count = rec.split(",")
        rows.append({"site_id": "S" + sid, "month": int(month),
                     "rate": pd.to_numeric(rate, errors="coerce"),
                     "interval_count": pd.to_numeric(count, errors="coerce")})
    m = pd.DataFrame(rows)
    # NREL reports soiling rate as a negative fraction per day (e.g. -0.0012).
    # Target used here: soiling loss in % per day (positive), e.g. 0.12
    m["soiling_pct_day"] = -m["rate"] * 100
    return m


def add_engineered(df, key):
    """Features built from the monthly climate. `key` = column(s) that identify a location."""
    df = df.sort_values(key + ["month"]).copy()
    g = df.groupby(key, sort=False)
    # rain in the previous calendar month (wraps Dec -> Jan); rain cleans panels
    df["rain_prev_month"] = g["PRECTOTCORR"].transform(lambda s: np.roll(s.values, 1))
    # yearly averages describe the general climate of the place
    df["aod_annual"] = g["AOD_55"].transform("mean")
    df["rain_annual"] = g["PRECTOTCORR"].transform("mean")
    # dust that stays on the panel: dusty air x long dry spells
    df["aod_x_dslr"] = df["AOD_55"] * df["dslr_mean"]
    # dry fraction of the month
    df["dry_fraction"] = 1 - df["rain_days"] / 30.4
    # month as a circle so December and January are close
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def main():
    sites, monthly = load_sites(), load_monthly()
    clim = pd.read_csv(RAW / "climate_raw.csv")
    us = clim[clim["name"] == "US"].drop(columns="name")
    pk = clim[clim["name"] != "US"].rename(columns={"name": "city"})

    df = monthly.merge(sites, on="site_id", how="left")
    # NASA POWER data sits on a 0.5 deg (lat) x 0.625 deg (lon) grid, so each site
    # takes the climate of the grid cell it falls in
    df["cell_lat"] = (df["lat"] / 0.5).round() * 0.5
    df["cell_lon"] = (df["lon"] / 0.625).round() * 0.625
    us = us.rename(columns={"lat": "cell_lat", "lon": "cell_lon"})
    df = df.merge(us, on=["cell_lat", "cell_lon", "month"], how="left")
    df["cell_id"] = df["cell_lat"].astype(str) + "_" + df["cell_lon"].astype(str)
    assert df[CLIMATE_COLS].notna().all().all(), "climate missing for some sites"
    df = add_engineered(df, ["site_id"])
    df["mounting_tracking"] = (df["mounting"] == "Tracking").astype(int)
    df["is_station"] = (df["type"] == "Soiling Station").astype(int)
    df["has_target"] = df["soiling_pct_day"].notna()
    df.to_csv(OUT / "site_month.csv", index=False)

    pk = add_engineered(pk, ["city"])
    # typical Pakistani rooftop: fixed panel at about 30 degrees (as in the Lahore study)
    pk["tilt"], pk["mounting_tracking"] = 30, 0
    pk.to_csv(OUT / "pakistan_features.csv", index=False)

    print("site-month rows:", len(df), "| with target:", int(df.has_target.sum()),
          "| sites:", df.site_id.nunique())
    print("Pakistan rows:", len(pk), "| cities:", pk.city.nunique())


if __name__ == "__main__":
    main()
