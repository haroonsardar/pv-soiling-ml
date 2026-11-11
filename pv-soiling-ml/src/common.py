"""Shared settings used by all steps."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
MOD = ROOT / "outputs" / "models"
for p in (FIG, TAB, MOD):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "soiling_pct_day"

# Features: only climate, air quality and panel design.
# Latitude/longitude are left out on purpose: they would let the model memorise
# U.S. places and would mean nothing for Pakistan.
CLIMATE = ["PRECTOTCORR", "RH2M", "T2M", "T2M_MAX", "WS2M", "WS10M", "AOD_55",
           "ALLSKY_SFC_SW_DWN", "QV2M", "rain_days", "dslr_mean", "max_dry_spell"]
ENGINEERED = ["rain_prev_month", "aod_annual", "rain_annual", "aod_x_dslr",
              "dry_fraction", "month_sin", "month_cos"]
DESIGN = ["tilt", "mounting_tracking"]
FEATURES = CLIMATE + ENGINEERED + DESIGN

NICE = {
    "PRECTOTCORR": "Rainfall (mm/day)", "RH2M": "Humidity (%)", "T2M": "Temperature (C)",
    "T2M_MAX": "Max temperature (C)", "WS2M": "Wind 2 m (m/s)", "WS10M": "Wind 10 m (m/s)",
    "AOD_55": "Aerosol optical depth", "ALLSKY_SFC_SW_DWN": "Solar radiation (kWh/m2/day)",
    "QV2M": "Specific humidity (g/kg)", "rain_days": "Rainy days (>=1 mm)",
    "dslr_mean": "Days since last rain", "max_dry_spell": "Longest dry spell (days)",
    "rain_prev_month": "Rain previous month", "aod_annual": "Yearly mean AOD",
    "rain_annual": "Yearly mean rain", "aod_x_dslr": "AOD x days since rain",
    "dry_fraction": "Dry fraction of month", "month_sin": "Month (sin)",
    "month_cos": "Month (cos)", "tilt": "Panel tilt (deg)", "mounting_tracking": "Tracking mount",
}


def region(row):
    """Simple region label used for error analysis."""
    if row["state"] != "CA":
        return "Other US" if row["state"] != "AZ" else "Arizona"
    if row["lat"] < 35:
        return "Southern California"
    return "Central California" if row["lat"] < 37.5 else "Northern California"


def style():
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                         "savefig.bbox": "tight"})
