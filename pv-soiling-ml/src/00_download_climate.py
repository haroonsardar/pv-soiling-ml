"""
Step 0: download climate data from NASA POWER (free, no login).

For every modelled site location and every Pakistani city it saves, per
calendar month:
  * 2001-2025 climatology: rainfall, humidity, temperature, max temperature,
    wind (2 m and 10 m), aerosol optical depth at 550 nm (from MERRA-2),
    solar radiation and specific humidity
  * from daily rainfall 2010-2019: mean number of rainy days (>= 1 mm),
    mean days since last rain, and mean longest dry spell in the month

Output: data/raw/climate_raw.csv
"""
import time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
P = "PRECTOTCORR,RH2M,T2M,T2M_MAX,WS2M,WS10M,AOD_55,ALLSKY_SFC_SW_DWN,QV2M"
MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
API = "https://power.larc.nasa.gov/api/temporal"
PAKISTAN = [("Lahore", 31.55, 74.34), ("Karachi", 24.86, 67.01), ("Multan", 30.2, 71.47),
            ("Islamabad", 33.68, 73.05), ("Bahawalpur", 29.4, 71.68), ("Peshawar", 34.01, 71.58),
            ("Quetta", 30.18, 66.97), ("Faisalabad", 31.42, 73.08), ("Hyderabad", 25.39, 68.37)]


def get(url, tries=4):
    for t in range(tries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(3 * (t + 1))
    raise RuntimeError(url)


def rain_features(daily):
    s = pd.Series(daily).sort_index()
    s.index = pd.to_datetime(s.index, format="%Y%m%d")
    s = s[s >= 0]
    rainy = s >= 1.0
    # days since last rain (counter resets on a rainy day)
    dslr, c = [], 30
    for r in rainy:
        c = 0 if r else c + 1
        dslr.append(min(c, 365))
    d = pd.DataFrame({"rainy": rainy, "dslr": dslr}, index=s.index)
    # longest run of dry days inside each month
    d["ym"] = d.index.to_period("M")
    d["run"] = (~d.rainy).groupby((d.rainy | (d.ym != d.ym.shift())).cumsum()).cumsum()
    out = []
    for m in range(1, 13):
        x = d[d.index.month == m]
        yrs = x.ym.nunique()
        out.append({"month": m, "rain_days": x.rainy.sum() / yrs, "dslr_mean": x.dslr.mean(),
                    "max_dry_spell": x.groupby("ym").run.max().mean()})
    return pd.DataFrame(out)


def main():
    sites = pd.read_csv(RAW / "sites.csv")
    ids = {"S" + r.split(",")[0] for r in open(RAW / "monthly_raw.txt").read().strip().split(";")}
    pts = sites[sites.site_id.isin(ids)][["lat", "lon"]].copy()
    # one request per NASA POWER grid cell (0.5 x 0.625 deg) instead of per site
    pts["lat"] = (pts["lat"] / 0.5).round() * 0.5
    pts["lon"] = (pts["lon"] / 0.625).round() * 0.625
    pts = pts.drop_duplicates()
    points = [("US", a, b) for a, b in pts.values] + PAKISTAN
    rows = []
    for i, (name, lat, lon) in enumerate(points, 1):
        c = get(f"{API}/climatology/point?parameters={P}&community=RE&longitude={lon}&latitude={lat}&format=JSON")
        d = get(f"{API}/daily/point?parameters=PRECTOTCORR&community=RE&longitude={lon}&latitude={lat}"
                f"&start=20100101&end=20191231&format=JSON")
        rf = rain_features(d["properties"]["parameter"]["PRECTOTCORR"])
        for m in range(1, 13):
            row = {"name": name, "lat": lat, "lon": lon, "month": m}
            row.update({p: c["properties"]["parameter"][p][MON[m - 1]] for p in P.split(",")})
            row.update(rf[rf.month == m].iloc[0].drop("month").round(2).to_dict())
            rows.append(row)
        print(f"{i}/{len(points)} {name} {lat},{lon}")
    pd.DataFrame(rows).to_csv(RAW / "climate_raw.csv", index=False)


if __name__ == "__main__":
    main()
