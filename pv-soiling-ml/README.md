# Predicting Solar Panel Soiling Losses from Climate Data using Machine Learning

MSc ML research project, Riphah International University, Lahore.
Student: Haroon Sardar (81660). Supervisor: Muhammad Yaseen.

## What the project does
Trains machine learning models on NREL's PV Soiling Map (real field data from U.S.
sites) to predict the monthly soiling rate (% of energy lost per day) from climate
and air-quality data only, tests them on sites the models never saw, and applies
them to nine Pakistani cities. The Lahore estimate is compared with the 0.8 %/day
measured by Ullah et al. (2020).

## Folder layout
```
data/raw/         original inputs (see "Data sources")
data/processed/   site_month.csv, pakistan_features.csv (made by step 1)
src/              Python scripts, run in order
outputs/figures/  all plots
outputs/tables/   all result tables (CSV/JSON)
outputs/models/   trained models (joblib)
```

## How to reproduce
```bash
pip install -r requirements.txt
cd src
python 00_download_climate.py   # optional: re-downloads NASA POWER data (needs internet)
python 01_prepare_data.py       # builds the site-month dataset
python 02_eda.py                # data dictionary, summary tables, EDA plots
python 03_models.py             # nested, leakage-safe CV of 4 models; final models; importance
python 04_pakistan.py           # Pakistan estimates and Lahore check
python 05_extra.py              # spatial kNN baseline, XGBoost without AOD, Wilcoxon tests (~15 min)
python 06_paper_figures.py      # IEEE-style figures for the paper
```
All random seeds are fixed (42). Step 3 takes a few minutes on a laptop.

## Data sources
| File | Source | Licence / access |
|---|---|---|
| `sites.csv`, `monthly_raw.txt` | NREL PV Soiling Map, file `soiling_data_2023_11_27.json`, DuraMAT Data Hub, project `nrel-soiling-map`, dataset `soiling-data-v2` (accessed 11 Nov 2026) | Public |
| `climate_raw.csv` | NASA POWER API v2.10 (climatology 2001-2025; daily rainfall 2010-2019). AOD_55 comes from MERRA-2 | Public, free |

`sites.csv` and `monthly_raw.txt` are a lossless flattening of the NREL JSON (site
attributes, and the monthly soiling rate with its interval count). Site names were
replaced by codes S001-S255.

## Key design decisions
* **Target:** monthly soiling rate x -100 = % energy lost per day.
* **Sites with only ">0.99" IWSR** (109 of 255) have no monthly data in the file, so they cannot be used as targets; this is reported as a limitation.
* **Features:** climate, air quality and panel design only. Latitude/longitude are not used, so the model cannot memorise U.S. places.
* **Climate resolution:** NASA POWER grid cell (0.5 x 0.625 deg); 146 sites fall in 35 cells.
* **Leakage control:** scaling inside a pipeline; tuning by inner GroupKFold on training folds only; outer GroupKFold by site (main) and by climate cell (strict). A random split is reported only to show how it inflates scores.

## Main results (site held out, out-of-fold)
| Model | MAE (%/day) | R² |
|---|---|---|
| Monthly-mean baseline | 0.047 | 0.03 |
| Spatial kNN (no climate) | 0.039 | 0.30 |
| Random Forest | 0.038 | 0.35 |
| XGBoost | 0.038 | 0.38 |

XGBoost vs Random Forest: p = 0.41; XGBoost vs spatial kNN: p = 0.61 (paired Wilcoxon on per-site MAE).
With whole climate cells held out, kNN keeps R² 0.30 while XGBoost falls to 0.05.
Lahore (Oct-Jan): measured 0.8 %/day, XGBoost 0.22, XGBoost without AOD 0.25.

## Citation
H. Sardar and M. Yaseen, "Can climate data alone predict photovoltaic soiling? A leakage-safe
machine learning study on U.S. field data and its transfer to Pakistan," 2026 (manuscript).

## Licence
Code: MIT. Data: see the licences of the original NREL and NASA POWER sources.
