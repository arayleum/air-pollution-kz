# Air Pollution Risk Assessment and Early Warning for Industrial Cities of Kazakhstan

Code and results for the article *"Development of an Air Pollution Risk Assessment and Early Warning Model Using Machine Learning for Industrial Cities in Kazakhstan"*
(A. Abdimutalip, A. Dukenbay — Astana IT University, Research Methods and Tools, 2026).

## What the code does
1. Loads hourly PM2.5 / PM10 and weather data for Temirtau, Pavlodar, Oskemen and Karaganda.
2. Cleans the data (removes negative / out-of-range / stuck-sensor values and PM2.5 > PM10, interpolates gaps of up to 3 h).
3. Builds features (lags 1 h / 24 h, hour of day, season, city).
4. Splits each city in time order: 70 % train / 15 % validation / 15 % test.
5. Trains a persistence baseline, XGBoost, Random Forest, SVR and LSTM for 1 h and 24 h ahead.
6. Reports MAE, RMSE, R² (mean ± sd over seeds 0–4, block-bootstrap 95 % CI) and Diebold–Mariano tests.
7. Converts forecasts into WHO-based risk classes (safe / moderate / dangerous) and reports Precision / Recall / F1.
8. Produces SHAP explanations and error analysis.

## Repository structure
```
notebook/   air_pollution_experiments.ipynb   – the full pipeline (Google Colab)
results/    tables/*.csv, figures/*.pdf, run_info.json – outputs used in the article
requirements.txt
```

## Data
No manual download is needed — the notebook loads everything itself:
- PM2.5 / PM10: AirData.kz open-data archive, https://github.com/qazybekb/AirDatakz-OpenData
  (Temirtau station 200, Pavlodar station 48, Oskemen station 240, Karaganda folder);
- weather (temperature, relative humidity, wind speed): Open-Meteo historical archive (ERA5), https://open-meteo.com

## How to run
1. Open `notebook/air_pollution_experiments.ipynb` in Google Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Runtime → Run all. Full run: about 21.5 min (Tesla T4 GPU, Intel Xeon 2.00 GHz).
4. At the end `results.zip` with all tables and figures is downloaded.

## Environment
Python 3.13.15 · NumPy 2.1.3 · pandas 2.2.3 · scikit-learn 1.6.1 · XGBoost 3.4.1 ·
TensorFlow 2.21.0 · SciPy 1.16.3 · SHAP 0.54.1. Random seeds: 0, 1, 2, 3, 4.

## Authors
Akbota Abdimutalip (BDA-2401), Arailym Dukenbay (BDA-2403). Supervisor: Mira Bukaeva.
