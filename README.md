# Air Pollution Risk Assessment and Early Warning for Industrial Cities of Kazakhstan

Code and results for the article *"Development of an Air Pollution Risk Assessment and Early Warning Model Using Machine Learning for Industrial Cities in Kazakhstan"*
(A. Abdimutalip, A. Dukenbay — Astana IT University, Research Methods and Tools, 2026).

## What the code does
1. Loads hourly PM2.5 / PM10 (AirData.kz) and weather data (Open-Meteo, ERA5) for Temirtau, Pavlodar, Oskemen and Karaganda.
2. Cleans the data (removes negative, out-of-range and stuck-sensor values and hours with PM2.5 > PM10; interpolates gaps of up to 3 h).
3. Builds features (lags 1 h / 24 h, hour of day, season, city) and splits each city in time order: 70 % train / 15 % validation / 15 % test.
4. Tunes and trains a persistence baseline, XGBoost, Random Forest, SVR and LSTM for 1 h and 24 h ahead (seeds 0–4).
5. Reports MAE, RMSE, R² (mean ± sd over seeds, block-bootstrap 95 % CI) and pairwise Diebold–Mariano tests.
6. Converts forecasts into WHO-based risk classes (safe / moderate / dangerous) and reports Precision / Recall / F1.
7. Runs additional analyses (by city, by season, hyperparameter settings, SHAP) and error analysis.

## Repository structure
```
notebook/air_pollution_experiments.ipynb   full pipeline (Google Colab): data -> all tables in results/tables
                                           and the working figures in results/figures
make_figures.py                            draws the article figures (results/paper_figures) from results/
results/tables/                            CSV outputs of the notebook
results/figures/                           figures drawn by the notebook (working versions)
results/paper_figures/                     figures exactly as they appear in the article
results/run_info.json                      versions, seeds, hardware, selected settings, run time
requirements.txt
LICENSE
```

## Where each result of the article comes from

| Article | File | Produced by |
|---|---|---|
| Table III – final dataset | `results/tables/table1_dataset.csv`, `cleaning_report.csv` | notebook |
| Table IV – main results | `results/tables/table2_main_results.csv`, `test_metrics_per_seed.csv` | notebook |
| Table V – risk classification | `results/tables/table5_risk_classification.csv` | notebook |
| Fig. 2 – RMSE with 95 % CI | `results/paper_figures/fig2_rmse_ci.pdf` | `make_figures.py` (from `table2_main_results.csv`) |
| Fig. 3 – Diebold–Mariano tests | `results/paper_figures/fig3_dm_tests.pdf` | `make_figures.py` (from `table4_diebold_mariano.csv`) |
| Fig. 4 – risk classification | `results/paper_figures/fig4_risk_classification.pdf` | `make_figures.py` (from `table5_risk_classification.csv` and the confusion matrices printed by the notebook) |
| Fig. 5 – results by city | `results/paper_figures/fig5_by_city.pdf` | `make_figures.py` (from `table6_by_city.csv`) |
| Fig. 6 – SHAP | `results/paper_figures/fig6_shap_xgb_pm25_h24.pdf` | notebook (`results/figures/fig5_shap_xgb_pm25_h24.pdf`), copied by `make_figures.py` |
| Results by season | `results/tables/table7_mae_by_season.csv` | notebook |
| Sensitivity to hyperparameters | `results/tables/tuning_validation.csv` (validation metrics of every tried setting) | notebook |
| Feature importance values | `results/tables/shap_importance.csv` | notebook |
| Error analysis | `results/tables/table9_top10_errors.csv`, `results/figures/fig6_worst_week_pm25.pdf` | notebook |

The CSV names follow the order in which the notebook creates them; there are no files `table3` or `table8` (nothing was removed).
The article numbers its tables and figures separately, as shown above.

## Data
No manual download is needed — the notebook loads everything itself:
- PM2.5 / PM10: AirData.kz open-data archive, https://github.com/qazybekb/AirDatakz-OpenData
  (Temirtau station 200, Pavlodar station 48, Oskemen station 240, Karaganda folder);
- weather (temperature, relative humidity, wind speed): Open-Meteo historical archive (ERA5), https://open-meteo.com

Data were downloaded in October 2026. The archive may be updated later, so a new run can give slightly different numbers.

## How to run
1. Open `notebook/air_pollution_experiments.ipynb` in Google Colab, select Runtime → Change runtime type → T4 GPU, and run all cells
   (about 21.5 min). At the end `results.zip` with `tables/`, `figures/` and `run_info.json` is downloaded.
2. To draw the article figures, add a new cell at the end of the notebook and run:
   ```
   !wget -q https://raw.githubusercontent.com/arayleum/air-pollution-kz/main/make_figures.py
   !python make_figures.py /content/results /content/results/paper_figures
   ```
   or run `python make_figures.py results results/paper_figures` locally on the unzipped results.

## Environment
Python 3.13.15 · NumPy 2.1.3 · pandas 2.2.3 · scikit-learn 1.6.1 · XGBoost 3.4.1 · TensorFlow 2.21.0 ·
SciPy 1.16.3 · SHAP 0.54.1 (see `run_info.json`). Hardware: Google Colab, Intel Xeon 2.00 GHz, NVIDIA Tesla T4.
Random seeds: 0–4 for the models, 42 for the bootstrap.

## Authors
Akbota Abdimutalip (BDA-2401), Arailym Dukenbay (BDA-2403). Supervisor: Mira Bukaeva.

## License
Code: MIT (see `LICENSE`). The data belong to their providers (AirData.kz, Open-Meteo) and are not redistributed here.
