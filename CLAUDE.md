# FinancialForecasting

Real-time USD/ZAR exchange rate forecasting dashboard. Loads pre-trained models and cached predictions, serves an interactive Streamlit web app, and supports live predictions via yfinance. Academic project (University of Zimbabwe, HFCSDA programme).

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| Dashboard | Streamlit | 1.56.0 |
| Deep learning | TensorFlow / Keras | 2.21.0 / 3.14.0 |
| Statistical model | Statsmodels (ARIMA) | 0.14.6 |
| Data & preprocessing | Pandas, NumPy, scikit-learn | 3.0.2 / 2.4.4 / 1.8.0 |
| Live data | yfinance | 1.3.0 |
| Plotting | Matplotlib, Seaborn | 3.10.8 / 0.13.2 |
| Serialisation | Pickle (stdlib), Keras `.keras` | — |

Python virtual environment is at `.venv/`.

## Key Directories

```
FinancialForecasting/
├── dashboard_app.py          # Entire application — data loading, pages, plotting
├── data/
│   ├── raw/USDZAR_raw.csv   # Historical rates 2014-2026 (~3 200 rows)
│   └── processed/           # Numpy arrays, scaler.pkl, config.json
├── models/                  # Trained model files (.keras, .pkl) + metrics JSON
└── outputs/                 # Pre-rendered PNG charts (EDA, forecasts, comparisons)
```

Key files:
- [dashboard_app.py](dashboard_app.py) — single-file app; all logic lives here
- [data/processed/config.json](data/processed/config.json) — ticker, lookback window (60), train/test split index (2555)
- [data/processed/scaler.pkl](data/processed/scaler.pkl) — MinMaxScaler; required for live inference
- [models/final_summary.json](models/final_summary.json) — aggregated metrics for all three models

## Running the App

```bash
# Activate virtual environment
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Unix

# Launch dashboard
streamlit run dashboard_app.py
```

No build step — models are pre-trained. All training artifacts are committed under `models/` and `data/processed/`.

## Models

Three independent forecasting models, all predicting the next-day USD/ZAR rate:

| Model | File | RMSE | MAE | MAPE |
|---|---|---|---|---|
| ARIMA(1,1,1) | `models/arima_model.pkl` | 0.3051 | 0.1313 | 0.74% |
| LSTM | `models/lstm_model.keras` | 0.3136 | 0.1321 | 0.75% |
| CNN-LSTM | `models/cnn_lstm_model.keras` | **0.2845** | 0.1720 | 0.97% |

CNN-LSTM has the lowest RMSE; ARIMA/LSTM have lower MAE/MAPE.

## Dashboard Pages

Defined by sidebar radio at [dashboard_app.py:72-81](dashboard_app.py#L72-L81):

1. **Overview** — dataset stats, rolling averages
2. **ARIMA Forecast** — test-set predictions vs actuals
3. **LSTM Forecast** — test-set predictions vs actuals
4. **CNN-LSTM Forecast** — test-set predictions vs actuals
5. **Model Comparison** — side-by-side RMSE/MAE/MAPE bar charts
6. **Live Forecast** — fetches latest data via yfinance, runs all three models

## Additional Documentation

Check these files when working on the relevant areas:

- [.claude/docs/architectural_patterns.md](.claude/docs/architectural_patterns.md) — data-flow pipeline, caching strategy, live-inference pattern, visualisation conventions, model colour palette
