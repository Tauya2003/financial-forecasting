# Architectural Patterns

## 1. Pre-Computed Results Pipeline

All model training happens offline. The dashboard is a pure inference/display layer.

```
USDZAR_raw.csv
  → MinMaxScaler (fit on train) → scaler.pkl
  → 60-day sequence windows → X_train.npy, X_test.npy
  → Three parallel model training runs
  → {arima,lstm,cnn_lstm}_predictions.npy + *_metrics.json
  → final_summary.json
```

The dashboard never trains or evaluates — it only loads these artefacts. Adding a new model means adding matching `.npy` and `_metrics.json` files plus wiring them into `load_all_data()` at [dashboard_app.py:20-68](../../dashboard_app.py#L20-L68).

## 2. Single Cached Load Function

All data (CSV, NumPy arrays, scaler, metrics JSON) is loaded once at startup inside `load_all_data()`, decorated with `@st.cache_data`.

- Location: [dashboard_app.py:20-68](../../dashboard_app.py#L20-L68)
- Returns one dict consumed by every page
- Array-length mismatches between models are resolved with `min_len = min(len(a), len(b), ...)`

When adding new data sources, add them to this single function rather than loading per-page.

## 3. Sidebar-Driven Page Routing

Navigation is a `st.sidebar.radio` followed by a flat `if/elif/else` chain.

- Radio definition: [dashboard_app.py:72-81](../../dashboard_app.py#L72-L81)
- No router abstraction; each branch renders directly inline

New pages require: one new radio option string + one new `elif` branch.

## 4. Live Inference Pattern

The Live Forecast page re-implements the same preprocessing pipeline used during training:

1. Fetch last N+60 days via `yfinance.download(ticker)`
2. Extract closing prices as a NumPy column
3. `scaler.transform(last_60.reshape(-1, 1))` — uses the persisted scaler, not a fresh fit
4. Reshape to `(1, 60, 1)` for Keras models; pass raw values to ARIMA
5. `scaler.inverse_transform(pred_scaled)` to recover ZAR price

- Location: [dashboard_app.py:124-203](../../dashboard_app.py#L124-L203)
- **Critical:** the scaler must never be refit on new data — always load from `data/processed/scaler.pkl`
- `config.json` lookback key (60) governs window size; both training and live inference must use the same value

## 5. Visualisation Conventions

All forecast plots follow the same scheme, applied consistently across every model page and the comparison page:

| Element | Style |
|---|---|
| Actual prices | solid line, blue |
| ARIMA forecast | dashed line, **orange** |
| LSTM forecast | dashed line, **green** |
| CNN-LSTM forecast | dashed line, **red** |
| Metric bar charts | value labels above each bar |

Plots are rendered with Matplotlib and passed to Streamlit via `st.pyplot(fig)`. No Plotly or Altair is used in the dashboard despite both being installed.

## 6. Metrics Storage Convention

Each model has its own `models/<name>_metrics.json` with keys `{model, RMSE, MAE, MAPE}`. These are loaded into the data dict and surfaced by the Model Comparison page. `models/final_summary.json` aggregates all three for quick reference.

When adding a model, follow the same JSON schema so the comparison page picks it up automatically.

## 7. Sequence Encoding

- Lookback window: 60 trading days (set in `config.json`)
- Features: closing price only (univariate), normalised to [0, 1] with MinMaxScaler
- Shape convention: `(batch, timesteps, features)` → `(N, 60, 1)`
- ARIMA receives raw (unscaled) prices; deep learning models receive the scaled sequences
