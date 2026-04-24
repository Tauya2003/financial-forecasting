
from pathlib import Path
import warnings

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
import yfinance as yf
import tensorflow as tf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
import dask.dataframe as dd
import dask.array as da
from streamlit_autorefresh import st_autorefresh

PROJECT_PATH = Path(__file__).parent

st.set_page_config(
    page_title="Financial Forecasting System",
    page_icon="📈",
    layout="wide"
)


@st.cache_data
def load_all_data():
    # Dask DataFrame for scalable CSV ingestion
    raw_dask = dd.read_csv(
        str(PROJECT_PATH / "data" / "raw" / "USDZAR_raw.csv"),
        skiprows=3,
        header=0,
        names=["date", "price", "rolling_30", "rolling_90", "returns"],
        assume_missing=True,
    )
    raw = raw_dask.compute()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.set_index("date")
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw.dropna(subset=["price"], inplace=True)

    with open(PROJECT_PATH / "data" / "processed" / "config.json") as f:
        config = json.load(f)

    # Dask arrays for parallel numerical data loading
    actual        = da.from_array(np.load(PROJECT_PATH / "data" / "processed" / "actual_prices.npy"),        chunks=200).compute()
    arima_pred    = da.from_array(np.load(PROJECT_PATH / "data" / "processed" / "arima_predictions.npy"),    chunks=200).compute()
    lstm_pred     = da.from_array(np.load(PROJECT_PATH / "data" / "processed" / "lstm_predictions.npy"),     chunks=200).compute()
    cnn_lstm_pred = da.from_array(np.load(PROJECT_PATH / "data" / "processed" / "cnn_lstm_predictions.npy"), chunks=200).compute()

    with open(PROJECT_PATH / "models" / "arima_metrics.json")    as f: arima_m    = json.load(f)
    with open(PROJECT_PATH / "models" / "lstm_metrics.json")     as f: lstm_m     = json.load(f)
    with open(PROJECT_PATH / "models" / "cnn_lstm_metrics.json") as f: cnn_lstm_m = json.load(f)

    split_idx  = config["split_idx"]
    lookback   = config["lookback"]
    test_dates = raw.index[split_idx + lookback:]

    min_len = min(
        len(test_dates),
        len(actual),
        len(arima_pred),
        len(lstm_pred),
        len(cnn_lstm_pred)
    )

    return {
        "raw"        : raw,
        "config"     : config,
        "test_dates" : test_dates[:min_len],
        "actual"     : actual[:min_len].flatten(),
        "arima_pred" : arima_pred[:min_len].flatten(),
        "lstm_pred"  : lstm_pred[:min_len].flatten(),
        "cnn_pred"   : cnn_lstm_pred[:min_len].flatten(),
        "arima_m"    : arima_m,
        "lstm_m"     : lstm_m,
        "cnn_lstm_m" : cnn_lstm_m,
    }


@st.cache_resource
def load_lstm_model():
    return tf.keras.models.load_model(str(PROJECT_PATH / "models" / "lstm_model.keras"))


@st.cache_resource
def load_cnn_lstm_model():
    return tf.keras.models.load_model(str(PROJECT_PATH / "models" / "cnn_lstm_best.keras"))


data = load_all_data()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Live Forecast",
    "Model Forecasts",
    "Performance Metrics",
    "Model Comparison",
    "About"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** USD/ZAR Exchange Rate")
st.sidebar.markdown("**Period:** 2014 to 2026")
_train_start = data["raw"].index[0].strftime("%b %Y")
_split_date  = data["raw"].index[data["config"]["split_idx"]].strftime("%b %Y")
_test_end    = data["raw"].index[-1].strftime("%b %Y")
st.sidebar.markdown(f"**Train:** {data['config']['train_size']} days ({_train_start} – {_split_date})")
st.sidebar.markdown(f"**Test:** {data['config']['test_size']} days ({_split_date} – {_test_end})")


# ══════════════════════════════════════════════════════════════
if page == "Home":
    st.title("Real-Time Financial Forecasting System")
    st.markdown("#### USD/ZAR Exchange Rate Forecasting using Big Data Analytics")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"**Total Trading Days**\n\n### {len(data['raw']):,}")
    col2.markdown(f"**Training Days**\n\n### {data['config']['train_size']:,}")
    col3.markdown(f"**Test Days**\n\n### {data['config']['test_size']:,}")
    col4.markdown(f"**Lookback Window**\n\n### {data['config']['lookback']} days")

    st.markdown("---")
    st.subheader("USD/ZAR Exchange Rate — Full History")
    price_vals = data["raw"]["price"].values.astype(float)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(data["raw"].index, price_vals, color="steelblue", lw=1.0)
    ax.axvline(x=data["raw"].index[data["config"]["split_idx"]],
               color="red", lw=1.5, ls="--", label="Train / Test split")
    ax.set_ylabel("USD/ZAR")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("System Architecture")
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("**DATA LAYER**")
            st.markdown("- yfinance API\n- Dask DataFrames & Arrays\n- MinMaxScaler\n- 60-day lookback sequences")
    with col2:
        with st.container(border=True):
            st.markdown("**MODEL LAYER**")
            st.markdown("- ARIMA(1,1,1)\n- LSTM (2 layers)\n- CNN-LSTM Hybrid")
    with col3:
        with st.container(border=True):
            st.markdown("**EVALUATION LAYER**")
            st.markdown("- RMSE\n- MAE\n- MAPE")


# ══════════════════════════════════════════════════════════════
elif page == "Live Forecast":
    st_autorefresh(interval=300_000, key="live_refresh")

    st.title("Live Forecast")
    st.markdown("#### Auto-refreshing every 5 minutes with latest market data")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("---")

    model_choice = st.selectbox(
        "Select forecasting model:",
        ["ARIMA(1,1,1)", "LSTM", "CNN-LSTM"]
    )

    try:
        with st.spinner("Fetching latest USD/ZAR data from Yahoo Finance..."):
            end   = datetime.today()
            start = end - timedelta(days=120)
            live  = yf.download("ZAR=X",
                                start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"),
                                progress=False)
            live  = live[["Close"]].dropna()
            live.columns = ["price"]
            live["price"] = pd.to_numeric(live["price"], errors="coerce")
            live.dropna(inplace=True)

        st.success(f"✅ Fetched {len(live)} trading days of live data")
        col_a, col_b = st.columns(2)
        col_a.markdown(f"**Latest price:** {live['price'].iloc[-1]:.4f} ZAR per USD")
        col_b.markdown(f"**As of:** {live.index[-1].date()}")

        LOOKBACK = data["config"]["lookback"]

        if len(live) < LOOKBACK:
            st.error(f"Not enough data — need {LOOKBACK} trading days, got {len(live)}. Try again later.")
        else:
            if model_choice == "ARIMA(1,1,1)":
                spinner_msg = "Fitting ARIMA(1,1,1) on recent market data — takes a few seconds..."
            else:
                spinner_msg = f"Running {model_choice} inference..."

            with st.spinner(spinner_msg):
                if model_choice == "ARIMA(1,1,1)":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        arima_live = ARIMAModel(live["price"].values, order=(1, 1, 1)).fit()
                    pred_price = float(np.array(arima_live.forecast(steps=1))[0])
                else:
                    with open(PROJECT_PATH / "data" / "processed" / "scaler.pkl", "rb") as f:
                        scaler = pickle.load(f)

                    last_60    = live["price"].values[-LOOKBACK:]
                    last_60_sc = scaler.transform(last_60.reshape(-1, 1))
                    X_live     = last_60_sc.reshape(1, LOOKBACK, 1)

                    if model_choice == "LSTM":
                        model = load_lstm_model()
                    else:
                        model = load_cnn_lstm_model()

                    pred_scaled = model.predict(X_live, verbose=0)
                    pred_price  = float(scaler.inverse_transform(pred_scaled)[0][0])

            current   = float(live["price"].iloc[-1])
            change    = pred_price - current
            direction = "UP" if change > 0 else "DOWN"
            emoji     = "📈" if change > 0 else "📉"

            st.markdown("---")
            st.subheader("Forecast Result")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current USD/ZAR", f"{current:.4f}")
            col2.metric("Predicted Next Day", f"{pred_price:.4f}", delta=f"{change:+.4f} ZAR")
            col3.markdown(f"**Direction**\n\n### {emoji} {direction}")

            st.markdown("---")
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(live.index[-LOOKBACK:],
                    live["price"].values[-LOOKBACK:],
                    color="steelblue", lw=1.5, label="Last 60 days")
            next_date = live.index[-1] + timedelta(days=1)
            ax.scatter(next_date, pred_price, color="red", s=120,
                       zorder=5, label=f"Forecast: {pred_price:.4f}")
            ax.axhline(y=pred_price, color="red", lw=0.8, ls="--", alpha=0.5)
            ax.set_title(f"{model_choice} — Live Forecast", fontweight="bold")
            ax.set_ylabel("USD/ZAR")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.info("Note: This is a prototype model. Not financial advice.")

    except Exception as e:
        st.error(f"Could not complete forecast: {e}")


# ══════════════════════════════════════════════════════════════
elif page == "Model Forecasts":
    st.title("Model Forecasts vs Actual")
    st.markdown("---")

    model_choice = st.selectbox(
        "Select model to view:",
        ["ARIMA(1,1,1)", "LSTM", "CNN-LSTM", "All Models"]
    )

    td  = data["test_dates"]
    act = data["actual"]

    def forecast_plot(title, pred, color):
        n   = min(len(td), len(act), len(pred))
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(td[:n], act[:n], label="Actual USD/ZAR",
                color="steelblue", lw=1.5)
        ax.plot(td[:n], pred[:n], label=title,
                color=color, lw=1.5, linestyle="--")
        ax.set_title(f"{title} — Forecast vs Actual", fontweight="bold")
        ax.set_ylabel("USD/ZAR")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    if model_choice == "ARIMA(1,1,1)":
        st.pyplot(forecast_plot("ARIMA(1,1,1)", data["arima_pred"], "orange"))
    elif model_choice == "LSTM":
        st.pyplot(forecast_plot("LSTM", data["lstm_pred"], "green"))
    elif model_choice == "CNN-LSTM":
        st.pyplot(forecast_plot("CNN-LSTM", data["cnn_pred"], "red"))
    else:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        for ax, pred, color, name in zip(
            axes,
            [data["arima_pred"], data["lstm_pred"], data["cnn_pred"]],
            ["orange", "green", "red"],
            ["ARIMA(1,1,1)", "LSTM", "CNN-LSTM"]
        ):
            n = min(len(td), len(act), len(pred))
            ax.plot(td[:n], act[:n], color="steelblue", lw=1.5, label="Actual")
            ax.plot(td[:n], pred[:n], color=color, lw=1.5, ls="--", label=name)
            ax.set_ylabel("USD/ZAR")
            ax.legend()
            ax.grid(alpha=0.3)
        axes[2].set_xlabel("Date")
        fig.tight_layout()
        st.pyplot(fig)
    plt.close("all")


# ══════════════════════════════════════════════════════════════
elif page == "Performance Metrics":
    st.title("Model Performance Metrics")
    st.markdown("---")

    df_metrics = pd.DataFrame([
        data["arima_m"],
        data["lstm_m"],
        data["cnn_lstm_m"]
    ]).set_index("model")
    df_metrics.columns = ["RMSE", "MAE", "MAPE (%)"]
    df_metrics = df_metrics.round(4)

    st.subheader("Performance Summary Table")
    st.dataframe(df_metrics.style.highlight_min(color="lightgreen"),
                 use_container_width=True)

    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **RMSE** (Root Mean Squared Error) — average prediction error in ZAR. "
            "Penalises large errors more than small ones. **Lower is better.**\n"
            "- **MAE** (Mean Absolute Error) — average absolute difference between predicted "
            "and actual price in ZAR. Easier to interpret than RMSE. **Lower is better.**\n"
            "- **MAPE** (Mean Absolute Percentage Error) — average error as a percentage of "
            "the actual price. Useful for comparing across different price scales. **Lower is better.**"
        )

    st.markdown("---")
    st.subheader("Metric Comparison Charts")
    models = ["ARIMA(1,1,1)", "LSTM", "CNN-LSTM"]
    colors = ["orange", "green", "red"]

    metrics_data = [
        ("RMSE (ZAR)",  [data["arima_m"]["RMSE"], data["lstm_m"]["RMSE"], data["cnn_lstm_m"]["RMSE"]]),
        ("MAE (ZAR)",   [data["arima_m"]["MAE"],  data["lstm_m"]["MAE"],  data["cnn_lstm_m"]["MAE"]]),
        ("MAPE (%)",    [data["arima_m"]["MAPE"], data["lstm_m"]["MAPE"], data["cnn_lstm_m"]["MAPE"]]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, vals) in zip(axes, metrics_data):
        bars = ax.bar(models, vals, color=colors, edgecolor="black", lw=0.8)
        ax.set_title(f"{metric}\n(lower is better)", fontweight="bold")
        ax.set_ylabel(metric)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=10)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout(pad=2.0)
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Head-to-Head Model Comparison")
    st.markdown("---")

    col1, col2 = st.columns(2)
    m1 = col1.selectbox("Model 1", ["ARIMA(1,1,1)", "LSTM", "CNN-LSTM"], index=0)
    m2 = col2.selectbox("Model 2", ["ARIMA(1,1,1)", "LSTM", "CNN-LSTM"], index=2)

    model_map = {
        "ARIMA(1,1,1)": (data["arima_pred"], "orange", data["arima_m"]),
        "LSTM"        : (data["lstm_pred"],  "green",  data["lstm_m"]),
        "CNN-LSTM"    : (data["cnn_pred"],   "red",    data["cnn_lstm_m"]),
    }

    if m1 == m2:
        st.warning("Please select two different models to compare.")
        st.stop()

    pred1, col1c, m1_metrics = model_map[m1]
    pred2, col2c, m2_metrics = model_map[m2]
    td  = data["test_dates"]
    act = data["actual"]
    n   = min(len(td), len(act), len(pred1), len(pred2))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(td[:n], act[:n],   label="Actual", color="steelblue", lw=1.5)
    ax.plot(td[:n], pred1[:n], label=m1, color=col1c, lw=1.5, ls="--")
    ax.plot(td[:n], pred2[:n], label=m2, color=col2c, lw=1.5, ls=":")
    ax.set_title(f"{m1} vs {m2}", fontweight="bold")
    ax.set_ylabel("USD/ZAR")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Metrics Side by Side")
    c1, c2 = st.columns(2)
    for col, name, metrics in [(c1, m1, m1_metrics), (c2, m2, m2_metrics)]:
        col.markdown(f"**{name}**")
        col.markdown(f"RMSE: **{metrics['RMSE']:.4f}**")
        col.markdown(f"MAE:  **{metrics['MAE']:.4f}**")
        col.markdown(f"MAPE: **{metrics['MAPE']:.4f}%**")


# ══════════════════════════════════════════════════════════════
elif page == "About":
    st.title("About This System")
    st.markdown("---")
    st.markdown("### Developing a Real-Time Financial Forecasting System Using Big Data Analytics")
    st.markdown("**Author:** Panashe Mutamba")
    st.markdown("**Programme:** HFCSDA — Faculty of BMSE")
    st.markdown("**Institution:** University of Zimbabwe")
    st.markdown("---")
    st.markdown("### Models Used")
    st.markdown("- **ARIMA(1,1,1)** — Traditional statistical baseline")
    st.markdown("- **LSTM** — Deep learning with temporal memory")
    st.markdown("- **CNN-LSTM** — Hybrid combining CNN feature extraction with LSTM temporal modelling")
    st.markdown("---")
    st.markdown("### Key Finding")
    st.markdown("CNN-LSTM achieves the lowest RMSE — 6.74% better than ARIMA and 9.27% better than standalone LSTM.")
    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown("`Python` · `TensorFlow/Keras` · `statsmodels` · `yfinance` · `Streamlit` · `Dask`")
    st.markdown("*Submitted in partial fulfilment of the requirements for the degree — 2026*")
