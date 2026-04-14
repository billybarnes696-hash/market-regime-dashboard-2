import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import grangercausalitytests


# =========================
# Streamlit setup
# =========================
st.set_page_config(page_title="SOXL Canary Lead Detector", layout="wide")
st.title("SOXL Canary Lead Detector")
st.caption("Granger causality + rolling lead detection + predictive ranking")


# =========================
# Sidebar controls
# =========================
DEFAULT_CANARIES = [
    "NVDA", "AMD", "AVGO", "SMH", "SOXX", "TSM",
    "MU", "AMAT", "LRCX", "KLAC", "ASML", "INTC"
]

target = st.sidebar.text_input("Target ticker", value="SOXL").upper()
canaries_raw = st.sidebar.text_area(
    "Candidate canaries (comma-separated)",
    value=", ".join(DEFAULT_CANARIES),
)
canaries = [x.strip().upper() for x in canaries_raw.split(",") if x.strip()]

interval = st.sidebar.selectbox(
    "Interval",
    options=["15m", "30m", "60m", "1d"],
    index=2,
)

if interval in {"15m", "30m", "60m"}:
    period_options = ["5d", "1mo", "3mo"]
    default_period = "3mo"
else:
    period_options = ["6mo", "1y", "2y", "5y"]
    default_period = "2y"

period = st.sidebar.selectbox(
    "Period",
    options=period_options,
    index=period_options.index(default_period),
)

max_lag = st.sidebar.slider("Max lag bars", min_value=1, max_value=10, value=5)
rolling_window = st.sidebar.slider("Rolling window (bars)", min_value=30, max_value=300, value=80, step=10)
test_size = st.sidebar.slider("Test split", min_value=0.10, max_value=0.50, value=0.30, step=0.05)
z_threshold = st.sidebar.slider("Extreme move Z-threshold", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
run_button = st.sidebar.button("Run analysis", type="primary")


# =========================
# Data utilities
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def download_prices(tickers, period, interval):
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if data is None or len(data) == 0:
        return pd.DataFrame()

    closes = {}
    if len(tickers) == 1:
        closes[tickers[0]] = data["Close"]
    else:
        lvl0 = set(data.columns.get_level_values(0))
        for t in tickers:
            if t in lvl0 and "Close" in data[t].columns:
                closes[t] = data[t]["Close"]

    out = pd.DataFrame(closes)
    out = out.sort_index().ffill().dropna(how="all")
    return out


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    rets = np.log(price_df / price_df.shift(1))
    return rets.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def safe_concat(a: pd.Series, b: pd.Series, names=("x", "y")) -> pd.DataFrame:
    return pd.concat([a.rename(names[0]), b.rename(names[1])], axis=1).dropna()


# =========================
# Core analytics
# =========================
def best_lagged_corr(target_ret: pd.Series, canary_ret: pd.Series, max_lag: int):
    rows = []
    for lag in range(1, max_lag + 1):
        df = safe_concat(canary_ret.shift(lag), target_ret, names=("x", "y"))
        if len(df) < 25:
            continue
        corr = df["x"].corr(df["y"])
        rows.append((lag, corr))

    if not rows:
        return np.nan, np.nan, pd.DataFrame(columns=["lag", "corr"])

    corr_df = pd.DataFrame(rows, columns=["lag", "corr"])
    best_idx = corr_df["corr"].abs().idxmax()
    best = corr_df.loc[best_idx]
    return int(best["lag"]), float(best["corr"]), corr_df


def predictive_regression(target_ret: pd.Series, canary_ret: pd.Series, max_lag: int, test_size: float):
    best = {
        "lag": np.nan,
        "r2": np.nan,
        "hit_rate": np.nan,
        "coef": np.nan,
        "intercept": np.nan,
    }

    y = target_ret.shift(-1)

    for lag in range(1, max_lag + 1):
        df = pd.concat(
            [canary_ret.shift(lag).rename("x"), y.rename("y")],
            axis=1
        ).dropna()

        if len(df) < 80:
            continue

        split = int(len(df) * (1 - test_size))
        train = df.iloc[:split]
        test = df.iloc[split:]

        if len(train) < 30 or len(test) < 20:
            continue

        model = LinearRegression()
        model.fit(train[["x"]], train["y"])
        pred = model.predict(test[["x"]])

        r2 = r2_score(test["y"], pred)
        hit = (np.sign(pred) == np.sign(test["y"])).mean()

        if pd.isna(best["r2"]) or r2 > best["r2"]:
            best = {
                "lag": lag,
                "r2": float(r2),
                "hit_rate": float(hit),
                "coef": float(model.coef_[0]),
                "intercept": float(model.intercept_),
            }

    return best


def event_study(target_ret: pd.Series, canary_ret: pd.Series, z_threshold: float = 1.5):
    mean = canary_ret.rolling(40).mean()
    std = canary_ret.rolling(40).std()
    z = (canary_ret - mean) / std
    fwd = target_ret.shift(-1)

    up = fwd[z > z_threshold].dropna()
    down = fwd[z < -z_threshold].dropna()

    return {
        "avg_fwd_after_up": float(up.mean()) if len(up) else np.nan,
        "avg_fwd_after_down": float(down.mean()) if len(down) else np.nan,
        "n_up": int(len(up)),
        "n_down": int(len(down)),
    }


def granger_score(target_ret: pd.Series, canary_ret: pd.Series, max_lag: int):
    """
    Test whether canary helps predict target.
    statsmodels expects array columns ordered [y, x],
    and tests whether x Granger-causes y.
    """
    df = safe_concat(target_ret, canary_ret, names=("y", "x"))
    if len(df) < max(60, max_lag * 12):
        return {
            "best_lag": np.nan,
            "best_pvalue": np.nan,
            "significant": False,
        }

    try:
        res = grangercausalitytests(df[["y", "x"]], maxlag=max_lag, verbose=False)
        rows = []
        for lag, out in res.items():
            pval = out[0]["ssr_ftest"][1]
            rows.append((lag, pval))

        gdf = pd.DataFrame(rows, columns=["lag", "pvalue"])
        idx = gdf["pvalue"].idxmin()
        best = gdf.loc[idx]
        return {
            "best_lag": int(best["lag"]),
            "best_pvalue": float(best["pvalue"]),
            "significant": bool(best["pvalue"] < 0.05),
        }
    except Exception:
        return {
            "best_lag": np.nan,
            "best_pvalue": np.nan,
            "significant": False,
        }


def rolling_lead_detection(target_ret: pd.Series, canary_ret: pd.Series, rolling_window: int, max_lag: int):
    """
    For each rolling window, find the lag with max absolute lagged correlation.
    """
    aligned = safe_concat(canary_ret, target_ret, names=("canary", "target"))
    if len(aligned) < rolling_window + max_lag + 5:
        return pd.DataFrame(columns=["date", "best_lag", "best_corr"])

    dates = []
    best_lags = []
    best_corrs = []

    for i in range(rolling_window + max_lag, len(aligned)):
        window = aligned.iloc[i - rolling_window:i].copy()

        lag_rows = []
        for lag in range(1, max_lag + 1):
            tmp = pd.concat(
                [
                    window["canary"].shift(lag).rename("x"),
                    window["target"].rename("y"),
                ],
                axis=1
            ).dropna()

            if len(tmp) < max(20, rolling_window // 2):
                continue

            corr = tmp["x"].corr(tmp["y"])
            lag_rows.append((lag, corr))

        if not lag_rows:
            continue

        lag_df = pd.DataFrame(lag_rows, columns=["lag", "corr"])
        idx = lag_df["corr"].abs().idxmax()
        best = lag_df.loc[idx]

        dates.append(aligned.index[i])
        best_lags.append(int(best["lag"]))
        best_corrs.append(float(best["corr"]))

    return pd.DataFrame({
        "date": dates,
        "best_lag": best_lags,
        "best_corr": best_corrs,
    })


def summarize_canaries(rets: pd.DataFrame, target: str, canaries, max_lag, rolling_window, test_size, z_threshold):
    target_ret = rets[target]
    results = []
    rolling_maps = {}

    for ticker in canaries:
        if ticker not in rets.columns:
            continue

        canary_ret = rets[ticker]

        corr_lag, corr_val, _ = best_lagged_corr(target_ret, canary_ret, max_lag)
        pred = predictive_regression(target_ret, canary_ret, max_lag, test_size)
        evt = event_study(target_ret, canary_ret, z_threshold)
        grg = granger_score(target_ret, canary_ret, max_lag)
        roll = rolling_lead_detection(target_ret, canary_ret, rolling_window, max_lag)
        rolling_maps[ticker] = roll

        if len(roll):
            mode_lag = roll["best_lag"].mode()
            rolling_dom_lag = int(mode_lag.iloc[0]) if len(mode_lag) else np.nan
            rolling_avg_corr = float(roll["best_corr"].mean())
        else:
            rolling_dom_lag = np.nan
            rolling_avg_corr = np.nan

        score = (
            (0 if pd.isna(pred["r2"]) else pred["r2"] * 100)
            + (0 if pd.isna(pred["hit_rate"]) else pred["hit_rate"] * 10)
            + (0 if pd.isna(corr_val) else abs(corr_val) * 5)
            + (6 if grg["significant"] else 0)
            + (0 if pd.isna(grg["best_pvalue"]) else max(0, 3 - 3 * grg["best_pvalue"]))
        )

        results.append({
            "ticker": ticker,
            "best_corr_lag": corr_lag,
            "best_corr": corr_val,
            "pred_lag": pred["lag"],
            "test_r2": pred["r2"],
            "hit_rate": pred["hit_rate"],
            "coef": pred["coef"],
            "granger_best_lag": grg["best_lag"],
            "granger_pvalue": grg["best_pvalue"],
            "granger_sig": grg["significant"],
            "rolling_dom_lag": rolling_dom_lag,
            "rolling_avg_corr": rolling_avg_corr,
            "avg_fwd_after_up": evt["avg_fwd_after_up"],
            "avg_fwd_after_down": evt["avg_fwd_after_down"],
            "n_up": evt["n_up"],
            "n_down": evt["n_down"],
            "score": score,
        })

    out = pd.DataFrame(results)
    if len(out):
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out, rolling_maps


# =========================
# Run app
# =========================
if run_button:
    tickers = [target] + canaries

    with st.spinner("Downloading prices and running lead analysis..."):
        prices = download_prices(tickers, period=period, interval=interval)

        if prices.empty or target not in prices.columns:
            st.error("No usable price data returned. Try fewer symbols or a shorter lookback.")
            st.stop()

        rets = compute_log_returns(prices)
        rets = rets.dropna(axis=1, how="all")

        if target not in rets.columns:
            st.error("Target ticker missing after return calculation.")
            st.stop()

        results, rolling_maps = summarize_canaries(
            rets=rets,
            target=target,
            canaries=canaries,
            max_lag=max_lag,
            rolling_window=rolling_window,
            test_size=test_size,
            z_threshold=z_threshold,
        )

    if results.empty:
        st.warning("No valid results. Try reducing max lag or shortening the symbol list.")
        st.stop()

    # Top summary
    top = results.iloc[0]
    st.subheader("Best current canary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker", top["ticker"])
    c2.metric("Predictive lag", "n/a" if pd.isna(top["pred_lag"]) else f"{int(top['pred_lag'])} bar(s)")
    c3.metric("Hit rate", "n/a" if pd.isna(top["hit_rate"]) else f"{top['hit_rate']:.1%}")
    c4.metric("Granger", "Yes" if bool(top["granger_sig"]) else "No")

    st.dataframe(
        results.style.format({
            "best_corr": "{:.3f}",
            "test_r2": "{:.4f}",
            "hit_rate": "{:.1%}",
            "coef": "{:.4f}",
            "granger_pvalue": "{:.4f}",
            "rolling_avg_corr": "{:.3f}",
            "avg_fwd_after_up": "{:.4%}",
            "avg_fwd_after_down": "{:.4%}",
            "score": "{:.2f}",
        }),
        use_container_width=True,
        height=420,
    )

    # Price chart
    st.subheader("Price context")
    px_df = prices[[col for col in [target] + canaries[:5] if col in prices.columns]].copy()
    px_df = px_df.reset_index().melt(id_vars=px_df.index.name or "Date", var_name="Ticker", value_name="Price")
    x_name = px_df.columns[0]
    fig_price = px.line(px_df, x=x_name, y="Price", color="Ticker")
    st.plotly_chart(fig_price, use_container_width=True)

    # Rolling lead detection
    st.subheader("Rolling lead detection")
    selected = st.selectbox("Show rolling lead for", results["ticker"].tolist(), index=0)
    roll = rolling_maps.get(selected, pd.DataFrame())

    if len(roll):
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=roll["date"],
            y=roll["best_lag"],
            mode="lines",
            name="Best lag",
        ))
        fig_roll.update_layout(
            xaxis_title="Date",
            yaxis_title="Best lag (bars)",
            height=350,
        )
        st.plotly_chart(fig_roll, use_container_width=True)

        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(
            x=roll["date"],
            y=roll["best_corr"],
            mode="lines",
            name="Rolling best corr",
        ))
        fig_corr.update_layout(
            xaxis_title="Date",
            yaxis_title="Correlation",
            height=350,
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough data for rolling lead detection with the current window.")

    # Interpretation block
    st.subheader("Interpretation")
    st.markdown(
        f"""
- **Best canary right now:** `{top['ticker']}`
- **Best predictive lag:** `{top['pred_lag']}` bar(s)
- **Directional hit rate:** `{top['hit_rate']:.1%}` if available
- **Granger significant:** `{bool(top['granger_sig'])}`
- **Rolling dominant lag:** `{top['rolling_dom_lag']}` bar(s)

Use this as a **leader board**, not a guarantee. A canary can lead for one regime and then stop.
        """
    )

else:
    st.info("Set your inputs on the left and click **Run analysis**.")
