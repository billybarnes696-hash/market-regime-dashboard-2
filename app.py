import warnings
warnings.filterwarnings("ignore")

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import grangercausalitytests


# =========================
# Streamlit setup
# =========================
st.set_page_config(page_title="SOXL Canary Lead Detector", layout="wide")
st.title("SOXL Canary Lead Detector")
st.caption("Granger causality + rolling lead detection + pinning / exhaustion model")


# =========================
# Sidebar controls
# =========================
DEFAULT_CANARIES = [
    "NVDA", "AMD", "AVGO", "SMH", "SOXX", "TSM",
    "MU", "AMAT", "LRCX", "KLAC", "ASML", "INTC"
]

target = st.sidebar.text_input("Target ticker", value="SOXL").upper().strip()
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
min_pinned_bars = st.sidebar.slider("Minimum pinned bars", min_value=2, max_value=8, value=3)
show_top_n = st.sidebar.slider("Top symbols in price chart", min_value=3, max_value=10, value=6)
run_button = st.sidebar.button("Run analysis", type="primary")


# =========================
# Data utilities
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def download_prices(tickers: List[str], period: str, interval: str) -> pd.DataFrame:
    """
    Safer downloader: requests one symbol at a time so a single Yahoo failure/rate limit
    does not kill the whole batch.
    """
    frames = []

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df is None or df.empty or "Close" not in df.columns:
                continue

            s = df["Close"].rename(ticker)
            frames.append(s)

            # Small pause helps reduce throttling on Streamlit Cloud.
            time.sleep(0.25)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index().ffill()
    out = out.dropna(how="all")
    return out


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    rets = np.log(price_df / price_df.shift(1))
    return rets.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def safe_concat(a: pd.Series, b: pd.Series, names=("x", "y")) -> pd.DataFrame:
    return pd.concat([a.rename(names[0]), b.rename(names[1])], axis=1).dropna()


# =========================
# Indicator helpers
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_tsi(close: pd.Series, fast: int = 4, slow: int = 2, signal: int = 4):
    mtm = close.diff()
    abs_mtm = mtm.abs()

    tsi_num = ema(ema(mtm, slow), fast)
    tsi_den = ema(ema(abs_mtm, slow), fast)
    tsi = 100 * (tsi_num / tsi_den.replace(0, np.nan))
    tsi_signal = ema(tsi, signal)
    return tsi, tsi_signal


def compute_cci(close: pd.Series, length: int = 15) -> pd.Series:
    sma = close.rolling(length).mean()
    mad = (close - sma).abs().rolling(length).mean()
    cci = (close - sma) / (0.015 * mad.replace(0, np.nan))
    return cci


def compute_bbpct(close: pd.Series, length: int = 20, n_std: float = 2.0) -> pd.Series:
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    bbpct = (close - lower) / (upper - lower).replace(0, np.nan)
    return bbpct


def consecutive_true_count(flag: pd.Series) -> pd.Series:
    out = np.zeros(len(flag), dtype=int)
    count = 0
    vals = flag.fillna(False).astype(bool).values

    for i, v in enumerate(vals):
        if v:
            count += 1
        else:
            count = 0
        out[i] = count

    return pd.Series(out, index=flag.index)


def compute_pinning_features(close: pd.Series) -> pd.DataFrame:
    tsi_424, tsi_sig = compute_tsi(close, fast=4, slow=2, signal=4)
    cci_15 = compute_cci(close, length=15)
    bbpct_20_2 = compute_bbpct(close, length=20, n_std=2.0)
    ema10 = ema(close, 10)

    tsi_pinned_flag = tsi_424 >= 95
    bb_pinned_flag = bbpct_20_2 >= 0.98
    consecutive_pinned_bars = consecutive_true_count(tsi_pinned_flag)

    cci_slope_3 = cci_15 - cci_15.shift(3)
    tsi_flat_high = tsi_pinned_flag & ((tsi_424 - tsi_424.shift(3)).abs() <= 2.0)
    price_change_while_tsi_flat = close.pct_change(3)
    price_flat_up_flag = price_change_while_tsi_flat >= 0
    cci_falling_flag = cci_slope_3 < 0

    divergence_flag = price_flat_up_flag & tsi_flat_high & cci_falling_flag

    out = pd.DataFrame({
        "close": close,
        "TSI_424": tsi_424,
        "TSI_424_signal": tsi_sig,
        "TSI_424_pinned_flag": tsi_pinned_flag.astype(int),
        "consecutive_pinned_bars": consecutive_pinned_bars,
        "CCI_15": cci_15,
        "CCI_slope_3": cci_slope_3,
        "BBPct_20_2": bbpct_20_2,
        "BBPct_20_2_pinned_flag": bb_pinned_flag.astype(int),
        "distance_from_ema10": (close / ema10) - 1.0,
        "price_change_while_tsi_flat": price_change_while_tsi_flat,
        "divergence_flag": divergence_flag.astype(int),
    })

    return out


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
            axis=1,
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
                axis=1,
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


def pinning_event_study(target_ret: pd.Series, close: pd.Series, min_pinned_bars: int = 3) -> Tuple[dict, pd.DataFrame]:
    feats = compute_pinning_features(close)

    event_flag = (
        (feats["TSI_424_pinned_flag"] == 1) &
        (feats["consecutive_pinned_bars"] >= min_pinned_bars) &
        (feats["BBPct_20_2_pinned_flag"] == 1) &
        (feats["divergence_flag"] == 1)
    )

    fwd1 = target_ret.shift(-1)
    fwd2 = target_ret.shift(-2).rolling(2).sum()

    hit_short_1 = (fwd1[event_flag] < 0).mean() if event_flag.sum() else np.nan
    hit_short_2 = (fwd2[event_flag] < 0).mean() if event_flag.sum() else np.nan

    avg_fwd1 = fwd1[event_flag].mean() if event_flag.sum() else np.nan
    avg_fwd2 = fwd2[event_flag].mean() if event_flag.sum() else np.nan

    latest = feats.iloc[-1] if len(feats) else None

    summary = {
        "pin_event_count": int(event_flag.sum()),
        "pin_hit_rate_1bar": float(hit_short_1) if pd.notna(hit_short_1) else np.nan,
        "pin_hit_rate_2bar": float(hit_short_2) if pd.notna(hit_short_2) else np.nan,
        "pin_avg_fwd1": float(avg_fwd1) if pd.notna(avg_fwd1) else np.nan,
        "pin_avg_fwd2": float(avg_fwd2) if pd.notna(avg_fwd2) else np.nan,
        "latest_TSI_424": float(latest["TSI_424"]) if latest is not None and pd.notna(latest["TSI_424"]) else np.nan,
        "latest_TSI_pinned": int(latest["TSI_424_pinned_flag"]) if latest is not None else 0,
        "latest_consecutive_pinned_bars": int(latest["consecutive_pinned_bars"]) if latest is not None and pd.notna(latest["consecutive_pinned_bars"]) else 0,
        "latest_CCI_15": float(latest["CCI_15"]) if latest is not None and pd.notna(latest["CCI_15"]) else np.nan,
        "latest_CCI_slope_3": float(latest["CCI_slope_3"]) if latest is not None and pd.notna(latest["CCI_slope_3"]) else np.nan,
        "latest_BBPct_20_2": float(latest["BBPct_20_2"]) if latest is not None and pd.notna(latest["BBPct_20_2"]) else np.nan,
        "latest_dist_from_ema10": float(latest["distance_from_ema10"]) if latest is not None and pd.notna(latest["distance_from_ema10"]) else np.nan,
        "latest_price_change_while_tsi_flat": float(latest["price_change_while_tsi_flat"]) if latest is not None and pd.notna(latest["price_change_while_tsi_flat"]) else np.nan,
        "latest_divergence_flag": int(latest["divergence_flag"]) if latest is not None else 0,
    }

    feats = feats.copy()
    feats["pin_event_flag"] = event_flag.astype(int)

    return summary, feats


def summarize_canaries(
    prices: pd.DataFrame,
    rets: pd.DataFrame,
    target: str,
    canaries: List[str],
    max_lag: int,
    rolling_window: int,
    test_size: float,
    z_threshold: float,
    min_pinned_bars: int,
):
    target_ret = rets[target]
    results = []
    rolling_maps: Dict[str, pd.DataFrame] = {}
    pinning_maps: Dict[str, pd.DataFrame] = {}

    for ticker in canaries:
        if ticker not in rets.columns or ticker not in prices.columns:
            continue

        canary_ret = rets[ticker]
        canary_close = prices[ticker]

        corr_lag, corr_val, _ = best_lagged_corr(target_ret, canary_ret, max_lag)
        pred = predictive_regression(target_ret, canary_ret, max_lag, test_size)
        evt = event_study(target_ret, canary_ret, z_threshold)
        grg = granger_score(target_ret, canary_ret, max_lag)
        roll = rolling_lead_detection(target_ret, canary_ret, rolling_window, max_lag)
        pin_sum, pin_feats = pinning_event_study(target_ret, canary_close, min_pinned_bars=min_pinned_bars)

        rolling_maps[ticker] = roll
        pinning_maps[ticker] = pin_feats

        if len(roll):
            mode_lag = roll["best_lag"].mode()
            rolling_dom_lag = int(mode_lag.iloc[0]) if len(mode_lag) else np.nan
            rolling_avg_corr = float(roll["best_corr"].mean())
        else:
            rolling_dom_lag = np.nan
            rolling_avg_corr = np.nan

        pinning_score = 0.0
        if pd.notna(pin_sum["pin_hit_rate_1bar"]):
            pinning_score += pin_sum["pin_hit_rate_1bar"] * 4
        if pd.notna(pin_sum["pin_hit_rate_2bar"]):
            pinning_score += pin_sum["pin_hit_rate_2bar"] * 4
        if pin_sum["latest_divergence_flag"] == 1:
            pinning_score += 2
        if pin_sum["latest_TSI_pinned"] == 1:
            pinning_score += 1

        score = (
            (0 if pd.isna(pred["r2"]) else pred["r2"] * 100)
            + (0 if pd.isna(pred["hit_rate"]) else pred["hit_rate"] * 10)
            + (0 if pd.isna(corr_val) else abs(corr_val) * 5)
            + (6 if grg["significant"] else 0)
            + (0 if pd.isna(grg["best_pvalue"]) else max(0, 3 - 3 * grg["best_pvalue"]))
            + pinning_score
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
            "pin_event_count": pin_sum["pin_event_count"],
            "pin_hit_rate_1bar": pin_sum["pin_hit_rate_1bar"],
            "pin_hit_rate_2bar": pin_sum["pin_hit_rate_2bar"],
            "pin_avg_fwd1": pin_sum["pin_avg_fwd1"],
            "pin_avg_fwd2": pin_sum["pin_avg_fwd2"],
            "latest_TSI_424": pin_sum["latest_TSI_424"],
            "latest_TSI_pinned": pin_sum["latest_TSI_pinned"],
            "latest_consecutive_pinned_bars": pin_sum["latest_consecutive_pinned_bars"],
            "latest_CCI_15": pin_sum["latest_CCI_15"],
            "latest_CCI_slope_3": pin_sum["latest_CCI_slope_3"],
            "latest_BBPct_20_2": pin_sum["latest_BBPct_20_2"],
            "latest_dist_from_ema10": pin_sum["latest_dist_from_ema10"],
            "latest_price_change_while_tsi_flat": pin_sum["latest_price_change_while_tsi_flat"],
            "latest_divergence_flag": pin_sum["latest_divergence_flag"],
            "score": score,
        })

    out = pd.DataFrame(results)
    if len(out):
        out = out.sort_values("score", ascending=False).reset_index(drop=True)

    return out, rolling_maps, pinning_maps


# =========================
# Visualization helpers
# =========================
def build_price_chart(prices: pd.DataFrame, tickers_to_show: List[str]):
    chart_df = prices[[c for c in tickers_to_show if c in prices.columns]].copy()
    if chart_df.empty:
        return None

    chart_df.index.name = "Time"
    chart_df = chart_df.reset_index()
    id_col = chart_df.columns[0]
    chart_df = chart_df.melt(id_vars=id_col, var_name="Ticker", value_name="Price")

    fig = px.line(chart_df, x=id_col, y="Price", color="Ticker")
    fig.update_layout(height=420)
    return fig


def build_indicator_chart(pinning_df: pd.DataFrame, symbol: str):
    if pinning_df.empty:
        return None

    chart_df = pinning_df.copy()
    chart_df.index.name = "Time"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["TSI_424"], mode="lines", name=f"{symbol} TSI 4,2,4"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["CCI_15"], mode="lines", name=f"{symbol} CCI 15", yaxis="y2"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BBPct_20_2"], mode="lines", name=f"{symbol} BB% 20,2", yaxis="y3"))

    pin_events = chart_df[chart_df["pin_event_flag"] == 1]
    if not pin_events.empty:
        fig.add_trace(
            go.Scatter(
                x=pin_events.index,
                y=pin_events["TSI_424"],
                mode="markers",
                name="Pin events",
                marker=dict(size=9, symbol="diamond"),
            )
        )

    fig.update_layout(
        height=450,
        yaxis=dict(title="TSI"),
        yaxis2=dict(title="CCI", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="BB%", overlaying="y", side="right", anchor="free", position=0.95, showgrid=False),
        legend=dict(orientation="h"),
        margin=dict(l=50, r=70, t=40, b=40),
    )
    return fig


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

        loaded = list(prices.columns)
        skipped = [t for t in tickers if t not in loaded]
        if skipped:
            st.warning(f"Skipped tickers due to missing data or rate limits: {', '.join(skipped)}")

        rets = compute_log_returns(prices)
        rets = rets.dropna(axis=1, how="all")

        if target not in rets.columns:
            st.error("Target ticker missing after return calculation.")
            st.stop()

        results, rolling_maps, pinning_maps = summarize_canaries(
            prices=prices,
            rets=rets,
            target=target,
            canaries=[c for c in canaries if c in prices.columns and c in rets.columns],
            max_lag=max_lag,
            rolling_window=rolling_window,
            test_size=test_size,
            z_threshold=z_threshold,
            min_pinned_bars=min_pinned_bars,
        )

    if results.empty:
        st.warning("No valid results. Try reducing max lag or shortening the symbol list.")
        st.stop()

    top = results.iloc[0]
    st.subheader("Best current canary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ticker", str(top["ticker"]))
    c2.metric("Predictive lag", "n/a" if pd.isna(top["pred_lag"]) else f"{int(top['pred_lag'])} bar(s)")
    c3.metric("Hit rate", "n/a" if pd.isna(top["hit_rate"]) else f"{top['hit_rate']:.1%}")
    c4.metric("Granger", "Yes" if bool(top["granger_sig"]) else "No")
    c5.metric("Pinned now", "Yes" if int(top["latest_TSI_pinned"]) == 1 else "No")

    st.subheader("Canary ranking")
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
            "pin_hit_rate_1bar": "{:.1%}",
            "pin_hit_rate_2bar": "{:.1%}",
            "pin_avg_fwd1": "{:.4%}",
            "pin_avg_fwd2": "{:.4%}",
            "latest_TSI_424": "{:.1f}",
            "latest_CCI_15": "{:.1f}",
            "latest_CCI_slope_3": "{:.1f}",
            "latest_BBPct_20_2": "{:.2f}",
            "latest_dist_from_ema10": "{:.2%}",
            "latest_price_change_while_tsi_flat": "{:.2%}",
            "score": "{:.2f}",
        }),
        width="stretch",
        height=460,
    )

    st.subheader("Price context")
    tickers_to_show = [target] + results["ticker"].head(show_top_n - 1).tolist()
    fig_price = build_price_chart(prices, tickers_to_show)
    if fig_price is not None:
        st.plotly_chart(fig_price, width="stretch")

    st.subheader("Rolling lead detection")
    selected = st.selectbox("Show rolling lead for", results["ticker"].tolist(), index=0)
    roll = rolling_maps.get(selected, pd.DataFrame())

    if len(roll):
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=roll["date"], y=roll["best_lag"], mode="lines", name="Best lag"))
        fig_roll.update_layout(xaxis_title="Date", yaxis_title="Best lag (bars)", height=340)
        st.plotly_chart(fig_roll, width="stretch")

        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(x=roll["date"], y=roll["best_corr"], mode="lines", name="Rolling best corr"))
        fig_corr.update_layout(xaxis_title="Date", yaxis_title="Correlation", height=340)
        st.plotly_chart(fig_corr, width="stretch")
    else:
        st.info("Not enough data for rolling lead detection with the current window.")

    st.subheader("Pinning / exhaustion panel")
    pin_df = pinning_maps.get(selected, pd.DataFrame())
    fig_pin = build_indicator_chart(pin_df, selected)
    if fig_pin is not None:
        st.plotly_chart(fig_pin, width="stretch")

    row = results.loc[results["ticker"] == selected].iloc[0]
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("TSI 4,2,4", "n/a" if pd.isna(row["latest_TSI_424"]) else f"{row['latest_TSI_424']:.1f}")
    p2.metric("Pinned bars", int(row["latest_consecutive_pinned_bars"]))
    p3.metric("CCI slope 3", "n/a" if pd.isna(row["latest_CCI_slope_3"]) else f"{row['latest_CCI_slope_3']:.1f}")
    p4.metric("BB% 20,2", "n/a" if pd.isna(row["latest_BBPct_20_2"]) else f"{row['latest_BBPct_20_2']:.2f}")
    p5.metric("Divergence", "Yes" if int(row["latest_divergence_flag"]) == 1 else "No")

    st.markdown("### Interpretation")
    st.markdown(
        f"""
- **Best current canary:** `{top['ticker']}`
- **Best predictive lag:** `{top['pred_lag']}` bar(s)
- **Directional hit rate:** `{top['hit_rate']:.1%}` if available
- **Granger significant:** `{bool(top['granger_sig'])}`
- **Rolling dominant lag:** `{top['rolling_dom_lag']}` bar(s)
- **Pinned exhaustion events for `{selected}`:** `{int(row['pin_event_count'])}`
- **1-bar SOXL fade rate after pin event:** `{row['pin_hit_rate_1bar']:.1%}` if available
- **2-bar SOXL fade rate after pin event:** `{row['pin_hit_rate_2bar']:.1%}` if available

This is built to catch your specific pattern: **TSI 4,2,4 pinned high, BB% stretched, price flat/up, and CCI fading**.
        """
    )
else:
    st.info("Set your inputs on the left and click **Run analysis**.")
