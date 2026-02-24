# app.py
# Streamlit: Optionable ETF EOD Screener + Backtest + StockCharts-style multi-panel chart
#
# What it does:
# - Builds an ETF universe (auto from StockAnalysis or manual tickers)
# - Filters to "optionable" ETFs (Yahoo options chain exists via yfinance)
# - Computes daily EOD score + probability (TSI/RSI/CCI/W%R/CMF + VWAP-proxy stretch + candle exhaust + simple divergence)
# - Backtests: next-day CLOSE drop >= X% after SCORE >= min_score
# - Shows a ranked screener table + filters (only diamonds, min historical accuracy)
# - Shows a StockCharts-like stacked chart for a selected ETF
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import pandas_ta as ta
from bs4 import BeautifulSoup
from datetime import date, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Optionable ETF EOD Screener", layout="wide")

# -----------------------------
# Universe: pull ETF tickers
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fetch_all_etf_tickers_from_stockanalysis() -> list[str]:
    """
    Pulls ETF tickers from StockAnalysis ETF directory page.
    This is convenient for building a broad ETF universe.
    """
    url = "https://stockanalysis.com/etf/"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # StockAnalysis generally has links like /etf/XYZ/
    tickers: set[str] = set()
    for a in soup.select('a[href^="/etf/"]'):
        href = a.get("href", "")
        parts = href.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "etf":
            t = parts[1].upper()
            if 1 <= len(t) <= 6 and t.isalnum():
                tickers.add(t)

    return sorted(tickers)


# -----------------------------
# Data fetch (daily bars)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_daily_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns=str.title)
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index)
    return df


# -----------------------------
# Optionable check (proxy)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def is_optionable_yf(ticker: str) -> bool:
    """
    Practical proxy: if Yahoo Finance exposes option expirations, treat as optionable.
    """
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        return bool(exps and len(exps) > 0)
    except Exception:
        return False


# -----------------------------
# Indicators + scoring
# -----------------------------
def compute_features(
    df: pd.DataFrame,
    tsi_fast=6, tsi_slow=3, tsi_signal=6,
    rsi_len=14, cci_len=20, willr_len=14,
    cmf_len=20,
    vwap_len=20,
    vwap_1=0.01, vwap_2=0.02,
    wick_thresh=0.50,
    min_score_trigger=6,
) -> pd.DataFrame:
    """
    Computes daily features + SCORE/PROB_PCT.
    Notes:
      - VWAP_PROXY is a *daily* volume-weighted rolling average (not true intraday VWAP).
      - Divergence is a simple 1-bar bearish divergence (higher close, lower RSI).
    """
    out = df.copy()

    # TSI (pandas_ta returns 2 cols: TSI_... and TSIs_... depending)
    tsi_df = ta.tsi(out["Close"], fast=tsi_fast, slow=tsi_slow, signal=tsi_signal)
    if tsi_df is not None and not tsi_df.empty:
        out = out.join(tsi_df)

    out["RSI"] = ta.rsi(out["Close"], length=rsi_len)
    out["CCI"] = ta.cci(out["High"], out["Low"], out["Close"], length=cci_len)
    out["WILLR"] = ta.willr(out["High"], out["Low"], out["Close"], length=willr_len)
    out["CMF"] = ta.cmf(out["High"], out["Low"], out["Close"], out["Volume"], length=cmf_len)

    # VWAP proxy (daily): sum(close*vol)/sum(vol) over vwap_len
    pv = out["Close"] * out["Volume"]
    out["VWAP_PROXY"] = pv.rolling(vwap_len).sum() / out["Volume"].rolling(vwap_len).sum()

    # VWAP stretch points 0/1/2
    out["VW_STRETCH"] = 0
    out.loc[out["Close"] > out["VWAP_PROXY"] * (1 + vwap_2), "VW_STRETCH"] = 2
    out.loc[
        (out["Close"] > out["VWAP_PROXY"] * (1 + vwap_1)) &
        (out["Close"] <= out["VWAP_PROXY"] * (1 + vwap_2)),
        "VW_STRETCH"
    ] = 1

    # Candle exhaustion: upper wick / range >= threshold
    upper_wick = out["High"] - np.maximum(out["Open"], out["Close"])
    rng = (out["High"] - out["Low"]).replace(0, np.nan)
    out["UPPER_WICK_PCT"] = (upper_wick / rng).clip(lower=0, upper=1)
    out["CANDLE_EXHAUST"] = (out["UPPER_WICK_PCT"] >= wick_thresh).astype(int)

    # Pull a usable "TSI" column
    tsi_main = [c for c in out.columns if c.lower().startswith("tsi_") and not c.lower().startswith("tsis_")]
    tsi_sig = [c for c in out.columns if c.lower().startswith("tsis_")]
    tsi_col = tsi_main[0] if tsi_main else (tsi_sig[0] if tsi_sig else None)
    out["TSI"] = out[tsi_col] if tsi_col else np.nan

    # Simple bearish divergence (1-bar): higher close, RSI lower
    out["BEAR_DIV"] = ((out["Close"] > out["Close"].shift(1)) & (out["RSI"] < out["RSI"].shift(1))).astype(int)

    # Binary conditions (extremes)
    out["S_TSI"] = (out["TSI"] > 95).astype(int)
    out["S_RSI"] = (out["RSI"] > 70).astype(int)
    out["S_CCI"] = (out["CCI"] > 100).astype(int)
    out["S_WILLR"] = (out["WILLR"] > -20).astype(int)  # W%R ranges [-100, 0]; overbought near 0
    out["S_CMF"] = (out["CMF"] < 0).astype(int)

    # Score: 5 binaries + VW(0/1/2) + candle(0/1) + div(0/1) => max 9
    out["SCORE"] = (
        out["S_TSI"] + out["S_RSI"] + out["S_CCI"] + out["S_WILLR"] + out["S_CMF"] +
        out["VW_STRETCH"] + out["CANDLE_EXHAUST"] + out["BEAR_DIV"]
    )
    out["MAX_SCORE"] = 9
    out["PROB_PCT"] = (out["SCORE"] / out["MAX_SCORE"] * 100).clip(0, 100)
    out["SIGNAL"] = (out["SCORE"] >= min_score_trigger)

    return out


def backtest_next_day_drop(out: pd.DataFrame, min_score: int, drop_pct: float) -> dict:
    """
    Backtest success definition:
      - SIGNAL day: SCORE >= min_score at today's close
      - HIT: next day CLOSE return <= -drop_pct
    """
    df = out.copy()
    df["NEXT_CLOSE"] = df["Close"].shift(-1)
    df["NEXT_RET_PCT"] = (df["NEXT_CLOSE"] / df["Close"] - 1) * 100
    df["SIGNAL"] = df["SCORE"] >= min_score
    df["HIT"] = df["SIGNAL"] & (df["NEXT_RET_PCT"] <= -drop_pct)

    signals = int(df["SIGNAL"].sum(skipna=True))
    wins = int(df["HIT"].sum(skipna=True))
    acc = (wins / signals * 100) if signals > 0 else np.nan

    latest = df.iloc[-1].copy()
    return {"df": df, "signals": signals, "wins": wins, "accuracy": acc, "latest": latest}


# -----------------------------
# StockCharts-style chart
# -----------------------------
def stockcharts_style_figure(df: pd.DataFrame, title: str, signal_score: int) -> go.Figure:
    dfp = df.dropna(subset=["Open", "High", "Low", "Close"]).tail(220).copy()
    has_prob = "PROB_PCT" in dfp.columns

    rows = 7 + (1 if has_prob else 0)
    row_heights = [0.38, 0.10, 0.10, 0.08, 0.08, 0.08, 0.10] + ([0.08] if has_prob else [])
    row_titles = [
        "Price (Candles) + VWAP Proxy",
        "Volume",
        "TSI",
        "RSI",
        "CCI",
        "Williams %R",
        "CMF",
    ] + (["Probability %"] if has_prob else [])

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=row_titles,
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=dfp.index, open=dfp["Open"], high=dfp["High"], low=dfp["Low"], close=dfp["Close"],
            name="Price",
        ),
        row=1, col=1
    )

    # VWAP proxy overlay
    if "VWAP_PROXY" in dfp.columns:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["VWAP_PROXY"], mode="lines", name="VWAP Proxy"), row=1, col=1)

    # Signal markers for SCORE >= signal_score
    if "SCORE" in dfp.columns:
        sig = dfp["SCORE"] >= signal_score
        fig.add_trace(
            go.Scatter(
                x=dfp.index[sig], y=dfp.loc[sig, "High"] * 1.01,
                mode="markers", name="Signal", marker=dict(symbol="triangle-down", size=9),
            ),
            row=1, col=1
        )

    # Volume
    fig.add_trace(go.Bar(x=dfp.index, y=dfp["Volume"], name="Volume"), row=2, col=1)

    # TSI
    if "TSI" in dfp.columns:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["TSI"], mode="lines", name="TSI"), row=3, col=1)
        fig.add_hline(y=95, row=3, col=1, line_dash="dot")

    # RSI
    if "RSI" in dfp.columns:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["RSI"], mode="lines", name="RSI"), row=4, col=1)
        fig.add_hline(y=70, row=4, col=1, line_dash="dot")
        fig.add_hline(y=30, row=4, col=1, line_dash="dot")

    # CCI
    if "CCI" in dfp.columns:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["CCI"], mode="lines", name="CCI"), row=5, col=1)
        fig.add_hline(y=100, row=5, col=1, line_dash="dot")
        fig.add_hline(y=-100, row=5, col=1, line_dash="dot")

    # Williams %R
    if "WILLR" in dfp.columns:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["WILLR"], mode="lines", name="Williams %R"), row=6, col=1)
        fig.add_hline(y=-20, row=6, col=1, line_dash="dot")
        fig.add_hline(y=-80, row=6, col=1, line_dash="dot")

    # CMF
    if "CMF" in dfp.columns:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["CMF"], mode="lines", name="CMF"), row=7, col=1)
        fig.add_hline(y=0, row=7, col=1, line_dash="dot")

    # Probability
    if has_prob:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["PROB_PCT"], mode="lines", name="Prob%"), row=8, col=1)
        fig.add_hline(y=90, row=8, col=1, line_dash="dot")

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=1100 if has_prob else 980,
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="left",
        legend_x=0,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    for r in range(1, rows + 1):
        fig.update_yaxes(showgrid=True, row=r, col=1)

    return fig


# -----------------------------
# UI
# -----------------------------
st.title("Optionable ETFs — EOD Probability Screener (Daily)")
st.caption("Scores ETFs on daily bars and backtests next-day CLOSE drop ≥ X%. No option chain/pricing/execution.")

with st.sidebar:
    st.header("Universe")
    use_auto_universe = st.checkbox("Auto-load ETF tickers (StockAnalysis)", value=True)
    max_etfs = st.slider("Max ETFs to scan this run", 25, 1200, 250, 25)
    manual_tickers = st.text_area("Manual tickers (if filled, overrides auto)", value="", height=70)

    st.header("Backtest")
    years = st.slider("Years of daily history", 3, 20, 10)
    min_score = st.slider("Min score to trigger", 3, 9, 6)
    drop_pct = st.slider("Next-day drop threshold (%)", 0.5, 3.0, 1.0, 0.1)

    st.header("Indicator params (daily)")
    tsi_fast = st.number_input("TSI fast", 1, 50, 6)
    tsi_slow = st.number_input("TSI slow", 1, 50, 3)
    tsi_signal = st.number_input("TSI signal", 1, 50, 6)
    rsi_len = st.number_input("RSI length", 2, 50, 14)
    cci_len = st.number_input("CCI length", 5, 50, 20)
    willr_len = st.number_input("Williams %R length", 5, 50, 14)
    cmf_len = st.number_input("CMF length", 5, 50, 20)

    st.header("VWAP proxy + candle")
    vwap_len = st.number_input("VWAP proxy length", 5, 100, 20)
    vwap_1 = st.number_input("VWAP stretch 1 (%)", 0.1, 10.0, 1.0, 0.1) / 100
    vwap_2 = st.number_input("VWAP stretch 2 (%)", 0.1, 10.0, 2.0, 0.1) / 100
    wick_thresh = st.number_input("Upper-wick exhaustion threshold", 0.1, 0.9, 0.50, 0.05)

    st.header("Screener filters")
    diamond_thr = st.slider("Diamond probability threshold (%)", 50, 100, 90)
    show_only_diamonds = st.checkbox("Show only Diamonds", value=False)
    min_hist_acc = st.slider("Min historical accuracy %", 0, 100, 0)

    st.header("Performance")
    optionable_check_limit = st.slider("Optionable check: max tickers to test", 25, 1200, 250, 25)

    run_btn = st.button("Run ETF Screener", type="primary")


if not run_btn:
    st.info("Configure the universe and parameters in the sidebar, then click **Run ETF Screener**.")
    st.stop()


# Universe selection
manual = [t.strip().upper() for t in manual_tickers.replace(",", " ").split() if t.strip()]
manual = list(dict.fromkeys(manual))

if manual:
    tickers = manual
else:
    if not use_auto_universe:
        st.error("Provide manual tickers or enable auto-load.")
        st.stop()
    all_etfs = fetch_all_etf_tickers_from_stockanalysis()
    tickers = all_etfs[:max_etfs]

# Date window
end_dt = date.today()
start_dt = end_dt - timedelta(days=365 * years + 30)

st.write(f"Universe: **{len(tickers)} ETFs** (this run). Filtering to **optionable** ETFs (options chain exists).")

# Step 1: optionable filter (limit to avoid super long runs)
tickers_for_optionable_check = tickers[:optionable_check_limit]
if len(tickers) > optionable_check_limit:
    st.warning(f"Optionable check limited to first {optionable_check_limit} tickers for runtime. "
               f"Increase the slider if you want to scan more.")

optionable: list[str] = []
progress = st.progress(0)
for i, tkr in enumerate(tickers_for_optionable_check, start=1):
    progress.progress(i / len(tickers_for_optionable_check))
    if is_optionable_yf(tkr):
        optionable.append(tkr)
progress.empty()

st.write(f"Found **{len(optionable)} optionable ETFs** in this batch.")

# Step 2: compute scores + backtests
rows: list[dict] = []
details: dict[str, dict] = {}

progress = st.progress(0)
for i, tkr in enumerate(optionable, start=1):
    progress.progress(i / max(1, len(optionable)))

    df = fetch_daily_ohlcv(tkr, start_dt.isoformat(), (end_dt + timedelta(days=1)).isoformat())
    if df.empty or len(df) < 160:
        continue

    feat = compute_features(
        df,
        tsi_fast=tsi_fast, tsi_slow=tsi_slow, tsi_signal=tsi_signal,
        rsi_len=rsi_len, cci_len=cci_len, willr_len=willr_len,
        cmf_len=cmf_len,
        vwap_len=vwap_len,
        vwap_1=vwap_1, vwap_2=vwap_2,
        wick_thresh=wick_thresh,
        min_score_trigger=min_score,
    )
    bt = backtest_next_day_drop(feat, min_score=min_score, drop_pct=drop_pct)
    latest = bt["latest"]

    prob_today = float(latest.get("PROB_PCT", np.nan))
    score_today = int(latest.get("SCORE", 0))

    rows.append({
        "Ticker": tkr,
        "Prob% (today)": prob_today,
        "Score (today)": score_today,
        "Signals": bt["signals"],
        "Wins": bt["wins"],
        "Accuracy%": float(bt["accuracy"]) if bt["signals"] > 0 else np.nan,

        # diagnostics
        "Close": float(latest.get("Close", np.nan)),
        "TSI": float(latest.get("TSI", np.nan)),
        "RSI": float(latest.get("RSI", np.nan)),
        "CCI": float(latest.get("CCI", np.nan)),
        "W%R": float(latest.get("WILLR", np.nan)),
        "CMF": float(latest.get("CMF", np.nan)),
        "VW_stretch_pts": int(latest.get("VW_STRETCH", 0)),
        "CandleExhaust": int(latest.get("CANDLE_EXHAUST", 0)),
        "BearDiv": int(latest.get("BEAR_DIV", 0)),
    })
    details[tkr] = bt

progress.empty()

if not rows:
    st.warning("No results (missing data or filters too strict). Try increasing max ETFs / history window.")
    st.stop()

res = pd.DataFrame(rows)
res = res.sort_values(
    ["Prob% (today)", "Score (today)", "Accuracy%"],
    ascending=[False, False, False],
    na_position="last"
).reset_index(drop=True)

# ---- optional screener filters ----
if show_only_diamonds:
    res = res[res["Prob% (today)"] >= float(diamond_thr)]

if min_hist_acc > 0:
    res = res[res["Accuracy%"].notna() & (res["Accuracy%"] >= float(min_hist_acc))]

res = res.reset_index(drop=True)

if res.empty:
    st.warning("No ETFs match your current filters. Try lowering the diamond threshold or min accuracy filter.")
    st.stop()

# Summary tiles
c1, c2, c3, c4 = st.columns(4)
c1.metric("Optionable ETFs scanned", f"{len(optionable)}")
c2.metric("Results shown", f"{len(res)}")
c3.metric(f"Diamonds (Prob ≥ {diamond_thr}%)", f"{int((res['Prob% (today)'] >= float(diamond_thr)).sum())}")
c4.metric("Top Prob% today", f"{float(res['Prob% (today)'].max()):.1f}%")

# Screener table
st.subheader("Screener Results")

def highlight(row):
    if pd.isna(row["Prob% (today)"]):
        return [""] * len(row)
    if row["Prob% (today)"] >= float(diamond_thr):
        return ["background-color: #ffd6d6"] * len(row)  # diamonds
    if row["Score (today)"] >= min_score:
        return ["background-color: #e7f7e7"] * len(row)  # signals (not diamond)
    return [""] * len(row)

styled = res.style.apply(highlight, axis=1).format({
    "Prob% (today)": "{:.1f}",
    "Accuracy%": "{:.1f}",
    "Close": "{:.2f}",
    "TSI": "{:.2f}",
    "RSI": "{:.2f}",
    "W%R": "{:.2f}",
    "CMF": "{:.3f}",
})

st.dataframe(styled, use_container_width=True, height=520)

diamonds = res.loc[res["Prob% (today)"] >= float(diamond_thr), "Ticker"].tolist()
st.write(f"**Diamonds (action list):** " + (", ".join(diamonds) if diamonds else "None"))

# Deep dive dashboard
st.subheader("StockCharts-style dashboard")
pick = st.selectbox("Select ETF", options=res["Ticker"].tolist(), index=0)

bt = details.get(pick)
if not bt:
    st.warning("No deep-dive data for that ETF (maybe it was filtered out or had missing history).")
    st.stop()

df_bt = bt["df"].copy()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Prob% (today)", f"{df_bt['PROB_PCT'].iloc[-1]:.1f}%")
k2.metric("Score (today)", f"{int(df_bt['SCORE'].iloc[-1])} / 9")
k3.metric("Signals (history)", f"{bt['signals']}")
k4.metric("Accuracy (history)", f"{(bt['accuracy'] if not np.isnan(bt['accuracy']) else 0):.1f}%")

fig = stockcharts_style_figure(df_bt, title=f"{pick} — Multi-panel (StockCharts-style)", signal_score=min_score)
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Recent rows (last 25)**")
cols = [
    "Open", "High", "Low", "Close", "Volume",
    "SCORE", "PROB_PCT", "SIGNAL", "NEXT_RET_PCT", "HIT",
    "TSI", "RSI", "CCI", "WILLR", "CMF",
    "VWAP_PROXY", "VW_STRETCH", "UPPER_WICK_PCT", "CANDLE_EXHAUST", "BEAR_DIV",
]
cols = [c for c in cols if c in df_bt.columns]
st.dataframe(df_bt[cols].tail(25), use_container_width=True, height=420)

st.caption(
    "Notes: Uses daily OHLCV. 'Optionable' is proxied by the presence of a Yahoo Finance options chain. "
    "VWAP is a rolling volume-weighted proxy (not intraday VWAP). Backtest target is next-day CLOSE drop."
)
