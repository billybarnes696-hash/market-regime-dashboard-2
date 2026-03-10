import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

# ------------------------------------------------
# Sidebar Configuration
# ------------------------------------------------

st.sidebar.title("Strategy Configuration")

years = st.sidebar.slider("Backtest Years",1,20,12)

pair = st.sidebar.selectbox(
"ETF Pair",
[
("SPXL","SPXS"),
("SSO","SDS"),
("SPUU","SPDN")
]
)

bull = pair[0]
bear = pair[1]

trend_ma = st.sidebar.slider("Trend Moving Average",100,300,150)

trade_cost = st.sidebar.slider("Transaction Cost (bps)",0,20,2)

# ------------------------------------------------
# Data Loader
# ------------------------------------------------

@st.cache_data(ttl=3600)
def load_data(start):

    tickers = [
        "SPY",
        bull,
        bear,
        "RSP",
        "SPXS",
        "SVOL",
        "HYG",
        "LQD",
        "VXX"
    ]

    df = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=True
    )["Close"]

    return df.dropna()

# ------------------------------------------------
# Indicator Engine
# ------------------------------------------------

def indicators(df):

    df["trend"] = df["SPY"] > df["SPY"].rolling(trend_ma).mean()

    df["breadth"] = (df["RSP"]/df["SPY"]).pct_change(20)

    df["fear"] = (df["SPXS"]/df["SVOL"]).pct_change(10)

    df["credit"] = (df["HYG"]/df["LQD"]).pct_change(20)

    df["internals"] = (df["SPY"]/df["VXX"]).pct_change(10)

    return df

# ------------------------------------------------
# Regime Score
# ------------------------------------------------

def regime(df):

    score = []

    for i in range(len(df)):

        s = 0

        if df["trend"].iloc[i]:
            s += 1

        if df["breadth"].iloc[i] > 0:
            s += 1

        if df["fear"].iloc[i] < 0:
            s += 1

        if df["credit"].iloc[i] > 0:
            s += 1

        if df["internals"].iloc[i] > 0:
            s += 1

        score.append(s)

    df["score"] = score

    return df

# ------------------------------------------------
# Signal Engine
# ------------------------------------------------

def signals(df):

    sig = []

    for i in range(len(df)):

        if df["score"].iloc[i] >= 3:
            sig.append(bull)

        elif df["score"].iloc[i] <= 1:
            sig.append(bear)

        else:
            sig.append(sig[-1] if len(sig)>0 else bear)

    df["signal"] = sig
    df["held"] = df["signal"].shift(1)

    return df

# ------------------------------------------------
# Backtest Engine
# ------------------------------------------------

def backtest(df):

    df["bull_ret"] = df[bull].pct_change()
    df["bear_ret"] = df[bear].pct_change()
    df["spy_ret"] = df["SPY"].pct_change()

    strat = []

    for i in range(len(df)):

        if df["held"].iloc[i] == bull:
            strat.append(df["bull_ret"].iloc[i])
        else:
            strat.append(df["bear_ret"].iloc[i])

    df["strategy_ret"] = strat

    cost = trade_cost / 10000

    turnover = (df["held"] != df["held"].shift()).astype(int)

    df["strategy_ret"] = df["strategy_ret"] - cost * turnover

    df["strategy"] = (1 + df["strategy_ret"].fillna(0)).cumprod() * 100
    df["buyhold"] = (1 + df["spy_ret"].fillna(0)).cumprod() * 100

    return df

# ------------------------------------------------
# Performance Metrics
# ------------------------------------------------

def metrics(df):

    years = (df.index[-1] - df.index[0]).days / 365

    strat = df["strategy"].iloc[-1] / df["strategy"].iloc[0]

    cagr = strat ** (1/years) - 1

    dd = (df["strategy"] / df["strategy"].cummax() - 1).min()

    sharpe = np.sqrt(252) * df["strategy_ret"].mean() / df["strategy_ret"].std()

    return cagr, dd, sharpe

# ------------------------------------------------
# Chart Builder
# ------------------------------------------------

def plot(df):

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6,0.4]
    )

    fig.add_trace(
        go.Scatter(x=df.index,y=df["SPY"],name="SPY"),
        row=1,col=1
    )

    held = df["held"]
    changes = (held != held.shift()).fillna(True)

    starts = list(df.index[changes])
    starts.append(df.index[-1])

    for i in range(len(starts)-1):

        s = starts[i]
        e = starts[i+1]

        color = "rgba(16,185,129,0.2)" if held.loc[s]==bull else "rgba(220,38,38,0.2)"

        fig.add_vrect(x0=s,x1=e,fillcolor=color,line_width=0)

    fig.add_trace(
        go.Scatter(x=df.index,y=df["strategy"],name="Strategy"),
        row=2,col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index,y=df["buyhold"],name="Buy & Hold"),
        row=2,col=1
    )

    return fig

# ------------------------------------------------
# App Execution
# ------------------------------------------------

start = datetime.now() - timedelta(days=365*years)

data = load_data(start)

data = indicators(data)

data = regime(data)

data = signals(data)

data = backtest(data)

cagr, dd, sharpe = metrics(data)

st.title("Leveraged Regime Dashboard")

last = data.iloc[-1]

st.metric("Current Position", last["signal"])

c1,c2,c3,c4 = st.columns(4)

c1.metric("Strategy Return", f"{data['strategy'].iloc[-1]-100:.1f}%")

c2.metric("Buy & Hold", f"{data['buyhold'].iloc[-1]-100:.1f}%")

c3.metric("CAGR", f"{cagr*100:.2f}%")

c4.metric("Max Drawdown", f"{dd:.1%}")

st.metric("Sharpe Ratio", f"{sharpe:.2f}")

fig = plot(data)

st.plotly_chart(fig,use_container_width=True)

st.subheader("Recent Data")

st.dataframe(data.tail(120))
