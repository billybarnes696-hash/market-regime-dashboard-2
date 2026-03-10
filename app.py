import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Market Breadth Dashboard", layout="wide")

# -----------------------------
# SETTINGS
# -----------------------------
START = "2016-01-01"

# Fast universe (~150 stocks)
UNIVERSE = [
"AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","TSLA","BRK-B","JPM",
"UNH","XOM","LLY","V","MA","AVGO","HD","PG","MRK","PEP","COST","ABBV",
"KO","BAC","CRM","ADBE","WMT","CSCO","TMO","ACN","MCD","DHR","LIN",
"ABT","ORCL","AMD","NFLX","TXN","CMCSA","VZ","PM","NKE","QCOM","INTC",
"UPS","LOW","SPGI","RTX","HON","SBUX","CAT","GE","INTU","AMAT","DE",
"IBM","GS","BLK","AMGN","PLD","NOW","MS","BKNG","MDT","BA","ISRG",
"ADI","GILD","LMT","TJX","VRTX","MO","PGR","MMC","SYK","ZTS","TGT",
"BDX","C","SO","REGN","SCHW","USB","DUK","PNC","FIS","ITW","CL",
"EL","HCA","SHW","ETN","AON","NSC","CI","CSX","EOG","SLB","EQIX",
"APD","WM","PSA","MPC","ADP","GD","ICE","FCX","TFC","NOC","KMB",
"GM","FDX","MET","ROP","PAYX","TRV","AEP","OXY","MAR","CMG","DOW",
"F","SPG","HLT","PSX","AFL","ALL","AZO","ROST","AIG","KMI"
]

# -----------------------------
# DATA DOWNLOAD
# -----------------------------
@st.cache_data
def load_data():

    prices = yf.download(
        UNIVERSE,
        start=START,
        auto_adjust=True,
        progress=False
    )["Close"]

    return prices

prices = load_data()

# -----------------------------
# ADVANCE / DECLINE
# -----------------------------
diff = prices.diff()

adv = (diff > 0).sum(axis=1)
dec = (diff < 0).sum(axis=1)

ad = adv - dec

# -----------------------------
# NYMO
# -----------------------------
ema19 = ad.ewm(span=19).mean()
ema39 = ad.ewm(span=39).mean()

nymo = ema19 - ema39

# -----------------------------
# NEW HIGHS / LOWS
# -----------------------------
high52 = prices.rolling(252).max()
low52 = prices.rolling(252).min()

new_high = (prices == high52).sum(axis=1)
new_low = (prices == low52).sum(axis=1)

nyhl = new_high - new_low

nyhl_ma10 = nyhl.rolling(10).mean()
nyhl_roc10 = nyhl - nyhl.shift(10)

# -----------------------------
# FEAR RATIO
# -----------------------------
fear = yf.download(["SPXS","SVOL"], start=START, auto_adjust=True)["Close"]
fear_ratio = fear["SPXS"] / fear["SVOL"]

# -----------------------------
# CREDIT STRESS
# -----------------------------
credit = yf.download(["HYG","LQD"], start=START, auto_adjust=True)["Close"]
credit_ratio = credit["HYG"] / credit["LQD"]

# -----------------------------
# BUILD MODEL
# -----------------------------
df = pd.DataFrame({
"NYMO": nymo,
"NYHL_MA10": nyhl_ma10,
"NYHL_ROC10": nyhl_roc10,
"FEAR": fear_ratio,
"CREDIT": credit_ratio
}).dropna()

# Stress score
score = pd.Series(0, index=df.index)

score += (df["NYMO"] < 0)
score += (df["NYHL_MA10"] < 0)
score += (df["NYHL_ROC10"] < -50)
score += (df["FEAR"].pct_change(10) > 0)
score += (df["CREDIT"].pct_change(20) < 0)

df["STRESS_SCORE"] = score

# -----------------------------
# SPX FOR COMPARISON
# -----------------------------
spx = yf.download("^GSPC", start=START, auto_adjust=True)["Close"]

# -----------------------------
# STREAMLIT DISPLAY
# -----------------------------
st.title("Market Breadth Risk Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("NYMO", round(df["NYMO"].iloc[-1],2))
col2.metric("NYHL MA10", round(df["NYHL_MA10"].iloc[-1],2))
col3.metric("Fear Ratio", round(df["FEAR"].iloc[-1],2))
col4.metric("Stress Score", int(df["STRESS_SCORE"].iloc[-1]))

st.line_chart(spx)

st.subheader("Market Stress Score")
st.line_chart(df["STRESS_SCORE"])

st.subheader("NYMO")
st.line_chart(df["NYMO"])

st.subheader("NYHL MA10")
st.line_chart(df["NYHL_MA10"])

st.subheader("NYHL ROC10")
st.line_chart(df["NYHL_ROC10"])
