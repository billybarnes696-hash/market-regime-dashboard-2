import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Market Regime Tracker", layout="centered")

# ─────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────
st.title("🎯 Breadth Confluence Tracker")
st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ─────────────────────────────────────────────────────────────
# SIDEBAR: INPUTS
# ─────────────────────────────────────────────────────────────
st.sidebar.header("📊 Indicator Values")

# OSCILLATORS (Intraday)
bpSPX_pctB = st.sidebar.number_input("BPSPX %B", value=0.02, format="%.2f")
bpSPX_ROC = st.sidebar.number_input("BPSPX ROC (%)", value=-29.87, format="%.2f")
BPNYA_pctB = st.sidebar.number_input("BPNYA %B", value=0.07, format="%.2f")
SPXA50R_pctB = st.sidebar.number_input("SPXA50R %B", value=0.01, format="%.2f")
OEXA50R_pctB = st.sidebar.number_input("OEXA50R %B", value=0.04, format="%.2f")
OEXA200R_pctB = st.sidebar.number_input("OEXA200R %B", value=0.05, format="%.2f")

# CUMULATIVE (Trend)
NYAD_slope = st.sidebar.selectbox("NYAD Slope", ["Falling", "Flat", "Rising"])
NYHL_value = st.sidebar.number_input("NYHL", value=-53, format="%d")

# EOD-ONLY (Use prior close)
NYMO = st.sidebar.number_input("NYMO (EOD)", value=-87.39, format="%.2f")
CPCE = st.sidebar.number_input("CPCE (EOD)", value=0.68, format="%.2f")

# CONTEXT
VXX_pctB = st.sidebar.number_input("VXX %B", value=0.93, format="%.2f")
SPX_pctB = st.sidebar.number_input("SPX %B", value=-0.09, format="%.2f")
HYG_IEF_RSI = st.sidebar.number_input("HYG:IEF RSI", value=45.80, format="%.2f")

# ─────────────────────────────────────────────────────────────
# SCORING LOGIC (FIXED)
# ─────────────────────────────────────────────────────────────
def score_indicator(name, value):
    """Returns +1 (bullish), -1 (bearish), or 0 (neutral)"""
    
    # Special cases first
    if name == "BPSPX_ROC":
        return 1 if value > 0 else -1
    
    if name == "NYAD_slope":
        return {"Rising": 1, "Flat": 0, "Falling": -1}[value]
    
    if name == "NYHL":
        return 1 if value > 0 else -1
    
    if name == "CPCE":
        return 1 if value < 0.45 else -1 if value > 0.90 else 0
    
    if name == "VXX_pctB":
        return 1 if value < 0.30 else -1 if value > 0.80 else 0
    
    if name == "HYG_IEF_RSI":
        return 1 if value > 50 else -1 if value < 40 else 0
    
    # %B indicators with safe dictionary access
    bull_thresholds = {
        "BPSPX_pctB": 0.30, "BPNYA_pctB": 0.25, "SPXA50R_pctB": 0.30,
        "OEXA50R_pctB": 0.30, "OEXA200R_pctB": 0.50, "SPX_pctB": 0.30
    }
    
    bear_thresholds = {
        "BPSPX_pctB": 0.20, "BPNYA_pctB": 0.15, "SPXA50R_pctB": 0.10,
        "OEXA50R_pctB": 0.10, "OEXA200R_pctB": 0.30, "SPX_pctB": 0.20
    }
    
    # Safe access with default
    bull = bull_thresholds.get(name, 0.30)
    bear = bear_thresholds.get(name, 0.20)
    
    if value >= bull:
        return 1
    elif value <= bear:
        return -1
    return 0

# Weights (must match values keys)
weights = {
    "BPSPX_pctB": 2.0, "BPSPX_ROC": 2.0, "BPNYA_pctB": 1.0,
    "SPXA50R_pctB": 1.0, "OEXA50R_pctB": 0.5, "OEXA200R_pctB": 0.5,
    "NYMO": 1.0, "NYAD_slope": 1.0, "NYHL": 1.0,
    "CPCE": 0.5, "VXX_pctB": 0.5, "SPX_pctB": 0.5, "HYG_IEF_RSI": 0.5
}

# Values (must match weights keys)
values = {
    "BPSPX_pctB": bpSPX_pctB, "BPSPX_ROC": bpSPX_ROC, "BPNYA_pctB": BPNYA_pctB,
    "SPXA50R_pctB": SPXA50R_pctB, "OEXA50R_pctB": OEXA50R_pctB,
    "OEXA200R_pctB": OEXA200R_pctB, "NYMO": NYMO, "NYAD_slope": NYAD_slope,
    "NYHL": NYHL_value, "CPCE": CPCE, "VXX_pctB": VXX_pctB, 
    "SPX_pctB": SPX_pctB, "HYG_IEF_RSI": HYG_IEF_RSI
}

# Calculate score
score = sum(score_indicator(k, v) * w for k, v, w in zip(values.keys(), values.values(), weights.values()))

# ─────────────────────────────────────────────────────────────
# SIGNAL LOGIC
# ─────────────────────────────────────────────────────────────
if score >= 7:
    signal, action = "🟢 STRONG BUY", "Add 75-100%"
elif score >= 4:
    signal, action = "🟢 BUY", "Add 50-75%"
elif score >= 0:
    signal, action = "🟡 HOLD", "Hold core, no adds"
elif score >= -4:
    signal, action = "🟡 CAUTION", "Reduce to 25-50%"
elif score >= -7:
    signal, action = "🔴 SELL", "Reduce to 0-25%"
else:
    signal, action = "🔴 CAPITULATION", "Watch for bounce"

# ─────────────────────────────────────────────────────────────
# MAIN DISPLAY
# ─────────────────────────────────────────────────────────────
st.metric("Confluence Score", f"{score:.2f} / 10")
st.markdown(f"### {signal}")
st.markdown(f"**Action:** {action}")

# ─────────────────────────────────────────────────────────────
# TRIGGERS
# ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔹 Key Triggers")

if score < -7:
    st.info("✅ **Bounce Setup** — Wait for BPSPX %B > 0.10 to confirm")
    st.warning("⚠️ **Risk** — If SPX < 6,500, stay defensive")
elif score < -4:
    st.warning("⚠️ **Weak Breadth** — Small size only, tight stops")
elif score >= 4:
    st.success("✅ **Breadth Confirmed** — Add on pullbacks")
else:
    st.info("⏳ **Neutral** — Wait for clearer signal")

# ─────────────────────────────────────────────────────────────
# BREAKDOWN
# ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Indicator Breakdown")

cols = st.columns(2)
for i, (name, value) in enumerate(values.items()):
    s = score_indicator(name, value)
    icon = "🟢" if s > 0 else "🔴" if s < 0 else "🟡"
    with cols[i % 2]:
        st.write(f"{icon} **{name}**: {value}")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ Not investment advice. Use with your own analysis.")
