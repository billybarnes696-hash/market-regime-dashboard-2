import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Market Regime Dashboard", layout="wide")

# === SIGNAL LOGIC (NO EXTERNAL DATA) ===
def get_signal():
    # Values from your StockCharts PDFs (Feb 24, 2026)
    nyhl_cum, nyhl_200 = 32941, 27441
    bpspx_rsi, bpspx_macd = 36.47, -0.470
    oexa_val, oexa_cci = 63.00, -163.74
    spy_vxx, vxx_sma50 = 24.05, 25.06
    hy_spread = 385
    
    # Simple logic
    regime_bull = nyhl_cum > nyhl_200
    bpspx_bear = bpspx_rsi < 40 and bpspx_macd < 0
    risk_off = spy_vxx < vxx_sma50
    
    if not regime_bull:
        return "SELL", "signal-sell", {'SPY':20,'IEF':60,'CASH':20}
    elif bpspx_bear and risk_off:
        return "WARNING", "signal-warning", {'SPY':40,'IEF':45,'CASH':15}
    elif oexa_val < 35 and oexa_cci < -100:
        return "BUY DIP", "signal-buy", {'SPY':80,'IEF':15,'CASH':5}
    else:
        return "HOLD", "signal-hold", {'SPY':65,'IEF':25,'CASH':10}

# === MAIN APP ===
def main():
    st.title("📊 Market Regime Dashboard")
    
    signal, sig_class, alloc = get_signal()
    
    # Signal banner
    st.markdown(f"<div style='background:{\"#dc2626\" if \"SELL\" in signal else \"#f97316\" if \"WARNING\" in signal else \"#f59e0b\" if \"HOLD\" in signal else \"#10b981\"};color:white;padding:1rem;border-radius:8px;font-weight:bold;font-size:1.3rem;text-align:center'>{signal}</div>", unsafe_allow_html=True)
    
    # Canary indicators
    st.subheader("🚨 Canary Indicators")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Credit Spreads", "385 bps", "✅ Normal")
    with c2: st.metric("Semis/SPX", "1.65", "✅ Strong")
    with c3: st.metric("Small/Large", "0.42", "⚠️ Weak")
    with c4: st.metric("McClellan", "-15.4", "✅ Normal")
    
    # Allocation
    st.subheader("🎯 Allocation")
    df = pd.DataFrame(list(alloc.items()), columns=['ETF','%'])
    fig = px.pie(df, values='%', names='ETF', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
    
    # Refresh button
    if st.button("🔄 Refresh"):
        st.rerun()

if __name__ == "__main__":
    main()
