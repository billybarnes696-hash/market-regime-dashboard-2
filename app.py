import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================
# CONFIG & STYLING
# ============================================
st.set_page_config(page_title="Market Breadth Dashboard", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;}
    .signal-buy {background: #10b981; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold;}
    .signal-hold {background: #f59e0b; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold;}
    .signal-warning {background: #ef4444; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold;}
    .signal-sell {background: #dc2626; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold;}
    .signal-dip {background: #3b82f6; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold;}
    .signal-improving {background: #8b5cf6; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA FETCHING (MOCK - REPLACE WITH REAL SOURCE)
# ============================================
@st.cache_data(ttl=3600)
def fetch_indicator_data():
    """
    Fetch current indicator readings.
    🔁 REPLACE THIS with your actual data source:
    - yfinance for SPY, IEF, VXX prices
    - StockCharts CSV export via manual upload
    - Alpha Vantage / Polygon API for breadth data
    """
    # Mock data based on your uploaded charts (Feb 2026 readings)
    return {
        'timestamp': datetime.now(),
        'spx_price': 6877.92,
        
        # $NYHL Cumulative
        'nyhl_cum': 32941,
        'nyhl_sma50': 29200,
        'nyhl_sma200': 27441,
        'nyhl_macd_hist': 24.045,
        'nyhl_tsi': 84.91,
        
        # $BPSPX (S&P 500 Bullish %)
        'bpspx_value': 60.00,
        'bpspx_sma50': 60.98,
        'bpspx_rsi14': 36.47,
        'bpspx_macd_hist': -0.470,
        'bpspx_cci100': 33.53,
        
        # $OEXA150R (% S&P 100 above 150-DMA)
        'oexa150r_value': 63.00,
        'oexa150r_cci14': -163.74,
        'oexa150r_macd_hist': -0.705,
        
        # $BPNYA (NYSE Bullish %)
        'bpnya_value': 60.83,
        'bpnya_sma50': 62.38,
        'bpnya_tsi': -34.47,
        'bpnya_cci20': -68.47,
        
        # SPY:VXX Ratio
        'spy_vxx_ratio': 24.05,
        'spy_vxx_sma50': 25.06,
        'spy_vxx_sma200': 18.50,
        'spy_vxx_macd_hist': -0.480,
        'spy_vxx_rsi14': 47.33,
        
        # ETF Prices (for allocation calc)
        'spy_price': 598.45,
        'ief_price': 96.32,
        'vxx_price': 14.87,
    }

# ============================================
# SIGNAL GENERATION LOGIC
# ============================================
def generate_signal(data):
    """
    Generate market signal based on breadth-momentum-risk framework.
    Returns: signal_label, signal_class, recommendation_dict
    """
    signals = []
    weights = []
    
    # === PRIMARY REGIME FILTER: $NYHL Cumulative vs 200-SMA ===
    if data['nyhl_cum'] > data['nyhl_sma200']:
        signals.append('bullish_regime')
        weights.append(3)  # High weight
    else:
        signals.append('bearish_regime')
        weights.append(3)
    
    # === EARLY WARNING: $BPSPX ===
    if data['bpspx_rsi14'] < 40 and data['bpspx_macd_hist'] < 0:
        signals.append('distribution_warning')
        weights.append(2)
    elif data['bpspx_rsi14'] > 60 and data['bpspx_macd_hist'] > 0:
        signals.append('accumulation_confirmed')
        weights.append(2)
    else:
        signals.append('neutral_bpspx')
        weights.append(1)
    
    # === ENTRY CONFIRMATION: $OEXA150R ===
    if data['oexa150r_value'] < 30 and data['oexa150r_cci14'] < -100:
        signals.append('oversold_buy_zone')
        weights.append(2)
    elif data['oexa150r_value'] > 70:
        signals.append('overbought_caution')
        weights.append(2)
    elif 45 <= data['oexa150r_value'] <= 55 and data['oexa150r_macd_hist'] > 0:
        signals.append('participation_improving')
        weights.append(1)
    else:
        signals.append('neutral_oexa')
        weights.append(1)
    
    # === RISK SENTIMENT: SPY:VXX Ratio ===
    if data['spy_vxx_ratio'] < data['spy_vxx_sma50'] and data['spy_vxx_macd_hist'] < 0:
        signals.append('risk_off_pressure')
        weights.append(2)
    elif data['spy_vxx_ratio'] > data['spy_vxx_sma50'] and data['spy_vxx_macd_hist'] > 0:
        signals.append('risk_on_confirmed')
        weights.append(2)
    else:
        signals.append('neutral_risk')
        weights.append(1)
    
    # === MOMENTUM DIVERGENCE: $BPNYA ===
    if data['bpnya_tsi'] < 0 and data['bpnya_cci20'] < -50:
        signals.append('momentum_weakness')
        weights.append(1)
    elif data['bpnya_tsi'] > 0 and data['bpnya_cci20'] > 50:
        signals.append('momentum_strength')
        weights.append(1)
    
    # === SIGNAL SCORING ===
    bullish_score = sum(w for s, w in zip(signals, weights) if 'bullish' in s or 'accumulation' in s or 'oversold_buy' in s or 'risk_on' in s or 'strength' in s or 'improving' in s)
    bearish_score = sum(w for s, w in zip(signals, weights) if 'bearish' in s or 'distribution' in s or 'overbought' in s or 'risk_off' in s or 'weakness' in s)
    
    # === FINAL SIGNAL LOGIC ===
    if data['nyhl_cum'] < data['nyhl_sma200'] and bearish_score >= 5:
        signal_label = "SELL"
        signal_class = "signal-sell"
        recommendation = {'SPY': 30, 'IEF': 40, 'VXX': 15, 'CASH': 15}
        
    elif data['bpspx_rsi14'] < 35 and data['oexa150r_value'] < 35 and data['nyhl_cum'] > data['nyhl_sma200']:
        signal_label = "TAKE A DIP"
        signal_class = "signal-dip"
        recommendation = {'SPY': 85, 'IEF': 10, 'VXX': 0, 'CASH': 5}
        
    elif bullish_score - bearish_score >= 4 and data['nyhl_cum'] > data['nyhl_sma200']:
        signal_label = "BUY"
        signal_class = "signal-buy"
        recommendation = {'SPY': 90, 'IEF': 5, 'VXX': 0, 'CASH': 5}
        
    elif bullish_score - bearish_score >= 1 and data['nyhl_cum'] > data['nyhl_sma200']:
        if 'improving' in ' '.join(signals):
            signal_label = "IMPROVING"
            signal_class = "signal-improving"
        else:
            signal_label = "HOLD"
            signal_class = "signal-hold"
        recommendation = {'SPY': 75, 'IEF': 15, 'VXX': 5, 'CASH': 5}
        
    elif bearish_score > bullish_score and data['nyhl_cum'] > data['nyhl_sma200']:
        signal_label = "WARNING"
        signal_class = "signal-warning"
        recommendation = {'SPY': 50, 'IEF': 30, 'VXX': 10, 'CASH': 10}
        
    elif 'distribution' in ' '.join(signals) and data['bpspx_rsi14'] < 45:
        signal_label = "LOOK TO SELL"
        signal_class = "signal-warning"
        recommendation = {'SPY': 40, 'IEF': 35, 'VXX': 15, 'CASH': 10}
        
    else:
        signal_label = "HOLD"
        signal_class = "signal-hold"
        recommendation = {'SPY': 70, 'IEF': 20, 'VXX': 5, 'CASH': 5}
    
    return signal_label, signal_class, recommendation, signals

# ============================================
# DASHBOARD UI
# ============================================
def main():
    st.title("📊 Market Breadth-Momentum-Risk Dashboard")
    st.markdown("*Framework to beat buy-and-hold SPY using $NYHL, $BPSPX, $OEXA150R, $BPNYA, SPY:VXX*")
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
    
    # Fetch data
    with st.spinner("Loading indicator data..."):
        data = fetch_indicator_data()
    
    # Generate signal
    signal_label, signal_class, recommendation, raw_signals = generate_signal(data)
    
    # === TOP ROW: PRIMARY SIGNAL & RECOMMENDATION ===
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### Current Signal: <div class='{signal_class}'>{signal_label}</div>", unsafe_allow_html=True)
        st.markdown(f"*Generated: {data['timestamp'].strftime('%Y-%m-%d %H:%M')}*")
    
    with col2:
        st.metric("SPX Price", f"${data['spx_price']:,.2f}")
    
    with col3:
        st.metric("Primary Regime", 
                  "🟢 BULLISH" if data['nyhl_cum'] > data['nyhl_sma200'] else "🔴 BEARISH",
                  delta=f"{data['nyhl_cum'] - data['nyhl_sma200']:,.0f} vs 200-SMA")
    
    # === ETF ALLOCATION RECOMMENDATION ===
    st.subheader("🎯 Recommended ETF Allocation")
    alloc_df = pd.DataFrame(list(recommendation.items()), columns=['ETF', 'Allocation %'])
    
    col1, col2 = st.columns([1, 2])
    with col1:
        # Pie chart
        fig_pie = px.pie(alloc_df, values='Allocation %', names='ETF', 
                         color='ETF',
                         color_discrete_map={'SPY': '#10b981', 'IEF': '#3b82f6', 'VXX': '#ef4444', 'CASH': '#6b7280'},
                         hole=0.4)
        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Allocation bars
        fig_bar = px.bar(alloc_df, x='Allocation %', y='ETF', orientation='h',
                         color='ETF',
                         color_discrete_map={'SPY': '#10b981', 'IEF': '#3b82f6', 'VXX': '#ef4444', 'CASH': '#6b7280'},
                         text='Allocation %')
        fig_bar.update_layout(showlegend=False, xaxis_title='Allocation %', yaxis_title='', 
                              xaxis=dict(range=[0, 100]), margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if recommendation['SPY'] >= 70:
            st.button("✅ Execute: Add SPY", type="primary", use_container_width=True)
        elif recommendation['SPY'] <= 40:
            st.button("⚠️ Execute: Reduce SPY", type="secondary", use_container_width=True)
        else:
            st.button("⏸️ Hold SPY Position", type="secondary", use_container_width=True)
    with col2:
        if recommendation['IEF'] >= 30:
            st.button("🛡️ Add IEF Hedge", use_container_width=True)
        else:
            st.button("IEF: Minimal", disabled=True, use_container_width=True)
    with col3:
        if recommendation['VXX'] >= 10:
            st.button("🔥 Add VXX Hedge", use_container_width=True)
        else:
            st.button("VXX: Not Needed", disabled=True, use_container_width=True)
    with col4:
        if recommendation['CASH'] >= 10:
            st.button("💵 Raise Cash", use_container_width=True)
        else:
            st.button("Cash: Minimal", disabled=True, use_container_width=True)
    
    # === INDICATOR CARDS ===
    st.subheader("🔍 Indicator Readings")
    
    # Row 1: Primary Regime + Early Warning
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>$NYHL Cumulative (Regime)</strong><br>
            Value: {data['nyhl_cum']:,}<br>
            vs 200-SMA: {data['nyhl_sma200']:,}<br>
            MACD Hist: {data['nyhl_macd_hist']:+.2f}<br>
            Status: {'✅ Bullish' if data['nyhl_cum'] > data['nyhl_sma200'] else '❌ Bearish'}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>$BPSPX (Early Warning)</strong><br>
            Value: {data['bpspx_value']}%<br>
            RSI(14): {data['bpspx_rsi14']:.1f}<br>
            MACD Hist: {data['bpspx_macd_hist']:+.3f}<br>
            Status: {'⚠️ Distribution' if data['bpspx_rsi14'] < 40 else '✅ Neutral/Bullish'}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <strong>SPY:VXX Ratio (Risk)</strong><br>
            Value: {data['spy_vxx_ratio']:.2f}<br>
            vs 50-SMA: {data['spy_vxx_sma50']:.2f}<br>
            RSI(14): {data['spy_vxx_rsi14']:.1f}<br>
            Status: {'⚠️ Risk-Off' if data['spy_vxx_ratio'] < data['spy_vxx_sma50'] else '✅ Risk-On'}
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2: Confirmation Indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>$OEXA150R (Entry Confirmation)</strong><br>
            % Above 150-DMA: {data['oexa150r_value']}%<br>
            CCI(14): {data['oexa150r_cci14']:.1f}<br>
            MACD Hist: {data['oexa150r_macd_hist']:+.3f}<br>
            Signal: {'🟢 Buy Zone' if data['oexa150r_value'] < 30 else '🟡 Neutral' if data['oexa150r_value'] < 70 else '🔴 Overbought'}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>$BPNYA (Momentum Cross-Check)</strong><br>
            Bullish %: {data['bpnya_value']}%<br>
            TSI(7,4,7): {data['bpnya_tsi']:.2f}<br>
            CCI(20): {data['bpnya_cci20']:.1f}<br>
            Signal: {'⚠️ Weakness' if data['bpnya_tsi'] < 0 else '✅ Strength'}
        </div>
        """, unsafe_allow_html=True)
    
    # === RAW SIGNAL BREAKDOWN ===
    with st.expander("🔎 View Raw Signal Logic"):
        st.write("**Signals Detected:**")
        for s in raw_signals:
            icon = "✅" if any(x in s for x in ['bullish', 'accumulation', 'oversold_buy', 'risk_on', 'strength', 'improving']) else "⚠️" if 'neutral' in s else "❌"
            st.write(f"{icon} {s.replace('_', ' ').title()}")
        
        st.write("\n**Scoring:**")
        st.write(f"- Bullish components: {sum(w for s, w in zip(raw_signals, [3,2,2,2,1,1]) if 'bullish' in s or 'accumulation' in s or 'oversold_buy' in s or 'risk_on' in s or 'strength' in s or 'improving' in s)}")
        st.write(f"- Bearish components: {sum(w for s, w in zip(raw_signals, [3,2,2,2,1,1]) if 'bearish' in s or 'distribution' in s or 'overbought' in s or 'risk_off' in s or 'weakness' in s)}")
    
    # === ALERTS & KEY LEVELS ===
    st.subheader("🚨 Key Levels to Watch")
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    
    with alert_col1:
        st.info(f"**$NYHL Caution**: Close below {data['nyhl_sma50']:,} (50-SMA)")
    with alert_col2:
        st.warning(f"**$BPSPX Warning**: RSI(14) < 35 OR value < 50")
    with alert_col3:
        st.error(f"**SPY:VXX Risk-Off**: Ratio breaks below {data['spy_vxx_sma200']:.2f} (200-SMA)")
    
    # === DATA UPLOAD OPTION (for real data) ===
    with st.sidebar:
        st.header("⚙️ Data Source")
        st.markdown("*Replace mock data with your source:*")
        
        uploaded_file = st.file_uploader("📤 Upload CSV (optional)", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ CSV loaded! (Implement parsing logic in fetch_indicator_data())")
                # TODO: Add parsing logic here
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
        
        st.markdown("---")
        st.markdown("**Quick Reference:**")
        st.markdown("""
        - **BUY**: NYHL>200SMA + BPSPX RSI>40 + OEXA>50 + VXX ratio rising
        - **HOLD**: NYHL>200SMA but mixed momentum
        - **WARNING**: NYHL>200SMA but BPSPX RSI<40 + VXX ratio falling  
        - **SELL**: NYHL<200SMA OR multiple bearish divergences
        - **TAKE A DIP**: NYHL>200SMA + BPSPX RSI<30 + OEXA<30
        """)
        
        st.markdown("---")
        st.caption("⚠️ Not investment advice. Backtest before live use.")

if __name__ == "__main__":
    main()
