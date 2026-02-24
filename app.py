import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================
# CONFIG & STYLING
# ============================================
st.set_page_config(page_title="Market Breadth Dashboard Pro", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;}
    .signal-buy {background: #10b981; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; text-align: center;}
    .signal-dip {background: #3b82f6; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; text-align: center;}
    .signal-improving {background: #8b5cf6; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; text-align: center;}
    .signal-hold {background: #f59e0b; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; text-align: center;}
    .signal-warning {background: #f97316; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; text-align: center;}
    .signal-looktosell {background: #ec4899; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; text-align: center;}
    .signal-sell {background: #dc2626; color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: bold; text-align: center;}
    .etf-card {background: #f8fafc; border-left: 4px solid #667eea; padding: 0.75rem; border-radius: 6px; margin: 0.25rem 0;}
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA FETCHING (MOCK - REPLACE WITH REAL SOURCE)
# ============================================
@st.cache_data(ttl=3600)
def fetch_indicator_data():
    """
    Fetch current indicator readings.
    🔁 REPLACE WITH: yfinance, Alpha Vantage, FRED API, or CSV upload
    """
    return {
        'timestamp': datetime.now(),
        'spx_price': 6877.92,
        
        # === CORE BREADTH INDICATORS ===
        # $NYHL Cumulative
        'nyhl_cum': 32941, 'nyhl_sma50': 29200, 'nyhl_sma200': 27441,
        'nyhl_macd_hist': 24.045, 'nyhl_tsi': 84.91,
        
        # $BPSPX (S&P 500 Bullish %)
        'bpspx_value': 60.00, 'bpspx_sma50': 60.98,
        'bpspx_rsi14': 36.47, 'bpspx_macd_hist': -0.470, 'bpspx_cci100': 33.53,
        
        # $OEXA150R (% S&P 100 above 150-DMA)
        'oexa150r_value': 63.00, 'oexa150r_cci14': -163.74, 'oexa150r_macd_hist': -0.705,
        
        # $BPNYA (NYSE Bullish %)
        'bpnya_value': 60.83, 'bpnya_sma50': 62.38,
        'bpnya_tsi': -34.47, 'bpnya_cci20': -68.47,
        
        # SPY:VXX Ratio
        'spy_vxx_ratio': 24.05, 'spy_vxx_sma50': 25.06, 'spy_vxx_sma200': 18.50,
        'spy_vxx_macd_hist': -0.480, 'spy_vxx_rsi14': 47.33,
        
        # === DAVE KELLER ADDITIONS ===
        # Sentiment
        'vix': 18.45,
        'aaii_bull_bear_spread': 5.2,  # Bull% - Bear%
        'naaim_exposure': 72.3,  # % long exposure
        
        # Breadth
        'mcclellan_osc': -15.4,  # -100 to +100
        'nyse_adl': 125000,  # Advance/Decline Line
        'spx_pct_above_200dma': 68.5,
        
        # Credit/Rates
        'hy_spread_bps': 385,  # High Yield vs Treasuries
        'tnx_10yr_yield': 4.25,
        
        # Leadership Ratios
        'smallcap_largecap_ratio': 0.42,  # IWM/SPY
        'equalweight_sp500_ratio': 0.88,  # RSP/SPY
        'semis_spx_ratio': 1.65,  # SOX/SPY
        
        # === ETF PRICES (for allocation) ===
        'spy_price': 598.45, 'ief_price': 96.32, 'vxx_price': 14.87,
        'iwm_price': 198.23, 'qqq_price': 485.67, 'tlt_price': 92.18,
        'gld_price': 234.56, 'shy_price': 82.45, 'lqd_price': 108.34,
        'efa_price': 78.92, 'eem_price': 41.23, 'dbc_price': 22.67,
    }

# ============================================
# SIGNAL GENERATION LOGIC (7-TIER SYSTEM)
# ============================================
def generate_signal(data):
    """
    Generate market signal: BUY, TAKE A DIP, IMPROVING, HOLD, WARNING, LOOK TO SELL, SELL
    Returns: signal_label, signal_class, recommendation_dict, raw_signals
    """
    signals = []
    weights = []
    
    # === PRIMARY REGIME: $NYHL Cumulative vs 200-SMA ===
    if data['nyhl_cum'] > data['nyhl_sma200']:
        signals.append('bullish_regime'); weights.append(4)
    else:
        signals.append('bearish_regime'); weights.append(4)
    
    # === EARLY WARNING: $BPSPX ===
    if data['bpspx_rsi14'] < 35:
        signals.append('bpspx_oversold'); weights.append(2)
    elif data['bpspx_rsi14'] > 65:
        signals.append('bpspx_overbought'); weights.append(2)
    if data['bpspx_macd_hist'] < 0 and data['bpspx_value'] < 50:
        signals.append('bpspx_distribution'); weights.append(2)
    
    # === ENTRY CONFIRMATION: $OEXA150R ===
    if data['oexa150r_value'] < 30 and data['oexa150r_cci14'] < -100:
        signals.append('oexa_oversold_buy'); weights.append(3)
    elif data['oexa150r_value'] > 75:
        signals.append('oexa_overbought'); weights.append(2)
    elif 45 <= data['oexa150r_value'] <= 55 and data['oexa150r_macd_hist'] > 0:
        signals.append('oexa_improving'); weights.append(1)
    
    # === RISK SENTIMENT: SPY:VXX + VIX ===
    if data['spy_vxx_ratio'] < data['spy_vxx_sma50'] and data['vix'] > 22:
        signals.append('risk_off_pressure'); weights.append(3)
    elif data['spy_vxx_ratio'] > data['spy_vxx_sma50'] and data['vix'] < 16:
        signals.append('risk_on_confirmed'); weights.append(2)
    
    # === SENTIMENT EXTREMES (Contrarian) ===
    if data['aaii_bull_bear_spread'] < -15 and data['vix'] > 25:
        signals.append('extreme_fear_oversold'); weights.append(3)
    elif data['aaii_bull_bear_spread'] > 25 and data['vix'] < 14:
        signals.append('extreme_complacency_overbought'); weights.append(3)
    if data['naaim_exposure'] > 85:
        signals.append('naaim_overbought'); weights.append(1)
    elif data['naaim_exposure'] < 35:
        signals.append('naaim_oversold'); weights.append(1)
    
    # === CREDIT STRESS (Leading Indicator) ===
    if data['hy_spread_bps'] > 450:
        signals.append('credit_stress_high'); weights.append(3)
    elif data['hy_spread_bps'] < 300:
        signals.append('credit_conditions_easy'); weights.append(1)
    
    # === LEADERSHIP ANALYSIS ===
    if data['equalweight_sp500_ratio'] < 0.85 and data['spx_price'] > data['nyhl_sma200']:
        signals.append('narrow_leadership_warning'); weights.append(2)
    if data['semis_spx_ratio'] < 1.5:
        signals.append('tech_leadership_weak'); weights.append(1)
    
    # === MOMENTUM DIVERGENCE ===
    if data['bpnya_tsi'] < 0 and data['bpspx_rsi14'] < 40:
        signals.append('momentum_divergence_bearish'); weights.append(2)
    
    # === MCCLELLAN OSCILLATOR ===
    if data['mcclellan_osc'] < -50:
        signals.append('mcclellan_oversold'); weights.append(2)
    elif data['mcclellan_osc'] > 50:
        signals.append('mcclellan_overbought'); weights.append(1)
    
    # === SCORING ===
    bullish_score = sum(w for s, w in zip(signals, weights) 
                       if any(x in s for x in ['bullish', 'oversold_buy', 'risk_on', 'extreme_fear', 
                                              'naaim_oversold', 'credit_easy', 'mcclellan_oversold', 'improving']))
    bearish_score = sum(w for s, w in zip(signals, weights) 
                       if any(x in s for x in ['bearish', 'overbought', 'distribution', 'risk_off', 
                                              'extreme_complacency', 'naaim_overbought', 'credit_stress', 
                                              'narrow_leadership', 'momentum_divergence', 'mcclellan_overbought']))
    
    # === FINAL 7-TIER SIGNAL LOGIC ===
    if data['nyhl_cum'] < data['nyhl_sma200'] and bearish_score >= 6:
        signal_label, signal_class = "SELL", "signal-sell"
        recommendation = _get_allocation('sell')
        
    elif 'distribution' in ' '.join(signals) and data['bpspx_rsi14'] < 45 and bearish_score > bullish_score:
        signal_label, signal_class = "LOOK TO SELL", "signal-looktosell"
        recommendation = _get_allocation('look_to_sell')
        
    elif bearish_score > bullish_score and data['nyhl_cum'] > data['nyhl_sma200']:
        signal_label, signal_class = "WARNING", "signal-warning"
        recommendation = _get_allocation('warning')
        
    elif bullish_score - bearish_score >= 1 and data['nyhl_cum'] > data['nyhl_sma200']:
        if 'improving' in ' '.join(signals) or 'oexa_improving' in signals:
            signal_label, signal_class = "IMPROVING", "signal-improving"
            recommendation = _get_allocation('improving')
        else:
            signal_label, signal_class = "HOLD", "signal-hold"
            recommendation = _get_allocation('hold')
            
    elif bullish_score - bearish_score >= 4 and data['nyhl_cum'] > data['nyhl_sma200']:
        signal_label, signal_class = "BUY", "signal-buy"
        recommendation = _get_allocation('buy')
        
    elif data['bpspx_rsi14'] < 35 and data['oexa150r_value'] < 35 and data['nyhl_cum'] > data['nyhl_sma200']:
        signal_label, signal_class = "TAKE A DIP", "signal-dip"
        recommendation = _get_allocation('take_a_dip')
        
    else:
        signal_label, signal_class = "HOLD", "signal-hold"
        recommendation = _get_allocation('hold')
    
    return signal_label, signal_class, recommendation, signals

def _get_allocation(regime):
    """Return ETF allocation dictionary based on regime"""
    allocations = {
        'buy': {'SPY': 70, 'QQQ': 15, 'IWM': 5, 'IEF': 5, 'GLD': 3, 'CASH': 2},
        'take_a_dip': {'SPY': 75, 'QQQ': 10, 'IWM': 5, 'IEF': 5, 'CASH': 5},
        'improving': {'SPY': 65, 'QQQ': 10, 'IWM': 5, 'IEF': 10, 'GLD': 5, 'CASH': 5},
        'hold': {'SPY': 55, 'QQQ': 10, 'IEF': 15, 'GLD': 8, 'SHY': 7, 'CASH': 5},
        'warning': {'SPY': 35, 'QQQ': 5, 'IEF': 25, 'TLT': 10, 'GLD': 10, 'VXX': 5, 'CASH': 10},
        'look_to_sell': {'SPY': 25, 'QQQ': 5, 'IEF': 20, 'TLT': 15, 'GLD': 12, 'VXX': 8, 'CASH': 15},
        'sell': {'SPY': 15, 'IEF': 25, 'TLT': 20, 'GLD': 15, 'SHY': 10, 'VXX': 10, 'CASH': 5},
    }
    return allocations.get(regime, allocations['hold'])

# ============================================
# DASHBOARD UI
# ============================================
def main():
    st.title("📊 Market Breadth-Momentum-Risk Dashboard Pro")
    st.markdown("*Framework to beat buy-and-hold SPY using breadth, sentiment, credit & leadership signals*")
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
    
    # Fetch data
    with st.spinner("Loading indicator data..."):
        data = fetch_indicator_data()
    
    # Generate signal
    signal_label, signal_class, recommendation, raw_signals = generate_signal(data)
    
    # === TOP ROW: PRIMARY SIGNAL ===
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### 🎯 Current Signal: <div class='{signal_class}'>{signal_label}</div>", unsafe_allow_html=True)
        st.caption(f"Generated: {data['timestamp'].strftime('%Y-%m-%d %H:%M')} | SPX: ${data['spx_price']:,.2f}")
    
    with col2:
        regime = "🟢 BULLISH" if data['nyhl_cum'] > data['nyhl_sma200'] else "🔴 BEARISH"
        st.metric("Primary Regime", regime, 
                  delta=f"{data['nyhl_cum'] - data['nyhl_sma200']:,.0f} vs 200-SMA")
    
    with col3:
        risk_status = "⚠️ Risk-Off" if data['spy_vxx_ratio'] < data['spy_vxx_sma50'] else "✅ Risk-On"
        st.metric("Risk Sentiment", risk_status, delta=f"VIX: {data['vix']:.1f}")
    
    # === ETF ALLOCATION RECOMMENDATION ===
    st.subheader("🎯 Recommended ETF Allocation")
    
    alloc_df = pd.DataFrame(list(recommendation.items()), columns=['ETF', 'Allocation %'])
    
    col1, col2 = st.columns([1, 2])
    with col1:
        # Pie chart with custom colors
        color_map = {'SPY': '#10b981', 'QQQ': '#3b82f6', 'IWM': '#8b5cf6', 'IEF': '#6366f1', 
                    'TLT': '#4f46e5', 'GLD': '#f59e0b', 'SHY': '#14b8a6', 'VXX': '#ef4444', 'CASH': '#6b7280'}
        fig_pie = px.pie(alloc_df, values='Allocation %', names='ETF', 
                         color='ETF', color_discrete_map=color_map, hole=0.4)
        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Horizontal bar chart
        fig_bar = px.bar(alloc_df, x='Allocation %', y='ETF', orientation='h',
                         color='ETF', color_discrete_map=color_map, text='Allocation %')
        fig_bar.update_layout(showlegend=False, xaxis_title='Allocation %', yaxis_title='', 
                              xaxis=dict(range=[0, 100]), margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Action buttons
    st.markdown("### ⚡ Quick Actions")
    cols = st.columns(5)
    actions = [
        ("✅ Add SPY/QQQ", recommendation['SPY'] + recommendation.get('QQQ', 0) >= 70),
        ("🛡️ Add IEF/TLT", recommendation.get('IEF', 0) + recommendation.get('TLT', 0) >= 25),
        ("🔥 Add VXX Hedge", recommendation.get('VXX', 0) >= 8),
        ("🥇 Add GLD", recommendation.get('GLD', 0) >= 8),
        ("💵 Raise Cash", recommendation.get('CASH', 0) >= 10),
    ]
    for i, (label, enabled) in enumerate(cols):
        with cols[i]:
            if enabled:
                st.button(label, type="primary" if i == 0 else "secondary", use_container_width=True)
            else:
                st.button(label, disabled=True, use_container_width=True)
    
    # === INDICATOR PANELS ===
    st.subheader("🔍 Indicator Dashboard")
    
    # Row 1: Core Breadth
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>$NYHL Cumulative (Regime)</strong><br>
            Value: {data['nyhl_cum']:,}<br>
            vs 200-SMA: {data['nyhl_sma200']:,}<br>
            MACD: {data['nyhl_macd_hist']:+.1f} | TSI: {data['nyhl_tsi']:.1f}<br>
            {'✅ Bullish' if data['nyhl_cum'] > data['nyhl_sma200'] else '❌ Bearish'}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>$BPSPX (Early Warning)</strong><br>
            Bullish %: {data['bpspx_value']}%<br>
            RSI(14): {data['bpspx_rsi14']:.1f} | MACD: {data['bpspx_macd_hist']:+.3f}<br>
            {'⚠️ Distribution' if data['bpspx_rsi14'] < 40 else '✅ Neutral/Bullish'}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <strong>$OEXA150R (Entry)</strong><br>
            % >150-DMA: {data['oexa150r_value']}%<br>
            CCI(14): {data['oexa150r_cci14']:.0f} | MACD: {data['oexa150r_macd_hist']:+.3f}<br>
            {'🟢 Buy Zone' if data['oexa150r_value'] < 30 else '🟡 Neutral' if data['oexa150r_value'] < 70 else '🔴 Overbought'}
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2: Dave Keller Additions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Sentiment (Contrarian)</strong><br>
            VIX: {data['vix']:.1f} | AAII Spread: {data['aaii_bull_bear_spread']:+.1f}%<br>
            NAAIM: {data['naaim_exposure']:.1f}%<br>
            {'⚠️ Complacent' if data['vix'] < 15 and data['aaii_bull_bear_spread'] > 20 else '✅ Balanced' if data['vix'] < 25 else '🟢 Fear = Opportunity'}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Credit & Rates</strong><br>
            HY Spread: {data['hy_spread_bps']} bps | 10Y Yield: {data['tnx_10yr_yield']:.2f}%<br>
            McClellan: {data['mcclellan_osc']:+.1f}<br>
            {'🔴 Stress' if data['hy_spread_bps'] > 450 else '⚠️ Watch' if data['hy_spread_bps'] > 350 else '✅ Normal'}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Leadership Analysis</strong><br>
            Small/Large: {data['smallcap_largecap_ratio']:.2f}<br>
            Equal Weight/SPX: {data['equalweight_sp500_ratio']:.2f}<br>
            Semis/SPX: {data['semis_spx_ratio']:.2f}<br>
            {'⚠️ Narrow' if data['equalweight_sp500_ratio'] < 0.88 else '✅ Broad Participation'}
        </div>
        """, unsafe_allow_html=True)
    
    # === RAW SIGNAL BREAKDOWN ===
    with st.expander("🔎 View Signal Logic Details"):
        st.write("**Detected Signals:**")
        for s in raw_signals:
            icon = "✅" if any(x in s for x in ['bullish', 'oversold_buy', 'risk_on', 'extreme_fear', 'naaim_oversold', 'credit_easy', 'mcclellan_oversold', 'improving']) else "⚠️" if 'neutral' in s or 'improving' in s else "❌"
            st.write(f"{icon} `{s.replace('_', ' ').title()}`")
        
        st.write("\n**Scoring Summary:**")
        st.write(f"- Bullish components: **{bullish_score}**")
        st.write(f"- Bearish components: **{bearish_score}**")
        st.write(f"- Net signal: **{'Bullish' if bullish_score > bearish_score else 'Bearish'}**")
    
    # === ALERTS & KEY LEVELS ===
    st.subheader("🚨 Key Levels to Watch")
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    with alert_col1:
        st.info(f"**$NYHL Caution**: Close below {data['nyhl_sma50']:,} (50-SMA)")
    with alert_col2:
        st.warning(f"**$BPSPX Warning**: RSI(14) < 35 OR value < 50")
    with alert_col3:
        st.error(f"**Credit Stress**: HY Spread > 450 bps OR VIX > 25")
    
    # === SIDEBAR: DATA SOURCE & SETTINGS ===
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selector
        source = st.selectbox("Data Source", ["Mock Data (Demo)", "yfinance", "CSV Upload", "API Integration"])
        if source == "CSV Upload":
            uploaded = st.file_uploader("📤 Upload CSV", type=['csv'])
            if uploaded:
                st.success("✅ CSV loaded! (Implement parsing in fetch_indicator_data())")
        
        # Signal sensitivity
        sensitivity = st.slider("Signal Sensitivity", 1, 10, 5, 
                               help="Lower = fewer signals, Higher = more responsive")
        
        # ETF selection toggle
        st.markdown("**Enabled ETFs:**")
        etf_toggles = {
            'SPY': st.checkbox('SPY (S&P 500)', True),
            'QQQ': st.checkbox('QQQ (Nasdaq 100)', True),
            'IWM': st.checkbox('IWM (Small Cap)', True),
            'IEF': st.checkbox('IEF (7-10Y Treasuries)', True),
            'TLT': st.checkbox('TLT (20+Y Treasuries)', True),
            'GLD': st.checkbox('GLD (Gold)', True),
            'VXX': st.checkbox('VXX (Volatility)', False),
            'SHY': st.checkbox('SHY (1-3Y Treasuries)', False),
            'CASH': st.checkbox('Cash Allocation', True),
        }
        
        st.markdown("---")
        st.markdown("**Signal Reference:**")
        st.markdown("""
        | Signal | Action |
        |--------|--------|
        | 🟢 BUY | Add equities, reduce hedges |
        | 🔵 TAKE A DIP | Buy oversold breadth + momentum |
        | 🟣 IMPROVING | Gradually add risk |
        | 🟡 HOLD | Maintain core allocation |
        | 🟠 WARNING | Trim equities, add hedges |
        | 🩷 LOOK TO SELL | Reduce risk, raise cash |
        | 🔴 SELL | Defensive allocation |
        """)
        
        st.caption("⚠️ Not investment advice. Backtest before live use.")

if __name__ == "__main__":
    main()
