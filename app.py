import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================
# CONFIG & STYLING
# ============================================
st.set_page_config(page_title="SPY/IEF Signal Dashboard", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .signal-buy {background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .signal-hold {background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .signal-warning {background: linear-gradient(135deg, #f97316, #ea580c); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .signal-sell {background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;}
    .bullish {color: #10b981; font-weight: bold;}
    .bearish {color: #ef4444; font-weight: bold;}
    .neutral {color: #6b7280; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA SOURCES
# ============================================

def fetch_yfinance_prices():
    """Fetch ETF and SPX prices from yfinance"""
    try:
        import yfinance as yf
        tickers = ['SPY', 'IEF', '^GSPC']
        data = yf.download(tickers, period='1d', progress=False)['Close'].iloc[-1]
        return {
            'spy_price': round(data['SPY'], 2),
            'ief_price': round(data['IEF'], 2),
            'spx_price': round(data['^GSPC'], 2),
        }
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return {'spy_price': 598.45, 'ief_price': 96.32, 'spx_price': 6877.92}

def fetch_fred_data():
    """Fetch Treasury yields from FRED API (requires API key)"""
    try:
        from fredapi import Fred
        fred = Fred(api_key=st.secrets.get('FRED_API_KEY', 'your_key_here'))
        tnx = fred.get_series('GS10')  # 10-Year Treasury
        return {'tnx_10yr_yield': round(tnx.iloc[-1], 2)}
    except:
        return {'tnx_10yr_yield': 4.25}

def get_mock_data():
    """Mock data based on your StockCharts PDFs (Feb 24, 2026)"""
    return {
        'timestamp': datetime.now(),
        'spx_price': 6877.92,
        'nyhl_cum': 32941, 'nyhl_sma50': 29200, 'nyhl_sma200': 27441,
        'nyhl_macd_hist': 24.045, 'nyhl_cci7': 98.30,
        'bpspx_value': 60.00, 'bpspx_sma50': 60.98,
        'bpspx_rsi14': 36.47, 'bpspx_macd_hist': -0.470,
        'oexa150r_value': 63.00, 'oexa150r_cci14': -163.74,
        'bpnya_value': 60.83, 'bpnya_sma50': 62.38,
        'bpnya_tsi': -34.47, 'bpnya_cci20': -68.47,
        'spy_vxx_ratio': 24.05, 'spy_vxx_sma50': 25.06,
        'vix': 18.45, 'hy_spread_bps': 385,
        'mcclellan_osc': -15.4,
        'spy_price': 598.45, 'ief_price': 96.32,
    }

@st.cache_data(ttl=3600)
def fetch_data(source='mock', manual_inputs=None):
    """Main data fetcher with source selection"""
    
    # Start with mock data as base
    data = get_mock_data()
    data['timestamp'] = datetime.now()
    
    # Override with yfinance prices if selected
    if source == 'yfinance':
        prices = fetch_yfinance_prices()
        data.update(prices)
    
    # Override with FRED data if available
    if source == 'fred':
        fred_data = fetch_fred_data()
        data.update(fred_data)
    
    # Override with manual inputs from sidebar
    if manual_inputs:
        data.update(manual_inputs)
    
    return data

# ============================================
# SIGNAL ANALYSIS ENGINE
# ============================================

def analyze_signals(data, indicator_toggles):
    """
    Generate signals based on enabled indicators only.
    Returns: signal_label, signal_class, allocation, signal_df, attribution, scores
    """
    signal_details = []
    
    # === 1. PRIMARY REGIME: $NYHL Cumulative (ALWAYS ON) ===
    nyhl_signal = "BULLISH" if data['nyhl_cum'] > data['nyhl_sma200'] else "BEARISH"
    nyhl_status = "✅" if nyhl_signal == "BULLISH" else "❌"
    signal_details.append({
        'Indicator': '$NYHL Cumulative vs 200-SMA',
        'Reading': f"{data['nyhl_cum']:,} vs {data['nyhl_sma200']:,}",
        'Signal': nyhl_signal,
        'Weight': 4,
        'Icon': nyhl_status,
        'Enabled': True
    })
    
    # === 2. EARLY WARNING: $BPSPX ===
    if indicator_toggles.get('bpspx', True):
        bpspx_signal = "NEUTRAL"
        if data['bpspx_rsi14'] < 35:
            bpspx_signal = "OVERSOLD (Bullish)"
            icon = "✅"
        elif data['bpspx_rsi14'] > 65:
            bpspx_signal = "OVERBOUGHT (Bearish)"
            icon = "❌"
        elif data['bpspx_macd_hist'] < 0 and data['bpspx_value'] < data['bpspx_sma50']:
            bpspx_signal = "DISTRIBUTION (Bearish)"
            icon = "❌"
        else:
            icon = "⚪"
        
        signal_details.append({
            'Indicator': '$BPSPX RSI(14) + MACD',
            'Reading': f"RSI:{data['bpspx_rsi14']:.1f} | MACD:{data['bpspx_macd_hist']:+.3f}",
            'Signal': bpspx_signal,
            'Weight': 3,
            'Icon': icon,
            'Enabled': True
        })
    
    # === 3. ENTRY CONFIRMATION: $OEXA150R ===
    if indicator_toggles.get('oexa150r', True):
        if data['oexa150r_value'] < 30 and data['oexa150r_cci14'] < -100:
            oexa_signal = "OVERSOLD BUY (Bullish)"
            icon = "✅"
        elif data['oexa150r_value'] > 75:
            oexa_signal = "OVERBOUGHT (Bearish)"
            icon = "❌"
        else:
            oexa_signal = "NEUTRAL"
            icon = "⚪"
        
        signal_details.append({
            'Indicator': '$OEXA150R % Above 150-DMA',
            'Reading': f"{data['oexa150r_value']}% | CCI:{data['oexa150r_cci14']:.0f}",
            'Signal': oexa_signal,
            'Weight': 3,
            'Icon': icon,
            'Enabled': True
        })
    
    # === 4. RISK SENTIMENT: SPY:VXX + VIX ===
    if indicator_toggles.get('spy_vxx', True):
        if data['spy_vxx_ratio'] < data['spy_vxx_sma50'] and data['vix'] > 22:
            risk_signal = "RISK-OFF (Bearish)"
            icon = "❌"
        elif data['spy_vxx_ratio'] > data['spy_vxx_sma50'] and data['vix'] < 16:
            risk_signal = "RISK-ON (Bullish)"
            icon = "✅"
        else:
            risk_signal = "NEUTRAL"
            icon = "⚪"
        
        signal_details.append({
            'Indicator': 'SPY:VXX Ratio + VIX',
            'Reading': f"Ratio:{data['spy_vxx_ratio']:.2f} | VIX:{data['vix']:.1f}",
            'Signal': risk_signal,
            'Weight': 3,
            'Icon': icon,
            'Enabled': True
        })
    
    # === 5. MOMENTUM CROSS-CHECK: $BPNYA ===
    if indicator_toggles.get('bpnya', True):
        if data['bpnya_tsi'] < 0 and data['bpnya_cci20'] < -50:
            bpnya_signal = "MOMENTUM WEAK (Bearish)"
            icon = "❌"
        elif data['bpnya_tsi'] > 0 and data['bpnya_cci20'] > 50:
            bpnya_signal = "MOMENTUM STRONG (Bullish)"
            icon = "✅"
        else:
            bpnya_signal = "NEUTRAL"
            icon = "⚪"
        
        signal_details.append({
            'Indicator': '$BPNYA TSI + CCI',
            'Reading': f"TSI:{data['bpnya_tsi']:.1f} | CCI:{data['bpnya_cci20']:.0f}",
            'Signal': bpnya_signal,
            'Weight': 2,
            'Icon': icon,
            'Enabled': True
        })
    
    # === 6. CREDIT STRESS: High Yield Spread ===
    if indicator_toggles.get('credit', True):
        if data['hy_spread_bps'] > 450:
            credit_signal = "STRESS (Bearish)"
            icon = "❌"
        elif data['hy_spread_bps'] < 300:
            credit_signal = "EASY (Bullish)"
            icon = "✅"
        else:
            credit_signal = "NORMAL"
            icon = "⚪"
        
        signal_details.append({
            'Indicator': 'High Yield Spread',
            'Reading': f"{data['hy_spread_bps']} bps",
            'Signal': credit_signal,
            'Weight': 3,
            'Icon': icon,
            'Enabled': True
        })
    
    # === 7. BREADTH MOMENTUM: McClellan ===
    if indicator_toggles.get('mcclellan', True):
        if data['mcclellan_osc'] < -50:
            mcc_signal = "OVERSOLD (Bullish)"
            icon = "✅"
        elif data['mcclellan_osc'] > 50:
            mcc_signal = "OVERBOUGHT (Bearish)"
            icon = "❌"
        else:
            mcc_signal = "NEUTRAL"
            icon = "⚪"
        
        signal_details.append({
            'Indicator': 'McClellan Oscillator',
            'Reading': f"{data['mcclellan_osc']:+.1f}",
            'Signal': mcc_signal,
            'Weight': 2,
            'Icon': icon,
            'Enabled': True
        })
    
    # === SCORING (Only enabled indicators) ===
    df = pd.DataFrame(signal_details)
    df_enabled = df[df['Enabled'] == True]
    
    bullish_score = df_enabled[df_enabled['Signal'].str.contains(
        'Bullish|OVERSOLD BUY|OVERSOLD \\(Bullish\\)|EASY', case=False, na=False
    )]['Weight'].sum()
    
    bearish_score = df_enabled[df_enabled['Signal'].str.contains(
        'Bearish|OVERBOUGHT|DISTRIBUTION|WEAK|STRESS|RISK-OFF', case=False, na=False
    )]['Weight'].sum()
    
    # === FINAL SIGNAL (4-TIER) ===
    if data['nyhl_cum'] < data['nyhl_sma200'] or (bearish_score >= 7 and bullish_score < 4):
        signal_label, signal_class = "SELL", "signal-sell"
        allocation = {'SPY': 20, 'IEF': 60, 'CASH': 20}
    elif bearish_score > bullish_score and data['nyhl_cum'] > data['nyhl_sma200']:
        signal_label, signal_class = "WARNING", "signal-warning"
        allocation = {'SPY': 40, 'IEF': 45, 'CASH': 15}
    elif data['bpspx_rsi14'] < 35 and data['oexa150r_value'] < 35 and data['nyhl_cum'] > data['nyhl_sma200']:
        signal_label, signal_class = "BUY DIP", "signal-buy"
        allocation = {'SPY': 80, 'IEF': 15, 'CASH': 5}
    else:
        signal_label, signal_class = "HOLD", "signal-hold"
        allocation = {'SPY': 65, 'IEF': 25, 'CASH': 10}
    
    # === ATTRIBUTION DATA FOR CHART ===
    attribution = []
    for _, row in df_enabled.iterrows():
        if 'Bullish' in row['Signal'] or 'OVERSOLD' in row['Signal'] or 'EASY' in row['Signal']:
            sentiment = 'Bullish'
        elif 'Bearish' in row['Signal'] or 'OVERBOUGHT' in row['Signal'] or 'STRESS' in row['Signal']:
            sentiment = 'Bearish'
        else:
            sentiment = 'Neutral'
        attribution.append({
            'Indicator': row['Indicator'].split(' ')[0],
            'Weight': row['Weight'],
            'Sentiment': sentiment
        })
    
    return signal_label, signal_class, allocation, df_enabled, attribution, bullish_score, bearish_score

# ============================================
# MANUAL DATA ENTRY WIDGETS
# ============================================

def manual_entry_widgets():
    """Sidebar widgets for manual data entry"""
    st.sidebar.markdown("### 📝 Manual Data Entry")
    st.sidebar.caption("Override mock data with your readings")
    
    manual = {}
    
    with st.sidebar.expander("$NYHL Cumulative"):
        manual['nyhl_cum'] = st.number_input("Value", value=32941, step=100)
        manual['nyhl_sma200'] = st.number_input("200-SMA", value=27441, step=100)
    
    with st.sidebar.expander("$BPSPX"):
        manual['bpspx_value'] = st.number_input("Bullish %", value=60.00, step=0.5)
        manual['bpspx_rsi14'] = st.number_input("RSI(14)", value=36.47, step=0.5)
        manual['bpspx_macd_hist'] = st.number_input("MACD Hist", value=-0.470, step=0.01)
        manual['bpspx_sma50'] = st.number_input("50-SMA", value=60.98, step=0.5)
    
    with st.sidebar.expander("$OEXA150R"):
        manual['oexa150r_value'] = st.number_input("% Above 150-DMA", value=63.00, step=0.5)
        manual['oexa150r_cci14'] = st.number_input("CCI(14)", value=-163.74, step=1.0)
    
    with st.sidebar.expander("SPY:VXX + VIX"):
        manual['spy_vxx_ratio'] = st.number_input("SPY:VXX Ratio", value=24.05, step=0.1)
        manual['spy_vxx_sma50'] = st.number_input("50-SMA", value=25.06, step=0.1)
        manual['vix'] = st.number_input("VIX", value=18.45, step=0.5)
    
    with st.sidebar.expander("$BPNYA"):
        manual['bpnya_tsi'] = st.number_input("TSI", value=-34.47, step=0.5)
        manual['bpnya_cci20'] = st.number_input("CCI(20)", value=-68.47, step=1.0)
    
    with st.sidebar.expander("Credit & Breadth"):
        manual['hy_spread_bps'] = st.number_input("HY Spread (bps)", value=385, step=5)
        manual['mcclellan_osc'] = st.number_input("McClellan", value=-15.4, step=0.5)
    
    return manual

def indicator_toggle_widgets():
    """Sidebar toggles for enabling/disabling indicators"""
    st.sidebar.markdown("### 🎛️ Indicator Toggles")
    st.sidebar.caption("Turn indicators on/off for signal calculation")
    
    toggles = {
        'bpspx': st.sidebar.checkbox('$BPSPX (Early Warning)', value=True),
        'oexa150r': st.sidebar.checkbox('$OEXA150R (Entry)', value=True),
        'spy_vxx': st.sidebar.checkbox('SPY:VXX + VIX (Risk)', value=True),
        'bpnya': st.sidebar.checkbox('$BPNYA (Momentum)', value=True),
        'credit': st.sidebar.checkbox('Credit Spread (Leading)', value=True),
        'mcclellan': st.sidebar.checkbox('McClellan (Breadth)', value=True),
    }
    
    return toggles

# ============================================
# MAIN DASHBOARD
# ============================================

def main():
    st.title("📊 SPY vs IEF Signal Dashboard")
    st.markdown("*Modular breadth-momentum framework with indicator toggles*")
    
    # === SIDEBAR CONFIGURATION ===
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        source = st.selectbox(
            "Data Source",
            ["Mock Data (Demo)", "yfinance (Prices)", "FRED API (Rates)", "Manual Entry"],
            help="Select where to fetch indicator data"
        )
        
        # Manual entry widgets
        manual_inputs = None
        if source == "Manual Entry":
            manual_inputs = manual_entry_widgets()
        
        # Indicator toggles
        indicator_toggles = indicator_toggle_widgets()
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Signal Logic:**")
        st.markdown("""
        | Signal | Trigger |
        |--------|---------|
        | 🟢 BUY DIP | NYHL>200SMA + BPSPX RSI<35 + OEXA<35 |
        | 🟡 HOLD | NYHL>200SMA, mixed momentum |
        | 🟠 WARNING | NYHL>200SMA but BPSPX RSI<40 + risk-off |
        | 🔴 SELL | NYHL<200SMA OR credit stress + distribution |
        """)
        
        st.caption("⚠️ Not investment advice. Test before live use.")
    
    # === FETCH DATA ===
    data = fetch_data(source='mock', manual_inputs=manual_inputs)
    
    # === GENERATE SIGNALS ===
    signal_label, signal_class, allocation, signal_df, attribution, bull_score, bear_score = analyze_signals(data, indicator_toggles)
    
    # === TOP SIGNAL BANNER ===
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<div class='{signal_class}'>{signal_label}</div>", unsafe_allow_html=True)
        st.caption(f"Updated: {data['timestamp'].strftime('%Y-%m-%d %H:%M')} | SPX: ${data['spx_price']:,.2f}")
    with col2:
        regime = "🟢 Bullish" if data['nyhl_cum'] > data['nyhl_sma200'] else "🔴 Bearish"
        st.metric("Primary Regime", regime)
    
    # === ALLOCATION (SPY/IEF ONLY) ===
    st.subheader("🎯 Recommended Allocation")
    alloc_df = pd.DataFrame([
        {'ETF': 'SPY', 'Allocation': allocation['SPY'], 'Color': '#10b981'},
        {'ETF': 'IEF', 'Allocation': allocation['IEF'], 'Color': '#3b82f6'},
        {'ETF': 'CASH', 'Allocation': allocation['CASH'], 'Color': '#6b7280'}
    ])
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(alloc_df, values='Allocation', names='ETF', 
                     color='ETF', color_discrete_map={'SPY':'#10b981','IEF':'#3b82f6','CASH':'#6b7280'},
                     hole=0.4)
        fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(alloc_df, x='Allocation', y='ETF', orientation='h',
                     color='ETF', color_discrete_map={'SPY':'#10b981','IEF':'#3b82f6','CASH':'#6b7280'},
                     text='Allocation', range_x=[0,100])
        fig.update_layout(showlegend=False, xaxis_title='% Allocation', yaxis_title='', margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # === ACTION BUTTONS ===
    st.markdown("### ⚡ Quick Actions")
    cols = st.columns(3)
    actions = [
        ("✅ Add SPY", allocation['SPY'] >= 65),
        ("🛡️ Add IEF", allocation['IEF'] >= 40),
        ("💵 Raise Cash", allocation['CASH'] >= 15),
    ]
    for idx, (label, enabled) in enumerate(actions):
        with cols[idx]:
            if enabled:
                st.button(label, type="primary" if idx==0 else "secondary", use_container_width=True)
            else:
                st.button(label, disabled=True, use_container_width=True)
    
    # === 🎯 SIGNAL ATTRIBUTION CHART ===
    st.subheader("🔍 Signal Attribution: What's Driving the Signal?")
    
    attr_df = pd.DataFrame(attribution)
    
    if not attr_df.empty:
        fig_attr = go.Figure()
        
        # Bullish bars (green, to the right)
        bullish = attr_df[attr_df['Sentiment'] == 'Bullish']
        if not bullish.empty:
            fig_attr.add_trace(go.Bar(
                y=bullish['Indicator'],
                x=bullish['Weight'],
                name='Bullish',
                orientation='h',
                marker_color='#10b981',
                text=bullish['Weight'],
                textposition='inside'
            ))
        
        # Bearish bars (red, to the left - negative values)
        bearish = attr_df[attr_df['Sentiment'] == 'Bearish']
        if not bearish.empty:
            fig_attr.add_trace(go.Bar(
                y=bearish['Indicator'],
                x=[-w for w in bearish['Weight']],
                name='Bearish',
                orientation='h',
                marker_color='#ef4444',
                text=bearish['Weight'],
                textposition='inside'
            ))
        
        # Neutral bars (gray, minimal)
        neutral = attr_df[attr_df['Sentiment'] == 'Neutral']
        if not neutral.empty:
            fig_attr.add_trace(go.Bar(
                y=neutral['Indicator'],
                x=[w*0.3 for w in neutral['Weight']],
                name='Neutral',
                orientation='h',
                marker_color='#9ca3af',
                showlegend=False
            ))
        
        fig_attr.update_layout(
            barmode='relative',
            xaxis_title='Signal Pressure ← Bearish | Bullish →',
            yaxis_title='',
            height=400,
            margin=dict(t=20, b=20, l=150, r=20),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_attr, use_container_width=True)
    else:
        st.warning("No indicators enabled for attribution chart")
    
    # Score summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Bullish Score", bull_score, delta=f"vs {bear_score} bearish")
    with col2:
        net = bull_score - bearish_score
        st.metric("Net Signal", f"{net:+d}", "Bullish" if net > 0 else "Bearish")
    with col3:
        st.metric("Confidence", f"{max(bull_score, bear_score)}/15", 
                  "High" if max(bull_score, bear_score) >= 10 else "Medium" if max(bull_score, bear_score) >= 7 else "Low")
    
    # === DETAILED SIGNAL BREAKDOWN TABLE ===
    st.subheader("📋 Full Signal Breakdown")
    
    display_df = signal_df.copy()
    display_df['Signal'] = display_df.apply(
        lambda row: f"{row['Icon']} {row['Signal']}", axis=1
    )
    
    st.dataframe(
        display_df[['Icon', 'Indicator', 'Reading', 'Signal', 'Weight']],
        column_config={
            "Icon": "Status",
            "Indicator": "Indicator",
            "Reading": "Current Reading",
            "Signal": "Interpretation",
            "Weight": "Importance"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # === KEY INDICATOR CARDS ===
    st.subheader("🔑 Key Readings at a Glance")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>$NYHL Cumulative (Regime)</strong><br>
            {data['nyhl_cum']:,} vs 200-SMA: {data['nyhl_sma200']:,}<br>
            <span class="{'bullish' if data['nyhl_cum'] > data['nyhl_sma200'] else 'bearish'}">
            {'✅ Bullish Regime' if data['nyhl_cum'] > data['nyhl_sma200'] else '❌ Bearish Regime'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>$BPSPX (Early Warning)</strong><br>
            {data['bpspx_value']}% | RSI: {data['bpspx_rsi14']:.1f}<br>
            <span class="{'bearish' if data['bpspx_rsi14'] < 40 else 'bullish' if data['bpspx_rsi14'] > 60 else 'neutral'}">
            {'⚠️ Distribution' if data['bpspx_rsi14'] < 40 else '✅ Neutral/Bullish'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Risk Sentiment</strong><br>
            VIX: {data['vix']:.1f} | SPY:VXX: {data['spy_vxx_ratio']:.2f}<br>
            <span class="{'bearish' if data['spy_vxx_ratio'] < data['spy_vxx_sma50'] else 'bullish'}">
            {'⚠️ Risk-Off' if data['spy_vxx_ratio'] < data['spy_vxx_sma50'] else '✅ Risk-On'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # === ALERTS ===
    st.subheader("🚨 Watch Levels")
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.info(f"**$NYHL Caution**: Close below {data['nyhl_sma200']:,} (200-SMA)")
    with ac2:
        st.warning(f"**$BPSPX Warning**: RSI(14) < 35 OR value < 50")
    with ac3:
        st.error(f"**Credit Stress**: HY Spread > 450 bps")
    
    # === EXPORT BUTTON ===
    st.subheader("💾 Export")
    csv_data = signal_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Signal Breakdown (CSV)",
        data=csv_data,
        file_name=f"signal_breakdown_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )

if __name__ == "__main__":
    main()
