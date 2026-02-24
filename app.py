import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

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
    .extreme-complacency {background: #dc2626; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: bold;}
    .extreme-fear {background: #10b981; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA FETCHING
# ============================================

@st.cache_data(ttl=3600)
def fetch_spy_history(years=15):
    """Fetch SPY historical data"""
    end = datetime.now()
    start = end - timedelta(days=years*365)
    spy = yf.download('SPY', start=start, end=end, progress=False)
    return spy

@st.cache_data(ttl=3600)
def fetch_put_call_data():
    """
    Fetch CPCE and CPC data from FRED or yfinance
    CPCE: CBOE Equity Put/Call Ratio
    CPC: CBOE Total Put/Call Ratio
    """
    try:
        # Try to fetch from FRED (Federal Reserve Economic Data)
        from fredapi import Fred
        fred = Fred(api_key=st.secrets.get('FRED_API_KEY', 'demo_key'))
        
        # CPCE - Equity Put/Call Ratio
        cpce = fred.get_series('CPCE')
        # CPC - Total Put/Call Ratio  
        cpc = fred.get_series('CPC')
        
        return cpce, cpc
    except:
        # Mock data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
        cpce = pd.Series(np.random.uniform(0.45, 0.85, len(dates)), index=dates, name='CPCE')
        cpc = pd.Series(np.random.uniform(0.65, 1.15, len(dates)), index=dates, name='CPC')
        return cpce, cpc

@st.cache_data(ttl=3600)
def generate_mock_breadth_data(spy_df):
    """Generate simulated breadth indicators"""
    df = spy_df.copy()
    
    # Simulate $NYHL Cumulative
    df['nyhl_cum'] = df['Close'].pct_change().rolling(20).sum().cumsum() * 1000 + 25000
    df['nyhl_sma200'] = df['nyhl_cum'].rolling(200).mean()
    
    # Simulate $BPSPX
    momentum = df['Close'].pct_change(20).rolling(10).mean()
    df['bpspx_value'] = 50 + momentum * 30 + np.random.normal(0, 5, len(df))
    df['bpspx_value'] = df['bpspx_value'].clip(10, 90)
    df['bpspx_rsi14'] = 50 + momentum * 40 + np.random.normal(0, 8, len(df))
    df['bpspx_rsi14'] = df['bpspx_rsi14'].clip(10, 90)
    df['bpspx_macd_hist'] = momentum * 2 + np.random.normal(0, 0.3, len(df))
    
    # Simulate $OEXA150R
    ma150 = df['Close'].rolling(150).mean()
    above_ma = (df['Close'] > ma150).astype(float)
    df['oexa150r_value'] = above_ma.rolling(20).mean() * 100 + np.random.normal(0, 8, len(df))
    df['oexa150r_value'] = df['oexa150r_value'].clip(10, 90)
    df['oexa150r_cci14'] = (df['Close'] - df['Close'].rolling(14).mean()) / df['Close'].rolling(14).std() * 100
    
    # Simulate SPY:VXX ratio
    returns = df['Close'].pct_change()
    vol = returns.rolling(20).std()
    df['spy_vxx_ratio'] = 1 / (vol * 100) + np.random.normal(0, 3, len(df))
    df['spy_vxx_sma50'] = df['spy_vxx_ratio'].rolling(50).mean()
    
    # Simulate VIX
    df['vix'] = 20 - momentum * 15 + np.random.normal(0, 4, len(df))
    df['vix'] = df['vix'].clip(10, 50)
    
    # Simulate credit spread
    df['hy_spread_bps'] = 350 - momentum * 100 + np.random.normal(0, 30, len(df))
    df['hy_spread_bps'] = df['hy_spread_bps'].clip(200, 600)
    
    # Simulate McClellan
    df['mcclellan_osc'] = momentum * 80 + np.random.normal(0, 20, len(df))
    
    return df.dropna()

# ============================================
# SIGNAL ENGINE
# ============================================

def analyze_signals_with_putcall(data, cpce_current, cpc_current):
    """Enhanced signal analysis including put/call ratios"""
    signal_details = []
    
    # === PRIMARY REGIME: $NYHL ===
    nyhl_signal = "BULLISH" if data['nyhl_cum'] > data['nyhl_sma200'] else "BEARISH"
    signal_details.append({
        'Indicator': '$NYHL Cumulative',
        'Reading': f"{data['nyhl_cum']:,} vs {data['nyhl_sma200']:,}",
        'Signal': nyhl_signal,
        'Weight': 4,
        'Icon': "✅" if nyhl_signal == "BULLISH" else "❌"
    })
    
    # === PUT/CALL RATIOS (CONTRARIAN) ===
    # CPCE Analysis
    if cpce_current < 0.55:
        cpce_signal = "EXTREME COMPLACENCY (Bearish)"
        cpce_icon = "❌"
        cpce_weight = 4
    elif cpce_current < 0.65:
        cpce_signal = "COMPLACENCY (Bearish)"
        cpce_icon = "⚠️"
        cpce_weight = 3
    elif cpce_current > 0.90:
        cpce_signal = "EXTREME FEAR (Bullish)"
        cpce_icon = "✅"
        cpce_weight = 4
    elif cpce_current > 0.80:
        cpce_signal = "FEAR (Bullish)"
        cpce_icon = "✅"
        cpce_weight = 3
    else:
        cpce_signal = "NEUTRAL"
        cpce_icon = "⚪"
        cpce_weight = 2
    
    signal_details.append({
        'Indicator': 'CPCE Put/Call Ratio',
        'Reading': f"{cpce_current:.3f}",
        'Signal': cpce_signal,
        'Weight': cpce_weight,
        'Icon': cpce_icon
    })
    
    # CPC Analysis
    if cpc_current < 0.65:
        cpc_signal = "COMPLACENCY (Bearish)"
        cpc_icon = "⚠️"
        cpc_weight = 3
    elif cpc_current > 1.10:
        cpc_signal = "PANIC (Bullish)"
        cpc_icon = "✅"
        cpc_weight = 4
    elif cpc_current > 0.95:
        cpc_signal = "CAUTION (Bullish)"
        cpc_icon = "✅"
        cpc_weight = 2
    else:
        cpc_signal = "NEUTRAL"
        cpc_icon = "⚪"
        cpc_weight = 2
    
    signal_details.append({
        'Indicator': 'CPC Total Put/Call',
        'Reading': f"{cpc_current:.3f}",
        'Signal': cpc_signal,
        'Weight': cpc_weight,
        'Icon': cpc_icon
    })
    
    # === $BPSPX ===
    if data['bpspx_rsi14'] < 35:
        bpspx_signal = "OVERSOLD (Bullish)"
        bpspx_icon = "✅"
        bpspx_weight = 3
    elif data['bpspx_rsi14'] > 65:
        bpspx_signal = "OVERBOUGHT (Bearish)"
        bpspx_icon = "❌"
        bpspx_weight = 3
    else:
        bpspx_signal = "NEUTRAL"
        bpspx_icon = "⚪"
        bpspx_weight = 2
    
    signal_details.append({
        'Indicator': '$BPSPX RSI(14)',
        'Reading': f"{data['bpspx_rsi14']:.1f}",
        'Signal': bpspx_signal,
        'Weight': bpspx_weight,
        'Icon': bpspx_icon
    })
    
    # === $OEXA150R ===
    if data['oexa150r_value'] < 30:
        oexa_signal = "OVERSOLD (Bullish)"
        oexa_icon = "✅"
        oexa_weight = 3
    elif data['oexa150r_value'] > 75:
        oexa_signal = "OVERBOUGHT (Bearish)"
        oexa_icon = "❌"
        oexa_weight = 3
    else:
        oexa_signal = "NEUTRAL"
        oexa_icon = "⚪"
        oexa_weight = 2
    
    signal_details.append({
        'Indicator': '$OEXA150R',
        'Reading': f"{data['oexa150r_value']:.1f}%",
        'Signal': oexa_signal,
        'Weight': oexa_weight,
        'Icon': oexa_icon
    })
    
    # === RISK SENTIMENT ===
    if data['spy_vxx_ratio'] < data['spy_vxx_sma50'] and data['vix'] > 22:
        risk_signal = "RISK-OFF (Bearish)"
        risk_icon = "❌"
        risk_weight = 3
    else:
        risk_signal = "NEUTRAL/RISK-ON"
        risk_icon = "✅"
        risk_weight = 2
    
    signal_details.append({
        'Indicator': 'SPY:VXX + VIX',
        'Reading': f"Ratio:{data['spy_vxx_ratio']:.2f} | VIX:{data['vix']:.1f}",
        'Signal': risk_signal,
        'Weight': risk_weight,
        'Icon': risk_icon
    })
    
    # === SCORING ===
    df = pd.DataFrame(signal_details)
    bullish_score = df[df['Signal'].str.contains('Bullish|OVERSOLD|FEAR', case=False, na=False)]['Weight'].sum()
    bearish_score = df[df['Signal'].str.contains('Bearish|OVERBOUGHT|COMPLACENCY', case=False, na=False)]['Weight'].sum()
    
    # === FINAL SIGNAL ===
    # Check for extreme complacency (market top warning)
    if cpce_current < 0.55 or (cpce_current < 0.65 and cpc_current < 0.70):
        signal_label, signal_class = "WARNING - COMPLACENCY", "signal-warning"
        allocation = {'SPY': 40, 'IEF': 45, 'CASH': 15}
    # Check for extreme fear (market bottom opportunity)
    elif cpce_current > 0.90 or cpc_current > 1.10:
        signal_label, signal_class = "BUY DIP - FEAR", "signal-buy"
        allocation = {'SPY': 80, 'IEF': 15, 'CASH': 5}
    # Normal signal logic
    elif data['nyhl_cum'] < data['nyhl_sma200'] or bearish_score >= 10:
        signal_label, signal_class = "SELL", "signal-sell"
        allocation = {'SPY': 20, 'IEF': 60, 'CASH': 20}
    elif bearish_score > bullish_score:
        signal_label, signal_class = "WARNING", "signal-warning"
        allocation = {'SPY': 40, 'IEF': 45, 'CASH': 15}
    elif bullish_score - bearish_score >= 3:
        signal_label, signal_class = "BUY", "signal-buy"
        allocation = {'SPY': 75, 'IEF': 20, 'CASH': 5}
    else:
        signal_label, signal_class = "HOLD", "signal-hold"
        allocation = {'SPY': 65, 'IEF': 25, 'CASH': 10}
    
    return signal_label, signal_class, allocation, df, bullish_score, bearish_score

# ============================================
# MAIN DASHBOARD
# ============================================

def main():
    st.title("📊 SPY vs IEF Signal Dashboard")
    st.markdown("*Breadth-Momentum framework with **Put/Call Ratio contrarian signals***")
    
    # === SIDEBAR ===
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Put/Call manual entry
        st.subheader("📊 Put/Call Ratios")
        cpce_manual = st.number_input("CPCE (Equity P/C)", min_value=0.30, max_value=1.50, value=0.62, step=0.01)
        cpc_manual = st.number_input("CPC (Total P/C)", min_value=0.40, max_value=2.00, value=0.85, step=0.01)
        
        st.info(f"""
        **Current Readings:**
        - CPCE: {cpce_manual:.3f} {'⚠️ COMPLACENT' if cpce_manual < 0.65 else '✅ NEUTRAL' if cpce_manual < 0.85 else '🟢 FEAR'}
        - CPC: {cpc_manual:.3f} {'⚠️ COMPLACENT' if cpc_manual < 0.70 else '✅ NEUTRAL' if cpc_manual < 0.95 else '🟢 FEAR'}
        """)
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.caption("⚠️ Not investment advice. Test before live use.")
    
    # === FETCH DATA ===
    cpce_series, cpc_series = fetch_put_call_data()
    
    # Use manual inputs or latest from series
    cpce_current = cpce_manual if cpce_manual != 0.62 else cpce_series.iloc[-1]
    cpc_current = cpc_manual if cpc_manual != 0.85 else cpc_series.iloc[-1]
    
    # Mock breadth data (replace with real data source)
    data = {
        'timestamp': datetime.now(),
        'spx_price': 6877.92,
        'nyhl_cum': 32941, 'nyhl_sma200': 27441,
        'bpspx_rsi14': 36.47, 'bpspx_value': 60.00,
        'oexa150r_value': 63.00, 'oexa150r_cci14': -163.74,
        'spy_vxx_ratio': 24.05, 'spy_vxx_sma50': 25.06,
        'vix': 18.45,
    }
    
    # === GENERATE SIGNALS ===
    signal_label, signal_class, allocation, signal_df, bull_score, bear_score = analyze_signals_with_putcall(
        data, cpce_current, cpc_current
    )
    
    # === DISPLAY CURRENT SIGNAL ===
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<div class='{signal_class}'>{signal_label}</div>", unsafe_allow_html=True)
        st.caption(f"Updated: {data['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    with col2:
        regime = "🟢 Bullish" if data['nyhl_cum'] > data['nyhl_sma200'] else "🔴 Bearish"
        st.metric("Primary Regime", regime)
    
    # === PUT/CALL RATIO VISUALIZATION ===
    st.subheader("📊 Put/Call Ratio Analysis (Contrarian Indicator)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # CPCE gauge
        cpce_color = "🔴" if cpce_current < 0.60 else "🟡" if cpce_current < 0.75 else "🟢"
        st.metric("CPCE Ratio", f"{cpce_color} {cpce_current:.3f}", 
                  "Extreme Complacency" if cpce_current < 0.55 else 
                  "Complacency" if cpce_current < 0.65 else
                  "Fear" if cpce_current > 0.90 else "Neutral")
    with col2:
        # CPC gauge
        cpc_color = "🔴" if cpc_current < 0.65 else "🟡" if cpc_current < 0.85 else "🟢"
        st.metric("CPC Ratio", f"{cpc_color} {cpc_current:.3f}",
                  "Complacency" if cpc_current < 0.70 else
                  "Fear" if cpc_current > 1.10 else "Neutral")
    with col3:
        # Combined signal
        if cpce_current < 0.60 or cpc_current < 0.70:
            st.metric("Sentiment", "⚠️ TOO BULLISH", "Market Top Risk")
        elif cpce_current > 0.85 or cpc_current > 1.05:
            st.metric("Sentiment", "🟢 TOO BEARISH", "Buying Opportunity")
        else:
            st.metric("Sentiment", "⚪ BALANCED", "Neutral")
    
    # Historical chart
    fig_pc = go.Figure()
    fig_pc.add_trace(go.Scatter(x=cpce_series.index, y=cpce_series, name='CPCE', line=dict(color='#ef4444')))
    fig_pc.add_trace(go.Scatter(x=cpc_series.index, y=cpc_series, name='CPC', line=dict(color='#3b82f6')))
    fig_pc.add_hline(y=0.60, line_dash="dash", line_color="red", annotation_text="Complacency Zone")
    fig_pc.add_hline(y=0.90, line_dash="dash", line_color="green", annotation_text="Fear Zone")
    fig_pc.update_layout(title="Put/Call Ratios - Historical Context", height=300, showlegend=True)
    st.plotly_chart(fig_pc, width="stretch")
    
    # === ALLOCATION ===
    st.subheader("🎯 Recommended Allocation")
    alloc_df = pd.DataFrame([
        {'ETF': 'SPY', 'Allocation': allocation['SPY']},
        {'ETF': 'IEF', 'Allocation': allocation['IEF']},
        {'ETF': 'CASH', 'Allocation': allocation['CASH']}
    ])
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(alloc_df, values='Allocation', names='ETF', 
                     color='ETF', color_discrete_map={'SPY':'#10b981','IEF':'#3b82f6','CASH':'#6b7280'},
                     hole=0.4)
        fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.bar(alloc_df, x='Allocation', y='ETF', orientation='h',
                     color='ETF', color_discrete_map={'SPY':'#10b981','IEF':'#3b82f6','CASH':'#6b7280'},
                     text='Allocation', range_x=[0,100])
        fig.update_layout(showlegend=False, xaxis_title='% Allocation', margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, width="stretch")
    
    # === SIGNAL BREAKDOWN ===
    st.subheader("📋 Signal Breakdown")
    st.dataframe(signal_df, use_container_width=True, hide_index=True)
    
    # Score summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Bullish Score", bull_score, delta=f"vs {bear_score} bearish")
    with col2:
        net = bull_score - bear_score
        st.metric("Net Signal", f"{net:+d}", "Bullish" if net > 0 else "Bearish")

if __name__ == "__main__":
    main()
