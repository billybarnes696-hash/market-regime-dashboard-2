import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="SPY/IEF Rotation Dashboard", layout="wide", page_icon="🔄")

st.markdown("""
<style>
    .signal-spy {background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .signal-ief {background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .signal-wait {background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA (MOCK - Replace with real source later)
# ============================================

@st.cache_data(ttl=3600)
def generate_mock_weekly_data(weeks=260):  # ~5 years of weekly data
    """Generate weekly mock data for swing/position trading"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=weeks, freq='W-FRI')
    
    # Mock SPY weekly returns (less noisy than daily)
    weekly_ret = np.random.normal(0.0015, 0.025, weeks)  # ~7.8% annual, 12.5% weekly vol
    prices = 100 * np.cumprod(1 + weekly_ret)
    spy_df = pd.DataFrame({'Close': prices}, index=dates)
    
    # Mock breadth indicators (weekly aggregation)
    df = spy_df.copy()
    ret = df['Close'].pct_change().fillna(0)
    mom = ret.rolling(4).mean()  # 4-week momentum
    
    # $NYHL Cumulative (smoothed)
    df['nyhl_cum'] = (mom.cumsum() * 20000 + 30000).ffill()
    df['nyhl_sma20'] = df['nyhl_cum'].rolling(20).mean()  # 20-week = ~5 months
    
    # $BPSPX (mean-reverting, weekly)
    df['bpspx_rsi'] = (50 + mom * 35).clip(25, 75)
    df['bpspx_macd'] = mom.rolling(3).mean() - mom.rolling(6).mean()
    
    # $OEXA150R (participation)
    df['oexa_val'] = (50 + mom * 80).clip(25, 75)
    df['oexa_cci'] = ((df['Close'] - df['Close'].rolling(4).mean()) / df['Close'].rolling(4).std().replace(0,1) * 100)
    
    # Risk sentiment
    vol = ret.rolling(4).std()
    df['spy_vxx'] = (1 / (vol * 100 + 0.01)).clip(12, 35)
    df['spy_vxx_sma10'] = df['spy_vxx'].rolling(10).mean()
    df['vix'] = (18 - mom * 12).clip(14, 35)
    
    # Canary indicators (weekly)
    df['hy_spread'] = (380 - mom * 80).clip(280, 520)
    df['mcclellan'] = mom * 70
    
    return df.dropna()

# ============================================
# SIGNAL ENGINE (WEEKLY/MONTHLY TIMEFRAME)
# ============================================

def get_rotation_signal_weekly(data):
    """
    Returns: 'SPY', 'IEF', or 'WAIT' for weekly/monthly rotations
    Designed for swing/position trading, not day trading
    """
    # Canary warnings (early warning system) - higher thresholds for weekly
    canary_score = 0
    if data.get('hy_spread', 380) > 470: canary_score += 2  # Credit stress (weekly threshold)
    if data.get('mcclellan', 0) < -40: canary_score += 1    # Breadth weak (weekly)
    if data.get('vix', 18) > 24 and data.get('spy_vxx', 25) < data.get('spy_vxx_sma10', 25): 
        canary_score += 1  # Risk-off (weekly)
    
    # Core signals (weekly timeframe)
    regime_bull = data['nyhl_cum'] > data['nyhl_sma20']  # 20-week SMA = ~5 months
    bpspx_bear = data['bpspx_rsi'] < 35 and data['bpspx_macd'] < 0  # Weekly oversold + momentum
    oexa_oversold = data['oexa_val'] < 30 and data['oexa_cci'] < -90  # Weekly oversold
    risk_off = data['spy_vxx'] < data['spy_vxx_sma10']
    
    # Decision logic (100% allocation, weekly signals)
    if canary_score >= 3 or (not regime_bull and risk_off):
        return "IEF", {'SPY': 0, 'IEF': 100}  # Defensive rotation
    elif oexa_oversold and regime_bull:
        return "SPY", {'SPY': 100, 'IEF': 0}  # Aggressive dip entry
    elif regime_bull and not bpspx_bear:
        return "SPY", {'SPY': 100, 'IEF': 0}  # Bullish trend continuation
    elif bpspx_bear or risk_off:
        return "WAIT", {'SPY': 50, 'IEF': 50}  # Transition zone (no action)
    else:
        return "WAIT", {'SPY': 50, 'IEF': 50}  # Default neutral

def generate_rotation_history_weekly(df):
    """Generate historical weekly rotation signals"""
    signals, allocations = [], []
    
    for idx, row in df.iterrows():
        data = row.to_dict()
        signal, alloc = get_rotation_signal_weekly(data)
        signals.append(signal)
        allocations.append(alloc)
    
    df = df.copy()
    df['signal'] = signals
    df['allocation'] = allocations
    df['SPY_%'] = df['allocation'].apply(lambda x: x['SPY'])
    df['IEF_%'] = df['allocation'].apply(lambda x: x['IEF'])
    
    # Calculate weekly returns
    df['spy_ret'] = df['Close'].pct_change().fillna(0)
    # Simplified IEF weekly return: positive when SPY negative (hedging)
    df['ief_ret'] = np.where(df['spy_ret'] < 0, -0.25 * df['spy_ret'], 0.0008)  # ~0.08% weekly drift
    
    def calc_ret(row):
        return (row['allocation']['SPY']/100 * row['spy_ret'] + 
                row['allocation']['IEF']/100 * row['ief_ret'])
    
    df['strat_ret'] = df.apply(calc_ret, axis=1)
    df['buyhold_cum'] = (1 + df['spy_ret']).cumprod() * 100
    df['strat_cum'] = (1 + df['strat_ret']).cumprod() * 100
    
    return df

# ============================================
# CURRENT SIGNAL (From Your Charts - Weekly View)
# ============================================

def get_current_rotation_weekly():
    """Current signal based on your StockCharts PDFs (weekly interpretation)"""
    data = {
        'nyhl_cum': 32941, 'nyhl_sma20': 30500,  # Weekly SMA approximation
        'bpspx_rsi': 36.47, 'bpspx_macd': -0.470,
        'oexa_val': 63.00, 'oexa_cci': -163.74,
        'vix': 18.45, 'spy_vxx': 24.05, 'spy_vxx_sma10': 24.80,
        'hy_spread': 385, 'mcclellan': -15.4,
    }
    return get_rotation_signal_weekly(data)

# ============================================
# TAB 1: CURRENT ROTATION (WEEKLY)
# ============================================

def render_current_tab():
    signal, alloc = get_current_rotation_weekly()
    
    # Big signal banner
    if signal == "SPY":
        sig_class = "signal-spy"
        msg = "🟢 ROTATE TO SPY (100%) • Weekly Signal"
    elif signal == "IEF":
        sig_class = "signal-ief"
        msg = "🔵 ROTATE TO IEF (100%) • Weekly Signal"
    else:
        sig_class = "signal-wait"
        msg = "🟡 HOLD CURRENT • No Weekly Change"
    
    st.markdown(f"<div class='{sig_class}'>{msg}</div>", unsafe_allow_html=True)
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d')} • Signals update weekly on Friday close")
    
    # Simple allocation display
    col1, col2 = st.columns(2)
    with col1:
        status = "✅ Full Exposure" if alloc['SPY'] == 100 else "⚠️ Reduced" if alloc['SPY'] == 50 else "❌ None"
        st.metric("SPY Allocation", f"{alloc['SPY']:.0f}%", status)
    with col2:
        status = "✅ Full Hedge" if alloc['IEF'] == 100 else "⚠️ Partial" if alloc['IEF'] == 50 else "❌ None"
        st.metric("IEF Allocation", f"{alloc['IEF']:.0f}%", status)
    
    # Why this signal? (weekly context)
    st.subheader("🚨 Why This Weekly Signal?")
    c1, c2, c3 = st.columns(3)
    with c1:
        regime = "✅ Bullish" if 32941 > 30500 else "❌ Bearish"
        st.markdown(f"""
        <div class="metric-card">
            <strong>Regime ($NYHL)</strong><br>
            {regime}<br>
            <small>20-week SMA filter</small>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        momentum = "⚠️ Weak" if 36.47 < 40 else "✅ Strong"
        st.markdown(f"""
        <div class="metric-card">
            <strong>Momentum ($BPSPX)</strong><br>
            {momentum}<br>
            <small>Weekly RSI + MACD</small>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        risk = "⚠️ Risk-Off" if 24.05 < 24.80 else "✅ Risk-On"
        st.markdown(f"""
        <div class="metric-card">
            <strong>Risk (SPY:VXX)</strong><br>
            {risk}<br>
            <small>10-week SMA filter</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Action guidance
    st.markdown("### ⚡ Weekly Action Guidance")
    if signal == "SPY":
        st.success("""
        **This Week**: Consider rotating to 100% SPY if you're not already.
        - Best executed on Friday close or Monday open
        - Hold through the week unless canaries flash warning
        - Next signal check: Next Friday
        """)
        st.button("✅ Execute: Buy SPY / Sell IEF", type="primary", use_container_width=True)
    elif signal == "IEF":
        st.warning("""
        **This Week**: Consider rotating to 100% IEF for defense.
        - Best executed on Friday close or Monday open
        - Hold through volatility, re-evaluate next Friday
        - Next signal check: Next Friday
        """)
        st.button("✅ Execute: Buy IEF / Sell SPY", type="primary", use_container_width=True)
    else:
        st.info("""
        **This Week**: No rotation needed. Hold current allocation.
        - Wait for clearer weekly signal
        - Re-check next Friday after market close
        - Use this week to review canary indicators
        """)
        st.button("⏸️ No Action Needed", disabled=True, use_container_width=True)

# ============================================
# TAB 2: HISTORICAL WEEKLY ROTATIONS
# ============================================

def render_backtest_tab():
    st.subheader("📈 Historical Weekly Rotations (~5 Years)")
    st.markdown("*Weekly signals only • Designed for swing/position trading*")
    
    if st.button("📊 Load Weekly History", type="primary"):
        with st.spinner("Calculating weekly signals..."):
            df = generate_mock_weekly_data(260)  # 5 years weekly
            results = generate_rotation_history_weekly(df)
            
            # Chart 1: Price with weekly rotation markers
            st.markdown("### SPY Weekly Price with Rotation Signals")
            spy_signals = results[results['signal'] == 'SPY']
            ief_signals = results[results['signal'] == 'IEF']
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=results.index, y=results['Close'], name='SPY Weekly', line=dict(color='#1f77b4', width=2)))
            if not spy_signals.empty:
                fig1.add_trace(go.Scatter(x=spy_signals.index, y=spy_signals['Close'], mode='markers', name='→ SPY', marker=dict(color='#10b981', size=12, symbol='triangle-up')))
            if not ief_signals.empty:
                fig1.add_trace(go.Scatter(x=ief_signals.index, y=ief_signals['Close'], mode='markers', name='→ IEF', marker=dict(color='#ef4444', size=12, symbol='x')))
            fig1.update_layout(height=450, hovermode='x unified', title="Weekly Signals Only (Less Noise)")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Chart 2: Allocation over time (weekly steps)
            st.markdown("### Weekly Allocation (100% SPY or IEF)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=results.index, y=results['SPY_%'], name='% in SPY', line=dict(color='#10b981', width=3, shape='hv')))
            fig2.add_trace(go.Scatter(x=results.index, y=results['IEF_%'], name='% in IEF', line=dict(color='#3b82f6', width=3, shape='hv')))
            fig2.update_layout(height=300, yaxis_title='% Allocation', hovermode='x unified', title="Step Changes Only (Weekly)")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Performance metrics (weekly compounding)
            total_bh = (results['buyhold_cum'].iloc[-1] - 100)
            total_strat = (results['strat_cum'].iloc[-1] - 100)
            years = 5
            cagr_bh = ((results['buyhold_cum'].iloc[-1] / 100) ** (1/years) - 1) * 100
            cagr_strat = ((results['strat_cum'].iloc[-1] / 100) ** (1/years) - 1) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return (5Y)", f"{total_strat:+.1f}%", delta=f"{total_strat - total_bh:+.1f}% vs BH")
            with col2:
                st.metric("CAGR", f"{cagr_strat:+.1f}%", delta=f"{cagr_strat - cagr_bh:+.1f}% vs BH")
            with col3:
                # Count rotations
                rotations = len(results[results['signal'].shift(1) != results['signal']])
                st.metric("Total Rotations (5Y)", f"{rotations}", delta=f"~{rotations/5:.1f} per year")
            
            st.success("✅ Weekly history loaded • Signals update Friday close")
    else:
        st.info("👆 Click to load weekly rotation history")

# ============================================
# TAB 3: WEEKLY POSITION LOG (Your CSV Request)
# ============================================

def render_positions_tab():
    st.subheader("📋 Weekly Position Log")
    st.markdown("*One row per week • Easy to review and export*")
    
    if st.button("📊 Load Weekly Log", type="primary"):
        with st.spinner("Loading..."):
            df = generate_mock_weekly_data(52)  # 1 year weekly
            results = generate_rotation_history_weekly(df)
            recent = results.tail(12).copy()  # Last 12 weeks
            
            # Clean table for weekly review
            display = recent.copy()
            display['Week'] = display.index.strftime('%Y-%m-%d')
            display['Signal'] = display['signal']
            display['SPY %'] = display['SPY_%'].apply(lambda x: f"{x:.0f}%")
            display['IEF %'] = display['IEF_%'].apply(lambda x: f"{x:.0f}%")
            display['SPY Weekly Return'] = display['spy_ret'].apply(lambda x: f"{x*100:+.2f}%")
            display['Strategy Weekly Return'] = display['strat_ret'].apply(lambda x: f"{x*100:+.2f}%")
            
            cols = ['Week', 'Signal', 'SPY %', 'IEF %', 'SPY Weekly Return', 'Strategy Weekly Return']
            st.dataframe(display[cols].sort_values('Week', ascending=False), use_container_width=True, hide_index=True)
            
            # Export
            export_df = recent[['SPY_%', 'IEF_%', 'spy_ret', 'strat_ret']].copy()
            export_df.columns = ['SPY %', 'IEF %', 'SPY Return', 'Strategy Return']
            csv = export_df.to_csv().encode('utf-8')
            st.download_button("📥 Export Weekly CSV", csv, "weekly_rotations.csv", "text/csv")
            
            st.info("""
            💡 **How to Use This Log**:
            - Review each Friday after market close
            - If Signal changed → consider rotation Monday
            - If Signal unchanged → hold current allocation
            - Edge comes from avoiding major drawdowns, not daily outperformance
            """)
    else:
        st.info("👆 Click to load weekly position log")

# ============================================
# MAIN
# ============================================

def main():
    st.title("🔄 SPY/IEF Weekly Rotation Dashboard")
    st.markdown("*Simple 100% allocation rotations • Weekly signals for swing/position trading*")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Current Weekly Signal", "📈 Weekly History", "📋 Weekly Log"])
    
    with tab1:
        render_current_tab()
    with tab2:
        render_backtest_tab()
    with tab3:
        render_positions_tab()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("**Signal Frequency:** Weekly (Friday close)")
        st.markdown("**Timeframe:** Swing/Position Trading (weeks to months)")
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("**Weekly Rotation Logic:**")
        st.markdown("""
        → **SPY (100%)**: Bullish regime + no canary warnings
        → **IEF (100%)**: Bearish regime OR credit stress + risk-off  
        → **WAIT (50/50)**: Mixed signals, no action needed
        """)
        st.markdown("---")
        st.caption("⚠️ Not investment advice • Test before live use")

if __name__ == "__main__":
    main()
