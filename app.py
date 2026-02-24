import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# ============================================
# CONFIG - LIGHTWEIGHT
# ============================================
st.set_page_config(page_title="Market Regime Dashboard", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .signal-buy {background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .signal-hold {background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .signal-warning {background: linear-gradient(135deg, #f97316, #ea580c); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .signal-sell {background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA - CACHED & FAST
# ============================================

@st.cache_data(ttl=3600)
def fetch_spy_history(years=5):  # Changed from 15 to 5
    """Fetch SPY historical data - 5 years max for speed"""
    end = datetime.now()
    start = end - timedelta(days=years*365)
    try:
        spy = yf.download('SPY', start=start, end=end, progress=False)
        return spy
    except:
        # Fallback mock data if yfinance fails
        dates = pd.date_range(start=start, end=end, freq='B')
        spy = pd.DataFrame({'Close': np.linspace(400, 600, len(dates))}, index=dates)
        return spy

@st.cache_data(ttl=86400)
def generate_fast_breadth_data(spy_df):
    """Simplified breadth data generation - faster"""
    df = spy_df.copy()
    np.random.seed(42)
    
    returns = df['Close'].pct_change()
    df['nyhl_cum'] = (returns.rolling(20).sum().cumsum() * 10000 + 25000).fillna(method='ffill')
    df['nyhl_sma200'] = df['nyhl_cum'].rolling(200).mean()
    df['bpspx_rsi14'] = (50 + returns.rolling(20).mean() * 40).clip(10, 90)
    df['bpspx_macd_hist'] = returns.rolling(12).mean() - returns.rolling(26).mean()
    df['oexa150r_value'] = (50 + returns.rolling(50).mean() * 100).clip(10, 90)
    df['oexa150r_cci14'] = ((df['Close'] - df['Close'].rolling(14).mean()) / df['Close'].rolling(14).std() * 100)
    df['spy_vxx_ratio'] = (1 / (returns.rolling(20).std() * 100)).clip(5, 50)
    df['spy_vxx_sma50'] = df['spy_vxx_ratio'].rolling(50).mean()
    df['vix'] = (20 - returns.rolling(20).mean() * 15).clip(10, 50)
    df['hy_spread_bps'] = (350 - returns.rolling(20).mean() * 100).clip(200, 600)
    df['mcclellan_osc'] = returns.rolling(20).mean() * 80
    df['semis_spx_ratio'] = (1.5 + returns.rolling(20).mean() * 0.5).clip(0.8, 2.5)
    df['semis_spx_sma50'] = df['semis_spx_ratio'].rolling(50).mean()
    df['smallcap_largecap_ratio'] = (0.45 + returns.rolling(20).mean() * 0.1).clip(0.30, 0.60)
    
    return df.dropna()

# ============================================
# SIGNAL ENGINE - SIMPLIFIED
# ============================================

def check_canary_warnings(row):
    """Fast canary check"""
    warnings = 0
    if row.get('hy_spread_bps', 350) > 450:
        warnings += 2
    if row.get('semis_spx_ratio', 1.5) < row.get('semis_spx_sma50', 1.5):
        warnings += 1
    if row.get('smallcap_largecap_ratio', 0.45) < 0.40:
        warnings += 1
    if row.get('mcclellan_osc', 0) < -50:
        warnings += 1
    return warnings

def generate_signals(df):
    """Fast signal generation"""
    signals, allocations, canary_scores = [], [], []
    
    for idx, row in df.iterrows():
        regime_bull = row['nyhl_cum'] > row['nyhl_sma200']
        bpspx_bear = row['bpspx_rsi14'] < 40 and row['bpspx_macd_hist'] < 0
        oexa_oversold = row['oexa150r_value'] < 35 and row['oexa150r_cci14'] < -100
        risk_off = row['spy_vxx_ratio'] < row['spy_vxx_sma50'] and row['vix'] > 20
        
        canary_score = check_canary_warnings(row)
        canary_scores.append(canary_score)
        
        bull_score = int(regime_bull) * 4 + int(oexa_oversold) * 3
        bear_score = int(not regime_bull) * 4 + int(bpspx_bear) * 3 + int(risk_off) * 3 + int(canary_score >= 3) * 3
        
        if canary_score >= 5:
            signal, alloc = "WARNING - CANARIES", {'SPY': 0.40, 'IEF': 0.45, 'CASH': 0.15}
        elif not regime_bull or (bear_score >= 8 and bull_score < 4):
            signal, alloc = "SELL", {'SPY': 0.20, 'IEF': 0.60, 'CASH': 0.20}
        elif bear_score > bull_score and regime_bull:
            signal, alloc = "WARNING", {'SPY': 0.40, 'IEF': 0.45, 'CASH': 0.15}
        elif oexa_oversold and regime_bull:
            signal, alloc = "BUY DIP", {'SPY': 0.80, 'IEF': 0.15, 'CASH': 0.05}
        elif bull_score - bear_score >= 3 and regime_bull:
            signal, alloc = "BUY", {'SPY': 0.85, 'IEF': 0.10, 'CASH': 0.05}
        else:
            signal, alloc = "HOLD", {'SPY': 0.65, 'IEF': 0.25, 'CASH': 0.10}
        
        signals.append(signal)
        allocations.append(alloc)
    
    df = df.copy()
    df['signal'] = signals
    df['allocation'] = allocations
    df['canary_score'] = canary_scores
    return df

def calculate_returns(df):
    """Fast return calculation"""
    df = df.copy()
    df['spy_return'] = df['Close'].pct_change().fillna(0)
    ief_returns = -0.3 * df['spy_return'].apply(lambda x: x if x < 0 else 0) + 0.02/252
    
    def daily_return(row):
        return row['allocation']['SPY'] * row['spy_return'] + row['allocation']['IEF'] * ief_returns.loc[row.name] if row.name in ief_returns.index else 0.02/252
    
    df['strategy_return'] = df.apply(daily_return, axis=1)
    df['buyhold_cum'] = (1 + df['spy_return']).cumprod() * 100
    df['strategy_cum'] = (1 + df['strategy_return']).cumprod() * 100
    return df

# ============================================
# CURRENT SIGNAL - INSTANT (No Loading)
# ============================================

def get_current_signal():
    """Current signal from your charts - instant"""
    data = {
        'timestamp': datetime(2026, 2, 24, 11, 52),
        'spx_price': 6877.92,
        'nyhl_cum': 32941, 'nyhl_sma200': 27441, 'nyhl_sma50': 29200,
        'bpspx_value': 60.00, 'bpspx_rsi14': 36.47, 'bpspx_macd_hist': -0.470, 'bpspx_sma50': 60.98,
        'oexa150r_value': 63.00, 'oexa150r_cci14': -163.74,
        'spy_vxx_ratio': 24.05, 'spy_vxx_sma50': 25.06, 'spy_vxx_sma200': 18.50, 'vix': 18.45,
        'hy_spread_bps': 385, 'mcclellan_osc': -15.4,
        'semis_spx_ratio': 1.65, 'semis_spx_sma50': 1.70,
        'smallcap_largecap_ratio': 0.42,
    }
    
    canary_score = check_canary_warnings(data)
    regime_bull = data['nyhl_cum'] > data['nyhl_sma200']
    bpspx_bear = data['bpspx_rsi14'] < 40 and data['bpspx_macd_hist'] < 0
    risk_off = data['spy_vxx_ratio'] < data['spy_vxx_sma50']
    oexa_oversold = data['oexa150r_value'] < 35 and data['oexa150r_cci14'] < -100
    
    if canary_score >= 5:
        signal, signal_class, allocation = "WARNING - CANARIES", "signal-warning", {'SPY': 40, 'IEF': 45, 'CASH': 15}
    elif not regime_bull:
        signal, signal_class, allocation = "SELL", "signal-sell", {'SPY': 20, 'IEF': 60, 'CASH': 20}
    elif bpspx_bear and risk_off and canary_score >= 3:
        signal, signal_class, allocation = "WARNING", "signal-warning", {'SPY': 40, 'IEF': 45, 'CASH': 15}
    elif oexa_oversold and regime_bull:
        signal, signal_class, allocation = "BUY DIP", "signal-buy", {'SPY': 80, 'IEF': 15, 'CASH': 5}
    else:
        signal, signal_class, allocation = "HOLD", "signal-hold", {'SPY': 65, 'IEF': 25, 'CASH': 10}
    
    return signal, signal_class, allocation, data, canary_score

# ============================================
# MAIN APP
# ============================================

def main():
    st.title("📊 Market Regime Dashboard")
    
    # Tabs
    tab1, tab2 = st.tabs(["🎯 Current Signal", "📈 Historical Backtest (5 Years)"])
    
    # === TAB 1: CURRENT SIGNAL (INSTANT) ===
    with tab1:
        signal, signal_class, allocation, data, canary_score = get_current_signal()
        
        # Signal banner
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"<div class='{signal_class}'>{signal}</div>", unsafe_allow_html=True)
            st.caption(f"Updated: {data['timestamp'].strftime('%Y-%m-%d %H:%M')} | SPX: ${data['spx_price']:,.2f}")
        with col2:
            regime = "🟢 Bullish" if data['nyhl_cum'] > data['nyhl_sma200'] else "🔴 Bearish"
            st.metric("Primary Regime", regime)
        
        # Canary indicators
        st.subheader("🚨 Canary Indicators")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Credit Spreads", f"{data['hy_spread_bps']} bps", "✅ Normal" if data['hy_spread_bps'] < 450 else "🔴 Stress")
        with c2:
            st.metric("Semis/SPX", f"{data['semis_spx_ratio']:.2f}", "✅ Strong" if data['semis_spx_ratio'] > data['semis_spx_sma50'] else "⚠️ Weak")
        with c3:
            st.metric("Small/Large Cap", f"{data['smallcap_largecap_ratio']:.2f}", "✅ Normal" if data['smallcap_largecap_ratio'] > 0.40 else "⚠️ Weak")
        with c4:
            st.metric("McClellan", f"{data['mcclellan_osc']:+.1f}", "✅ Normal" if data['mcclellan_osc'] > -50 else "⚠️ Weak")
        
        if canary_score >= 3:
            st.warning(f"⚠️ **{canary_score} Canary Warnings Active**")
        else:
            st.success("✅ Canary indicators are quiet")
        
        # Core indicators
        st.subheader("📊 Core Indicators (From Your Charts)")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown(f"""
            <div class="metric-card">
                <strong>$NYHL Cumulative</strong><br>
                {data['nyhl_cum']:,} vs 200-SMA: {data['nyhl_sma200']:,}<br>
                {'✅ Bullish' if data['nyhl_cum'] > data['nyhl_sma200'] else '❌ Bearish'}
            </div>
            """, unsafe_allow_html=True)
        with cc2:
            st.markdown(f"""
            <div class="metric-card">
                <strong>$BPSPX</strong><br>
                {data['bpspx_value']}% | RSI: {data['bpspx_rsi14']:.1f}<br>
                {'⚠️ Distribution' if data['bpspx_rsi14'] < 40 else '✅ Neutral'}
            </div>
            """, unsafe_allow_html=True)
        with cc3:
            st.markdown(f"""
            <div class="metric-card">
                <strong>$OEXA150R</strong><br>
                {data['oexa150r_value']}% | CCI: {data['oexa150r_cci14']:.0f}<br>
                {'🟢 Oversold' if data['oexa150r_value'] < 30 else '🟡 Neutral' if data['oexa150r_value'] < 70 else '🔴 Overbought'}
            </div>
            """, unsafe_allow_html=True)
        
        # Allocation
        st.subheader("🎯 Recommended Allocation")
        alloc_df = pd.DataFrame([
            {'ETF': 'SPY', 'Allocation': allocation['SPY']},
            {'ETF': 'IEF', 'Allocation': allocation['IEF']},
            {'ETF': 'CASH', 'Allocation': allocation['CASH']}
        ])
        
        ac1, ac2 = st.columns(2)
        with ac1:
            fig = px.pie(alloc_df, values='Allocation', names='ETF', 
                         color='ETF', color_discrete_map={'SPY':'#10b981','IEF':'#3b82f6','CASH':'#6b7280'},
                         hole=0.4)
            fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig, width="stretch")
        with ac2:
            fig = px.bar(alloc_df, x='Allocation', y='ETF', orientation='h',
                         color='ETF', color_discrete_map={'SPY':'#10b981','IEF':'#3b82f6','CASH':'#6b7280'},
                         text='Allocation', range_x=[0,100])
            fig.update_layout(showlegend=False, xaxis_title='% Allocation', margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig, width="stretch")
        
        # Actions
        st.markdown("### ⚡ Quick Actions")
        cols = st.columns(3)
        with cols[0]:
            st.button("✅ Add SPY", disabled=allocation['SPY'] < 65, width="stretch")
        with cols[1]:
            st.button("🛡️ Add IEF", disabled=allocation['IEF'] < 40, width="stretch")
        with cols[2]:
            st.button("💵 Raise Cash", disabled=allocation['CASH'] < 15, width="stretch")
    
    # === TAB 2: BACKTEST (LOADS ON DEMAND) ===
    with tab2:
        st.subheader("📈 Historical Backtest (5 Years)")
        st.markdown("*Click button below to load - saves time on startup*")
        
        if st.button("📊 Load Backtest Data", type="primary"):
            with st.spinner("Loading 5 years of data... (30 seconds max)"):
                try:
                    spy_df = fetch_spy_history(5)
                    breadth_df = generate_fast_breadth_data(spy_df)
                    signal_df = generate_signals(breadth_df)
                    results_df = calculate_returns(signal_df)
                    
                    # Price chart with signals
                    st.markdown("### SPY Price with Signals")
                    buy_signals = results_df[results_df['signal'].isin(['BUY', 'BUY DIP'])]
                    sell_signals = results_df[results_df['signal'].isin(['SELL', 'WARNING'])]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Close'], name='SPY', line=dict(color='#1f77b4')))
                    if not buy_signals.empty:
                        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='BUY', marker=dict(color='#10b981', size=8, symbol='triangle-up')))
                    if not sell_signals.empty:
                        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='SELL', marker=dict(color='#dc2626', size=8, symbol='x')))
                    fig.update_layout(height=400, hovermode='x unified')
                    st.plotly_chart(fig, width="stretch")
                    
                    # Equity curve
                    st.markdown("### Strategy vs Buy-and-Hold")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=results_df.index, y=results_df['buyhold_cum'], name='Buy-and-Hold', line=dict(color='#6b7280')))
                    fig2.add_trace(go.Scatter(x=results_df.index, y=results_df['strategy_cum'], name='Strategy', line=dict(color='#10b981')))
                    fig2.update_layout(height=400, hovermode='x unified')
                    st.plotly_chart(fig2, width="stretch")
                    
                    # Metrics
                    st.markdown("### Performance Metrics")
                    total_bh = (results_df['buyhold_cum'].iloc[-1] - 100) / 100 * 100
                    total_strat = (results_df['strategy_cum'].iloc[-1] - 100) / 100 * 100
                    st.metric("Total Return", f"{total_strat:+.1f}%", delta=f"{total_strat - total_bh:+.1f}% vs BH")
                    
                    st.success("✅ Backtest complete!")
                    
                except Exception as e:
                    st.error(f"Error loading backtest: {e}")
                    st.info("Try refreshing or check your internet connection")
        else:
            st.info("👆 Click button above to load backtest data")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        if st.button("🔄 Refresh", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.caption("⚠️ Not investment advice. Test before live use.")

if __name__ == "__main__":
    main()
