import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================
# CONFIG & STYLING
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
# DATA GENERATION
# ============================================

@st.cache_data(ttl=3600)
def generate_mock_spy_data(days=1260):
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    returns = np.random.normal(0.0003, 0.012, days)
    for i in range(1, len(returns)):
        returns[i] += 0.05 * returns[i-1]
    vol = np.abs(np.random.normal(1, 0.3, days))
    returns = returns * vol
    prices = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({'Close': prices}, index=dates)
    return df

@st.cache_data(ttl=3600)
def generate_breadth_indicators(spy_df):
    df = spy_df.copy()
    returns = df['Close'].pct_change().fillna(0)
    
    df['nyhl_cum'] = (returns.rolling(20).sum().cumsum() * 15000 + 28000).ffill()
    df['nyhl_sma200'] = df['nyhl_cum'].rolling(200).mean()
    df['nyhl_sma50'] = df['nyhl_cum'].rolling(50).mean()
    
    momentum = returns.rolling(20).mean()
    df['bpspx_value'] = (55 + momentum * 35 + np.random.normal(0, 6, len(df))).clip(15, 85)
    df['bpspx_rsi14'] = (50 + momentum * 45 + np.random.normal(0, 9, len(df))).clip(15, 85)
    df['bpspx_macd_hist'] = momentum.rolling(12).mean() - momentum.rolling(26).mean()
    df['bpspx_sma50'] = df['bpspx_value'].rolling(50).mean()
    
    ma150 = df['Close'].rolling(150).mean()
    above = (df['Close'] > ma150).astype(float)
    df['oexa150r_value'] = (above.rolling(25).mean() * 100 + np.random.normal(0, 10, len(df))).clip(15, 85)
    df['oexa150r_cci14'] = ((df['Close'] - df['Close'].rolling(14).mean()) / df['Close'].rolling(14).std().replace(0, 1) * 100)
    
    vol = returns.rolling(20).std()
    df['spy_vxx_ratio'] = (1 / (vol * 100 + 0.01) + np.random.normal(0, 4, len(df))).clip(8, 45)
    df['spy_vxx_sma50'] = df['spy_vxx_ratio'].rolling(50).mean()
    df['vix'] = (18 - momentum * 18 + np.random.normal(0, 5, len(df))).clip(12, 45)
    
    df['hy_spread_bps'] = (380 - momentum * 120 + np.random.normal(0, 35, len(df))).clip(220, 580)
    df['mcclellan_osc'] = momentum * 90 + np.random.normal(0, 22, len(df))
    df['semis_spx_ratio'] = (1.55 + momentum * 0.6 + np.random.normal(0, 0.25, len(df))).clip(0.9, 2.3)
    df['semis_spx_sma50'] = df['semis_spx_ratio'].rolling(50).mean()
    df['smallcap_largecap_ratio'] = (0.46 + momentum * 0.12 + np.random.normal(0, 0.06, len(df))).clip(0.32, 0.58)
    
    return df.dropna()

# ============================================
# SIGNAL ENGINE
# ============================================

def check_canary_warnings(row):
    warnings = 0
    details = []
    if row.get('hy_spread_bps', 380) > 460:
        warnings += 2
        details.append("Credit Stress")
    if row.get('semis_spx_ratio', 1.5) < row.get('semis_spx_sma50', 1.5):
        warnings += 1
        details.append("Semis Weak")
    if row.get('smallcap_largecap_ratio', 0.46) < 0.41:
        warnings += 1
        details.append("Small Cap Weak")
    if row.get('mcclellan_osc', 0) < -55:
        warnings += 1
        details.append("Breadth Weak")
    return warnings, details

def generate_signals(df):
    signals, allocations, canary_scores = [], [], []
    
    for idx, row in df.iterrows():
        regime_bull = row['nyhl_cum'] > row['nyhl_sma200']
        bpspx_bear = (row['bpspx_rsi14'] < 38 and row['bpspx_macd_hist'] < 0)
        oexa_oversold = row['oexa150r_value'] < 33 and row['oexa150r_cci14'] < -110
        risk_off = row['spy_vxx_ratio'] < row['spy_vxx_sma50'] and row['vix'] > 21
        
        canary_score, canary_details = check_canary_warnings(row)
        canary_scores.append(canary_score)
        
        bull_score = int(regime_bull) * 4 + int(oexa_oversold) * 3
        bear_score = int(not regime_bull) * 4 + int(bpspx_bear) * 3 + int(risk_off) * 3 + int(canary_score >= 3) * 3
        
        if canary_score >= 5:
            signal, alloc = "WARNING - CANARIES", {'SPY': 0.40, 'IEF': 0.45, 'CASH': 0.15}
        elif not regime_bull or (bear_score >= 9 and bull_score < 4):
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

def calculate_strategy_returns(df):
    df = df.copy()
    df['spy_return'] = df['Close'].pct_change().fillna(0)
    ief_returns = np.where(df['spy_return'] < 0, -0.35 * df['spy_return'], 0.02/252)
    
    def calc_daily_return(row):
        idx = row.name
        spy_ret = row['spy_return']
        ief_ret = ief_returns[df.index.get_loc(idx)] if idx in df.index else 0.02/252
        return row['allocation']['SPY'] * spy_ret + row['allocation']['IEF'] * ief_ret
    
    df['strategy_return'] = df.apply(calc_daily_return, axis=1)
    df['buyhold_cum'] = (1 + df['spy_return']).cumprod() * 100
    df['strategy_cum'] = (1 + df['strategy_return']).cumprod() * 100
    df['SPY_%'] = df['allocation'].apply(lambda x: x['SPY'] * 100)
    df['IEF_%'] = df['allocation'].apply(lambda x: x['IEF'] * 100)
    df['CASH_%'] = df['allocation'].apply(lambda x: x['CASH'] * 100)
    return df

# ============================================
# CURRENT SIGNAL
# ============================================

def get_current_signal():
    data = {
        'timestamp': datetime(2026, 2, 24, 11, 52),
        'spx_price': 6877.92,
        'nyhl_cum': 32941, 'nyhl_sma200': 27441,
        'bpspx_value': 60.00, 'bpspx_rsi14': 36.47, 'bpspx_macd_hist': -0.470, 'bpspx_sma50': 60.98,
        'oexa150r_value': 63.00, 'oexa150r_cci14': -163.74,
        'spy_vxx_ratio': 24.05, 'spy_vxx_sma50': 25.06, 'vix': 18.45,
        'hy_spread_bps': 385, 'mcclellan_osc': -15.4,
        'semis_spx_ratio': 1.65, 'semis_spx_sma50': 1.70,
        'smallcap_largecap_ratio': 0.42,
    }
    
    canary_score, canary_details = check_canary_warnings(data)
    regime_bull = data['nyhl_cum'] > data['nyhl_sma200']
    bpspx_bear = data['bpspx_rsi14'] < 40 and data['bpspx_macd_hist'] < 0
    risk_off = data['spy_vxx_ratio'] < data['spy_vxx_sma50']
    
    if canary_score >= 5:
        signal, sig_class, alloc = "WARNING - CANARIES", "signal-warning", {'SPY': 40, 'IEF': 45, 'CASH': 15}
    elif not regime_bull:
        signal, sig_class, alloc = "SELL", "signal-sell", {'SPY': 20, 'IEF': 60, 'CASH': 20}
    elif bpspx_bear and risk_off and canary_score >= 3:
        signal, sig_class, alloc = "WARNING", "signal-warning", {'SPY': 40, 'IEF': 45, 'CASH': 15}
    else:
        signal, sig_class, alloc = "HOLD", "signal-hold", {'SPY': 65, 'IEF': 25, 'CASH': 10}
    
    return signal, sig_class, alloc, data, canary_score, canary_details

# ============================================
# TABS
# ============================================

def render_current_tab():
    signal, sig_class, alloc, data, canary_score, canary_details = get_current_signal()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<div class='{sig_class}'>{signal}</div>", unsafe_allow_html=True)
        st.caption(f"Updated: {data['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    with col2:
        regime = "🟢 Bullish" if data['nyhl_cum'] > data['nyhl_sma200'] else "🔴 Bearish"
        st.metric("Primary Regime", regime)
    
    st.subheader("🚨 Canary Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Credit Spreads", f"{data['hy_spread_bps']} bps", "✅ Normal" if data['hy_spread_bps'] < 450 else "🔴 Stress")
    with c2: st.metric("Semis/SPX", f"{data['semis_spx_ratio']:.2f}", "✅ Strong" if data['semis_spx_ratio'] > data['semis_spx_sma50'] else "⚠️ Weak")
    with c3: st.metric("Small/Large", f"{data['smallcap_largecap_ratio']:.2f}", "✅ Normal" if data['smallcap_largecap_ratio'] > 0.40 else "⚠️ Weak")
    with c4: st.metric("McClellan", f"{data['mcclellan_osc']:+.1f}", "✅ Normal" if data['mcclellan_osc'] > -50 else "⚠️ Weak")
    
    if canary_score >= 3:
        st.warning(f"⚠️ **{canary_score} Canary Warnings**: {', '.join(canary_details)}")
    else:
        st.success("✅ Canary indicators quiet")
    
    st.subheader("🎯 Allocation")
    alloc_df = pd.DataFrame([{'ETF': 'SPY', 'Allocation': alloc['SPY']}, {'ETF': 'IEF', 'Allocation': alloc['IEF']}, {'ETF': 'CASH', 'Allocation': alloc['CASH']}])
    ac1, ac2 = st.columns(2)
    with ac1:
        fig = px.pie(alloc_df, values='Allocation', names='ETF', color='ETF', color_discrete_map={'SPY':'#10b981','IEF':'#3b82f6','CASH':'#6b7280'}, hole=0.4)
        fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
    with ac2:
        fig = px.bar(alloc_df, x='Allocation', y='ETF', orientation='h', color='ETF', color_discrete_map={'SPY':'#10b981','IEF':'#3b82f6','CASH':'#6b7280'}, text='Allocation', range_x=[0,100])
        fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

def render_backtest_tab():
    st.subheader("📈 Historical Backtest (~5 Years)")
    if st.button("📊 Load Backtest Data", type="primary"):
        with st.spinner("Loading..."):
            spy_df = generate_mock_spy_data(1260)
            breadth_df = generate_breadth_indicators(spy_df)
            signal_df = generate_signals(breadth_df)
            results_df = calculate_strategy_returns(signal_df)
            
            st.markdown("### SPY Price with Signals")
            buy_sig = results_df[results_df['signal'].isin(['BUY', 'BUY DIP'])]
            sell_sig = results_df[results_df['signal'].isin(['SELL', 'WARNING'])]
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=results_df.index, y=results_df['Close'], name='SPY', line=dict(color='#1f77b4')))
            if not buy_sig.empty: fig1.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig['Close'], mode='markers', name='BUY', marker=dict(color='#10b981', size=8, symbol='triangle-up')))
            if not sell_sig.empty: fig1.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig['Close'], mode='markers', name='SELL', marker=dict(color='#dc2626', size=8, symbol='x')))
            fig1.update_layout(height=450, hovermode='x unified')
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("### Strategy vs Buy-and-Hold")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=results_df.index, y=results_df['buyhold_cum'], name='Buy-and-Hold', line=dict(color='#6b7280')))
            fig2.add_trace(go.Scatter(x=results_df.index, y=results_df['strategy_cum'], name='Strategy', line=dict(color='#10b981')))
            fig2.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig2, use_container_width=True)
            
            total_bh = (results_df['buyhold_cum'].iloc[-1] - 100) / 100 * 100
            total_strat = (results_df['strategy_cum'].iloc[-1] - 100) / 100 * 100
            st.metric("Total Return", f"{total_strat:+.1f}%", delta=f"{total_strat - total_bh:+.1f}% vs BH")
            st.success("✅ Backtest complete!")
    else:
        st.info("👆 Click button to load")

def render_position_tracker_tab():
    st.subheader("📋 Daily Position Tracker")
    st.markdown("*See EXACTLY what % was in SPY/IEF/CASH each day*")
    
    if st.button("📊 Load Position History", type="primary"):
        with st.spinner("Loading..."):
            try:
                spy_df = generate_mock_spy_data(252)
                breadth_df = generate_breadth_indicators(spy_df)
                signal_df = generate_signals(breadth_df)
                results_df = calculate_strategy_returns(signal_df)
                
                recent = results_df.tail(90).copy()
                recent['Date'] = [d.strftime('%Y-%m-%d') for d in recent.index.date]
                recent['SPY Return'] = recent['spy_return'].apply(lambda x: f"{x*100:+.2f}%")
                recent['Strategy Return'] = recent['strategy_return'].apply(lambda x: f"{x*100:+.2f}%")
                recent['SPY %'] = recent['SPY_%'].apply(lambda x: f"{x:.0f}%")
                recent['IEF %'] = recent['IEF_%'].apply(lambda x: f"{x:.0f}%")
                recent['CASH %'] = recent['CASH_%'].apply(lambda x: f"{x:.0f}%")
                recent['Signal'] = recent['signal']  # FIX: Create capitalized version
                
                display_cols = ['Date', 'Signal', 'SPY %', 'IEF %', 'CASH %', 'SPY Return', 'Strategy Return']
                st.dataframe(recent[display_cols].sort_values('Date', ascending=False), use_container_width=True, hide_index=True, height=400)
                
                st.markdown("### Allocation History")
                fig_alloc = go.Figure()
                fig_alloc.add_trace(go.Scatter(x=recent['Date'], y=recent['SPY_%'], name='SPY', stackgroup='one', line=dict(color='#10b981')))
                fig_alloc.add_trace(go.Scatter(x=recent['Date'], y=recent['IEF_%'], name='IEF', stackgroup='one', line=dict(color='#3b82f6')))
                fig_alloc.add_trace(go.Scatter(x=recent['Date'], y=recent['CASH_%'], name='CASH', stackgroup='one', line=dict(color='#6b7280')))
                fig_alloc.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(fig_alloc, use_container_width=True)
                
                st.markdown("### Signal Changes")
                signal_changes = recent[recent['signal'].shift(1) != recent['signal']].tail(10)  # FIX: lowercase
                if not signal_changes.empty:
                    st.dataframe(signal_changes[['Date', 'signal', 'SPY_%', 'IEF_%']], use_container_width=True, hide_index=True)
                else:
                    st.info("No signal changes in last 90 days")
                
                st.success("✅ Position history loaded!")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("👆 Click button to load")

# ============================================
# MAIN
# ============================================

def main():
    st.title("📊 Market Regime Dashboard")
    tab1, tab2, tab3 = st.tabs(["🎯 Current Signal", "📈 Historical Backtest", "📋 Daily Position Tracker"])
    with tab1: render_current_tab()
    with tab2: render_backtest_tab()
    with tab3: render_position_tracker_tab()
    
    with st.sidebar:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("⚠️ Not investment advice")

if __name__ == "__main__":
    main()
