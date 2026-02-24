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
# DATA GENERATION (MOCK - NO EXTERNAL DEPS)
# ============================================

@st.cache_data(ttl=3600)
def generate_mock_spy_data(days=1260):  # ~5 years of trading days
    """Generate realistic mock SPY data for backtesting"""
    np.random.seed(42)  # Fixed seed for consistency
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    # Simulate realistic price movement with trends and volatility clusters
    returns = np.random.normal(0.0003, 0.012, days)  # ~7.5% annual return, 19% vol
    # Add momentum autocorrelation
    for i in range(1, len(returns)):
        returns[i] += 0.05 * returns[i-1]
    # Add volatility clustering
    vol = np.abs(np.random.normal(1, 0.3, days))
    returns = returns * vol
    
    prices = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({'Close': prices}, index=dates)
    return df

@st.cache_data(ttl=3600)
def generate_breadth_indicators(spy_df):
    """Generate correlated breadth indicators (mock but realistic)"""
    df = spy_df.copy()
    returns = df['Close'].pct_change().fillna(0)
    
    # $NYHL Cumulative: trending breadth
    df['nyhl_cum'] = (returns.rolling(20).sum().cumsum() * 15000 + 28000).ffill()
    df['nyhl_sma200'] = df['nyhl_cum'].rolling(200).mean()
    df['nyhl_sma50'] = df['nyhl_cum'].rolling(50).mean()
    
    # $BPSPX: mean-reverting bullish %
    momentum = returns.rolling(20).mean()
    df['bpspx_value'] = (55 + momentum * 35 + np.random.normal(0, 6, len(df))).clip(15, 85)
    df['bpspx_rsi14'] = (50 + momentum * 45 + np.random.normal(0, 9, len(df))).clip(15, 85)
    df['bpspx_macd_hist'] = momentum.rolling(12).mean() - momentum.rolling(26).mean()
    df['bpspx_sma50'] = df['bpspx_value'].rolling(50).mean()
    
    # $OEXA150R: % above 150-DMA
    ma150 = df['Close'].rolling(150).mean()
    above = (df['Close'] > ma150).astype(float)
    df['oexa150r_value'] = (above.rolling(25).mean() * 100 + np.random.normal(0, 10, len(df))).clip(15, 85)
    df['oexa150r_cci14'] = ((df['Close'] - df['Close'].rolling(14).mean()) / df['Close'].rolling(14).std().replace(0, 1) * 100)
    
    # Risk sentiment
    vol = returns.rolling(20).std()
    df['spy_vxx_ratio'] = (1 / (vol * 100 + 0.01) + np.random.normal(0, 4, len(df))).clip(8, 45)
    df['spy_vxx_sma50'] = df['spy_vxx_ratio'].rolling(50).mean()
    df['vix'] = (18 - momentum * 18 + np.random.normal(0, 5, len(df))).clip(12, 45)
    
    # Canary indicators
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
    """Check canary indicators for early warning signals"""
    warnings = 0
    details = []
    
    # Tier 1: Credit spreads
    if row.get('hy_spread_bps', 380) > 460:
        warnings += 2
        details.append("Credit Stress")
    
    # Tier 2: Leadership
    if row.get('semis_spx_ratio', 1.5) < row.get('semis_spx_sma50', 1.5):
        warnings += 1
        details.append("Semis Weak")
    if row.get('smallcap_largecap_ratio', 0.46) < 0.41:
        warnings += 1
        details.append("Small Cap Weak")
    
    # Tier 3: Breadth momentum
    if row.get('mcclellan_osc', 0) < -55:
        warnings += 1
        details.append("Breadth Weak")
    
    return warnings, details

def generate_signals(df):
    """Generate historical signals"""
    signals, allocations, canary_scores = [], [], []
    
    for idx, row in df.iterrows():
        # Core logic
        regime_bull = row['nyhl_cum'] > row['nyhl_sma200']
        bpspx_bear = (row['bpspx_rsi14'] < 38 and row['bpspx_macd_hist'] < 0) or \
                     (row['bpspx_value'] < row['bpspx_sma50'] and row['bpspx_macd_hist'] < 0)
        oexa_oversold = row['oexa150r_value'] < 33 and row['oexa150r_cci14'] < -110
        risk_off = row['spy_vxx_ratio'] < row['spy_vxx_sma50'] and row['vix'] > 21
        
        # Canary check
        canary_score, canary_details = check_canary_warnings(row)
        canary_scores.append(canary_score)
        
        # Scoring
        bull_score = int(regime_bull) * 4 + int(oexa_oversold) * 3 + int(row['mcclellan_osc'] > 25) * 2
        bear_score = (int(not regime_bull) * 4 + int(bpspx_bear) * 3 + 
                     int(risk_off) * 3 + int(canary_score >= 3) * 3)
        
        # Signal decision
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
    """Calculate strategy vs buy-and-hold returns"""
    df = df.copy()
    df['spy_return'] = df['Close'].pct_change().fillna(0)
    
    # Simplified IEF return (hedging effect)
    ief_returns = np.where(df['spy_return'] < 0, -0.35 * df['spy_return'], 0.02/252)
    
    def calc_daily_return(row):
        idx = row.name
        spy_ret = row['spy_return']
        ief_ret = ief_returns[df.index.get_loc(idx)] if idx in df.index else 0.02/252
        return row['allocation']['SPY'] * spy_ret + row['allocation']['IEF'] * ief_ret
    
    df['strategy_return'] = df.apply(calc_daily_return, axis=1)
    df['buyhold_cum'] = (1 + df['spy_return']).cumprod() * 100
    df['strategy_cum'] = (1 + df['strategy_return']).cumprod() * 100
    return df

# ============================================
# CURRENT SIGNAL (From Your Charts)
# ============================================

def get_current_signal():
    """Current signal based on your StockCharts PDFs"""
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
    
    canary_score, canary_details = check_canary_warnings(data)
    regime_bull = data['nyhl_cum'] > data['nyhl_sma200']
    bpspx_bear = data['bpspx_rsi14'] < 40 and data['bpspx_macd_hist'] < 0
    risk_off = data['spy_vxx_ratio'] < data['spy_vxx_sma50']
    oexa_oversold = data['oexa150r_value'] < 35 and data['oexa150r_cci14'] < -100
    
    if canary_score >= 5:
        signal, sig_class, alloc = "WARNING - CANARIES", "signal-warning", {'SPY': 40, 'IEF': 45, 'CASH': 15}
    elif not regime_bull:
        signal, sig_class, alloc = "SELL", "signal-sell", {'SPY': 20, 'IEF': 60, 'CASH': 20}
    elif bpspx_bear and risk_off and canary_score >= 3:
        signal, sig_class, alloc = "WARNING", "signal-warning", {'SPY': 40, 'IEF': 45, 'CASH': 15}
    elif oexa_oversold and regime_bull:
        signal, sig_class, alloc = "BUY DIP", "signal-buy", {'SPY': 80, 'IEF': 15, 'CASH': 5}
    else:
        signal, sig_class, alloc = "HOLD", "signal-hold", {'SPY': 65, 'IEF': 25, 'CASH': 10}
    
    return signal, sig_class, alloc, data, canary_score, canary_details

# ============================================
# CURRENT SIGNAL TAB
# ============================================

def render_current_tab():
    signal, sig_class, alloc, data, canary_score, canary_details = get_current_signal()
    
    # Signal banner
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<div class='{sig_class}'>{signal}</div>", unsafe_allow_html=True)
        st.caption(f"Updated: {data['timestamp'].strftime('%Y-%m-%d %H:%M')} | SPX: ${data['spx_price']:,.2f}")
    with col2:
        regime = "🟢 Bullish" if data['nyhl_cum'] > data['nyhl_sma200'] else "🔴 Bearish"
        st.metric("Primary Regime", regime)
    
    # Canary indicators
    st.subheader("🚨 Canary Indicators (Early Warning)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        credit_status = "✅ Normal" if data['hy_spread_bps'] < 450 else "🔴 Stress"
        st.metric("Credit Spreads", f"{data['hy_spread_bps']} bps", credit_status)
    with c2:
        semis_status = "✅ Strong" if data['semis_spx_ratio'] > data['semis_spx_sma50'] else "⚠️ Weak"
        st.metric("Semis/SPX", f"{data['semis_spx_ratio']:.2f}", semis_status)
    with c3:
        sc_status = "✅ Normal" if data['smallcap_largecap_ratio'] > 0.40 else "⚠️ Weak"
        st.metric("Small/Large Cap", f"{data['smallcap_largecap_ratio']:.2f}", sc_status)
    with c4:
        mcc_status = "✅ Normal" if data['mcclellan_osc'] > -50 else "⚠️ Weak"
        st.metric("McClellan", f"{data['mcclellan_osc']:+.1f}", mcc_status)
    
    if canary_score >= 3:
        st.warning(f"⚠️ **{canary_score} Canary Warnings**: {', '.join(canary_details)}")
    else:
        st.success("✅ Canary indicators quiet - no early warnings")
    
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
        {'ETF': 'SPY', 'Allocation': alloc['SPY']},
        {'ETF': 'IEF', 'Allocation': alloc['IEF']},
        {'ETF': 'CASH', 'Allocation': alloc['CASH']}
    ])
    
    ac1, ac2 = st.columns(2)
    with ac1:
        fig = px.pie(alloc_df, values='Allocation', names='ETF', 
                     color='ETF', color_discrete_map={'SPY':'#10b981','IEF':'#3b82f6','CASH':'#6b7280'},
                     hole=0.4)
        fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
    with ac2:
        fig = px.bar(alloc_df, x='Allocation', y='ETF', orientation='h',
                     color='ETF', color_discrete_map={'SPY':'#10b981','IEF':'#3b82f6','CASH':'#6b7280'},
                     text='Allocation', range_x=[0,100])
        fig.update_layout(showlegend=False, xaxis_title='% Allocation', margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Actions
    st.markdown("### ⚡ Quick Actions")
    cols = st.columns(3)
    with cols[0]:
        st.button("✅ Add SPY", disabled=alloc['SPY'] < 65, use_container_width=True)
    with cols[1]:
        st.button("🛡️ Add IEF", disabled=alloc['IEF'] < 40, use_container_width=True)
    with cols[2]:
        st.button("💵 Raise Cash", disabled=alloc['CASH'] < 15, use_container_width=True)

# ============================================
# BACKTEST TAB
# ============================================

def render_backtest_tab():
    st.subheader("📈 Historical Backtest (~5 Years)")
    st.markdown("*Mock breadth data for demonstration • Click button to load*")
    
    if st.button("📊 Load Backtest Data", type="primary"):
        with st.spinner("Generating data and calculating signals..."):
            try:
                # Generate data
                spy_df = generate_mock_spy_data(1260)  # ~5 years
                breadth_df = generate_breadth_indicators(spy_df)
                signal_df = generate_signals(breadth_df)
                results_df = calculate_strategy_returns(signal_df)
                
                # Chart 1: Price with signals
                st.markdown("### SPY Price with Strategy Signals")
                buy_sig = results_df[results_df['signal'].isin(['BUY', 'BUY DIP'])]
                sell_sig = results_df[results_df['signal'].isin(['SELL', 'WARNING', 'WARNING - CANARIES'])]
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=results_df.index, y=results_df['Close'], name='SPY', line=dict(color='#1f77b4', width=1.5)))
                if not buy_sig.empty:
                    fig1.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig['Close'], mode='markers', name='BUY', marker=dict(color='#10b981', size=9, symbol='triangle-up')))
                if not sell_sig.empty:
                    fig1.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig['Close'], mode='markers', name='SELL/WARNING', marker=dict(color='#dc2626', size=9, symbol='x')))
                fig1.update_layout(height=450, hovermode='x unified', legend=dict(orientation='h', y=1.02))
                st.plotly_chart(fig1, use_container_width=True)
                
                # Chart 2: Equity curve
                st.markdown("### Strategy vs Buy-and-Hold Performance")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=results_df.index, y=results_df['buyhold_cum'], name='Buy-and-Hold SPY', line=dict(color='#6b7280', width=2)))
                fig2.add_trace(go.Scatter(x=results_df.index, y=results_df['strategy_cum'], name='Breadth-Momentum Strategy', line=dict(color='#10b981', width=2.5)))
                fig2.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(fig2, use_container_width=True)
                
                # Metrics
                st.markdown("### Performance Summary")
                total_bh = (results_df['buyhold_cum'].iloc[-1] - 100) / 100 * 100
                total_strat = (results_df['strategy_cum'].iloc[-1] - 100) / 100 * 100
                years = 5
                cagr_bh = ((results_df['buyhold_cum'].iloc[-1] / 100) ** (1/years) - 1) * 100
                cagr_strat = ((results_df['strategy_cum'].iloc[-1] / 100) ** (1/years) - 1) * 100
                
                def max_dd(cum):
                    peak = cum.cummax()
                    dd = (cum - peak) / peak * 100
                    return dd.min()
                
                mdd_bh = max_dd(results_df['buyhold_cum'])
                mdd_strat = max_dd(results_df['strategy_cum'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", f"{total_strat:+.1f}%", delta=f"{total_strat - total_bh:+.1f}% vs BH")
                with col2:
                    st.metric("CAGR", f"{cagr_strat:+.1f}%", delta=f"{cagr_strat - cagr_bh:+.1f}% vs BH")
                with col3:
                    st.metric("Max Drawdown", f"{mdd_strat:.1f}%", delta=f"{mdd_strat - mdd_bh:+.1f}% vs BH")
                
                st.success("✅ Backtest complete!")
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Try refreshing the page")
    else:
        st.info("👆 Click button above to load backtest data")

# ============================================
# MAIN APP
# ============================================

def main():
    st.title("📊 Market Regime Dashboard")
    st.markdown("*Breadth-Momentum framework with Canary Indicator Early Warning*")
    
    # Tabs
    tab1, tab2 = st.tabs(["🎯 Current Signal", "📈 Historical Backtest"])
    
    with tab1:
        render_current_tab()
    
    with tab2:
        render_backtest_tab()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("**Core Indicators:**")
        st.markdown("- $NYHL Cumulative (Regime)\n- $BPSPX (Early Warning)\n- $OEXA150R (Entry)\n- SPY:VXX Ratio (Risk)")
        st.markdown("**Canary Indicators:**")
        st.markdown("- 🔴 High Yield Spreads\n- 🟠 Semis/SPX Ratio\n- 🟠 Small/Large Cap\n- 🟡 McClellan Oscillator")
        st.markdown("---")
        st.caption("⚠️ Not investment advice. Test before live use.")

if __name__ == "__main__":
    main()
