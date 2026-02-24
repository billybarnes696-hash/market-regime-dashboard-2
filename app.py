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
st.set_page_config(page_title="Market Regime Dashboard", layout="wide", page_icon="📊")

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
    .canary-warning {background: #dc2626; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: bold; font-size: 0.9rem;}
    .canary-ok {background: #10b981; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: bold; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA FETCHING
# ============================================

@st.cache_data(ttl=3600)
def fetch_spy_history(years=15):
    """Fetch SPY historical data for backtesting"""
    end = datetime.now()
    start = end - timedelta(days=years*365)
    spy = yf.download('SPY', start=start, end=end, progress=False)
    return spy

@st.cache_data(ttl=86400)
def fetch_ief_history(years=15):
    """Fetch IEF historical data for backtesting"""
    end = datetime.now()
    start = end - timedelta(days=years*365)
    ief = yf.download('IEF', start=start, end=end, progress=False)
    return ief

@st.cache_data(ttl=86400)
def generate_historical_breadth_data(spy_df):
    """
    Generate simulated historical breadth indicators.
    In production, replace with real StockCharts CSV data.
    """
    df = spy_df.copy()
    np.random.seed(42)  # Fixed seed for consistency
    
    # Simulate $NYHL Cumulative (trending with SPY)
    returns = df['Close'].pct_change()
    df['nyhl_cum'] = (returns.rolling(20).sum().cumsum() * 10000 + 25000).fillna(method='ffill')
    df['nyhl_sma200'] = df['nyhl_cum'].rolling(200).mean()
    df['nyhl_sma50'] = df['nyhl_cum'].rolling(50).mean()
    
    # Simulate $BPSPX (mean-reverting around 50)
    momentum = df['Close'].pct_change(20).rolling(10).mean()
    df['bpspx_value'] = (50 + momentum * 30 + np.random.normal(0, 5, len(df))).clip(10, 90)
    df['bpspx_rsi14'] = (50 + momentum * 40 + np.random.normal(0, 8, len(df))).clip(10, 90)
    df['bpspx_macd_hist'] = momentum * 2 + np.random.normal(0, 0.3, len(df))
    df['bpspx_sma50'] = df['bpspx_value'].rolling(50).mean()
    
    # Simulate $OEXA150R (% above 150-DMA)
    ma150 = df['Close'].rolling(150).mean()
    above_ma = (df['Close'] > ma150).astype(float)
    df['oexa150r_value'] = (above_ma.rolling(20).mean() * 100 + np.random.normal(0, 8, len(df))).clip(10, 90)
    df['oexa150r_cci14'] = ((df['Close'] - df['Close'].rolling(14).mean()) / df['Close'].rolling(14).std() * 100)
    
    # Simulate SPY:VXX Ratio (inverse of volatility)
    vol = returns.rolling(20).std()
    df['spy_vxx_ratio'] = (1 / (vol * 100) + np.random.normal(0, 3, len(df))).clip(5, 50)
    df['spy_vxx_sma50'] = df['spy_vxx_ratio'].rolling(50).mean()
    df['spy_vxx_sma200'] = df['spy_vxx_ratio'].rolling(200).mean()
    
    # Simulate VIX
    df['vix'] = (20 - momentum * 15 + np.random.normal(0, 4, len(df))).clip(10, 50)
    
    # Simulate Credit Spread (Canary Indicator)
    df['hy_spread_bps'] = (350 - momentum * 100 + np.random.normal(0, 30, len(df))).clip(200, 600)
    
    # Simulate McClellan Oscillator (Canary Indicator)
    df['mcclellan_osc'] = momentum * 80 + np.random.normal(0, 20, len(df))
    
    # Simulate Semis/SPX Ratio (Canary Indicator)
    df['semis_spx_ratio'] = (1.5 + momentum * 0.5 + np.random.normal(0, 0.2, len(df))).clip(0.8, 2.5)
    df['semis_spx_sma50'] = df['semis_spx_ratio'].rolling(50).mean()
    
    # Simulate Small/Large Cap Ratio (Canary Indicator)
    df['smallcap_largecap_ratio'] = (0.45 + momentum * 0.1 + np.random.normal(0, 0.05, len(df))).clip(0.30, 0.60)
    
    return df.dropna()

# ============================================
# CANARY INDICATOR CHECK
# ============================================

def check_canary_warnings(row):
    """Check canary indicators for early warning signals"""
    warnings = 0
    details = []
    
    # Tier 1: Credit spreads (highest priority)
    if row['hy_spread_bps'] > 450:
        warnings += 2
        details.append("🔴 Credit Stress")
    
    # Tier 2: Leadership deterioration
    if 'semis_spx_ratio' in row and row['semis_spx_ratio'] < row.get('semis_spx_sma50', 1.5):
        warnings += 1
        details.append("🟠 Semis Weakness")
    
    if 'smallcap_largecap_ratio' in row and row['smallcap_largecap_ratio'] < 0.40:
        warnings += 1
        details.append("🟠 Small Cap Weakness")
    
    # Tier 3: Breadth momentum
    if 'mcclellan_osc' in row and row['mcclellan_osc'] < -50:
        warnings += 1
        details.append("🟡 Breadth Weakness")
    
    return warnings, details

# ============================================
# SIGNAL GENERATION (HISTORICAL)
# ============================================

def generate_historical_signals(df):
    """Generate historical signals for each day"""
    signals = []
    allocations = []
    canary_scores = []
    
    for idx, row in df.iterrows():
        # Primary regime
        regime_bull = row['nyhl_cum'] > row['nyhl_sma200']
        
        # Early warning
        bpspx_bear = (row['bpspx_rsi14'] < 40 and row['bpspx_macd_hist'] < 0) or \
                     (row['bpspx_value'] < row['bpspx_sma50'] and row['bpspx_macd_hist'] < 0)
        
        # Entry confirmation
        oexa_oversold = row['oexa150r_value'] < 35 and row['oexa150r_cci14'] < -100
        
        # Risk sentiment
        risk_off = row['spy_vxx_ratio'] < row['spy_vxx_sma50'] and row['vix'] > 20
        
        # Canary check
        canary_score, canary_details = check_canary_warnings(row)
        canary_scores.append(canary_score)
        
        # Scoring
        bull_score = int(regime_bull) * 4 + int(oexa_oversold) * 3 + int(row['mcclellan_osc'] > 20) * 2
        bear_score = (int(not regime_bull) * 4 + int(bpspx_bear) * 3 + 
                     int(risk_off) * 3 + int(canary_score >= 3) * 3)
        
        # Signal logic with canary weighting
        if canary_score >= 5:
            signal = "WARNING - CANARIES"
            alloc = {'SPY': 0.40, 'IEF': 0.45, 'CASH': 0.15}
        elif not regime_bull or (bear_score >= 8 and bull_score < 4):
            signal = "SELL"
            alloc = {'SPY': 0.20, 'IEF': 0.60, 'CASH': 0.20}
        elif bear_score > bull_score and regime_bull:
            signal = "WARNING"
            alloc = {'SPY': 0.40, 'IEF': 0.45, 'CASH': 0.15}
        elif oexa_oversold and regime_bull:
            signal = "BUY DIP"
            alloc = {'SPY': 0.80, 'IEF': 0.15, 'CASH': 0.05}
        elif bull_score - bear_score >= 3 and regime_bull:
            signal = "BUY"
            alloc = {'SPY': 0.85, 'IEF': 0.10, 'CASH': 0.05}
        else:
            signal = "HOLD"
            alloc = {'SPY': 0.65, 'IEF': 0.25, 'CASH': 0.10}
        
        signals.append(signal)
        allocations.append(alloc)
    
    df = df.copy()
    df['signal'] = signals
    df['allocation'] = allocations
    df['canary_score'] = canary_scores
    return df

def calculate_strategy_returns(df, ief_df=None):
    """Calculate strategy vs buy-and-hold returns"""
    df = df.copy()
    df['spy_return'] = df['Close'].pct_change().fillna(0)
    
    # IEF returns if available
    if ief_df is not None:
        ief_returns = ief_df['Close'].pct_change().reindex(df.index).fillna(0)
    else:
        # Simplified IEF return (positive when SPY negative)
        ief_returns = -0.3 * df['spy_return'].apply(lambda x: x if x < 0 else 0) + 0.02/252
    
    # Strategy returns
    def daily_return(row):
        spy_ret = row['spy_return']
        ief_ret = ief_returns.loc[row.name] if row.name in ief_returns.index else 0.02/252
        return row['allocation']['SPY'] * spy_ret + row['allocation']['IEF'] * ief_ret
    
    df['strategy_return'] = df.apply(daily_return, axis=1)
    df['buyhold_cum'] = (1 + df['spy_return']).cumprod() * 100
    df['strategy_cum'] = (1 + df['strategy_return']).cumprod() * 100
    
    return df

# ============================================
# CURRENT SIGNAL (Based on Your Charts)
# ============================================

def get_current_signal():
    """Get current signal based on your uploaded charts (Feb 24, 2026)"""
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
    
    # Canary check
    canary_score, canary_details = check_canary_warnings(data)
    
    # Generate signal
    regime_bull = data['nyhl_cum'] > data['nyhl_sma200']
    bpspx_bear = data['bpspx_rsi14'] < 40 and data['bpspx_macd_hist'] < 0
    risk_off = data['spy_vxx_ratio'] < data['spy_vxx_sma50']
    oexa_oversold = data['oexa150r_value'] < 35 and data['oexa150r_cci14'] < -100
    
    if canary_score >= 5:
        signal = "WARNING - CANARIES"
        signal_class = "signal-warning"
        allocation = {'SPY': 40, 'IEF': 45, 'CASH': 15}
    elif not regime_bull:
        signal = "SELL"
        signal_class = "signal-sell"
        allocation = {'SPY': 20, 'IEF': 60, 'CASH': 20}
    elif bpspx_bear and risk_off and canary_score >= 3:
        signal = "WARNING"
        signal_class = "signal-warning"
        allocation = {'SPY': 40, 'IEF': 45, 'CASH': 15}
    elif oexa_oversold and regime_bull:
        signal = "BUY DIP"
        signal_class = "signal-buy"
        allocation = {'SPY': 80, 'IEF': 15, 'CASH': 5}
    else:
        signal = "HOLD"
        signal_class = "signal-hold"
        allocation = {'SPY': 65, 'IEF': 25, 'CASH': 10}
    
    return signal, signal_class, allocation, data, canary_score, canary_details

# ============================================
# CURRENT SIGNAL TAB
# ============================================

def render_current_signal_tab():
    """Render the current signal dashboard"""
    signal, signal_class, allocation, data, canary_score, canary_details = get_current_signal()
    
    # Top signal banner
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<div class='{signal_class}'>{signal}</div>", unsafe_allow_html=True)
        st.caption(f"Updated: {data['timestamp'].strftime('%Y-%m-%d %H:%M')} | SPX: ${data['spx_price']:,.2f}")
    with col2:
        regime = "🟢 Bullish" if data['nyhl_cum'] > data['nyhl_sma200'] else "🔴 Bearish"
        st.metric("Primary Regime", regime)
    
    # Canary indicators status
    st.subheader("🚨 Canary Indicators")
    canary_cols = st.columns(4)
    with canary_cols[0]:
        credit_status = "🔴 Stress" if data['hy_spread_bps'] > 450 else "✅ Normal"
        st.metric("Credit Spreads", f"{data['hy_spread_bps']} bps", credit_status)
    with canary_cols[1]:
        semis_status = "⚠️ Weak" if data['semis_spx_ratio'] < data['semis_spx_sma50'] else "✅ Strong"
        st.metric("Semis/SPX", f"{data['semis_spx_ratio']:.2f}", semis_status)
    with canary_cols[2]:
        smallcap_status = "⚠️ Weak" if data['smallcap_largecap_ratio'] < 0.40 else "✅ Normal"
        st.metric("Small/Large Cap", f"{data['smallcap_largecap_ratio']:.2f}", smallcap_status)
    with canary_cols[3]:
        mcclellan_status = "⚠️ Weak" if data['mcclellan_osc'] < -50 else "✅ Normal"
        st.metric("McClellan", f"{data['mcclellan_osc']:+.1f}", mcclellan_status)
    
    if canary_score >= 3:
        st.warning(f"⚠️ **{canary_score} Canary Warnings Active**: {', '.join(canary_details)}")
    else:
        st.success("✅ Canary indicators are quiet - no early warning signals")
    
    # Core indicators
    st.subheader("📊 Core Indicators (From Your Charts)")
    core_cols = st.columns(3)
    with core_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <strong>$NYHL Cumulative</strong><br>
            {data['nyhl_cum']:,} vs 200-SMA: {data['nyhl_sma200']:,}<br>
            {'✅ Bullish Regime' if data['nyhl_cum'] > data['nyhl_sma200'] else '❌ Bearish Regime'}
        </div>
        """, unsafe_allow_html=True)
    with core_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <strong>$BPSPX</strong><br>
            {data['bpspx_value']}% | RSI: {data['bpspx_rsi14']:.1f}<br>
            {'⚠️ Distribution' if data['bpspx_rsi14'] < 40 else '✅ Neutral/Bullish'}
        </div>
        """, unsafe_allow_html=True)
    with core_cols[2]:
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
    
    # Action buttons
    st.markdown("### ⚡ Quick Actions")
    cols = st.columns(3)
    with cols[0]:
        st.button("✅ Add SPY", disabled=allocation['SPY'] < 65, width="stretch")
    with cols[1]:
        st.button("🛡️ Add IEF", disabled=allocation['IEF'] < 40, width="stretch")
    with cols[2]:
        st.button("💵 Raise Cash", disabled=allocation['CASH'] < 15, width="stretch")

# ============================================
# BACKTEST TAB
# ============================================

def render_backtest_tab():
    """Render the historical backtest dashboard"""
    st.subheader("📈 Historical Backtest (15 Years)")
    st.markdown("*Visual backtest: See if signals caught major tops and bottoms*")
    
    with st.spinner("Loading 15 years of data and calculating signals..."):
        # Fetch and process data
        spy_df = fetch_spy_history(15)
        ief_df = fetch_ief_history(15)
        breadth_df = generate_historical_breadth_data(spy_df)
        signal_df = generate_historical_signals(breadth_df)
        results_df = calculate_strategy_returns(signal_df, ief_df)
    
    # Chart 1: SPY Price with Signal Markers
    st.markdown("### 🔹 SPY Price with Buy/Sell Signals")
    
    buy_signals = results_df[results_df['signal'].isin(['BUY', 'BUY DIP'])]
    sell_signals = results_df[results_df['signal'].isin(['SELL', 'WARNING', 'WARNING - CANARIES'])]
    
    fig_price = go.Figure()
    
    # SPY Price line
    fig_price.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df['Close'],
        name='SPY Price',
        line=dict(color='#1f77b4', width=1.5)
    ))
    
    # Buy signals (green triangles)
    if not buy_signals.empty:
        fig_price.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            name='BUY Signals',
            marker=dict(color='#10b981', size=10, symbol='triangle-up')
        ))
    
    # Sell signals (red X)
    if not sell_signals.empty:
        fig_price.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            name='SELL/WARNING Signals',
            marker=dict(color='#dc2626', size=10, symbol='x')
        ))
    
    fig_price.update_layout(
        title='SPY Price with Strategy Signals (2011-2026)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_price, width="stretch")
    
    # Chart 2: Equity Curve Comparison
    st.markdown("### 🔹 Strategy vs. Buy-and-Hold Performance")
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df['buyhold_cum'],
        name='Buy-and-Hold SPY',
        line=dict(color='#6b7280', width=2)
    ))
    fig_equity.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df['strategy_cum'],
        name='Breadth-Momentum Strategy',
        line=dict(color='#10b981', width=2.5)
    ))
    
    fig_equity.update_layout(
        title='Cumulative Returns ($100 Starting Value)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_equity, width="stretch")
    
    # Performance Metrics
    st.markdown("### 🔹 Performance Summary (15 Years)")
    
    total_return_bh = (results_df['buyhold_cum'].iloc[-1] - 100) / 100 * 100
    total_return_strat = (results_df['strategy_cum'].iloc[-1] - 100) / 100 * 100
    
    years = 15
    cagr_bh = ((results_df['buyhold_cum'].iloc[-1] / 100) ** (1/years) - 1) * 100
    cagr_strat = ((results_df['strategy_cum'].iloc[-1] / 100) ** (1/years) - 1) * 100
    
    # Max drawdown
    def max_dd(cum):
        peak = cum.cummax()
        dd = (cum - peak) / peak * 100
        return dd.min()
    
    mdd_bh = max_dd(results_df['buyhold_cum'])
    mdd_strat = max_dd(results_df['strategy_cum'])
    
    # Sharpe ratio
    excess_bh = results_df['spy_return'] - 0.02/252
    excess_strat = results_df['strategy_return'] - 0.02/252
    sharpe_bh = (excess_bh.mean() / excess_bh.std()) * np.sqrt(252) if excess_bh.std() > 0 else 0
    sharpe_strat = (excess_strat.mean() / excess_strat.std()) * np.sqrt(252) if excess_strat.std() > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", f"{total_return_strat:+.1f}%", 
                  delta=f"{total_return_strat - total_return_bh:+.1f}% vs BH")
    with col2:
        st.metric("CAGR", f"{cagr_strat:+.1f}%", 
                  delta=f"{cagr_strat - cagr_bh:+.1f}% vs BH")
    with col3:
        st.metric("Max Drawdown", f"{mdd_strat:.1f}%", 
                  delta=f"{mdd_strat - mdd_bh:+.1f}% vs BH")
    with col4:
        st.metric("Sharpe Ratio", f"{sharpe_strat:.2f}", 
                  delta=f"{sharpe_strat - sharpe_bh:+.2f} vs BH")
    
    # Signal Distribution
    st.markdown("### 🔹 Signal Distribution")
    signal_counts = results_df['signal'].value_counts()
    fig_dist = px.bar(
        x=signal_counts.index,
        y=signal_counts.values,
        color=signal_counts.index,
        color_discrete_map={'BUY': '#10b981', 'BUY DIP': '#3b82f6', 'HOLD': '#f59e0b', 
                           'WARNING': '#f97316', 'WARNING - CANARIES': '#ec4899', 'SELL': '#dc2626'},
        labels={'x': 'Signal', 'y': 'Number of Days'}
    )
    fig_dist.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_dist, width="stretch")
    
    # Canary Score Distribution
    st.markdown("### 🔹 Canary Warning Score Distribution")
    canary_dist = results_df['canary_score'].value_counts().sort_index()
    fig_canary = px.bar(
        x=canary_dist.index.astype(str),
        y=canary_dist.values,
        color=canary_dist.index,
        color_continuous_scale='RdYlGn_r',
        labels={'x': 'Canary Score', 'y': 'Number of Days'}
    )
    fig_canary.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_canary, width="stretch")
    
    # Disclaimer
    st.warning("""
    ⚠️ **Note**: This backtest uses *simulated* breadth indicators for demonstration. 
    Real historical data from StockCharts would provide more accurate results.
    Past performance does not guarantee future results.
    """)

# ============================================
# MAIN APP
# ============================================

def main():
    st.title("📊 Market Regime Dashboard")
    st.markdown("*Breadth-Momentum framework with **Canary Indicator Early Warning System***")
    
    # Tabs
    tab1, tab2 = st.tabs(["🎯 Current Signal", "📈 Historical Backtest"])
    
    with tab1:
        render_current_signal_tab()
    
    with tab2:
        render_backtest_tab()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if st.button("🔄 Refresh Data", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Core Indicators:**")
        st.markdown("""
        - $NYHL Cumulative (Regime)
        - $BPSPX (Early Warning)
        - $OEXA150R (Entry)
        - SPY:VXX Ratio (Risk)
        """)
        
        st.markdown("**Canary Indicators:**")
        st.markdown("""
        - 🔴 High Yield Spreads
        - 🟠 Semis/SPX Ratio
        - 🟠 Small/Large Cap Ratio
        - 🟡 McClellan Oscillator
        """)
        
        st.markdown("---")
        st.caption("⚠️ Not investment advice. Test before live use.")

if __name__ == "__main__":
    main()
