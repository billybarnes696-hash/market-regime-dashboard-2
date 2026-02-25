import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="SPY/IEF Signal Dashboard", layout="wide", page_icon="🔄")

st.markdown("""
<style>
    .signal-spy {background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .signal-ief {background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
</style>
""", unsafe_allow_html=True)

# ============================================
# REAL DATA FROM YFINANCE
# ============================================

@st.cache_data(ttl=3600)
def fetch_real_data(period="5y"):
    """Fetch real SPY and IEF data from yfinance"""
    try:
        # Download SPY and IEF
        spy = yf.download("SPY", period=period, progress=False)
        ief = yf.download("IEF", period=period, progress=False)
        
        if spy.empty or ief.empty:
            st.error("Failed to fetch data from yfinance")
            return None, None
        
        # Clean data
        spy = spy[['Close']].rename(columns={'Close': 'SPY'})
        ief = ief[['Close']].rename(columns={'Close': 'IEF'})
        
        # Align dates
        df = pd.concat([spy, ief], axis=1).dropna()
        
        return df, spy, ief
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None

def calculate_smooth_signal(df):
    """
    Calculate smooth composite signal from price data.
    Uses PPO-style calculation on SPY with IEF as confirmation.
    """
    # Calculate returns
    df = df.copy()
    df['spy_ret'] = df['SPY'].pct_change()
    df['ief_ret'] = df['IEF'].pct_change()
    
    # Momentum indicators
    df['spy_sma20'] = df['SPY'].rolling(20).mean()
    df['spy_sma50'] = df['SPY'].rolling(50).mean()
    df['spy_sma200'] = df['SPY'].rolling(200).mean()
    
    # PPO-style signal (like your chart)
    df['ema12'] = df['SPY'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['SPY'].ewm(span=26, adjust=False).mean()
    df['ppo'] = ((df['ema12'] - df['ema26']) / df['ema26']) * 100
    df['ppo_signal'] = df['ppo'].ewm(span=9, adjust=False).mean()
    df['ppo_hist'] = df['ppo'] - df['ppo_signal']
    
    # Relative strength SPY vs IEF
    df['spy_ief_ratio'] = df['SPY'] / df['IEF']
    df['ratio_sma20'] = df['spy_ief_ratio'].rolling(20).mean()
    df['ratio_signal'] = (df['spy_ief_ratio'] - df['ratio_sma20']) / df['ratio_sma20'] * 100
    
    # Volatility (inverse of risk sentiment)
    df['volatility'] = df['spy_ret'].rolling(20).std()
    df['vol_signal'] = (df['volatility'].rolling(20).mean() - df['volatility']) / df['volatility'].rolling(20).std()
    
    # Trend strength
    df['trend'] = (df['SPY'] - df['spy_sma200']) / df['spy_sma200'] * 100
    
    # === COMPOSITE SIGNAL ===
    # Combine multiple factors (smoothed like PPO)
    composite = (
        df['ppo_hist'] * 0.40 +           # PPO histogram (momentum)
        df['ratio_signal'] * 0.25 +        # SPY/IEF relative strength
        df['trend'] * 0.20 +               # Long-term trend
        df['vol_signal'] * 0.10 +          # Volatility signal
        df['ppo'] * 0.05                   # PPO line
    )
    
    # Smooth the composite (like PPO 5,13,0)
    composite_smooth = composite.ewm(span=5, adjust=False).mean()
    
    # Normalize around zero
    composite_smooth = (composite_smooth - composite_smooth.rolling(50, min_periods=1).mean())
    
    df['signal_line'] = composite_smooth
    
    # Generate signals
    df['signal'] = np.where(df['signal_line'] > 0, 'SPY', 'IEF')
    
    return df

def calculate_strategy_returns(df):
    """Calculate strategy returns based on signal"""
    df = df.copy()
    
    # Create allocations
    df['SPY_%'] = np.where(df['signal'] == 'SPY', 100, 0)
    df['IEF_%'] = np.where(df['signal'] == 'IEF', 100, 0)
    
    # Calculate daily returns
    df['spy_ret'] = df['SPY'].pct_change().fillna(0)
    df['ief_ret'] = df['IEF'].pct_change().fillna(0)
    
    # Strategy return: 100% in SPY or 100% in IEF based on signal
    df['strategy_ret'] = (
        (df['SPY_%'] / 100) * df['spy_ret'].shift(1) + 
        (df['IEF_%'] / 100) * df['ief_ret'].shift(1)
    ).fillna(0)
    
    # Cumulative returns (starting at 100)
    df['buyhold_cum'] = (1 + df['spy_ret']).cumprod() * 100
    df['strat_cum'] = (1 + df['strategy_ret']).cumprod() * 100
    
    # Find crossovers
    df['crossed_above'] = (df['signal_line'] > 0) & (df['signal_line'].shift(1) <= 0)
    df['crossed_below'] = (df['signal_line'] < 0) & (df['signal_line'].shift(1) >= 0)
    
    return df

# ============================================
# CURRENT SIGNAL
# ============================================

def get_current_signal():
    """Get latest signal from real data"""
    try:
        df, spy, ief = fetch_real_data(period="6mo")
        if df is None:
            return None, None, None
        
        df = calculate_smooth_signal(df)
        df = calculate_strategy_returns(df)
        
        latest = df.iloc[-1]
        signal = latest['signal']
        
        if signal == 'SPY':
            alloc = {'SPY': 100, 'IEF': 0}
        else:
            alloc = {'SPY': 0, 'IEF': 100}
        
        return signal, alloc, latest['signal_line']
    except:
        return None, None, None

# ============================================
# TAB 1: CURRENT SIGNAL
# ============================================

def render_current_tab():
    signal, alloc, score = get_current_signal()
    
    if signal is None:
        st.error("Unable to fetch current signal. Please try again.")
        return
    
    if signal == "SPY":
        st.markdown(f"""
        <div class="signal-spy">
            🟢 ROTATE TO SPY (100%)<br>
            <small>Signal: {score:+.2f} • Above zero threshold</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="signal-ief">
            🔴 ROTATE TO IEF (100%)<br>
            <small>Signal: {score:+.2f} • Below zero threshold</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### ⚡ Execute")
    if signal == "SPY":
        st.button("✅ Buy SPY / Sell IEF", type="primary", use_container_width=True)
    else:
        st.button("✅ Buy IEF / Sell SPY", type="primary", use_container_width=True)

# ============================================
# TAB 2: REAL DATA CHART
# ============================================

def render_chart_tab():
    st.subheader("📈 Real SPY/IEF Performance with Signal")
    st.markdown("*Actual data from yfinance • Cross above zero = SPY | Cross below zero = IEF*")
    
    if st.button("📊 Load Real Data Chart", type="primary"):
        with st.spinner("Fetching real data from yfinance..."):
            df, spy, ief = fetch_real_data(period="5y")
            
            if df is None:
                st.error("Failed to fetch data. Please try again.")
                return
            
            # Calculate signal
            df = calculate_smooth_signal(df)
            df = calculate_strategy_returns(df)
            
            # === CHART 1: SPY Price with Signals ===
            fig1 = go.Figure()
            
            # SPY price
            fig1.add_trace(go.Scatter(
                x=df.index,
                y=df['SPY'],
                name='SPY Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Color background by signal
            prev_signal = None
            for i in range(len(df)-1):
                curr_signal = df['signal'].iloc[i]
                if curr_signal != prev_signal:
                    start_idx = i
                    end_idx = i + 1
                    while end_idx < len(df) and df['signal'].iloc[end_idx] == curr_signal:
                        end_idx += 1
                    
                    color = 'rgba(16, 185, 129, 0.10)' if curr_signal == 'SPY' else 'rgba(220, 38, 38, 0.10)'
                    fig1.add_vrect(
                        x0=df.index[start_idx],
                        x1=df.index[min(end_idx, len(df)-1)],
                        fillcolor=color,
                        opacity=0.5,
                        layer="below",
                        line_width=0
                    )
                    prev_signal = curr_signal
            
            # Add crossover markers
            buy_signals = df[df['crossed_above']]
            sell_signals = df[df['crossed_below']]
            
            if not buy_signals.empty:
                fig1.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['SPY'],
                    mode='markers',
                    name='→ SPY (Cross Above)',
                    marker=dict(color='#10b981', size=10, symbol='triangle-up')
                ))
            
            if not sell_signals.empty:
                fig1.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['SPY'],
                    mode='markers',
                    name='→ IEF (Cross Below)',
                    marker=dict(color='#dc2626', size=10, symbol='triangle-down')
                ))
            
            fig1.update_layout(
                title='SPY Price with Rotation Signals (Real Data)',
                xaxis_title='Date',
                yaxis_title='SPY Price ($)',
                height=450,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # === CHART 2: Signal Line ===
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=df.index,
                y=df['signal_line'],
                name='Smooth Signal Line',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig2.update_layout(
                title='Composite Signal Line (PPO-Style)',
                xaxis_title='Date',
                yaxis_title='Signal Value',
                height=300,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # === CHART 3: Performance Comparison ===
            fig3 = go.Figure()
            
            fig3.add_trace(go.Scatter(
                x=df.index,
                y=df['buyhold_cum'],
                name='Buy-and-Hold SPY',
                line=dict(color='#6b7280', width=2)
            ))
            
            fig3.add_trace(go.Scatter(
                x=df.index,
                y=df['strat_cum'],
                name='Strategy (SPY/IEF Rotation)',
                line=dict(color='#10b981', width=2.5)
            ))
            
            fig3.update_layout(
                title='Real Performance Comparison ($100 Starting Value)',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # === PERFORMANCE METRICS ===
            total_bh = (df['buyhold_cum'].iloc[-1] - 100)
            total_strat = (df['strat_cum'].iloc[-1] - 100)
            
            # Calculate CAGR
            years = (df.index[-1] - df.index[0]).days / 365.25
            cagr_bh = ((df['buyhold_cum'].iloc[-1] / 100) ** (1/years) - 1) * 100
            cagr_strat = ((df['strat_cum'].iloc[-1] / 100) ** (1/years) - 1) * 100
            
            # Max drawdown
            def max_dd(cum):
                peak = cum.cummax()
                dd = (cum - peak) / peak * 100
                return dd.min()
            
            mdd_bh = max_dd(df['buyhold_cum'])
            mdd_strat = max_dd(df['strat_cum'])
            
            # Count rotations
            rotations = len(df[df['crossed_above']]) + len(df[df['crossed_below']])
            
            st.markdown("### 📊 Real Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{total_strat:+.1f}%", 
                         delta=f"{total_strat - total_bh:+.1f}% vs Buy-and-Hold")
            with col2:
                st.metric("CAGR", f"{cagr_strat:+.1f}%", 
                         delta=f"{cagr_strat - cagr_bh:+.1f}% vs Buy-and-Hold")
            with col3:
                st.metric("Max Drawdown", f"{mdd_strat:.1f}%", 
                         delta=f"{mdd_strat - mdd_bh:+.1f}% vs Buy-and-Hold")
            with col4:
                st.metric("Signal Rotations", f"{rotations}", 
                         delta=f"~{rotations/years:.1f} per year")
            
            st.success("✅ Real data loaded from yfinance • Actual SPY and IEF prices")
    else:
        st.info("👆 Click button to load real data from yfinance")

# ============================================
# MAIN
# ============================================

def main():
    st.title("🔄 SPY/IEF Signal Dashboard (REAL DATA)")
    st.markdown("*Real historical data from yfinance • Smooth PPO-style signal • Zero-line crossovers*")
    
    tab1, tab2 = st.tabs(["🎯 Current Signal", "📊 Real Data Chart"])
    
    with tab1:
        render_current_tab()
    with tab2:
        render_chart_tab()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("**Data Source:** yfinance (real-time)")
        st.markdown("**Signal Logic:**")
        st.markdown("""
        Composite = 
        - 40% PPO Histogram (momentum)
        - 25% SPY/IEF Ratio (relative strength)
        - 20% Trend vs 200-SMA
        - 10% Volatility signal
        - 5% PPO line
        
        **Smoothing:** EMA(5) + normalization
        
        **Threshold:** 
        - Cross ABOVE zero → SPY (100%)
        - Cross BELOW zero → IEF (100%)
        """)
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("⚠️ Not investment advice • Real data • Test before live use")

if __name__ == "__main__":
    main()
