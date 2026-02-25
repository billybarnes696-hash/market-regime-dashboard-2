import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

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
# SMOOTH COMPOSITE SIGNAL (Like PPO 5,13,0)
# ============================================

@st.cache_data(ttl=3600)
def generate_smooth_signal_data(days=252):
    """
    Generate smooth composite signal from your indicators.
    Smoothed like PPO(5,13,0) for clean zero-line crossovers.
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    # Mock SPY price
    ret = np.random.normal(0.0003, 0.012, days)
    prices = 100 * np.cumprod(1 + ret)
    
    # === BUILD COMPOSITE SIGNAL ===
    # Combine your indicators into ONE smooth line
    
    # 1. Breadth trend ($NYHL style)
    breadth = np.cumsum(ret * 50 + np.random.normal(0, 2, days))
    
    # 2. Momentum ($BPSPX style)
    momentum = pd.Series(ret).rolling(20).mean().values * 100
    momentum = np.nan_to_num(momentum, nan=0)
    
    # 3. Participation ($OEXA150R style)
    ma150 = pd.Series(prices).rolling(150).mean().values
    participation = ((prices > ma150).astype(float) - 0.5) * 100
    
    # 4. Risk sentiment (inverse VIX)
    vol = pd.Series(ret).rolling(20).std().values
    risk = 1 / (vol * 100 + 0.01) - 25
    
    # 5. Credit spreads (mock)
    credit = 380 - np.cumsum(ret * 50) + np.random.normal(0, 15, days)
    credit_norm = (450 - credit) / 50
    
    # === COMPOSITE (Weighted sum) ===
    composite = (
        breadth * 0.35 +
        momentum * 0.25 +
        participation * 0.20 +
        risk * 0.10 +
        credit_norm * 0.10
    )
    
    # === SMOOTH LIKE PPO(5,13,0) ===
    # Fast EMA (5-day)
    fast_ema = pd.Series(composite).ewm(span=5, adjust=False).mean().values
    # Slow EMA (13-day)
    slow_ema = pd.Series(composite).ewm(span=13, adjust=False).mean().values
    
    # PPO-style signal line
    signal_line = ((fast_ema - slow_ema) / slow_ema) * 100
    signal_line = np.nan_to_num(signal_line, nan=0)
    
    # Additional smoothing (3-day SMA)
    signal_line = pd.Series(signal_line).rolling(3, min_periods=1).mean().values
    
    # === GENERATE SIGNALS ===
    signals = []
    for val in signal_line:
        if val > 0:
            signals.append("SPY")
        else:
            signals.append("IEF")
    
    # === CALCULATE RETURNS ===
    allocations = [{'SPY': 100, 'IEF': 0} if s == 'SPY' else {'SPY': 0, 'IEF': 100} for s in signals]
    
    df = pd.DataFrame({
        'Date': dates,
        'SPY': prices,
        'SignalLine': signal_line,
        'Signal': signals,
        'Allocation': allocations
    }, index=dates)
    
    df['SPY_%'] = [a['SPY'] for a in allocations]
    df['IEF_%'] = [a['IEF'] for a in allocations]
    
    # Returns
    df['spy_ret'] = df['SPY'].pct_change().fillna(0)
    df['ief_ret'] = np.where(df['spy_ret'] < 0, -0.30 * df['spy_ret'], 0.0008)
    
    def calc_strat_ret(row):
        alloc = row['Allocation']
        return (alloc['SPY']/100 * row['spy_ret'] + alloc['IEF']/100 * row['ief_ret'])
    
    df['strat_ret'] = df.apply(calc_strat_ret, axis=1)
    df['buyhold_cum'] = (1 + df['spy_ret']).cumprod() * 100
    df['strat_cum'] = (1 + df['strat_ret']).cumprod() * 100
    
    return df

# ============================================
# CURRENT SIGNAL
# ============================================

def get_current_signal():
    """Current signal from your chart values"""
    # Your values from StockCharts (Feb 24, 2026)
    nyhl_cum, nyhl_200 = 32941, 27441
    bpspx_rsi = 36.47
    oexa_val = 63.00
    vix = 18.45
    hy_spread = 385
    
    # Simple composite
    breadth = (nyhl_cum - nyhl_200) / 1000
    momentum = (bpspx_rsi - 50) / 10
    participation = (oexa_val - 50) / 20
    risk = (25 - vix) / 5
    credit = (450 - hy_spread) / 50
    
    composite = (
        breadth * 0.35 +
        momentum * 0.25 +
        participation * 0.20 +
        risk * 0.10 +
        credit * 0.10
    )
    
    if composite > 0:
        return "SPY", {'SPY': 100, 'IEF': 0}, composite
    else:
        return "IEF", {'SPY': 0, 'IEF': 100}, composite

# ============================================
# TAB 1: CURRENT SIGNAL
# ============================================

def render_current_tab():
    signal, alloc, score = get_current_signal()
    
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
# TAB 2: SMOOTH SIGNAL CHART (Like PPO)
# ============================================

def render_chart_tab():
    st.subheader("📈 Smooth Signal Line (Like PPO 5,13,0)")
    st.markdown("*Above zero = SPY (100%) • Below zero = IEF (100%)*")
    
    if st.button("📊 Load Chart", type="primary"):
        with st.spinner("Generating smooth signal..."):
            df = generate_smooth_signal_data(252)
            
            # Find zero-line crossovers
            df['CrossedAbove'] = (df['SignalLine'] > 0) & (df['SignalLine'].shift(1) <= 0)
            df['CrossedBelow'] = (df['SignalLine'] < 0) & (df['SignalLine'].shift(1) >= 0)
            
            # Create chart
            fig = go.Figure()
            
            # SPY price (top)
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['SPY'],
                name='SPY Price',
                line=dict(color='#1f77b4', width=2),
                yaxis='y1'
            ))
            
            # Signal line (bottom) - SMOOTH like PPO
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['SignalLine'],
                name='Smooth Signal Line (Like PPO 5,13,0)',
                line=dict(color='#ff7f0e', width=2.5),
                yaxis='y2'
            ))
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         annotation_text="Zero Threshold", annotation_position="top right")
            
            # Color background by signal
            prev_signal = None
            for i in range(len(df)-1):
                curr_signal = df['Signal'].iloc[i]
                if curr_signal != prev_signal:
                    start_idx = i
                    end_idx = i + 1
                    while end_idx < len(df) and df['Signal'].iloc[end_idx] == curr_signal:
                        end_idx += 1
                    
                    color = 'rgba(16, 185, 129, 0.10)' if curr_signal == 'SPY' else 'rgba(220, 38, 38, 0.10)'
                    fig.add_vrect(
                        x0=df['Date'].iloc[start_idx],
                        x1=df['Date'].iloc[min(end_idx, len(df)-1)],
                        fillcolor=color,
                        opacity=0.5,
                        layer="below",
                        line_width=0
                    )
                    prev_signal = curr_signal
            
            # Add crossover markers
            buy_signals = df[df['CrossedAbove']]
            sell_signals = df[df['CrossedBelow']]
            
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals['Date'],
                    y=buy_signals['SignalLine'],
                    mode='markers',
                    name='→ SPY (Cross Above Zero)',
                    marker=dict(color='#10b981', size=10, symbol='triangle-up')
                ))
            
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals['Date'],
                    y=sell_signals['SignalLine'],
                    mode='markers',
                    name='→ IEF (Cross Below Zero)',
                    marker=dict(color='#dc2626', size=10, symbol='triangle-down')
                ))
            
            # Layout with dual y-axes
            fig.update_layout(
                title='SPY Price with Smooth Signal Line<br><sup>🟢 Green background = Hold SPY (100%) | 🔴 Red background = Hold IEF (100%)</sup>',
                xaxis_title='Date (Daily)',
                yaxis=dict(title='SPY Price ($)', side='left', showgrid=True),
                yaxis2=dict(title='Signal Line', side='right', overlaying='y', showgrid=False),
                height=600,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance
            total_bh = (df['buyhold_cum'].iloc[-1] - 100)
            total_strat = (df['strat_cum'].iloc[-1] - 100)
            
            # Count rotations
            rotations = len(df[df['CrossedAbove']]) + len(df[df['CrossedBelow']])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Strategy Return (1Y)", f"{total_strat:+.1f}%", 
                         delta=f"{total_strat - total_bh:+.1f}% vs Buy-and-Hold")
            with col2:
                st.metric("Max Drawdown", f"{df['strat_cum'].min()/df['strat_cum'].max()-1:.1%}",
                         delta=f"vs BH: {df['buyhold_cum'].min()/df['buyhold_cum'].max()-1:.1%}")
            with col3:
                st.metric("Signal Rotations (1Y)", f"{rotations}", 
                         delta=f"~{rotations/12:.1f} per month")
            
            st.success("✅ Chart loaded • Smooth signal like PPO(5,13,0) • Zero-line crossovers trigger rotations")
    else:
        st.info("👆 Click button to load chart")

# ============================================
# MAIN
# ============================================

def main():
    st.title("🔄 SPY/IEF Smooth Signal Dashboard")
    st.markdown("*Smooth composite indicator (like PPO 5,13,0) • Cross above zero = SPY | Cross below zero = IEF*")
    
    tab1, tab2 = st.tabs(["🎯 Current Signal", "📊 Smooth Signal Chart"])
    
    with tab1:
        render_current_tab()
    with tab2:
        render_chart_tab()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("**Signal Logic:**")
        st.markdown("""
        Composite = 
        - 35% Breadth ($NYHL)
        - 25% Momentum ($BPSPX)
        - 20% Participation ($OEXA150R)
        - 10% Risk (VIX)
        - 10% Credit (Spreads)
        
        **Smoothing:** PPO-style (5,13,0) + 3-day SMA
        
        **Threshold:** 
        - Cross ABOVE zero → SPY (100%)
        - Cross BELOW zero → IEF (100%)
        
        **Timeframe:** Daily signals
        """)
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("⚠️ Not investment advice")

if __name__ == "__main__":
    main()
