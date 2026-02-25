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
# YOUR ACTUAL INDICATOR VALUES (From Uploaded Charts)
# ============================================

def get_current_indicator_values():
    """Pull values directly from your StockCharts PDFs (Feb 24, 2026)"""
    return {
        # $NYHL Cumulative
        'nyhl_cum': 32941,
        'nyhl_sma200': 27441,
        
        # $BPSPX
        'bpspx_value': 60.00,
        'bpspx_rsi14': 36.47,
        'bpspx_macd_hist': -0.470,
        
        # $OEXA150R
        'oexa150r_value': 63.00,
        'oexa150r_cci14': -163.74,
        
        # SPY:VXX Ratio
        'spy_vxx_ratio': 24.05,
        'spy_vxx_sma50': 25.06,
        
        # VIX
        'vix': 18.45,
        
        # Credit Spreads (mock - replace with real)
        'hy_spread': 385,
    }

# ============================================
# SIMPLE COMPOSITE SIGNAL (Your Multi-Tool)
# ============================================

def calculate_signal_score(data):
    """
    Simple weighted score from YOUR indicators.
    Positive = SPY zone | Negative = IEF zone
    """
    # Normalize each indicator around 0
    breadth = (data['nyhl_cum'] - data['nyhl_sma200']) / 1000  # $NYHL trend
    momentum = (data['bpspx_rsi14'] - 50) / 10  # $BPSPX RSI centered
    participation = (data['oexa150r_value'] - 50) / 20  # $OEXA150R centered
    risk = (25 - data['vix']) / 5  # Lower VIX = positive
    credit = (450 - data['hy_spread']) / 50  # Lower spreads = positive
    
    # Weighted composite (adjust weights to tune)
    score = (
        breadth * 0.40 +      # 40% breadth trend
        momentum * 0.25 +      # 25% momentum
        participation * 0.15 + # 15% participation
        risk * 0.10 +          # 10% risk sentiment
        credit * 0.10          # 10% credit canary
    )
    
    return score

def get_rotation_signal(score):
    """Simple threshold: >0 = SPY, <0 = IEF"""
    if score > 0:
        return "SPY", {'SPY': 100, 'IEF': 0}
    else:
        return "IEF", {'SPY': 0, 'IEF': 100}

# ============================================
# MOCK HISTORICAL DATA (For Chart Demo)
# ============================================

@st.cache_data(ttl=3600)
def generate_simple_chart_data(days=252):
    """Generate 1 year of simple mock data for the chart"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    # Mock SPY price
    ret = np.random.normal(0.0003, 0.012, days)
    prices = 100 * np.cumprod(1 + ret)
    
    # Mock composite signal (correlated with price but lagging slightly)
    signal = np.cumsum(ret * 50 + np.random.normal(0, 2, days))
    
    # Generate signals based on zero crossover
    signals = ['SPY' if s > 0 else 'IEF' for s in signal]
    
    df = pd.DataFrame({
        'Date': dates,
        'SPY': prices,
        'Signal': signals,
        'SignalScore': signal
    }, index=dates)
    
    return df

# ============================================
# TAB 1: CURRENT SIGNAL (Simple)
# ============================================

def render_current_tab():
    # Get your actual values
    data = get_current_indicator_values()
    score = calculate_signal_score(data)
    signal, alloc = get_rotation_signal(score)
    
    # Big clear signal
    if signal == "SPY":
        st.markdown(f"""
        <div class="signal-spy">
            🟢 ROTATE TO SPY (100%)<br>
            <small>Signal Score: {score:+.2f} • Above zero threshold</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="signal-ief">
            🔴 ROTATE TO IEF (100%)<br>
            <small>Signal Score: {score:+.2f} • Below zero threshold</small>
        </div>
        """, unsafe_allow_html=True)
    
    # What's driving the score?
    st.markdown("### 🔍 What's Moving the Signal?")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        val = (data['nyhl_cum'] - data['nyhl_sma200']) / 1000
        st.metric("$NYHL Trend", f"{val:+.1f}", "✅ +" if val > 0 else "❌ -")
    with c2:
        val = (data['bpspx_rsi14'] - 50) / 10
        st.metric("$BPSPX Momentum", f"{val:+.1f}", "✅ +" if val > 0 else "❌ -")
    with c3:
        val = (data['oexa150r_value'] - 50) / 20
        st.metric("$OEXA Participation", f"{val:+.1f}", "✅ +" if val > 0 else "❌ -")
    with c4:
        val = (25 - data['vix']) / 5
        st.metric("VIX Risk", f"{val:+.1f}", "✅ +" if val > 0 else "❌ -")
    with c5:
        val = (450 - data['hy_spread']) / 50
        st.metric("Credit Spreads", f"{val:+.1f}", "✅ +" if val > 0 else "❌ -")
    
    # Action
    st.markdown("### ⚡ Execute")
    if signal == "SPY":
        st.button("✅ Buy SPY / Sell IEF", type="primary", use_container_width=True)
    else:
        st.button("✅ Buy IEF / Sell SPY", type="primary", use_container_width=True)

# ============================================
# TAB 2: SIMPLE CHART (What You Asked For)
# ============================================

def render_chart_tab():
    st.subheader("📈 SPY Price with Signal Markers")
    st.markdown("*Green ▲ = Rotate to SPY (100%) • Red ▼ = Rotate to IEF (100%)*")
    
    if st.button("📊 Load Chart", type="primary"):
        with st.spinner("Generating..."):
            df = generate_simple_chart_data(252)
            
            # Find signal change points for markers
            df['SignalChanged'] = df['Signal'] != df['Signal'].shift(1)
            markers = df[df['SignalChanged']].copy()
            
            # Create chart
            fig = go.Figure()
            
            # SPY price line
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['SPY'],
                name='SPY Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add BUY markers (green triangles) where signal changed to SPY
            buy_markers = markers[markers['Signal'] == 'SPY']
            if not buy_markers.empty:
                fig.add_trace(go.Scatter(
                    x=buy_markers['Date'],
                    y=buy_markers['SPY'],
                    mode='markers',
                    name='→ SPY (Buy)',
                    marker=dict(color='#10b981', size=12, symbol='triangle-up')
                ))
            
            # Add SELL markers (red X) where signal changed to IEF
            sell_markers = markers[markers['Signal'] == 'IEF']
            if not sell_markers.empty:
                fig.add_trace(go.Scatter(
                    x=sell_markers['Date'],
                    y=sell_markers['SPY'],
                    mode='markers',
                    name='→ IEF (Sell)',
                    marker=dict(color='#dc2626', size=12, symbol='x')
                ))
            
            # Color background by current signal zone (simple method)
            prev_signal = None
            for i in range(len(df)-1):
                curr_signal = df['Signal'].iloc[i]
                if curr_signal != prev_signal:
                    start_idx = i
                    end_idx = i + 1
                    while end_idx < len(df) and df['Signal'].iloc[end_idx] == curr_signal:
                        end_idx += 1
                    
                    color = 'rgba(16, 185, 129, 0.08)' if curr_signal == 'SPY' else 'rgba(220, 38, 38, 0.08)'
                    fig.add_vrect(
                        x0=df['Date'].iloc[start_idx],
                        x1=df['Date'].iloc[min(end_idx, len(df)-1)],
                        fillcolor=color,
                        opacity=0.5,
                        layer="below",
                        line_width=0
                    )
                    prev_signal = curr_signal
            
            fig.update_layout(
                title='SPY Price with Rotation Signals<br><sup>🟢 Green zone = Hold SPY | 🔴 Red zone = Hold IEF</sup>',
                xaxis_title='Date',
                yaxis_title='SPY Price ($)',
                height=500,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Simple performance note
            st.info("""
            💡 **How to Read This Chart**:
            - When background turns 🟢 GREEN → Rotate to 100% SPY
            - When background turns 🔴 RED → Rotate to 100% IEF
            - Green ▲ markers = Entry points for SPY
            - Red ▼ markers = Entry points for IEF
            - Edge comes from avoiding major drawdowns, not catching every up move
            """)
    else:
        st.info("👆 Click button to load chart")

# ============================================
# MAIN
# ============================================

def main():
    st.title("🔄 SPY/IEF Signal Dashboard")
    st.markdown("*Simple composite signal from your indicators • Green = SPY, Red = IEF*")
    
    tab1, tab2 = st.tabs(["🎯 Current Signal", "📊 Signal Chart"])
    
    with tab1:
        render_current_tab()
    with tab2:
        render_chart_tab()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("**Signal Logic:**")
        st.markdown("""
        Composite Score = 
        - 40% $NYHL trend
        - 25% $BPSPX momentum  
        - 15% $OEXA participation
        - 10% VIX risk
        - 10% Credit spreads
        
        **Threshold:** Score > 0 → SPY (100%) | Score < 0 → IEF (100%)
        """)
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("⚠️ Not investment advice")

if __name__ == "__main__":
    main()
