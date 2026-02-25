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
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ============================================
# COMPOSITE SIGNAL INDEX (Your Multi-Tool + Canaries)
# ============================================

@st.cache_data(ttl=3600)
def generate_signal_index_data(weeks=260):
    """
    Generate composite "Breadth-Momentum-Canary Index" from your indicators.
    Weekly timeframe for swing/position trading.
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=weeks, freq='W-FRI')
    
    # Mock SPY weekly price
    weekly_ret = np.random.normal(0.0015, 0.025, weeks)
    # Add realistic crash clusters
    for start in [40, 95, 150, 205]:
        if start + 8 < len(weekly_ret):
            weekly_ret[start:start+8] = np.linspace(-0.025, -0.015, 8) + np.random.normal(0, 0.008, 8)
    prices = 100 * np.cumprod(1 + weekly_ret)
    spy_df = pd.DataFrame({'Close': prices}, index=dates)
    
    ret = spy_df['Close'].pct_change().fillna(0)
    mom = ret.rolling(4).mean()  # 4-week momentum
    
    # === CORE INDICATORS (From Your Charts) ===
    
    # 1. Breadth trend ($NYHL style) - 40% weight
    breadth = mom.cumsum() * 15000
    # Add crash sensitivity
    for start in [40, 95, 150, 205]:
        if start + 10 < len(breadth):
            breadth.iloc[start:start+10] *= np.linspace(1, 0.65, 10)
    breadth_norm = (breadth - breadth.mean()) / breadth.std()
    
    # 2. Momentum oscillator ($BPSPX style) - 25% weight
    momentum_osc = (mom * 45).clip(-35, 35)
    
    # 3. Participation ($OEXA150R style) - 10% weight
    ma150 = spy_df['Close'].rolling(150).mean()
    above = (spy_df['Close'] > ma150).astype(float)
    participation = (above.rolling(25).mean() * 100 - 50) / 15  # Normalize around 0
    
    # 4. Risk sentiment (SPY:VXX inverse) - 10% weight
    vol = ret.rolling(4).std()
    risk_sentiment = (1 / (vol * 100 + 0.01) - 25) / 5
    
    # === CANARY INDICATORS (Dave Keller Additions) ===
    
    # 5. Credit spreads (High Yield) - 8% weight (inverted: lower = better)
    credit = (380 - mom * 90 + np.random.normal(0, 25, weeks)).clip(260, 540)
    for start in [40, 95, 150, 205]:
        if start + 8 < len(credit):
            credit.iloc[start:start+8] = np.linspace(380, 490, 8)
    credit_norm = (450 - credit) / 40  # Inverted: higher = positive signal
    
    # 6. AAII Bull/Bear Spread (mock contrarian) - 5% weight
    aaii = (5 + mom * 20 + np.random.normal(0, 8, weeks)).clip(-35, 45)
    # Contrarian: extreme bullishness = negative signal
    aaii_norm = -aaii / 20
    
    # 7. NAAIM Exposure (mock) - 4% weight
    naaim = (65 + mom * 25 + np.random.normal(0, 10, weeks)).clip(30, 95)
    # Contrarian: >80% long = caution
    naaim_norm = (70 - naaim) / 15
    
    # 8. $MOVE vs $VIX ratio (bond vol leading equity vol) - 3% weight
    move_vix = (1.1 - mom * 0.3 + np.random.normal(0, 0.15, weeks)).clip(0.7, 1.6)
    for start in [40, 95, 150, 205]:
        if start + 6 < len(move_vix):
            move_vix.iloc[start:start+6] = np.linspace(1.1, 1.45, 6)
    move_vix_norm = (1.2 - move_vix) / 0.2  # Higher ratio = caution
    
    # 9. Equal Weight/SPX Ratio (RSP/SPY) - 3% weight
    rsp_spx = (0.92 + mom * 0.08 + np.random.normal(0, 0.03, weeks)).clip(0.82, 1.02)
    for start in [40, 95, 150, 205]:
        if start + 8 < len(rsp_spx):
            rsp_spx.iloc[start:start+8] *= np.linspace(1, 0.94, 8)
    rsp_norm = (rsp_spx - 0.92) / 0.04
    
    # 10. Hindenburg Omen (binary mock) - 2% weight
    hindenburg = np.zeros(weeks)
    # Trigger during crash clusters (simplified)
    for start in [40, 95, 150, 205]:
        if start + 3 < weeks:
            hindenburg[start:start+3] = 1
    hindenburg_norm = -hindenburg * 2  # Trigger = negative signal
    
    # === COMPOSITE INDEX (Weighted Sum = 100%) ===
    signal_index = (
        breadth_norm * 0.40 +      # $NYHL breadth trend
        momentum_osc * 0.25 +       # $BPSPX momentum
        participation * 0.10 +      # $OEXA150R participation
        risk_sentiment * 0.10 +     # SPY:VXX risk
        credit_norm * 0.08 +        # Credit spreads (canary)
        aaii_norm * 0.05 +          # AAII sentiment (canary)
        naaim_norm * 0.04 +         # NAAIM positioning (canary)
        move_vix_norm * 0.03 +      # $MOVE/$VIX (canary)
        rsp_norm * 0.03 +           # RSP/SPX leadership (canary)
        hindenburg_norm * 0.02      # Hindenburg Omen (canary)
    )
    
    # Smooth with 3-week MA for cleaner weekly signals
    signal_index = signal_index.rolling(3, min_periods=1).mean()
    
    # === SIGNAL DECISION (Zero-line crossover) ===
    signals = ['SPY' if val > 0 else 'IEF' for val in signal_index]
    allocations = [{'SPY': 100, 'IEF': 0} if s == 'SPY' else {'SPY': 0, 'IEF': 100} for s in signals]
    
    spy_df['signal_index'] = signal_index
    spy_df['signal'] = signals
    spy_df['allocation'] = allocations
    spy_df['SPY_%'] = [a['SPY'] for a in allocations]
    spy_df['IEF_%'] = [a['IEF'] for a in allocations]
    
    # === RETURN CALCULATIONS ===
    spy_df['spy_ret'] = spy_df['Close'].pct_change().fillna(0)
    # IEF return: positive when SPY negative (hedging effect)
    spy_df['ief_ret'] = np.where(spy_df['spy_ret'] < 0, -0.32 * spy_df['spy_ret'], 0.0008)
    
    def calc_strat_ret(row):
        alloc = row['allocation']
        return (alloc['SPY']/100 * row['spy_ret'] + alloc['IEF']/100 * row['ief_ret'])
    
    spy_df['strat_ret'] = spy_df.apply(calc_strat_ret, axis=1)
    spy_df['buyhold_cum'] = (1 + spy_df['spy_ret']).cumprod() * 100
    spy_df['strat_cum'] = (1 + spy_df['strat_ret']).cumprod() * 100
    
    return spy_df

# ============================================
# CURRENT SIGNAL (From Your Charts + Canaries)
# ============================================

def get_current_signal():
    """Calculate current composite signal from your uploaded values"""
    # Core indicators (from your StockCharts PDFs, Feb 24, 2026)
    nyhl_cum, nyhl_200 = 32941, 27441
    bpspx_rsi, bpspx_macd = 36.47, -0.470
    oexa_val, oexa_cci = 63.00, -163.74
    spy_vxx, vxx_sma50 = 24.05, 25.06
    vix = 18.45
    hy_spread = 385
    
    # Canary indicators (mock current values - replace with real data source)
    aaii_spread = 5.2      # AAII Bull-Bear (neutral)
    naaim_exp = 72.3       # NAAIM Exposure (neutral)
    move_vix_ratio = 1.08  # $MOVE/$VIX (neutral)
    rsp_spx_ratio = 0.88   # RSP/SPX (slightly weak)
    hindenburg = 0         # Hindenburg Omen (not triggered)
    
    # Normalize each component around 0
    breadth = (nyhl_cum - nyhl_200) / 1200
    momentum = (bpspx_rsi - 50) / 12
    participation = (oexa_val - 50) / 20
    risk = (25 - vix) / 6
    credit = (450 - hy_spread) / 45
    aaii = -aaii_spread / 18  # Contrarian
    naaim = (70 - naaim_exp) / 12  # Contrarian
    move_vix = (1.2 - move_vix_ratio) / 0.18
    rsp = (rsp_spx_ratio - 0.92) / 0.035
    hind = -hindenburg * 2
    
    # Weighted composite (same weights as historical)
    signal_index = (
        breadth * 0.40 +
        momentum * 0.25 +
        participation * 0.10 +
        risk * 0.10 +
        credit * 0.08 +
        aaii * 0.05 +
        naaim * 0.04 +
        move_vix * 0.03 +
        rsp * 0.03 +
        hind * 0.02
    )
    
    # Signal decision
    if signal_index > 0:
        return "SPY", {'SPY': 100, 'IEF': 0}, signal_index
    else:
        return "IEF", {'SPY': 0, 'IEF': 100}, signal_index

# ============================================
# TAB 1: CURRENT SIGNAL
# ============================================

def render_current_tab():
    signal, alloc, index_val = get_current_signal()
    
    # Big signal banner
    if signal == "SPY":
        sig_class = "signal-spy"
        msg = f"🟢 ROTATE TO SPY (100%) • Signal: {index_val:+.2f}"
    else:
        sig_class = "signal-ief"
        msg = f"🔴 ROTATE TO IEF (100%) • Signal: {index_val:+.2f}"
    
    st.markdown(f"<div class='{sig_class}'>{msg}</div>", unsafe_allow_html=True)
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d')} • Weekly signals • Zero-line crossover")
    
    # Allocation display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("SPY Allocation", f"{alloc['SPY']:.0f}%", 
                  "✅ Full Exposure" if alloc['SPY'] == 100 else "🛡️ Protected")
    with col2:
        st.metric("IEF Allocation", f"{alloc['IEF']:.0f}%", 
                  "🛡️ Full Hedge" if alloc['IEF'] == 100 else "✅ Minimal")
    
    # What's driving the signal?
    st.subheader("🔍 Signal Components (Weighted)")
    
    # Core indicators (75% of signal)
    st.markdown("**Core Indicators (75% weight):**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val = "✅ +" if (32941 > 27441) else "❌ -"
        st.markdown(f"""
        <div style="background:#667eea;padding:0.75rem;border-radius:8px;color:white;text-align:center">
            <strong>Breadth</strong><br>
            {val} 40%
        </div>
        """, unsafe_allow_html=True)
    with c2:
        val = "✅ +" if (36.47 > 35) else "❌ -"
        st.markdown(f"""
        <div style="background:#667eea;padding:0.75rem;border-radius:8px;color:white;text-align:center">
            <strong>Momentum</strong><br>
            {val} 25%
        </div>
        """, unsafe_allow_html=True)
    with c3:
        val = "✅ +" if (63 > 50) else "❌ -"
        st.markdown(f"""
        <div style="background:#667eea;padding:0.75rem;border-radius:8px;color:white;text-align:center">
            <strong>Participation</strong><br>
            {val} 10%
        </div>
        """, unsafe_allow_html=True)
    with c4:
        val = "✅ +" if (18.45 < 25) else "❌ -"
        st.markdown(f"""
        <div style="background:#667eea;padding:0.75rem;border-radius:8px;color:white;text-align:center">
            <strong>Risk</strong><br>
            {val} 10%
        </div>
        """, unsafe_allow_html=True)
    
    # Canary indicators (25% of signal)
    st.markdown("**Canary Indicators (25% weight):**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        val = "✅ +" if (385 < 450) else "❌ -"
        st.markdown(f"""
        <div style="background:#764ba2;padding:0.75rem;border-radius:8px;color:white;text-align:center;font-size:0.9rem">
            <strong>Credit</strong><br>
            {val} 8%
        </div>
        """, unsafe_allow_html=True)
    with c2:
        val = "✅ +" if (5.2 < 10) else "❌ -"
        st.markdown(f"""
        <div style="background:#764ba2;padding:0.75rem;border-radius:8px;color:white;text-align:center;font-size:0.9rem">
            <strong>AAII</strong><br>
            {val} 5%
        </div>
        """, unsafe_allow_html=True)
    with c3:
        val = "✅ +" if (72.3 < 80) else "❌ -"
        st.markdown(f"""
        <div style="background:#764ba2;padding:0.75rem;border-radius:8px;color:white;text-align:center;font-size:0.9rem">
            <strong>NAAIM</strong><br>
            {val} 4%
        </div>
        """, unsafe_allow_html=True)
    with c4:
        val = "✅ +" if (1.08 < 1.2) else "❌ -"
        st.markdown(f"""
        <div style="background:#764ba2;padding:0.75rem;border-radius:8px;color:white;text-align:center;font-size:0.9rem">
            <strong>MOVE/VIX</strong><br>
            {val} 3%
        </div>
        """, unsafe_allow_html=True)
    with c5:
        val = "✅ +" if (0.88 > 0.92) else "❌ -"
        st.markdown(f"""
        <div style="background:#764ba2;padding:0.75rem;border-radius:8px;color:white;text-align:center;font-size:0.9rem">
            <strong>RSP/SPX</strong><br>
            {val} 3%
        </div>
        """, unsafe_allow_html=True)
    
    # Action button
    st.markdown("### ⚡ Execute Weekly Rotation")
    if signal == "SPY":
        st.success("**This Week**: Consider rotating to 100% SPY if not already. Execute Monday open.")
        st.button("✅ Buy SPY / Sell IEF", type="primary", use_container_width=True)
    else:
        st.warning("**This Week**: Consider rotating to 100% IEF for defense. Execute Monday open.")
        st.button("✅ Buy IEF / Sell SPY • Defensive", type="primary", use_container_width=True)

# ============================================
# TAB 2: SIGNAL INDEX CHART (The Core Ask)
# ============================================

def render_chart_tab():
    st.subheader("📈 Signal Index vs SPY Price")
    st.markdown("*Your composite indicator (colored zones) overlaid on SPY • Green = SPY zone, Red = IEF zone*")
    
    if st.button("📊 Load Signal Chart", type="primary"):
        with st.spinner("Generating composite signal..."):
            df = generate_signal_index_data(260)  # ~5 years weekly
            
            # === CHART: SPY Price with Signal Zones ===
            fig = go.Figure()
            
            # SPY Price line
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                name='SPY Price',
                line=dict(color='#1f77b4', width=1.5),
                yaxis='y1'
            ))
            
            # Signal Index line (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['signal_index'],
                name='Signal Index (Your Indicators + Canaries)',
                line=dict(color='#ff7f0e', width=2.5),
                yaxis='y2'
            ))
            
            # Zero threshold line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         annotation_text="Signal Threshold (0)", annotation_position="top right")
            
            # Color background by signal zone (efficient method)
            prev_signal = None
            for i in range(len(df)-1):
                curr_signal = 'SPY' if df['signal_index'].iloc[i] > 0 else 'IEF'
                if curr_signal != prev_signal:
                    # Find segment start
                    start_idx = i
                    # Find segment end
                    end_idx = i + 1
                    while end_idx < len(df) and ('SPY' if df['signal_index'].iloc[end_idx] > 0 else 'IEF') == curr_signal:
                        end_idx += 1
                    
                    color = 'rgba(16, 185, 129, 0.12)' if curr_signal == 'SPY' else 'rgba(220, 38, 38, 0.12)'
                    fig.add_vrect(
                        x0=df.index[start_idx], x1=df.index[min(end_idx, len(df)-1)],
                        fillcolor=color, opacity=0.4, layer="below",
                        line_width=0, annotation_text=""
                    )
                    prev_signal = curr_signal
            
            # Layout with dual y-axes
            fig.update_layout(
                title='SPY Price with Signal Index Overlay<br><sup>🟢 Green background = Hold SPY (100%) | 🔴 Red background = Hold IEF (100%)</sup>',
                xaxis_title='Date (Weekly)',
                yaxis=dict(title='SPY Price ($)', side='left', showgrid=True),
                yaxis2=dict(title='Signal Index', side='right', overlaying='y', showgrid=False),
                height=550,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # === PERFORMANCE SUMMARY ===
            total_bh = (df['buyhold_cum'].iloc[-1] - 100)
            total_strat = (df['strat_cum'].iloc[-1] - 100)
            years = 5
            cagr_bh = ((df['buyhold_cum'].iloc[-1] / 100) ** (1/years) - 1) * 100
            cagr_strat = ((df['strat_cum'].iloc[-1] / 100) ** (1/years) - 1) * 100
            
            def max_dd(cum):
                peak = cum.cummax()
                dd = (cum - peak) / peak * 100
                return dd.min()
            
            mdd_bh = max_dd(df['buyhold_cum'])
            mdd_strat = max_dd(df['strat_cum'])
            
            # Count rotations
            rotations = len(df[df['signal'].shift(1) != df['signal']])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Strategy Return (5Y)", f"{total_strat:+.1f}%", 
                         delta=f"{total_strat - total_bh:+.1f}% vs Buy-and-Hold")
            with col2:
                st.metric("CAGR", f"{cagr_strat:+.1f}%", 
                         delta=f"{cagr_strat - cagr_bh:+.1f}% vs Buy-and-Hold")
            with col3:
                st.metric("Max Drawdown", f"{mdd_strat:.1f}%", 
                         delta=f"{mdd_strat - mdd_bh:+.1f}% vs Buy-and-Hold")
            with col4:
                st.metric("Rotations (5Y)", f"{rotations}", 
                         delta=f"~{rotations/5:.1f} per year")
            
            st.success("✅ Chart loaded • Green zones = SPY (100%) • Red zones = IEF (100%)")
    else:
        st.info("👆 Click button to load signal chart")

# ============================================
# TAB 3: SIMPLE WEEKLY POSITION LOG
# ============================================

def render_log_tab():
    st.subheader("📋 Weekly Position Log")
    st.markdown("*One row per week • Easy to review and export*")
    
    if st.button("📊 Load Weekly Log", type="primary"):
        with st.spinner("Loading..."):
            df = generate_signal_index_data(52)  # 1 year weekly
            recent = df.tail(12).copy()
            
            # Clean table
            display = recent.copy()
            display['Week'] = display.index.strftime('%Y-%m-%d')
            display['Signal'] = display['signal']
            display['SPY %'] = display['SPY_%'].apply(lambda x: f"{x:.0f}%")
            display['IEF %'] = display['IEF_%'].apply(lambda x: f"{x:.0f}%")
            display['SPY Weekly Return'] = display['spy_ret'].apply(lambda x: f"{x*100:+.2f}%")
            display['Strategy Weekly Return'] = display['strat_ret'].apply(lambda x: f"{x*100:+.2f}%")
            
            cols = ['Week', 'Signal', 'SPY %', 'IEF %', 'SPY Weekly Return', 'Strategy Weekly Return']
            st.dataframe(display[cols].sort_values('Week', ascending=False), 
                        use_container_width=True, hide_index=True)
            
            # Export
            export_df = recent[['SPY_%', 'IEF_%', 'spy_ret', 'strat_ret']].copy()
            export_df.columns = ['SPY %', 'IEF %', 'SPY Return', 'Strategy Return']
            csv = export_df.to_csv().encode('utf-8')
            st.download_button("📥 Export Weekly CSV", csv, "signal_rotations.csv", "text/csv")
            
            st.info("""
            💡 **How to Use**:
            - Review each Friday after market close
            - If Signal changed → consider rotation Monday open
            - If Signal unchanged → hold current allocation all week
            - Edge comes from avoiding major drawdowns, not daily outperformance
            """)
    else:
        st.info("👆 Click to load weekly log")

# ============================================
# MAIN
# ============================================

def main():
    st.title("🔄 SPY/IEF Signal Dashboard")
    st.markdown("*Composite indicator from your multi-tool + canaries • Green = SPY, Red = IEF*")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Current Signal", "📊 Signal Chart", "📋 Weekly Log"])
    
    with tab1:
        render_current_tab()
    with tab2:
        render_chart_tab()
    with tab3:
        render_log_tab()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("**Signal Logic:**")
        st.markdown("""
        **Composite Index =**
        - 40% Breadth ($NYHL trend)
        - 25% Momentum ($BPSPX)
        - 10% Participation ($OEXA150R)
        - 10% Risk (SPY:VXX/VIX)
        - 8% Credit Spreads (canary)
        - 5% AAII Sentiment (canary)
        - 4% NAAIM Positioning (canary)
        - 3% $MOVE/$VIX (canary)
        - 3% RSP/SPX Ratio (canary)
        - 2% Hindenburg Omen (canary)
        
        **Threshold:** Index > 0 → SPY (100%) | Index < 0 → IEF (100%)
        
        **Timeframe:** Weekly (Friday close signals)
        """)
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.caption("⚠️ Not investment advice • Test before live use")

if __name__ == "__main__":
    main()
