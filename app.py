import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="SPY/IEF Signal Dashboard (Real Data)", layout="wide", page_icon="🔄")

st.markdown("""
<style>
    .signal-spy {background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .signal-ief {background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 0.75rem; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.3rem;}
    .smallnote {opacity: 0.9; font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

# ============================================
# HELPERS
# ============================================

def zscore(s: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score with safe handling."""
    mu = s.rolling(window, min_periods=max(20, window//5)).mean()
    sd = s.rolling(window, min_periods=max(20, window//5)).std()
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())

@st.cache_data(ttl=3600)
def fetch_adj_close(tickers, start, end) -> pd.DataFrame:
    px = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    # yfinance returns different shapes depending on count of tickers
    if isinstance(tickers, str) or len(tickers) == 1:
        adj = px["Close"].to_frame(tickers if isinstance(tickers, str) else tickers[0])
    else:
        # with auto_adjust=True, "Close" is adjusted close
        adj = px["Close"].copy()
    adj = adj.dropna(how="all")
    adj.index = pd.to_datetime(adj.index)
    return adj

def build_composite_and_signal(
    closes: pd.DataFrame,
    weights: dict,
    zwin: int,
    fast: int,
    slow: int,
    smooth_sma: int,
    use_ppo: bool,
    ppo_eps: float,
) -> pd.DataFrame:
    """
    Build composite from real series:
    - Breadth proxy: SPY vs 200D MA (percent above)
    - Momentum: SPY 20D return
    - Participation proxy: RSP/SPY ratio (equal-weight participation)
    - Risk: inverse VIX (lower VIX => higher risk-on)
    - Credit: HYG/LQD ratio (junk vs IG)
    """
    df = closes.copy()

    required = ["SPY", "IEF", "^VIX", "RSP", "HYG", "LQD"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required tickers in data: {missing}")

    spy = df["SPY"]
    vix = df["^VIX"]
    rsp = df["RSP"]
    hyg = df["HYG"]
    lqd = df["LQD"]

    # Components
    spy_ma200 = spy.rolling(200, min_periods=200).mean()
    breadth = (spy / spy_ma200 - 1.0) * 100.0                         # % above 200D

    momentum = spy.pct_change(20) * 100.0                             # 20D return (%)

    participation = (rsp / spy).pct_change(20) * 100.0                # 20D change in RSP/SPY (%)

    risk = -(vix)                                                     # inverse VIX (lower VIX is "higher risk-on")

    credit = (hyg / lqd)                                              # junk vs IG ratio

    # Normalize components to comparable scale
    breadth_z = zscore(breadth, window=zwin)
    momentum_z = zscore(momentum, window=zwin)
    participation_z = zscore(participation, window=zwin)
    risk_z = zscore(risk, window=zwin)
    credit_z = zscore(credit.pct_change(20), window=zwin)             # focus on trend in credit ratio

    comp = (
        weights["breadth"] * breadth_z +
        weights["momentum"] * momentum_z +
        weights["participation"] * participation_z +
        weights["risk"] * risk_z +
        weights["credit"] * credit_z
    )

    comp = comp.dropna()

    # Smooth regime line (PPO-like or MACD-like)
    fast_ema = ema(comp, fast)
    slow_ema = ema(comp, slow)

    if use_ppo:
        denom = slow_ema.abs().clip(lower=ppo_eps)
        regime = (fast_ema - slow_ema) / denom * 100.0
    else:
        regime = (fast_ema - slow_ema)  # MACD-style difference

    regime = regime.rolling(smooth_sma, min_periods=1).mean()

    out = pd.DataFrame(index=comp.index)
    out["SPY"] = df.loc[out.index, "SPY"]
    out["IEF"] = df.loc[out.index, "IEF"]
    out["RegimeLine"] = regime
    out["Composite"] = comp

    # Signal today (t): if >0 -> SPY else IEF
    out["Signal"] = np.where(out["RegimeLine"] > 0, "SPY", "IEF")

    # Lookahead fix: trade on next bar (t+1)
    out["Signal_lag"] = out["Signal"].shift(1)
    out = out.dropna(subset=["Signal_lag"])

    # Returns
    out["spy_ret"] = out["SPY"].pct_change().fillna(0.0)
    out["ief_ret"] = out["IEF"].pct_change().fillna(0.0)

    out["strat_ret"] = np.where(out["Signal_lag"] == "SPY", out["spy_ret"], out["ief_ret"])
    out["buyhold_ret"] = out["spy_ret"]

    out["strat_cum"] = (1.0 + out["strat_ret"]).cumprod() * 100.0
    out["buyhold_cum"] = (1.0 + out["buyhold_ret"]).cumprod() * 100.0

    # Crossovers on the actual regime line (today)
    out["CrossedAbove"] = (out["RegimeLine"] > 0) & (out["RegimeLine"].shift(1) <= 0)
    out["CrossedBelow"] = (out["RegimeLine"] < 0) & (out["RegimeLine"].shift(1) >= 0)

    return out

# ============================================
# UI
# ============================================

def render_current_tab(result: pd.DataFrame):
    last = result.iloc[-1]
    signal_now = last["Signal"]          # today's computed signal
    alloc = {"SPY": 100, "IEF": 0} if signal_now == "SPY" else {"SPY": 0, "IEF": 100}
    score = float(last["RegimeLine"])

    if signal_now == "SPY":
        st.markdown(f"""
        <div class="signal-spy">
            🟢 ROTATE TO SPY (100%)<br>
            <span class="smallnote">Regime line: {score:+.2f} • Above zero</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### ⚡ Execute")
        st.button("✅ Buy SPY / Sell IEF", type="primary", use_container_width=True)
    else:
        st.markdown(f"""
        <div class="signal-ief">
            🔴 ROTATE TO IEF (100%)<br>
            <span class="smallnote">Regime line: {score:+.2f} • Below zero</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### ⚡ Execute")
        st.button("✅ Buy IEF / Sell SPY", type="primary", use_container_width=True)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Today Signal", signal_now)
    with c2:
        st.metric("Next-Day Position (no lookahead)", last["Signal_lag"])
    with c3:
        st.metric("Composite (z-weighted)", f"{float(last['Composite']):+.2f}")
    with c4:
        st.metric("Last Close Date", str(result.index[-1].date()))

def render_chart_tab(result: pd.DataFrame):
    st.subheader("📈 SPY Price + Smooth Regime Line (Real Data)")
    st.markdown("*Above zero = SPY next day • Below zero = IEF next day • Signal uses lag to avoid lookahead*")

    fig = go.Figure()

    # SPY price
    fig.add_trace(go.Scatter(
        x=result.index, y=result["SPY"],
        name="SPY (Adj Close)",
        line=dict(width=2),
        yaxis="y1"
    ))

    # Regime line
    fig.add_trace(go.Scatter(
        x=result.index, y=result["RegimeLine"],
        name="Regime Line",
        line=dict(width=2.5),
        yaxis="y2"
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray",
                  annotation_text="Zero Threshold", annotation_position="top right")

    # Shade background by *lagged* signal (what you actually hold)
    hold = result["Signal_lag"]
    changes = (hold != hold.shift(1)).fillna(True)
    blocks = result.loc[changes, ["RegimeLine"]].copy()
    block_starts = list(blocks.index)
    block_starts.append(result.index[-1])

    for i in range(len(block_starts) - 1):
        start = block_starts[i]
        end = block_starts[i + 1]
        sig = hold.loc[start]
        color = "rgba(16, 185, 129, 0.10)" if sig == "SPY" else "rgba(220, 38, 38, 0.10)"
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.5, layer="below", line_width=0)

    # Cross markers (regime cross today)
    buys = result[result["CrossedAbove"]]
    sells = result[result["CrossedBelow"]]

    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["RegimeLine"],
            mode="markers", name="Cross Above → SPY",
            marker=dict(size=10, symbol="triangle-up")
        ))
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["RegimeLine"],
            mode="markers", name="Cross Below → IEF",
            marker=dict(size=10, symbol="triangle-down")
        ))

    fig.update_layout(
        title="SPY with Smooth Regime Line (Real Data)",
        xaxis_title="Date (Daily)",
        yaxis=dict(title="SPY Price", side="left", showgrid=True),
        yaxis2=dict(title="Regime Line", side="right", overlaying="y", showgrid=False),
        height=650,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Performance metrics
    strat_total = float(result["strat_cum"].iloc[-1] - 100.0)
    bh_total = float(result["buyhold_cum"].iloc[-1] - 100.0)
    mdd_strat = max_drawdown(result["strat_cum"])
    mdd_bh = max_drawdown(result["buyhold_cum"])

    rotations = int(result["CrossedAbove"].sum() + result["CrossedBelow"].sum())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Strategy Return", f"{strat_total:+.1f}%", delta=f"{(strat_total - bh_total):+.1f}% vs Buy/Hold")
    with col2:
        st.metric("Max Drawdown", f"{mdd_strat:.1%}", delta=f"BH {mdd_bh:.1%}")
    with col3:
        st.metric("Rotations", f"{rotations}", delta=f"~{rotations/12:.1f}/month (approx)")

    with st.expander("Show recent signals"):
        tail = result[["SPY", "IEF", "RegimeLine", "Signal", "Signal_lag", "strat_cum", "buyhold_cum"]].tail(20)
        st.dataframe(tail, use_container_width=True)

# ============================================
# MAIN
# ============================================

def main():
    st.title("🔄 SPY/IEF Smooth Signal Dashboard (Real Data)")
    st.markdown("*Real daily data • Composite proxies • Smooth regime line • No lookahead (signal applies next day)*")

    with st.sidebar:
        st.header("⚙️ Settings")

        years = st.slider("History (years)", 1, 20, 5)
        end = datetime.now().date() + timedelta(days=1)  # include latest close if available
        start = (datetime.now() - timedelta(days=int(365.25 * years))).date()

        st.subheader("Composite Weights")
        w_breadth = st.slider("Breadth (SPY vs 200MA)", 0.0, 1.0, 0.35, 0.05)
        w_mom = st.slider("Momentum (SPY 20D return)", 0.0, 1.0, 0.25, 0.05)
        w_part = st.slider("Participation (RSP/SPY)", 0.0, 1.0, 0.20, 0.05)
        w_risk = st.slider("Risk (inverse VIX)", 0.0, 1.0, 0.10, 0.05)
        w_credit = st.slider("Credit (HYG/LQD)", 0.0, 1.0, 0.10, 0.05)

        w_sum = w_breadth + w_mom + w_part + w_risk + w_credit
        if abs(w_sum - 1.0) > 1e-6:
            st.warning(f"Weights sum to {w_sum:.2f}. They will be normalized automatically.")

        st.subheader("Smoothing")
        fast = st.slider("Fast EMA", 2, 30, 5)
        slow = st.slider("Slow EMA", 5, 60, 13)
        smooth_sma = st.slider("Extra SMA smoothing", 1, 10, 3)

        use_ppo = st.toggle("Use PPO-style normalization", value=True)
        zwin = st.slider("Z-score window (days)", 60, 504, 252, 21)

        if st.button("🔄 Refresh data / clear cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption("⚠️ Not investment advice.")

    # Normalize weights if needed
    weights = {
        "breadth": w_breadth,
        "momentum": w_mom,
        "participation": w_part,
        "risk": w_risk,
        "credit": w_credit,
    }
    ws = sum(weights.values())
    if ws <= 0:
        st.error("Weights sum to 0. Increase at least one weight.")
        return
    weights = {k: v / ws for k, v in weights.items()}

    tickers = ["SPY", "IEF", "^VIX", "RSP", "HYG", "LQD"]

    try:
        closes = fetch_adj_close(tickers, start=str(start), end=str(end))
        result = build_composite_and_signal(
            closes=closes,
            weights=weights,
            zwin=zwin,
            fast=fast,
            slow=slow,
            smooth_sma=smooth_sma,
            use_ppo=use_ppo,
            ppo_eps=1e-6,
        )
    except Exception as e:
        st.error(f"Error building signal: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["🎯 Current Signal", "📊 Chart + Backtest"])
    with tab1:
        render_current_tab(result)
    with tab2:
        render_chart_tab(result)

if __name__ == "__main__":
    main()
