import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ============================================
# CONFIG / STYLE
# ============================================
st.set_page_config(page_title="SPY/IEF Signal Dashboard", layout="wide", page_icon="🔄")

st.markdown(
    """
<style>
    .signal-spy {background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 0.85rem; border-radius: 10px; font-weight: 800; text-align: center; font-size: 1.35rem;}
    .signal-ief {background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 0.85rem; border-radius: 10px; font-weight: 800; text-align: center; font-size: 1.35rem;}
    .subtle {opacity: 0.9; font-size: 0.95rem; font-weight: 500;}
    .pill {display:inline-block; padding:0.15rem 0.55rem; border-radius:999px; font-weight:700; font-size:0.85rem; margin-left:0.35rem;}
    .pill-green {background:#d1fae5; color:#065f46;}
    .pill-red {background:#fee2e2; color:#7f1d1d;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================
# DATA
# ============================================

@st.cache_data(ttl=3600)
def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily adjusted closes from yfinance.
    auto_adjust=True -> Close is adjusted (dividends/splits)
    """
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

    if isinstance(tickers, str) or len(tickers) == 1:
        close = px["Close"].to_frame(tickers if isinstance(tickers, str) else tickers[0])
    else:
        close = px["Close"].copy()

    close = close.dropna(how="all")
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    return close


# ============================================
# SIGNAL BUILDERS
# ============================================

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def build_raw_signal(
    closes: pd.DataFrame,
    mode: str,
    fast: int,
    slow: int,
    smooth_span: int,
    zwin: int,
) -> pd.Series:
    """
    Build a RAW signal (can be anything). We’ll later transform it into an oscillator LOOK.

    mode:
      - "PPO_on_SPY": classic PPO on SPY close (still transformed to your preferred look)
      - "Composite_Proxies": real-data composite from proxies (RSP, VIX, HYG/LQD, SPY trend/mom)
    """
    spy = closes["SPY"].copy()

    if mode == "PPO_on_SPY":
        fe = ema(spy, fast)
        se = ema(spy, slow)
        raw = (fe - se) / se * 100.0

    elif mode == "Composite_Proxies":
        # Requires these tickers to exist
        required = ["RSP", "^VIX", "HYG", "LQD"]
        missing = [t for t in required if t not in closes.columns]
        if missing:
            raise ValueError(f"Missing tickers for composite mode: {missing}")

        rsp = closes["RSP"]
        vix = closes["^VIX"]
        hyg = closes["HYG"]
        lqd = closes["LQD"]

        # Components (all real, daily)
        # Breadth/trend proxy: SPY vs 200MA (% above)
        ma200 = spy.rolling(200, min_periods=200).mean()
        breadth = (spy / ma200 - 1.0) * 100.0

        # Momentum: 20D return
        momentum = spy.pct_change(20) * 100.0

        # Participation proxy: equal-weight vs cap-weight
        participation = (rsp / spy).pct_change(20) * 100.0

        # Risk: inverse VIX
        risk = -vix

        # Credit: junk vs IG trend
        credit = (hyg / lqd).pct_change(20) * 100.0

        def zscore(x: pd.Series, window: int) -> pd.Series:
            mu = x.rolling(window, min_periods=max(40, window // 5)).mean()
            sd = x.rolling(window, min_periods=max(40, window // 5)).std()
            z = (x - mu) / sd
            return z.replace([np.inf, -np.inf], np.nan)

        comp = (
            0.35 * zscore(breadth, zwin) +
            0.25 * zscore(momentum, zwin) +
            0.20 * zscore(participation, zwin) +
            0.10 * zscore(risk, zwin) +
            0.10 * zscore(credit, zwin)
        )
        raw = comp

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Smooth the raw signal a bit (this is NOT the “look”; it’s just noise reduction)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.ewm(span=smooth_span, adjust=False).mean()
    return raw


def make_oscillator_look(raw: pd.Series, look_window: int, tanh_scale: float, tanh_sensitivity: float) -> pd.Series:
    """
    This is the key: regardless of what 'raw' is (PPO/TSI/TRIX/composite),
    transform it into a clean, bounded, StockCharts-looking oscillator around zero.

    Steps:
      1) normalize by rolling std (keeps amplitude stable over time)
      2) squash with tanh (prevents blowups; keeps oscillator in a consistent band)
      3) scale to a nice visual range (like +/- 3)
    """
    std = raw.rolling(look_window, min_periods=max(30, look_window // 3)).std()
    z = raw / std.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    osc = tanh_scale * np.tanh(z / tanh_sensitivity)
    osc = pd.Series(osc, index=raw.index)
    return osc


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def build_strategy_df(
    closes: pd.DataFrame,
    osc: pd.Series,
) -> pd.DataFrame:
    """
    Signals:
      osc > 0 => SPY
      osc < 0 => IEF
    Execution: next day (lag 1)
    """
    df = pd.DataFrame(index=closes.index)
    df["SPY"] = closes["SPY"]
    df["IEF"] = closes["IEF"]
    df["Osc"] = osc

    df = df.dropna(subset=["SPY", "IEF", "Osc"])

    df["Signal"] = np.where(df["Osc"] > 0, "SPY", "IEF")
    df["Signal_lag"] = df["Signal"].shift(1)  # trade next day (no lookahead)
    df = df.dropna(subset=["Signal_lag"])

    df["spy_ret"] = df["SPY"].pct_change().fillna(0.0)
    df["ief_ret"] = df["IEF"].pct_change().fillna(0.0)

    df["strat_ret"] = np.where(df["Signal_lag"] == "SPY", df["spy_ret"], df["ief_ret"])
    df["buyhold_ret"] = df["spy_ret"]

    df["strat_cum"] = (1.0 + df["strat_ret"]).cumprod() * 100.0
    df["buyhold_cum"] = (1.0 + df["buyhold_ret"]).cumprod() * 100.0

    df["CrossedAbove"] = (df["Osc"] > 0) & (df["Osc"].shift(1) <= 0)
    df["CrossedBelow"] = (df["Osc"] < 0) & (df["Osc"].shift(1) >= 0)

    return df


# ============================================
# PLOTTING
# ============================================

def plot_stockcharts_style(df: pd.DataFrame, show_line: bool = True) -> go.Figure:
    """
    StockCharts-like look:
      - Top pane: SPY price
      - Bottom pane: oscillator bars around zero
      - Background shading: held position (Signal_lag)
      - Triangles: zero-cross events
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.06
    )

    # Price
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["SPY"],
            name="SPY (Adj Close)",
            line=dict(width=2)
        ),
        row=1, col=1
    )

    # Oscillator as bars (this is the “PPO/TSI/TRIX panel look”)
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Osc"],
            name="Signal (osc)",
            opacity=0.9
        ),
        row=2, col=1
    )

    if show_line:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Osc"],
                name="Signal line",
                line=dict(width=2)
            ),
            row=2, col=1
        )

    # Zero line (bold-ish)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Background shading by HELD signal (lagged)
    hold = df["Signal_lag"]
    changes = (hold != hold.shift(1)).fillna(True)
    starts = list(df.index[changes])
    starts.append(df.index[-1])

    for i in range(len(starts) - 1):
        start = starts[i]
        end = starts[i + 1]
        sig = hold.loc[start]
        color = "rgba(16,185,129,0.12)" if sig == "SPY" else "rgba(220,38,38,0.12)"
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.5, layer="below", line_width=0)

    # Cross markers
    buys = df[df["CrossedAbove"]]
    sells = df[df["CrossedBelow"]]

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index, y=buys["Osc"],
                mode="markers",
                name="Cross Above → SPY",
                marker=dict(size=10, symbol="triangle-up")
            ),
            row=2, col=1
        )

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index, y=sells["Osc"],
                mode="markers",
                name="Cross Below → IEF",
                marker=dict(size=10, symbol="triangle-down")
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=740,
        hovermode="x unified",
        bargap=0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="SPY Price", row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Signal (0-line)", row=2, col=1, showgrid=False)

    return fig


# ============================================
# APP
# ============================================

def main():
    st.title("🔄 SPY/IEF Signal Dashboard (Real Data, Clear Oscillator Look)")
    st.markdown(
        "**Rule:** Signal above 0 → **SPY** • below 0 → **IEF** (executed **next day**, no lookahead).  \n"
        "You can build the underlying signal however you want — this app makes it **look like a clean oscillator panel**."
    )

    with st.sidebar:
        st.header("⚙️ Settings")

        years = st.slider("History (years)", 1, 20, 5)

        mode = st.selectbox(
            "Raw signal source",
            ["Composite_Proxies", "PPO_on_SPY"],
            index=0,
            help="You said you don’t care if it’s PPO/TSI/TRIX—pick any raw source; the oscillator look is handled separately."
        )

        st.subheader("Raw signal smoothing")
        fast = st.slider("Fast (if PPO mode)", 2, 30, 5)
        slow = st.slider("Slow (if PPO mode)", 5, 80, 13)
        smooth_span = st.slider("Extra smoothing (EMA span)", 1, 30, 5)

        st.subheader("Oscillator LOOK controls (this is the magic)")
        look_window = st.slider("Look window (rolling std days)", 30, 252, 126, 7)
        tanh_scale = st.slider("Visual range (+/-)", 1.0, 8.0, 3.0, 0.5)
        tanh_sensitivity = st.slider("Sensitivity (lower = more swing)", 0.5, 3.0, 1.5, 0.1)
        show_line = st.toggle("Overlay signal line on bars", value=True)

        st.subheader("Composite settings (if composite mode)")
        zwin = st.slider("Z-score window (days)", 60, 504, 252, 21)

        if st.button("🔄 Refresh / clear cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption("⚠️ Not investment advice.")

    # Dates
    end = datetime.now().date() + timedelta(days=1)
    start = (datetime.now() - timedelta(days=int(365.25 * years))).date()

    # Tickers needed
    tickers = ["SPY", "IEF"]
    if mode == "Composite_Proxies":
        tickers += ["RSP", "^VIX", "HYG", "LQD"]

    # Fetch + build
    try:
        closes = fetch_prices(tickers, start=str(start), end=str(end))
        # Ensure required columns exist (sometimes yfinance returns missing ^VIX on some days)
        closes = closes.dropna(subset=["SPY", "IEF"], how="any")

        raw = build_raw_signal(
            closes=closes,
            mode=mode,
            fast=fast,
            slow=slow,
            smooth_span=smooth_span,
            zwin=zwin,
        )

        osc = make_oscillator_look(
            raw=raw,
            look_window=look_window,
            tanh_scale=tanh_scale,
            tanh_sensitivity=tanh_sensitivity,
        )

        df = build_strategy_df(closes=closes, osc=osc)

        if df.empty or len(df) < 50:
            st.error("Not enough data after indicator warmup. Increase history years.")
            return

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # Tabs
    tab1, tab2 = st.tabs(["🎯 Current Signal", "📊 Clear Oscillator Chart + Backtest"])

    with tab1:
        last = df.iloc[-1]
        signal_now = last["Signal"]  # today
        held_now = last["Signal_lag"]  # what you actually hold today (no lookahead)
        score = float(last["Osc"])

        if signal_now == "SPY":
            st.markdown(
                f"""
                <div class="signal-spy">
                    🟢 SIGNAL ABOVE ZERO → SPY (100%)
                    <div class="subtle">Osc: {score:+.2f} • Today signal = {signal_now} • Held (lagged) = {held_now}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="signal-ief">
                    🔴 SIGNAL BELOW ZERO → IEF (100%)
                    <div class="subtle">Osc: {score:+.2f} • Today signal = {signal_now} • Held (lagged) = {held_now}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Last SPY Close", f"{last['SPY']:.2f}")
        with c2:
            st.metric("Last IEF Close", f"{last['IEF']:.2f}")
        with c3:
            st.metric("Today Signal", signal_now)
        with c4:
            st.metric("Held Today (no lookahead)", held_now)

        st.markdown("---")
        st.markdown(
            "### Quick Read\n"
            "- **Green shading** on the chart = you were **in SPY** (based on lagged signal)\n"
            "- **Red shading** on the chart = you were **in IEF**\n"
            "- ▲ / ▼ markers show **zero-cross flips**"
        )

    with tab2:
        st.subheader("📊 Price + Oscillator Panel (Daily, Clear, Visual)")
        st.caption("This is the StockCharts-style *look* you asked for: oscillator pane, bold zero line, shaded regimes.")

        fig = plot_stockcharts_style(df, show_line=show_line)
        fig.update_layout(
            title=f"SPY Price + Clean Oscillator Signal ({mode})<br>"
                  f"<sup>Above 0 = SPY • Below 0 = IEF • Trades next day (Signal_lag)</sup>"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Perf
        strat_total = float(df["strat_cum"].iloc[-1] - 100.0)
        bh_total = float(df["buyhold_cum"].iloc[-1] - 100.0)
        mdd_strat = max_drawdown(df["strat_cum"])
        mdd_bh = max_drawdown(df["buyhold_cum"])
        rotations = int(df["CrossedAbove"].sum() + df["CrossedBelow"].sum())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Strategy Return", f"{strat_total:+.1f}%", delta=f"{(strat_total - bh_total):+.1f}% vs Buy/Hold")
        with col2:
            st.metric("Buy/Hold Return", f"{bh_total:+.1f}%")
        with col3:
            st.metric("Max Drawdown", f"{mdd_strat:.1%}", delta=f"BH {mdd_bh:.1%}")
        with col4:
            st.metric("Rotations", f"{rotations}")

        with st.expander("Show recent rows"):
            show = df[["SPY", "IEF", "Osc", "Signal", "Signal_lag", "strat_cum", "buyhold_cum"]].tail(25)
            st.dataframe(show, use_container_width=True)

if __name__ == "__main__":
    main()
