import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Regime Backtester v1.0",
    layout="wide",
    page_icon="📈",
)

st.markdown(
    """
<style>
    .signal-long {background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 0.85rem; border-radius: 10px; font-weight: 800; text-align: center; font-size: 1.15rem;}
    .signal-cash {background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 0.85rem; border-radius: 10px; font-weight: 800; text-align: center; font-size: 1.15rem;}
    .signal-bear {background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 0.85rem; border-radius: 10px; font-weight: 800; text-align: center; font-size: 1.15rem;}
    .subtle {opacity: 0.92; font-size: 0.92rem; font-weight: 500;}
    .warnbox {background:#fff7ed; border:1px solid #fdba74; color:#9a3412; padding:0.8rem 1rem; border-radius:10px; margin-bottom:0.75rem;}
    .notebox {background:#eff6ff; border:1px solid #93c5fd; color:#1d4ed8; padding:0.8rem 1rem; border-radius:10px; margin-bottom:0.75rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)

def rolling_z(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=max(30, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(30, window // 4)).std()
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)

def annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return np.nan
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    if yrs <= 0:
        return np.nan
    total = equity.iloc[-1] / equity.iloc[0]
    if total <= 0:
        return np.nan
    return total ** (1 / yrs) - 1

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def sharpe_ratio(ret: pd.Series, rf_daily: float = 0.0) -> float:
    x = ret.dropna() - rf_daily
    if len(x) < 2 or x.std() == 0:
        return np.nan
    return float(np.sqrt(252) * x.mean() / x.std())

def sortino_ratio(ret: pd.Series, rf_daily: float = 0.0) -> float:
    x = ret.dropna() - rf_daily
    downside = x[x < 0]
    if len(x) < 2 or len(downside) < 2 or downside.std() == 0:
        return np.nan
    return float(np.sqrt(252) * x.mean() / downside.std())

def calmar_ratio(equity: pd.Series) -> float:
    cagr = annualized_return(equity)
    mdd = abs(max_drawdown(equity))
    if pd.isna(cagr) or mdd == 0:
        return np.nan
    return float(cagr / mdd)

def win_rate(ret: pd.Series) -> float:
    x = ret.dropna()
    if len(x) == 0:
        return np.nan
    return float((x > 0).mean())

def avg_win_loss(ret: pd.Series):
    x = ret.dropna()
    wins = x[x > 0]
    losses = x[x < 0]
    aw = float(wins.mean()) if len(wins) else np.nan
    al = float(losses.mean()) if len(losses) else np.nan
    return aw, al

def time_in_market(long_w: pd.Series, bear_w: pd.Series) -> tuple[float, float, float]:
    n = len(long_w)
    if n == 0:
        return np.nan, np.nan, np.nan
    long_pct = float((long_w > 0).mean())
    bear_pct = float((bear_w > 0).mean())
    cash_pct = float(((long_w == 0) & (bear_w == 0)).mean())
    return long_pct, bear_pct, cash_pct

def run_length_condition(cond: pd.Series, n_days: int) -> pd.Series:
    if n_days <= 1:
        return cond.fillna(False)
    return cond.fillna(False).rolling(n_days, min_periods=n_days).sum().eq(n_days)

# ============================================================
# DATA
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    px = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if px.empty:
        return pd.DataFrame()

    close = px["Close"].copy() if isinstance(px.columns, pd.MultiIndex) else px["Close"].to_frame(tickers[0])
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    close = close.dropna(how="all")
    return close

# ============================================================
# INDICATORS / SCORE
# ============================================================
def build_features(closes: pd.DataFrame, trend_ma: int, zwin: int, vol_lookback: int) -> pd.DataFrame:
    df = pd.DataFrame(index=closes.index)
    for c in closes.columns:
        df[c] = closes[c]

    df["spy_ma"] = df["SPY"].rolling(trend_ma, min_periods=trend_ma).mean()
    df["trend_up"] = df["SPY"] > df["spy_ma"]
    df["trend_dn"] = df["SPY"] < df["spy_ma"]

    df["rsp_spy"] = safe_div(df["RSP"], df["SPY"])
    df["hyg_lqd"] = safe_div(df["HYG"], df["LQD"])
    df["spy_vxx"] = safe_div(df["SPY"], df["VXX"])
    df["spxs_svol"] = safe_div(df["SPXS"], df["SVOL"])

    df["breadth_mom"] = df["rsp_spy"].pct_change(20)
    df["credit_mom"] = df["hyg_lqd"].pct_change(20)
    df["internals_mom"] = df["spy_vxx"].pct_change(10)
    df["fear_mom"] = -df["spxs_svol"].pct_change(10)

    # Risk / vol features
    df["spy_ret"] = df["SPY"].pct_change()
    df["spy_vol_20"] = df["spy_ret"].rolling(vol_lookback, min_periods=max(10, vol_lookback // 2)).std() * np.sqrt(252)
    df["vxx_ret"] = df["VXX"].pct_change()
    df["vxx_spike"] = df["VXX"].pct_change(5)

    # Z-score normalize for comparability
    df["breadth_z"] = rolling_z(df["breadth_mom"], zwin)
    df["credit_z"] = rolling_z(df["credit_mom"], zwin)
    df["internals_z"] = rolling_z(df["internals_mom"], zwin)
    df["fear_z"] = rolling_z(df["fear_mom"], zwin)

    return df

def build_weighted_score(
    df: pd.DataFrame,
    w_trend: float,
    w_breadth: float,
    w_credit: float,
    w_internals: float,
    w_fear: float,
    z_cut: float,
) -> pd.DataFrame:
    out = df.copy()

    out["s_trend"] = np.where(out["trend_up"], 1.0, np.where(out["trend_dn"], -1.0, 0.0))
    out["s_breadth"] = np.where(out["breadth_z"] > z_cut, 1.0, np.where(out["breadth_z"] < -z_cut, -1.0, 0.0))
    out["s_credit"] = np.where(out["credit_z"] > z_cut, 1.0, np.where(out["credit_z"] < -z_cut, -1.0, 0.0))
    out["s_internals"] = np.where(out["internals_z"] > z_cut, 1.0, np.where(out["internals_z"] < -z_cut, -1.0, 0.0))
    out["s_fear"] = np.where(out["fear_z"] > z_cut, 1.0, np.where(out["fear_z"] < -z_cut, -1.0, 0.0))

    out["weighted_raw"] = (
        w_trend * out["s_trend"]
        + w_breadth * out["s_breadth"]
        + w_credit * out["s_credit"]
        + w_internals * out["s_internals"]
        + w_fear * out["s_fear"]
    )

    # Convert to intuitive score buckets 0..5
    max_w = max(w_trend + w_breadth + w_credit + w_internals + w_fear, 1e-9)
    norm = out["weighted_raw"] / max_w  # -1 .. +1
    out["score_cont"] = norm
    out["score_0_5"] = pd.cut(
        norm,
        bins=[-np.inf, -0.60, -0.20, 0.20, 0.60, np.inf],
        labels=[0, 1, 2, 4, 5],
    ).astype(float)
    out["score_0_5"] = out["score_0_5"].fillna(2.0)

    return out

# ============================================================
# SIGNAL ENGINE
# ============================================================
def build_regime_states(
    df: pd.DataFrame,
    long_enter_score: float,
    neutral_score: float,
    short_enter_score: float,
    confirm_days: int,
    defensive_mode: str,
) -> pd.DataFrame:
    out = df.copy()

    score = out["score_0_5"]

    long_ready = (score >= long_enter_score) & out["trend_up"]
    short_ready = (score <= short_enter_score) & out["trend_dn"]
    neutral_ready = (score == neutral_score)

    long_confirm = run_length_condition(long_ready, confirm_days)
    short_confirm = run_length_condition(short_ready, confirm_days)
    neutral_confirm = run_length_condition(neutral_ready, max(1, confirm_days // 2))

    state = pd.Series(index=out.index, dtype="object")
    state.iloc[0] = "CASH"

    for i in range(1, len(out)):
        prev = state.iloc[i - 1]

        if long_confirm.iloc[i]:
            state.iloc[i] = "LONG"
        elif short_confirm.iloc[i]:
            if defensive_mode in ["SDS", "SPXS", "Custom"]:
                state.iloc[i] = "BEAR"
            else:
                state.iloc[i] = "CASH"
        elif neutral_confirm.iloc[i]:
            state.iloc[i] = "CASH"
        else:
            state.iloc[i] = prev

    out["Signal"] = state
    out["Held"] = out["Signal"].shift(1)
    return out.dropna(subset=["Held"])

# ============================================================
# BACKTEST
# ============================================================
def backtest_strategy(
    df: pd.DataFrame,
    bull_ticker: str,
    bear_ticker: str,
    cash_ticker: str,
    defensive_mode: str,
    custom_def_ticker: str,
    max_bear_alloc: float,
    vol_target_enabled: bool,
    vol_target_annual: float,
    vol_min_scale: float,
    trade_cost_bps: float,
    compare_nonleveraged: bool,
):
    out = df.copy()

    actual_cash_ticker = custom_def_ticker if defensive_mode == "Custom" else cash_ticker
    actual_bear_ticker = None
    if defensive_mode == "SDS":
        actual_bear_ticker = "SDS"
    elif defensive_mode == "SPXS":
        actual_bear_ticker = "SPXS"
    elif defensive_mode == "Custom":
        actual_bear_ticker = custom_def_ticker if custom_def_ticker in out.columns else None

    out["bull_ret"] = out[bull_ticker].pct_change().fillna(0.0)
    out["cash_ret"] = out[actual_cash_ticker].pct_change().fillna(0.0)
    out["spy_ret"] = out["SPY"].pct_change().fillna(0.0)

    if actual_bear_ticker is not None and actual_bear_ticker in out.columns:
        out["bear_ret"] = out[actual_bear_ticker].pct_change().fillna(0.0)
    else:
        out["bear_ret"] = 0.0

    # Vol targeting multiplier
    if vol_target_enabled:
        realized = out["spy_vol_20"].replace(0, np.nan)
        vt = (vol_target_annual / realized).clip(lower=vol_min_scale, upper=1.0).fillna(1.0)
        # additional cap if VXX spiking
        vxx_penalty = np.where(out["vxx_spike"] > 0.15, 0.50, np.where(out["vxx_spike"] > 0.08, 0.75, 1.0))
        out["risk_scale"] = (vt * vxx_penalty).clip(lower=vol_min_scale, upper=1.0)
    else:
        out["risk_scale"] = 1.0

    out["bull_w"] = 0.0
    out["bear_w"] = 0.0
    out["cash_w"] = 0.0

    for i in range(len(out)):
        held = out["Held"].iloc[i]
        rs = float(out["risk_scale"].iloc[i])

        if held == "LONG":
            out.iat[i, out.columns.get_loc("bull_w")] = rs
            out.iat[i, out.columns.get_loc("cash_w")] = 1.0 - rs
        elif held == "BEAR" and actual_bear_ticker is not None:
            bw = min(max_bear_alloc, rs)
            out.iat[i, out.columns.get_loc("bear_w")] = bw
            out.iat[i, out.columns.get_loc("cash_w")] = 1.0 - bw
        else:
            out.iat[i, out.columns.get_loc("cash_w")] = 1.0

    out["gross_ret"] = (
        out["bull_w"] * out["bull_ret"]
        + out["bear_w"] * out["bear_ret"]
        + out["cash_w"] * out["cash_ret"]
    )

    out["turnover"] = (
        out["Held"].ne(out["Held"].shift(1)).fillna(False).astype(int)
        + out["bull_w"].sub(out["bull_w"].shift(1).fillna(0)).abs()
        + out["bear_w"].sub(out["bear_w"].shift(1).fillna(0)).abs()
        + out["cash_w"].sub(out["cash_w"].shift(1).fillna(0)).abs()
    )

    cost = (trade_cost_bps / 10000.0) * out["turnover"].clip(lower=0.0)
    out["net_ret"] = out["gross_ret"] - cost

    out["strategy_eq"] = (1.0 + out["net_ret"]).cumprod() * 100.0
    out["buyhold_eq"] = (1.0 + out["spy_ret"]).cumprod() * 100.0

    # Optional comparison proxy strategy
    if compare_nonleveraged:
        bear_proxy_ret = 0.0
        if actual_bear_ticker is not None:
            bear_proxy_ret = out["SPY"].pct_change().fillna(0.0) * -1.0
        proxy_ret = (
            out["bull_w"] * out["spy_ret"]
            + out["bear_w"] * bear_proxy_ret
            + out["cash_w"] * out["cash_ret"]
        )
        proxy_cost = (trade_cost_bps / 10000.0) * out["turnover"].clip(lower=0.0)
        out["proxy_ret"] = proxy_ret - proxy_cost
        out["proxy_eq"] = (1.0 + out["proxy_ret"]).cumprod() * 100.0

    return out, actual_cash_ticker, actual_bear_ticker

# ============================================================
# METRICS TABLES
# ============================================================
def summarize_performance(df: pd.DataFrame, eq_col: str, ret_col: str, spy_eq_col: str = "buyhold_eq") -> dict:
    cagr = annualized_return(df[eq_col])
    mdd = max_drawdown(df[eq_col])
    shrp = sharpe_ratio(df[ret_col])
    sortino = sortino_ratio(df[ret_col])
    calmar = calmar_ratio(df[eq_col])
    wr = win_rate(df[ret_col])
    aw, al = avg_win_loss(df[ret_col])
    long_pct, bear_pct, cash_pct = time_in_market(df["bull_w"], df["bear_w"])
    return {
        "CAGR": cagr,
        "Max DD": mdd,
        "Sharpe": shrp,
        "Sortino": sortino,
        "Calmar": calmar,
        "Win Rate": wr,
        "Avg Win": aw,
        "Avg Loss": al,
        "Time Long": long_pct,
        "Time Bear": bear_pct,
        "Time Cash": cash_pct,
        "Final Return": df[eq_col].iloc[-1] / df[eq_col].iloc[0] - 1.0,
        "SPY Final Return": df[spy_eq_col].iloc[-1] / df[spy_eq_col].iloc[0] - 1.0,
    }

def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.1%}"

def fmt_num(x):
    return "—" if pd.isna(x) else f"{x:.2f}"

def build_comparison_table(
    closes: pd.DataFrame,
    base_features: pd.DataFrame,
    signal_params: dict,
    bt_params: dict,
    pairs: list[tuple[str, str]],
):
    rows = []
    for bull, bear in pairs:
        req = ["SPY", bull, "RSP", "SPXS", "SVOL", "HYG", "LQD", "VXX", bt_params["cash_ticker"]]
        if bt_params["defensive_mode"] == "SDS":
            req.append("SDS")
        elif bt_params["defensive_mode"] == "SPXS":
            req.append("SPXS")
        elif bt_params["defensive_mode"] == "Custom":
            req.append(bt_params["custom_def_ticker"])

        if any(r not in closes.columns for r in set(req)):
            continue

        sig_df = build_regime_states(base_features.copy(), **signal_params)
        bt_df, _, _ = backtest_strategy(
            sig_df,
            bull_ticker=bull,
            bear_ticker=bear,
            cash_ticker=bt_params["cash_ticker"],
            defensive_mode=bt_params["defensive_mode"],
            custom_def_ticker=bt_params["custom_def_ticker"],
            max_bear_alloc=bt_params["max_bear_alloc"],
            vol_target_enabled=bt_params["vol_target_enabled"],
            vol_target_annual=bt_params["vol_target_annual"],
            vol_min_scale=bt_params["vol_min_scale"],
            trade_cost_bps=bt_params["trade_cost_bps"],
            compare_nonleveraged=False,
        )
        s = summarize_performance(bt_df, "strategy_eq", "net_ret")
        rows.append(
            {
                "Pair": f"{bull}/{bear}",
                "CAGR": s["CAGR"],
                "Sharpe": s["Sharpe"],
                "Sortino": s["Sortino"],
                "Calmar": s["Calmar"],
                "Max DD": s["Max DD"],
                "Final Return": s["Final Return"],
                "SPY Return": s["SPY Final Return"],
            }
        )
    comp = pd.DataFrame(rows)
    if not comp.empty:
        comp = comp.sort_values("Sharpe", ascending=False)
    return comp

# ============================================================
# MONTE CARLO / WALK FORWARD
# ============================================================
def monte_carlo_start_dates(
    closes: pd.DataFrame,
    feature_builder_kwargs: dict,
    score_builder_kwargs: dict,
    signal_params: dict,
    bt_params: dict,
    bull_ticker: str,
    bear_ticker: str,
    n_runs: int = 100,
    min_years: int = 5,
):
    idx = closes.index
    if len(idx) < 252 * min_years:
        return pd.DataFrame()

    rng = np.random.default_rng(42)
    max_start_pos = max(1, len(idx) - 252 * min_years)
    picks = rng.integers(0, max_start_pos, size=n_runs)

    vals = []
    for p in picks:
        sub = closes.loc[idx[p]:].copy()
        feats = build_features(sub, **feature_builder_kwargs)
        feats = build_weighted_score(feats, **score_builder_kwargs)
        sig = build_regime_states(feats, **signal_params)
        bt, _, _ = backtest_strategy(
            sig,
            bull_ticker=bull_ticker,
            bear_ticker=bear_ticker,
            cash_ticker=bt_params["cash_ticker"],
            defensive_mode=bt_params["defensive_mode"],
            custom_def_ticker=bt_params["custom_def_ticker"],
            max_bear_alloc=bt_params["max_bear_alloc"],
            vol_target_enabled=bt_params["vol_target_enabled"],
            vol_target_annual=bt_params["vol_target_annual"],
            vol_min_scale=bt_params["vol_min_scale"],
            trade_cost_bps=bt_params["trade_cost_bps"],
            compare_nonleveraged=False,
        )
        vals.append(
            {
                "start_date": bt.index[0],
                "cagr": annualized_return(bt["strategy_eq"]),
                "sharpe": sharpe_ratio(bt["net_ret"]),
                "max_dd": max_drawdown(bt["strategy_eq"]),
            }
        )
    return pd.DataFrame(vals)

def walk_forward_analysis(
    closes: pd.DataFrame,
    feature_builder_kwargs: dict,
    score_builder_kwargs: dict,
    signal_params: dict,
    bt_params: dict,
    bull_ticker: str,
    bear_ticker: str,
):
    if len(closes) < 300:
        return None, None

    split = int(len(closes) * 0.70)
    train = closes.iloc[:split].copy()
    test = closes.iloc[split:].copy()

    train_feats = build_features(train, **feature_builder_kwargs)
    train_feats = build_weighted_score(train_feats, **score_builder_kwargs)
    train_sig = build_regime_states(train_feats, **signal_params)
    train_bt, _, _ = backtest_strategy(
        train_sig,
        bull_ticker=bull_ticker,
        bear_ticker=bear_ticker,
        cash_ticker=bt_params["cash_ticker"],
        defensive_mode=bt_params["defensive_mode"],
        custom_def_ticker=bt_params["custom_def_ticker"],
        max_bear_alloc=bt_params["max_bear_alloc"],
        vol_target_enabled=bt_params["vol_target_enabled"],
        vol_target_annual=bt_params["vol_target_annual"],
        vol_min_scale=bt_params["vol_min_scale"],
        trade_cost_bps=bt_params["trade_cost_bps"],
        compare_nonleveraged=False,
    )

    test_feats = build_features(test, **feature_builder_kwargs)
    test_feats = build_weighted_score(test_feats, **score_builder_kwargs)
    test_sig = build_regime_states(test_feats, **signal_params)
    test_bt, _, _ = backtest_strategy(
        test_sig,
        bull_ticker=bull_ticker,
        bear_ticker=bear_ticker,
        cash_ticker=bt_params["cash_ticker"],
        defensive_mode=bt_params["defensive_mode"],
        custom_def_ticker=bt_params["custom_def_ticker"],
        max_bear_alloc=bt_params["max_bear_alloc"],
        vol_target_enabled=bt_params["vol_target_enabled"],
        vol_target_annual=bt_params["vol_target_annual"],
        vol_min_scale=bt_params["vol_min_scale"],
        trade_cost_bps=bt_params["trade_cost_bps"],
        compare_nonleveraged=False,
    )

    return train_bt, test_bt

# ============================================================
# PLOT
# ============================================================
def plot_dashboard(df: pd.DataFrame, bull_ticker: str, bear_label: str) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.20, 0.30],
        vertical_spacing=0.05,
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["SPY"], name="SPY", line=dict(width=2)), row=1, col=1)

    held = df["Held"]
    changes = (held != held.shift(1)).fillna(True)
    starts = list(df.index[changes])
    starts.append(df.index[-1])

    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        current = held.loc[s]
        if current == "LONG":
            col = "rgba(16,185,129,0.12)"
        elif current == "BEAR":
            col = "rgba(220,38,38,0.12)"
        else:
            col = "rgba(245,158,11,0.12)"
        fig.add_vrect(x0=s, x1=e, fillcolor=col, opacity=0.55, layer="below", line_width=0)

    fig.add_trace(go.Scatter(x=df.index, y=df["score_0_5"], name="Score", line=dict(width=2)), row=2, col=1)
    fig.add_hline(y=4, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=2, line_dash="dot", line_color="orange", row=2, col=1)
    fig.add_hline(y=1, line_dash="dash", line_color="red", row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["strategy_eq"], name="Strategy", line=dict(width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["buyhold_eq"], name="SPY Buy/Hold", line=dict(width=2)), row=3, col=1)
    if "proxy_eq" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["proxy_eq"], name="Non-Levered Proxy", line=dict(width=1.75)), row=3, col=1)

    fig.update_layout(
        height=900,
        hovermode="x unified",
        bargap=0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=f"Regime Backtester — {bull_ticker} / {bear_label} / Cash",
    )
    fig.update_yaxes(title_text="SPY Price", row=1, col=1)
    fig.update_yaxes(title_text="Regime Score", row=2, col=1)
    fig.update_yaxes(title_text="Equity (Start=100)", row=3, col=1)
    return fig

# ============================================================
# APP
# ============================================================
def main():
    st.title("📈 Regime Backtester v1.0")

    st.markdown(
        '<div class="warnbox"><strong>Leveraged ETFs decay over time; this strategy is for tactical allocation, not long-term holding.</strong></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notebox"><strong>If strategy only works on specific date ranges, it is likely overfit.</strong></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Backtest Setup")

        backtest_years = st.slider("History (years)", 3, 20, 10)
        pair_name = st.selectbox("ETF Pair", ["SSO/SDS", "SPXL/SPXS", "SPUU/SPDN"], index=0)

        pair_map = {
            "SSO/SDS": ("SSO", "SDS"),
            "SPXL/SPXS": ("SPXL", "SPXS"),
            "SPUU/SPDN": ("SPUU", "SPDN"),
        }
        bull_ticker, pair_bear_ticker = pair_map[pair_name]

        st.subheader("Defensive Asset")
        defensive_mode = st.selectbox("Defensive Asset", ["Cash/SHY", "SDS", "SPXS", "Custom"], index=0)
        cash_ticker = st.selectbox("Cash Proxy", ["SHY", "SGOV", "BIL"], index=0)
        custom_def_ticker = st.text_input("Custom Defensive Ticker", value="SHY").strip().upper()
        compare_nonleveraged = st.toggle("Show non-leveraged proxy comparison", value=True)

        st.subheader("Trend / Hysteresis")
        trend_ma = st.slider("SPY Trend MA", 100, 300, 200, 10)
        confirm_days = st.slider("Confirmation days before switch", 1, 10, 3)
        long_enter_score = st.slider("Go LONG when score ≥", 3.0, 5.0, 4.0, 1.0)
        short_enter_score = st.slider("Go SHORT when score ≤", 0.0, 2.0, 1.0, 1.0)

        st.subheader("Regime Score Weights")
        w_trend = st.slider("Weight: Trend", 0.0, 5.0, 2.0, 0.25)
        w_breadth = st.slider("Weight: Breadth", 0.0, 5.0, 1.50, 0.25)
        w_credit = st.slider("Weight: Credit", 0.0, 5.0, 1.25, 0.25)
        w_internals = st.slider("Weight: Internals", 0.0, 5.0, 1.25, 0.25)
        w_fear = st.slider("Weight: Fear", 0.0, 5.0, 1.00, 0.25)
        z_cut = st.slider("Z-score threshold", 0.0, 1.5, 0.25, 0.05)
        zwin = st.slider("Z-score window", 60, 504, 252, 21)

        st.subheader("Risk Management")
        max_bear_alloc = st.slider("Max allocation to bear", 0.0, 1.0, 0.60, 0.05)
        vol_target_enabled = st.toggle("Enable volatility target", value=True)
        vol_target_annual = st.slider("Target annualized vol", 0.05, 0.30, 0.15, 0.01)
        vol_lookback = st.slider("Vol lookback (days)", 10, 60, 20, 5)
        vol_min_scale = st.slider("Minimum risk scale", 0.10, 1.00, 0.35, 0.05)

        st.subheader("Realism")
        trade_cost_bps = st.slider("Trading cost (bps)", 0.0, 20.0, 2.0, 0.5)

        st.subheader("Validation")
        monte_carlo_enabled = st.toggle("Monte Carlo (100 random start dates)", value=False)
        walk_forward_enabled = st.toggle("Walk-Forward Analysis (70/30 split)", value=False)

        if st.button("🔄 Refresh / clear cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    end = datetime.now().date() + timedelta(days=1)
    start = (datetime.now() - timedelta(days=int(365.25 * backtest_years))).date()

    required = {
        "SPY", "RSP", "SPXS", "SVOL", "HYG", "LQD", "VXX",
        bull_ticker, pair_bear_ticker, cash_ticker
    }

    if defensive_mode == "SDS":
        required.add("SDS")
    elif defensive_mode == "SPXS":
        required.add("SPXS")
    elif defensive_mode == "Custom":
        required.add(custom_def_ticker)

    tickers = sorted(required)

    closes = fetch_adj_close(tickers, start=str(start), end=str(end))
    if closes.empty:
        st.error("No data returned. Try a different ticker or shorter period.")
        st.stop()

    missing = [t for t in tickers if t not in closes.columns]
    if missing:
        st.error(f"Missing required data: {missing}")
        st.stop()

    feature_builder_kwargs = {
        "trend_ma": trend_ma,
        "zwin": zwin,
        "vol_lookback": vol_lookback,
    }
    score_builder_kwargs = {
        "w_trend": w_trend,
        "w_breadth": w_breadth,
        "w_credit": w_credit,
        "w_internals": w_internals,
        "w_fear": w_fear,
        "z_cut": z_cut,
    }
    signal_params = {
        "long_enter_score": long_enter_score,
        "neutral_score": 2.0,
        "short_enter_score": short_enter_score,
        "confirm_days": confirm_days,
        "defensive_mode": defensive_mode,
    }
    bt_params = {
        "cash_ticker": cash_ticker,
        "defensive_mode": defensive_mode,
        "custom_def_ticker": custom_def_ticker,
        "max_bear_alloc": max_bear_alloc,
        "vol_target_enabled": vol_target_enabled,
        "vol_target_annual": vol_target_annual,
        "vol_min_scale": vol_min_scale,
        "trade_cost_bps": trade_cost_bps,
    }

    features = build_features(closes, **feature_builder_kwargs)
    features = build_weighted_score(features, **score_builder_kwargs)
    sig_df = build_regime_states(features, **signal_params)
    bt_df, actual_cash_ticker, actual_bear_ticker = backtest_strategy(
        sig_df,
        bull_ticker=bull_ticker,
        bear_ticker=pair_bear_ticker,
        cash_ticker=cash_ticker,
        defensive_mode=defensive_mode,
        custom_def_ticker=custom_def_ticker,
        max_bear_alloc=max_bear_alloc,
        vol_target_enabled=vol_target_enabled,
        vol_target_annual=vol_target_annual,
        vol_min_scale=vol_min_scale,
        trade_cost_bps=trade_cost_bps,
        compare_nonleveraged=compare_nonleveraged,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Current", "📊 Backtest", "📋 Comparison", "🧪 Validation"])

    with tab1:
        last = bt_df.iloc[-1]
        held_today = last["Held"]
        score_today = float(last["score_0_5"])

        if held_today == "LONG":
            st.markdown(
                f"""<div class="signal-long">🟢 LONG {bull_ticker}
                <div class="subtle">score={score_today:.1f} • trend_up={bool(last['trend_up'])} • risk_scale={float(last['risk_scale']):.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        elif held_today == "BEAR":
            label = actual_bear_ticker if actual_bear_ticker else "CASH"
            st.markdown(
                f"""<div class="signal-bear">🔴 DEFENSIVE / BEAR → {label}
                <div class="subtle">score={score_today:.1f} • trend_dn={bool(last['trend_dn'])} • bear_w={float(last['bear_w']):.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="signal-cash">🟡 CASH / NEUTRAL → {actual_cash_ticker}
                <div class="subtle">score={score_today:.1f} • cash_w={float(last['cash_w']):.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("SPY", f"{last['SPY']:.2f}")
        with c2:
            st.metric("Held today", held_today)
        with c3:
            st.metric("Risk Scale", f"{float(last['risk_scale']):.2f}")
        with c4:
            st.metric("Bull Weight", f"{float(last['bull_w']):.0%}")
        with c5:
            st.metric("Bear Weight", f"{float(last['bear_w']):.0%}")

        st.dataframe(
            bt_df[[
                "score_0_5", "weighted_raw", "trend_up", "trend_dn",
                "breadth_z", "credit_z", "internals_z", "fear_z",
                "Signal", "Held", "bull_w", "bear_w", "cash_w", "risk_scale"
            ]].tail(20),
            use_container_width=True,
        )

    with tab2:
        fig = plot_dashboard(bt_df, bull_ticker=bull_ticker, bear_label=(actual_bear_ticker if actual_bear_ticker else actual_cash_ticker))
        st.plotly_chart(fig, use_container_width=True)

        strat_stats = summarize_performance(bt_df, "strategy_eq", "net_ret")
        spy_stats = {
            "CAGR": annualized_return(bt_df["buyhold_eq"]),
            "Max DD": max_drawdown(bt_df["buyhold_eq"]),
            "Sharpe": sharpe_ratio(bt_df["spy_ret"]),
            "Sortino": sortino_ratio(bt_df["spy_ret"]),
            "Calmar": calmar_ratio(bt_df["buyhold_eq"]),
            "Final Return": bt_df["buyhold_eq"].iloc[-1] / bt_df["buyhold_eq"].iloc[0] - 1.0,
        }

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategy Return", fmt_pct(strat_stats["Final Return"]), delta=f"{(strat_stats['Final Return'] - spy_stats['Final Return']):+.1%} vs SPY")
        m2.metric("Strategy CAGR", fmt_pct(strat_stats["CAGR"]), delta=f"SPY {fmt_pct(spy_stats['CAGR'])}")
        m3.metric("Strategy Sharpe", fmt_num(strat_stats["Sharpe"]), delta=f"SPY {fmt_num(spy_stats['Sharpe'])}")
        m4.metric("Max Drawdown", fmt_pct(strat_stats["Max DD"]), delta=f"SPY {fmt_pct(spy_stats['Max DD'])}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Sortino", fmt_num(strat_stats["Sortino"]))
        m6.metric("Calmar", fmt_num(strat_stats["Calmar"]))
        m7.metric("Win Rate", fmt_pct(strat_stats["Win Rate"]))
        m8.metric("Time in Market", fmt_pct(1.0 - strat_stats["Time Cash"]))

        m9, m10, m11 = st.columns(3)
        m9.metric("Avg Win", fmt_pct(strat_stats["Avg Win"]))
        m10.metric("Avg Loss", fmt_pct(strat_stats["Avg Loss"]))
        m11.metric("Time in Bear", fmt_pct(strat_stats["Time Bear"]))

        with st.expander("Recent rows (last 6 months)"):
            cols = [
                "SPY", bull_ticker, "score_0_5", "weighted_raw", "Signal", "Held",
                "bull_w", "bear_w", "cash_w", "risk_scale", "gross_ret", "net_ret",
                "strategy_eq", "buyhold_eq", "turnover"
            ]
            cols = [c for c in cols if c in bt_df.columns]
            recent_start = bt_df.index.max() - pd.DateOffset(months=6)
            recent_df = bt_df.loc[bt_df.index >= recent_start, cols]
            st.dataframe(recent_df, use_container_width=True)

    with tab3:
        pairs = [("SSO", "SDS"), ("SPXL", "SPXS"), ("SPUU", "SPDN")]
        comp = build_comparison_table(closes, features, signal_params, bt_params, pairs)
        if comp.empty:
            st.info("Comparison table unavailable because some pair data is missing in the selected date range.")
        else:
            disp = comp.copy()
            for c in ["CAGR", "Max DD", "Final Return", "SPY Return"]:
                disp[c] = disp[c].map(fmt_pct)
            for c in ["Sharpe", "Sortino", "Calmar"]:
                disp[c] = disp[c].map(fmt_num)
            st.dataframe(disp, use_container_width=True)

    with tab4:
        if monte_carlo_enabled:
            mc = monte_carlo_start_dates(
                closes=closes,
                feature_builder_kwargs=feature_builder_kwargs,
                score_builder_kwargs=score_builder_kwargs,
                signal_params=signal_params,
                bt_params=bt_params,
                bull_ticker=bull_ticker,
                bear_ticker=pair_bear_ticker,
                n_runs=100,
                min_years=min(5, backtest_years),
            )
            if not mc.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("Monte Carlo Median CAGR", fmt_pct(mc["cagr"].median()))
                c2.metric("Monte Carlo Median Sharpe", fmt_num(mc["sharpe"].median()))
                c3.metric("Monte Carlo Median Max DD", fmt_pct(mc["max_dd"].median()))
                st.dataframe(mc, use_container_width=True)

        if walk_forward_enabled:
            train_bt, test_bt = walk_forward_analysis(
                closes=closes,
                feature_builder_kwargs=feature_builder_kwargs,
                score_builder_kwargs=score_builder_kwargs,
                signal_params=signal_params,
                bt_params=bt_params,
                bull_ticker=bull_ticker,
                bear_ticker=pair_bear_ticker,
            )
            if train_bt is not None and test_bt is not None:
                train_stats = summarize_performance(train_bt, "strategy_eq", "net_ret")
                test_stats = summarize_performance(test_bt, "strategy_eq", "net_ret")
                wf = pd.DataFrame(
                    [
                        {
                            "Segment": "Train 70%",
                            "CAGR": train_stats["CAGR"],
                            "Sharpe": train_stats["Sharpe"],
                            "Sortino": train_stats["Sortino"],
                            "Calmar": train_stats["Calmar"],
                            "Max DD": train_stats["Max DD"],
                            "Final Return": train_stats["Final Return"],
                        },
                        {
                            "Segment": "Test 30%",
                            "CAGR": test_stats["CAGR"],
                            "Sharpe": test_stats["Sharpe"],
                            "Sortino": test_stats["Sortino"],
                            "Calmar": test_stats["Calmar"],
                            "Max DD": test_stats["Max DD"],
                            "Final Return": test_stats["Final Return"],
                        },
                    ]
                )
                disp = wf.copy()
                for c in ["CAGR", "Max DD", "Final Return"]:
                    disp[c] = disp[c].map(fmt_pct)
                for c in ["Sharpe", "Sortino", "Calmar"]:
                    disp[c] = disp[c].map(fmt_num)
                st.dataframe(disp, use_container_width=True)

if __name__ == "__main__":
    main()
