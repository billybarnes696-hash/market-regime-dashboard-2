import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="NYMO/NYHL Cumulative Oscillator", layout="wide")

# -----------------------------
# Helper Functions
# -----------------------------
def parse_csv_data(file_bytes, filename):
    if file_bytes is None:
        return None
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), header=None, names=['Date', 'V1', 'V2', 'V3', 'V4', 'Extra'],
                         skipinitialspace=True, on_bad_lines='skip')
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
        for col in ['V1', 'V2', 'V3', 'V4']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
        if df.empty: return None

        sample = df.head(20)
        is_single = (sample['V1'] == sample['V2']).all() and (sample['V1'] == sample['V3']).all()

        if is_single:
            df['Value'] = df['V1']
        else:
            df['Value'] = df[['V1', 'V2', 'V3', 'V4']].mean(axis=1)
        return df[['Date', 'Value']]
    except Exception as e:
        st.error(f"❌ Parsing Error for {filename}: {e}")
        return None

def calc_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

def calc_tsi(series, long=25, short=13, signal=7):
    diff = series.diff()
    abs_diff = diff.abs()
    double_sm = diff.ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean()
    double_abs = abs_diff.ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean()
    tsi = 100 * double_sm / double_abs.replace(0, np.nan)
    sig = tsi.ewm(span=signal, adjust=False).mean()
    return tsi, sig

def calc_wr(high, low, close, window=14):
    hh = high.rolling(window).max()
    ll = low.rolling(window).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)

def process_data(base_df, w_nymo, w_nyhl, rsi_win, cci_win, tsi_long, tsi_short, wr_win):
    df = base_df.copy()
    df['Cum_NYMO'] = df['Value_nymo'].cumsum()
    df['Cum_NYHL'] = df['Value_nyhl'].cumsum()
    df['Composite'] = (df['Cum_NYMO'] * (w_nymo/100)) + (df['Cum_NYHL'] * (w_nyhl/100))

    window_hl = 5
    df['High'] = df['Composite'].rolling(window_hl, center=True).max()
    df['Low'] = df['Composite'].rolling(window_hl, center=True).min()
    df['Close'] = df['Composite']

    df['RSI'] = calc_rsi(df['Close'], rsi_win)
    df['CCI'] = calc_cci(df['High'], df['Low'], df['Close'], cci_win)
    df['TSI'], df['TSI_Signal'] = calc_tsi(df['Close'], tsi_long, tsi_short, 7)
    df['WR'] = calc_wr(df['High'], df['Low'], df['Close'], wr_win)
    return df

# -----------------------------
# UI & Layout
# -----------------------------
st.title("📈 Cumulative NYMO & NYHL Oscillator Dashboard")
st.caption("Upload data files to generate the technical dashboard. Adjust weights/params in the sidebar to update live.")

col1, col2 = st.columns(2)
with col1:
    nymo_file = st.file_uploader("Upload _NYmo.csv", type=['csv'], key="nymo_uploader")
with col2:
    nyhl_file = st.file_uploader("Upload _NYHL.csv", type=['csv'], key="nyhl_uploader")

# 🔒 PERSIST DATA ACROSS SLIDER CHANGES
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
    st.session_state.file_key = None
    st.session_state.data_loaded = False

if nymo_file is not None and nyhl_file is not None:
    current_key = (nymo_file.name, nyhl_file.name, nymo_file.size, nyhl_file.size)
    if st.session_state.file_key != current_key:
        df_nymo = parse_csv_data(nymo_file.getvalue(), "_NYmo.csv")
        df_nyhl = parse_csv_data(nyhl_file.getvalue(), "_NYHL.csv")
        if df_nymo is not None and df_nyhl is not None:
            st.session_state.raw_df = pd.merge(df_nymo, df_nyhl, on='Date', suffixes=('_nymo', '_nyhl')).dropna().sort_values('Date').reset_index(drop=True)
            st.session_state.file_key = current_key
            st.session_state.data_loaded = True
        else:
            st.warning("Failed to parse files. Check CSV format.")
            st.session_state.data_loaded = False
else:
    st.session_state.data_loaded = False

if not st.session_state.data_loaded:
    st.info("👆 Please upload both CSV files to begin.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Oscillator Weights")
    w_nymo = st.slider("NYMO Weight %", 0, 100, 50, key="w_nymo")
    w_nyhl = 100 - w_nymo
    st.markdown(f"**NYMO:** `{w_nymo/100:.2f}` | **NYHL:** `{w_nyhl/100:.2f}`")

    st.header("📊 Indicator Settings")
    rsi_win = st.number_input("RSI Window", 2, 50, 14)
    cci_win = st.number_input("CCI Window", 2, 50, 20)
    tsi_long = st.number_input("TSI Long", 2, 50, 25)
    tsi_short = st.number_input("TSI Short", 2, 50, 13)
    wr_win = st.number_input("Williams %R Window", 2, 50, 14)

    st.divider()
    st.header("📅 Time Window")
    view_range = st.selectbox("Select View", ["6 Months", "1 Year", "2 Years", "3 Years", "4 Years", "Custom"])

# 🔄 RECALCULATE LIVE ON SLIDER/INPUT CHANGE
df_processed = process_data(
    st.session_state.raw_df, w_nymo, w_nyhl, rsi_win, cci_win, tsi_long, tsi_short, wr_win
)

# 📅 Date filtering
latest_date = df_processed['Date'].max()
earliest_date = df_processed['Date'].min()
if view_range == "Custom":
    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("Start", value=latest_date - pd.DateOffset(years=1), min_value=earliest_date, max_value=latest_date)
    end_date = c2.date_input("End", value=latest_date, min_value=earliest_date, max_value=latest_date)
else:
    offset_map = {"6 Months": pd.DateOffset(months=6), "1 Year": pd.DateOffset(years=1),
                  "2 Years": pd.DateOffset(years=2), "3 Years": pd.DateOffset(years=3), "4 Years": pd.DateOffset(years=4)}
    start_date = max(earliest_date, latest_date - offset_map[view_range])
    end_date = latest_date
    st.sidebar.caption(f"🔍 Showing: `{start_date.date()}` → `{end_date.date()}`")

df_view = df_processed[(df_processed['Date'] >= pd.Timestamp(start_date)) & (df_processed['Date'] <= pd.Timestamp(end_date))].copy()

# -----------------------------
# Display & Plotting (5 SEPARATE SUBPLOTS)
# -----------------------------
st.dataframe(df_view[['Date', 'Cum_NYMO', 'Cum_NYHL', 'Composite', 'RSI', 'TSI', 'CCI', 'WR']].tail(15), use_container_width=True)

fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    row_heights=[0.35, 0.15, 0.15, 0.15, 0.2],
                    subplot_titles=["Cumulative Oscillators", "RSI", "TSI", "CCI", "Williams %R"])

# Row 1
fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Cum_NYMO'], name='Cum NYMO', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Cum_NYHL'], name='Cum NYHL', line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Composite'], name='Weighted Composite', line=dict(color='black', width=2)), row=1, col=1)

# Row 2: RSI
fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

# Row 3: TSI
fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['TSI'], name='TSI', line=dict(color='teal')), row=3, col=1)
fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['TSI_Signal'], name='TSI Signal', line=dict(color='gray', dash='dot')), row=3, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

# Row 4: CCI (Separate)
fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['CCI'], name='CCI', line=dict(color='cyan')), row=4, col=1)
fig.add_hline(y=100, line_dash="dash", line_color="red", row=4, col=1)  # Standard CCI Overbought
fig.add_hline(y=-100, line_dash="dash", line_color="green", row=4, col=1) # Standard CCI Oversold

# Row 5: Williams %R (Separate)
fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['WR'], name='Williams %R', line=dict(color='brown')), row=5, col=1)
fig.add_hline(y=-20, line_dash="dash", line_color="red", row=5, col=1)   # WR Overbought
fig.add_hline(y=-80, line_dash="dash", line_color="green", row=5, col=1) # WR Oversold

fig.update_layout(height=1000, hovermode="x unified", legend_tracegroupgap=150)
st.plotly_chart(fig, use_container_width=True)
