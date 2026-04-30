import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="NYMO/NYHL Cumulative Oscillator", layout="wide")

# -----------------------------
# Helper Functions
# -----------------------------
def parse_ny_file(filepath):
    """
    Parses the specific format: MM/DD/YYYY, Val, Val, Val, Val, 0
    Handles both single-value rows (NYMO) and multi-value rows (NYHL).
    Returns a DataFrame with 'Date' and 'Value'.
    For NYHL, it averages the 4 values to get a single daily sentiment score.
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        # Read without header, assuming comma separation
        df = pd.read_csv(filepath, header=None, names=['Date', 'V1', 'V2', 'V3', 'V4', 'Zero'], skipinitialspace=True)
        
        # Clean Date
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Determine if it's NYMO (single value) or NYHL (4 values)
        # If V2, V3, V4 are NaN or identical to V1, it's likely single value
        # Based on your samples:
        # NYMO: 11/26/2021, -70.120, -70.120, -70.120, -70.120, 0 (Repeated values)
        # NYHL: 07/28/2022, 11.0000, 18.0000, -6.0000, -6.0000, 0 (Distinct values)
        
        # Check if V1 == V2 == V3 == V4 for the first few valid rows
        is_single_value = True
        sample = df.head(10)
        if not sample.empty:
            if not (sample['V1'] == sample['V2']).all() or not (sample['V1'] == sample['V3']).all():
                is_single_value = False
                
        if is_single_value:
            df['Value'] = df['V1']
        else:
            # Average the 4 components for a composite daily score
            df['Value'] = df[['V1', 'V2', 'V3', 'V4']].mean(axis=1)
            
        return df[['Date', 'Value']]
    except Exception as e:
        st.error(f"Error parsing file {filepath}: {e}")
        return None

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_cci(df, window=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

def calculate_tsi(series, long=25, short=13, signal=7):
    diff = series.diff()
    abs_diff = diff.abs()
    
    double_smoothed = diff.ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean()
    double_abs = abs_diff.ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean()
    
    tsi = 100 * (double_smoothed / double_abs)
    signal_line = tsi.ewm(span=signal, adjust=False).mean()
    return tsi, signal_line

def calculate_williams_r(df, window=14):
    highest_high = df['High'].rolling(window=window).max()
    lowest_low = df['Low'].rolling(window=window).min()
    wr = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
    return wr

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.title("⚙️ Settings")

# Weights
st.sidebar.header("Oscillator Weights")
w_nymo = st.sidebar.slider("NYMO Weight %", 0, 100, 50)
w_nyhl = st.sidebar.slider("NYHL Weight %", 0, 100, 50)

# Normalize weights to sum to 1
total_w = w_nymo + w_nyhl
if total_w == 0: total_w = 1
norm_w_nymo = w_nymo / total_w
norm_w_nyhl = w_nyhl / total_w

st.sidebar.markdown(f"**Normalized:** NYMO: `{norm_w_nymo:.2f}` | NYHL: `{norm_w_nyhl:.2f}`")

# Indicator Parameters
st.sidebar.header("Indicator Parameters")
rsi_win = st.sidebar.number_input("RSI Window", 2, 100, 14)
cci_win = st.sidebar.number_input("CCI Window", 2, 100, 20)
tsi_long = st.sidebar.number_input("TSI Long", 2, 100, 25)
tsi_short = st.sidebar.number_input("TSI Short", 2, 100, 13)
wr_win = st.sidebar.number_input("Williams %R Window", 2, 100, 14)

# -----------------------------
# Main Logic
# -----------------------------
st.title("📈 Cumulative NYMO & NYHL Oscillator Dashboard")
st.caption("Upload `_NYmo.csv` and `_NYHL.csv` to begin.")

# File Uploaders
col1, col2 = st.columns(2)
with col1:
    uploaded_nymo = st.file_uploader("Upload _NYmo.csv", type=['csv'])
with col2:
    uploaded_nyhl = st.file_uploader("Upload _NYHL.csv", type=['csv'])

# Save uploads temporarily to parse
if uploaded_nymo is not None:
    with open("_NYmo_temp.csv", "wb") as f:
        f.write(uploaded_nymo.getbuffer())
    df_nymo = parse_ny_file("_NYmo_temp.csv")
else:
    df_nymo = None

if uploaded_nyhl is not None:
    with open("_NYHL_temp.csv", "wb") as f:
        f.write(uploaded_nyhl.getbuffer())
    df_nyhl = parse_ny_file("_NYHL_temp.csv")
else:
    df_nyhl = None

if df_nymo is not None and df_nyhl is not None:
    # Merge DataFrames
    df_merged = pd.merge(df_nymo, df_nyhl, on='Date', suffixes=('_nymo', '_nyhl'))
    df_merged = df_merged.dropna().sort_values('Date').reset_index(drop=True)
    
    # Calculate Cumulative Oscillators
    # Start from 0 or a baseline
    df_merged['Cum_NYMO'] = df_merged['Value_nymo'].cumsum()
    df_merged['Cum_NYHL'] = df_merged['Value_nyhl'].cumsum()
    
    # Weighted Composite
    df_merged['Composite'] = (df_merged['Cum_NYMO'] * norm_w_nymo) + (df_merged['Cum_NYHL'] * norm_w_nyhl)
    
    # Calculate Technical Indicators on the Composite Series
    # Note: RSI, CCI, Williams %R typically require OHLC. 
    # Since we only have a single composite line, we will treat the Composite as the "Close".
    # For High/Low, we will use a rolling max/min of the Composite to approximate volatility bands for CCI/WR.
    
    comp_series = df_merged['Composite']
    
    # Approximate High/Low for CCI/WR using rolling windows around the composite
    window_hl = 5 # Small window to approximate local H/L
    df_merged['Comp_High'] = comp_series.rolling(window=window_hl, center=True).max()
    df_merged['Comp_Low'] = comp_series.rolling(window=window_hl, center=True).min()
    df_merged['Comp_Close'] = comp_series
    
    # RSI
    df_merged['RSI'] = calculate_rsi(comp_series, rsi_win)
    
    # CCI
    df_merged['CCI'] = calculate_cci(df_merged[['Comp_High', 'Comp_Low', 'Comp_Close']], cci_win)
    
    # TSI
    tsi_val, tsi_sig = calculate_tsi(comp_series, tsi_long, tsi_short, 7)
    df_merged['TSI'] = tsi_val
    df_merged['TSI_Signal'] = tsi_sig
    
    # Williams %R
    df_merged['Williams_R'] = calculate_williams_r(df_merged[['Comp_High', 'Comp_Low', 'Comp_Close']], wr_win)
    
    # -----------------------------
    # Display Data
    # -----------------------------
    st.dataframe(df_merged.tail(10), use_container_width=True)
    
    # -----------------------------
    # Plotting
    # -----------------------------
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.4, 0.2, 0.2, 0.2],
                        subplot_titles=("Cumulative Oscillators", "RSI", "TSI", "Williams %R / CCI"))
    
    # Row 1: Cumulative Oscillators
    fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged['Cum_NYMO'], name='Cum NYMO', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged['Cum_NYHL'], name='Cum NYHL', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged['Composite'], name='Weighted Composite', line=dict(color='black', width=2)), row=1, col=1)
    
    # Row 2: RSI
    fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged['RSI'], name=f'RSI ({rsi_win})', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Row 3: TSI
    fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged['TSI'], name=f'TSI ({tsi_long},{tsi_short})', line=dict(color='teal')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged['TSI_Signal'], name='TSI Signal', line=dict(color='gray', dash='dot')), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    # Row 4: Williams %R and CCI
    fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged['Williams_R'], name=f'Williams %R ({wr_win})', line=dict(color='brown')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged['CCI'], name=f'CCI ({cci_win})', line=dict(color='cyan')), row=4, col=1)
    fig.add_hline(y=-20, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=-80, line_dash="dash", line_color="green", row=4, col=1)
    
    fig.update_layout(height=900, title_text="Technical Analysis of Composite Oscillator", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

elif df_nymo is None or df_nyhl is None:
    st.warning("Please upload both CSV files to generate the dashboard.")
