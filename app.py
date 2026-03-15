import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="RSP / URSP Breadth Model", layout="wide", page_icon="📈")

st.markdown(
    """
    <style>
    .big-card {
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: #111827;
        color: white;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }
    .green-card {
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: linear-gradient(135deg, #065f46, #047857);
        color: white;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }
    .yellow-card {
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: linear-gradient(135deg, #92400e, #b45309);
        color: white;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }
    .red-card {
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: linear-gradient(135deg, #991b1b, #b91c1c);
        color: white;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.88;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .small-note {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_STOCKCHART_URLS = [
    "https://stockcharts.com/public/3423650/tenpp/1",
    "https://stockcharts.com/public/3423650/tenpp/2",
    "https://stockcharts.com/public/3423650/tenpp/3",
]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://stockcharts.com/",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lowered:
            return lowered[key]
    for cand in candidates:
        key = cand.strip().lower()
        for existing in df.columns:
            existing_l = str(existing).strip().lower()
            if key in existing_l or existing_l in key:
                return existing
    return None


def get_latest_value(df: pd.DataFrame, candidates: List[str], default):
    col = find_col(df, candidates)
    if col is None:
        return default
    series = df[col].dropna()
    if series.empty:
        return default
    return series.iloc[-1]


def to_float(value, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if pd.isna(value):
            return default
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
        return float(value)
    except Exception:
        return default


def infer_vxx_mode(vxx_level: float) -> str:
    if vxx_level >= 35:
        return "Spike above upper Bollinger / panic"
    if vxx_level >= 28:
        return "Strong spike"
    if vxx_level >= 22:
        return "Elevated but ordinary"
    if vxx_level >= 17:
        return "Flat / declining"
    return "Complacent / very low"


def fetch_url_text(url: str) -> str:
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return ""


def extract_indicator_value_from_text(text: str, indicator_name: str) -> Optional[float]:
    if not text:
        return None
    patterns = {
        "BPSPX": [r"[$]?BPSPX[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)"],
        "BPNYA": [r"[$]?BPNYA[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)"],
        "NYMO": [r"[$]?NYMO[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)"],
        "NYSI": [r"[$]?NYSI[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)"],
        "SPXA50R": [r"[$]?SPXA50R[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)"],
        "CPCE": [r"[$]?CPCE[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)"],
        "VXX": [r"\bVXX[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)"],
        "RSP:SPY": [r"RSP:SPY[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)", r"RSP/SPY[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)"],
        "HYG:IEF": [r"HYG:IEF[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)", r"HYG/IEF[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)"],
        "HYG:TLT": [r"HYG:TLT[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)", r"HYG/TLT[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)"],
        "RSP": [r"\bRSP[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)"],
        "URSP": [r"\bURSP[^0-9-]*([-]?[0-9]+(?:\.[0-9]+)?)"],
    }
    for pattern in patterns.get(indicator_name, []):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return to_float(match.group(1), None)
    return None


def scrape_stockcharts_values(urls: List[str]) -> Tuple[Dict[str, Optional[float]], List[str]]:
    combined_text = []
    notes = []
    for url in urls:
        txt = fetch_url_text(url)
        if txt:
            combined_text.append(txt)
            notes.append(f"Fetched: {url}")
        else:
            notes.append(f"Failed: {url}")
    full_text = " ".join(combined_text)
    scraped = {}
    for indicator in ["BPSPX", "BPNYA", "NYMO", "NYSI", "SPXA50R", "CPCE", "VXX", "RSP:SPY", "HYG:IEF", "HYG:TLT", "RSP", "URSP"]:
        scraped[indicator] = extract_indicator_value_from_text(full_text, indicator)
    return scraped, notes


def build_scrape_validation_rows(scraped_values: Dict[str, Optional[float]], reference_values: Dict[str, float], tolerance: float) -> pd.DataFrame:
    rows = []
    for key, ref_val in reference_values.items():
        scraped_val = scraped_values.get(key, None)
        if scraped_val is None:
            rows.append({"Field": key, "Scraped": "N/A", "Reference": round(ref_val, 4), "Diff": "N/A", "Match": "⚪"})
            continue
        diff = scraped_val - ref_val
        rows.append({
            "Field": key,
            "Scraped": round(scraped_val, 4),
            "Reference": round(ref_val, 4),
            "Diff": round(diff, 4),
            "Match": "🟢" if abs(diff) <= tolerance else "🔴",
        })
    return pd.DataFrame(rows)


def score_nymo_setup(nymo: float) -> int:
    if nymo <= -90:
        return 10
    if nymo <= -75:
        return 8
    if nymo <= -60:
        return 6
    if nymo <= -40:
        return 3
    if nymo < 40:
        return 0
    if nymo < 60:
        return -2
    if nymo < 75:
        return -4
    if nymo < 90:
        return -6
    return -8


def score_cpce(cpce: float) -> int:
    if cpce >= 0.75:
        return 7
    if cpce >= 0.65:
        return 6
    if cpce >= 0.58:
        return 5
    if cpce >= 0.50:
        return 3
    if cpce >= 0.42:
        return 1
    if cpce >= 0.36:
        return 0
    if cpce >= 0.30:
        return -2
    return -4


def score_bpspx_pctb_rsi_roc(pctb: float, rsi: float, roc: float) -> int:
    score = 0
    if pctb <= 0.05:
        score += 4
    elif pctb <= 0.10:
        score += 3
    elif pctb <= 0.20:
        score += 1
    if rsi <= 30:
        score += 2
    elif rsi <= 40:
        score += 1
    if roc <= -20:
        score += 2
    elif roc <= -10:
        score += 1
    return min(score, 8)


def score_bpnya_pctb_rsi_roc(pctb: float, rsi: float, roc: float) -> int:
    score = 0
    if pctb <= 0.08:
        score += 3
    elif pctb <= 0.15:
        score += 2
    elif pctb <= 0.25:
        score += 1
    if rsi <= 35:
        score += 1
    if roc <= -15:
        score += 1
    return min(score, 5)


def score_vxx(spike_level: str) -> int:
    return {
        "Complacent / very low": -2,
        "Flat / declining": 0,
        "Elevated but ordinary": 1,
        "Strong spike": 3,
        "Spike above upper Bollinger / panic": 5,
    }[spike_level]


def score_participation_damage(spxa50r: float) -> int:
    if spxa50r < 25:
        return 5
    if spxa50r <= 35:
        return 4
    if spxa50r <= 45:
        return 2
    if spxa50r <= 55:
        return 1
    if spxa50r <= 70:
        return 0
    return -2


def score_nymo_hook(nymo_today: float, nymo_yesterday: float, rising_2_days: bool) -> int:
    if nymo_yesterday <= -60 and nymo_today > nymo_yesterday and rising_2_days:
        return 8
    if nymo_yesterday <= -60 and nymo_today > nymo_yesterday:
        return 8
    if nymo_today <= -60 and math.isclose(nymo_today, nymo_yesterday, abs_tol=1.0):
        return 2
    if nymo_today < nymo_yesterday and nymo_today <= -60:
        return 0
    if nymo_today < nymo_yesterday:
        return -3
    return 3


def score_nyhl_improvement(mode: str) -> int:
    return {
        "Sharp improvement": 6,
        "Modest improvement": 4,
        "Flat": 1,
        "Worsening": 0,
        "Worsening sharply while price bounces": -2,
    }[mode]


def score_rsp_price(mode: str) -> int:
    return {
        "Close above prior day high": 8,
        "Intraday break above prior high but weak close": 5,
        "Close green above open but below prior high": 3,
        "Doji / no confirmation": 1,
        "Close red": 0,
        "Close below prior day low": -4,
    }[mode]


def score_rsp_spy_ratio(mode: str) -> int:
    return {
        "Up and above 5-day average": 7,
        "Up on day": 5,
        "Flat": 2,
        "Down slightly": 0,
        "Down hard": -3,
    }[mode]


def score_breadth_spread(confirm_count: int, price_up_breadth_worse: bool) -> int:
    if price_up_breadth_worse:
        return -3
    if confirm_count >= 3:
        return 6
    if confirm_count == 2:
        return 4
    if confirm_count == 1:
        return 2
    return 0


def score_weekly_nysi(mode: str) -> int:
    return {
        "Rising and above signal/MA": 8,
        "Rising but below signal/MA": 5,
        "Flattening": 2,
        "Falling": 0,
        "Falling sharply": -3,
    }[mode]


def score_weekly_bp_trend(mode: str) -> int:
    return {
        "Both rising": 6,
        "One rising, one flat": 4,
        "Both flat": 2,
        "One falling": 1,
        "Both falling": 0,
    }[mode]


def score_weekly_rsp_structure(mode: str) -> int:
    return {
        "Above 10W and 40W, both rising": 6,
        "Above 10W, testing 40W": 4,
        "Reclaiming 10W after washout": 3,
        "Below both but stabilizing": 1,
        "Below both and declining": 0,
    }[mode]


def score_weekly_rsp_spy(mode: str) -> int:
    return {
        "Rising and above key MA": 5,
        "Bottoming / stabilizing": 3,
        "Sideways": 2,
        "Falling": 0,
        "Breaking support": -2,
    }[mode]


def score_risk_appetite(mode: str) -> int:
    return {
        "Both HYG ratios rising": 5,
        "One rising, one flat": 3,
        "Mixed": 2,
        "Both weak": 0,
        "Both falling sharply": -2,
    }[mode]


def classify_total(total_score: int) -> str:
    if total_score < 35:
        return "Breakdown / avoid longs"
    if total_score < 50:
        return "Weak bounce only / mostly cash"
    if total_score < 65:
        return "Tactical long bias / RSP favored"
    if total_score < 80:
        return "Strong long bias / RSP core, URSP add-ons"
    return "Full risk-on breadth thrust"


def trade_recommendation(setup: int, confirmation: int, regime: int, total: int) -> Tuple[str, str, str]:
    if total < 50 or confirmation < 15:
        return "CASH / NO LONG", "red-card", "Breadth is not strong enough to justify RSP or URSP exposure."
    if setup >= 22 and confirmation >= 22 and regime >= 16 and total >= 65:
        return "RSP + URSP", "green-card", "Setup, confirmation, and regime all support tactical leverage."
    if setup >= 20 and confirmation >= 16 and total >= 50:
        return "RSP", "yellow-card", "Good tactical long setup, but leverage is not fully earned yet."
    return "LIGHT RSP / CASH", "yellow-card", "Some positives exist, but the model still wants caution."


def position_sizing(total: int, use_ursp: bool) -> Dict[str, str]:
    rsp_size = "0%"
    ursp_size = "0%"
    if 50 <= total <= 57:
        rsp_size = "25%"
    elif 58 <= total <= 64:
        rsp_size = "50%"
    elif 65 <= total <= 74:
        rsp_size = "75%"
    elif total >= 75:
        rsp_size = "100%"
    if use_ursp:
        if 65 <= total <= 69:
            ursp_size = "20%"
        elif 70 <= total <= 74:
            ursp_size = "30%"
        elif 75 <= total <= 79:
            ursp_size = "40%"
        elif total >= 80:
            ursp_size = "50%"
    return {"RSP": rsp_size, "URSP": ursp_size}


def build_notes(setup: int, confirmation: int, regime: int, total: int) -> List[str]:
    notes: List[str] = []
    if setup >= 25:
        notes.append("Oversold setup is strong. A bounce is plausible.")
    elif setup < 20:
        notes.append("Setup is weak. Avoid assuming a reflex rally.")
    if confirmation >= 22:
        notes.append("Confirmation is strong. Price and breadth are starting to align.")
    elif confirmation < 16:
        notes.append("Confirmation is weak. Treat any bounce as suspect until internals improve.")
    if regime >= 16:
        notes.append("Weekly regime is supportive enough for tactical leverage.")
    else:
        notes.append("Weekly regime is not strong. Favor RSP over URSP.")
    if total >= 80:
        notes.append("Rare breadth thrust zone. This is the only zone where larger URSP sizing is justified.")
    elif total >= 65:
        notes.append("Strong long zone. URSP can be used, but still tactically.")
    elif total >= 50:
        notes.append("RSP zone. Lean long, but do not overpress leverage.")
    else:
        notes.append("Sub-50 total score. Default to cash or very small exposure.")
    return notes


@dataclass
class ScoreResult:
    name: str
    score: int
    max_score: int
    description: str


st.sidebar.title("Breadth Inputs")
st.sidebar.caption("Manual mode or upload your daily / weekly CSV exports.")
mode = st.sidebar.radio("Input mode", ["Manual", "CSV Auto-Load"], index=0)

auto_loaded = False
mapping_notes: List[str] = []

defaults = {
    "nymo": -87.72,
    "cpce": 0.58,
    "bpspx_pctb": 0.02,
    "bpspx_rsi": 24.83,
    "bpspx_roc": -30.19,
    "bpnya_pctb": 0.07,
    "bpnya_rsi": 28.00,
    "bpnya_roc": -22.00,
    "vxx_mode": "Strong spike",
    "spxa50r": 31.0,
    "nymo_yesterday": -92.00,
    "rising_2_days": False,
    "nyhl_mode": "Modest improvement",
    "rsp_price_mode": "Close green above open but below prior high",
    "rsp_spy_mode": "Up on day",
    "breadth_confirm_count": 2,
    "price_up_breadth_worse": False,
    "weekly_nysi_mode": "Flattening",
    "weekly_bp_trend_mode": "One rising, one flat",
    "weekly_rsp_structure_mode": "Reclaiming 10W after washout",
    "weekly_rsp_spy_mode": "Bottoming / stabilizing",
    "risk_appetite_mode": "Mixed",
}

if mode == "CSV Auto-Load":
    daily_file = st.sidebar.file_uploader("Upload daily CSV", type=["csv"])
    weekly_file = st.sidebar.file_uploader("Upload weekly CSV", type=["csv"])
    st.sidebar.caption("You can upload one or both. The app will auto-map common column names and let you override anything.")
    if daily_file is not None or weekly_file is not None:
        auto_loaded = True
        daily_df = clean_columns(pd.read_csv(daily_file)) if daily_file is not None else None
        weekly_df = clean_columns(pd.read_csv(weekly_file)) if weekly_file is not None else None
        if daily_df is not None:
            defaults["nymo"] = to_float(get_latest_value(daily_df, ["NYMO", "$NYMO"], defaults["nymo"]), defaults["nymo"])
            defaults["cpce"] = to_float(get_latest_value(daily_df, ["CPCE", "$CPCE"], defaults["cpce"]), defaults["cpce"])
            defaults["bpspx_pctb"] = to_float(get_latest_value(daily_df, ["BPSPX %B", "BPSPX_pctB"], defaults["bpspx_pctb"]), defaults["bpspx_pctb"])
            defaults["bpspx_rsi"] = to_float(get_latest_value(daily_df, ["BPSPX RSI", "BPSPX_RSI"], defaults["bpspx_rsi"]), defaults["bpspx_rsi"])
            defaults["bpspx_roc"] = to_float(get_latest_value(daily_df, ["BPSPX ROC", "BPSPX ROC (%)"], defaults["bpspx_roc"]), defaults["bpspx_roc"])
            defaults["bpnya_pctb"] = to_float(get_latest_value(daily_df, ["BPNYA %B", "BPNYA_pctB"], defaults["bpnya_pctb"]), defaults["bpnya_pctb"])
            defaults["bpnya_rsi"] = to_float(get_latest_value(daily_df, ["BPNYA RSI", "BPNYA_RSI"], defaults["bpnya_rsi"]), defaults["bpnya_rsi"])
            defaults["bpnya_roc"] = to_float(get_latest_value(daily_df, ["BPNYA ROC", "BPNYA ROC (%)"], defaults["bpnya_roc"]), defaults["bpnya_roc"])
            vxx_val = to_float(get_latest_value(daily_df, ["VXX", "$VXX"], None), None)
            if vxx_val is not None:
                defaults["vxx_mode"] = infer_vxx_mode(vxx_val)
            defaults["spxa50r"] = to_float(get_latest_value(daily_df, ["SPXA50R", "$SPXA50R"], defaults["spxa50r"]), defaults["spxa50r"])
            mapping_notes.append("Daily CSV auto-loaded for key fields where found.")
        if weekly_df is not None:
            mapping_notes.append("Weekly CSV upload detected. Use it for regime confirmation inputs.")

st.sidebar.subheader("Daily Setup")
nymo = st.sidebar.number_input("NYMO (today)", value=float(defaults["nymo"]), format="%.2f")
cpce = st.sidebar.number_input("CPCE", value=float(defaults["cpce"]), format="%.2f")
bpspx_pctb = st.sidebar.number_input("BPSPX %B", value=float(defaults["bpspx_pctb"]), format="%.2f")
bpspx_rsi = st.sidebar.number_input("BPSPX RSI", value=float(defaults["bpspx_rsi"]), format="%.2f")
bpspx_roc = st.sidebar.number_input("BPSPX ROC (%)", value=float(defaults["bpspx_roc"]), format="%.2f")
bpnya_pctb = st.sidebar.number_input("BPNYA %B", value=float(defaults["bpnya_pctb"]), format="%.2f")
bpnya_rsi = st.sidebar.number_input("BPNYA RSI", value=float(defaults["bpnya_rsi"]), format="%.2f")
bpnya_roc = st.sidebar.number_input("BPNYA ROC (%)", value=float(defaults["bpnya_roc"]), format="%.2f")
vxx_mode = st.sidebar.selectbox("VXX condition", ["Complacent / very low", "Flat / declining", "Elevated but ordinary", "Strong spike", "Spike above upper Bollinger / panic"], index=3)
spxa50r = st.sidebar.number_input("SPXA50R", value=float(defaults["spxa50r"]), format="%.2f")

st.sidebar.subheader("Daily Confirmation")
nymo_yesterday = st.sidebar.number_input("NYMO (yesterday)", value=float(defaults["nymo_yesterday"]), format="%.2f")
rising_2_days = st.sidebar.checkbox("NYMO rising 2 straight days from below -60", value=bool(defaults["rising_2_days"]))
nyhl_mode = st.sidebar.selectbox("NYHL trend", ["Sharp improvement", "Modest improvement", "Flat", "Worsening", "Worsening sharply while price bounces"], index=1)
rsp_price_mode = st.sidebar.selectbox("RSP price trigger", ["Close above prior day high", "Intraday break above prior high but weak close", "Close green above open but below prior high", "Doji / no confirmation", "Close red", "Close below prior day low"], index=2)
rsp_spy_mode = st.sidebar.selectbox("RSP:SPY ratio", ["Up and above 5-day average", "Up on day", "Flat", "Down slightly", "Down hard"], index=1)
breadth_confirm_count = st.sidebar.slider("Breadth confirmations", 0, 3, int(defaults["breadth_confirm_count"]))
price_up_breadth_worse = st.sidebar.checkbox("Price up but breadth worse underneath", value=bool(defaults["price_up_breadth_worse"]))

st.sidebar.subheader("Weekly Regime")
weekly_nysi_mode = st.sidebar.selectbox("Weekly NYSI trend", ["Rising and above signal/MA", "Rising but below signal/MA", "Flattening", "Falling", "Falling sharply"], index=2)
weekly_bp_trend_mode = st.sidebar.selectbox("Weekly BPSPX/BPNYA trend", ["Both rising", "One rising, one flat", "Both flat", "One falling", "Both falling"], index=1)
weekly_rsp_structure_mode = st.sidebar.selectbox("Weekly RSP structure", ["Above 10W and 40W, both rising", "Above 10W, testing 40W", "Reclaiming 10W after washout", "Below both but stabilizing", "Below both and declining"], index=2)
weekly_rsp_spy_mode = st.sidebar.selectbox("Weekly RSP:SPY structure", ["Rising and above key MA", "Bottoming / stabilizing", "Sideways", "Falling", "Breaking support"], index=1)
risk_appetite_mode = st.sidebar.selectbox("HYG:IEF / HYG:TLT risk appetite", ["Both HYG ratios rising", "One rising, one flat", "Mixed", "Both weak", "Both falling sharply"], index=2)

setup_components = [
    ScoreResult("NYMO setup", score_nymo_setup(nymo), 10, "Daily washout / stretched condition."),
    ScoreResult("CPCE", score_cpce(cpce), 7, "Fear tailwind, not a trigger."),
    ScoreResult("BPSPX core", score_bpspx_pctb_rsi_roc(bpspx_pctb, bpspx_rsi, bpspx_roc), 8, "Bullish percent washout using %B + RSI + ROC."),
    ScoreResult("BPNYA core", score_bpnya_pctb_rsi_roc(bpnya_pctb, bpnya_rsi, bpnya_roc), 5, "NYSE breadth washout."),
    ScoreResult("VXX stress", score_vxx(vxx_mode), 5, "Volatility stress backdrop."),
]
setup_score = int(clamp(sum(x.score for x in setup_components) + score_participation_damage(spxa50r), 0, 35))

confirmation_components = [
    ScoreResult("NYMO hook", score_nymo_hook(nymo, nymo_yesterday, rising_2_days), 8, "Turning up from oversold matters more than being oversold."),
    ScoreResult("NYHL improvement", score_nyhl_improvement(nyhl_mode), 6, "New highs/lows must improve with price."),
    ScoreResult("RSP price trigger", score_rsp_price(rsp_price_mode), 8, "RSP should confirm with price, not just SPX."),
    ScoreResult("RSP:SPY", score_rsp_spy_ratio(rsp_spy_mode), 7, "Equal-weight relative strength is critical."),
    ScoreResult("Breadth spread", score_breadth_spread(breadth_confirm_count, price_up_breadth_worse), 6, "Protects against cap-weight masking weak internals."),
]
confirmation_score = int(clamp(sum(x.score for x in confirmation_components), 0, 35))

regime_components = [
    ScoreResult("Weekly NYSI", score_weekly_nysi(weekly_nysi_mode), 8, "Intermediate breadth regime."),
    ScoreResult("Weekly BP trend", score_weekly_bp_trend(weekly_bp_trend_mode), 6, "Direction matters more than level."),
    ScoreResult("Weekly RSP structure", score_weekly_rsp_structure(weekly_rsp_structure_mode), 6, "Trend quality of equal-weight."),
    ScoreResult("Weekly RSP:SPY", score_weekly_rsp_spy(weekly_rsp_spy_mode), 5, "Equal-weight leadership backdrop."),
    ScoreResult("Risk appetite", score_risk_appetite(risk_appetite_mode), 5, "Credit risk appetite filter."),
]
regime_score = int(clamp(sum(x.score for x in regime_components), 0, 30))

total_score = int(clamp(setup_score + confirmation_score + regime_score, 0, 100))
classification = classify_total(total_score)
recommendation, card_class, recommendation_note = trade_recommendation(setup_score, confirmation_score, regime_score, total_score)
use_ursp = recommendation == "RSP + URSP"
sizes = position_sizing(total_score, use_ursp)
notes = build_notes(setup_score, confirmation_score, regime_score, total_score)

st.title("📊 RSP / URSP Breadth Confluence Model")
st.caption("Manual scoring or CSV auto-load for deciding between cash, RSP, and URSP using Setup / Confirmation / Regime.")

if auto_loaded and mapping_notes:
    st.success("CSV auto-load is active. Review the mapped fields below and override anything that looks off.")
    for note in mapping_notes:
        st.caption(f"• {note}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div class='big-card'><div class='metric-label'>Setup</div><div class='metric-value'>{setup_score}/35</div><div class='small-note'>Oversold potential</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='big-card'><div class='metric-label'>Confirmation</div><div class='metric-value'>{confirmation_score}/35</div><div class='small-note'>Turn quality</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='big-card'><div class='metric-label'>Regime</div><div class='metric-value'>{regime_score}/30</div><div class='small-note'>Weekly backdrop</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='big-card'><div class='metric-label'>Total</div><div class='metric-value'>{total_score}/100</div><div class='small-note'>{classification}</div></div>", unsafe_allow_html=True)

st.markdown(f"<div class='{card_class}'><div class='metric-label'>Recommendation</div><div class='metric-value'>{recommendation}</div><div class='small-note'>{recommendation_note}</div></div>", unsafe_allow_html=True)

st.subheader("Positioning Map")
p1, p2, p3 = st.columns(3)
with p1:
    st.metric("Suggested RSP Size", sizes["RSP"])
with p2:
    st.metric("Suggested URSP Size", sizes["URSP"])
with p3:
    st.metric("Regime Class", classification)

st.info("Rule set: RSP is the default long vehicle. URSP must be earned with strong confirmation and supportive regime.")

left, right = st.columns(2)
with left:
    st.subheader("Setup Components")
    st.dataframe(pd.DataFrame({"Component": [c.name for c in setup_components], "Score": [c.score for c in setup_components], "Max": [c.max_score for c in setup_components], "Why it matters": [c.description for c in setup_components]}), use_container_width=True, hide_index=True)
    st.subheader("Confirmation Components")
    st.dataframe(pd.DataFrame({"Component": [c.name for c in confirmation_components], "Score": [c.score for c in confirmation_components], "Max": [c.max_score for c in confirmation_components], "Why it matters": [c.description for c in confirmation_components]}), use_container_width=True, hide_index=True)
with right:
    st.subheader("Regime Components")
    st.dataframe(pd.DataFrame({"Component": [c.name for c in regime_components], "Score": [c.score for c in regime_components], "Max": [c.max_score for c in regime_components], "Why it matters": [c.description for c in regime_components]}), use_container_width=True, hide_index=True)
    st.subheader("Daily Notes")
    for note in notes:
        st.write(f"- {note}")

st.subheader("Scrape Validation Check")
st.caption("This validates the numbers you see in the app against values scraped from the public StockCharts URLs. It is a sanity check for URL parsing, not a substitute for chart review.")

val_col1, val_col2 = st.columns([2, 1])
with val_col1:
    url_text = st.text_area("StockCharts URLs to validate", value="\n".join(DEFAULT_STOCKCHART_URLS), height=100)
with val_col2:
    scrape_tolerance = st.number_input("Validation tolerance", min_value=0.0, value=0.05, step=0.01, format="%.2f")
    run_validation = st.button("Run scrape validation")

if run_validation:
    urls_to_check = [u.strip() for u in url_text.splitlines() if u.strip()]
    scraped_values, scrape_notes = scrape_stockcharts_values(urls_to_check)
    for note in scrape_notes:
        st.caption(note)
    reference_values = {
        "BPSPX": bpspx_pctb,
        "BPNYA": bpnya_pctb,
        "NYMO": nymo,
        "SPXA50R": spxa50r,
        "CPCE": cpce,
    }
    validation_df = build_scrape_validation_rows(scraped_values, reference_values, scrape_tolerance)
    st.dataframe(validation_df, use_container_width=True, hide_index=True)
    checked = int((validation_df["Reference"] != "N/A").sum()) if not validation_df.empty else 0
    matched = int((validation_df["Match"] == "🟢").sum()) if not validation_df.empty else 0
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Fields checked", checked)
    with s2:
        st.metric("Matches", matched)
    with s3:
        st.metric("Match rate", f"{(matched / checked * 100):.1f}%" if checked else "0.0%")
    st.info("If the match rate is poor, trust the PDF or your manual reading over URL scraping. Public StockCharts pages are image-heavy, so regex extraction can be incomplete or stale.")

st.subheader("Action Checklist")
check_col1, check_col2 = st.columns(2)
with check_col1:
    st.markdown("**Buy RSP when:**")
    st.write("- Setup >= 20")
    st.write("- Confirmation >= 16")
    st.write("- Total >= 50")
    st.write("- RSP price trigger is positive")
    st.write("- Breadth is not deteriorating underneath")
    st.markdown("**Buy URSP only when:**")
    st.write("- Setup >= 22")
    st.write("- Confirmation >= 22")
    st.write("- Regime >= 16")
    st.write("- Total >= 65")
    st.write("- RSP:SPY confirms")
with check_col2:
    st.markdown("**Stand down / trim when:**")
    st.write("- Total drops below 50")
    st.write("- Confirmation drops below 15")
    st.write("- RSP:SPY weakens while price appears stable")
    st.write("- NYMO re-rolls lower")
    st.write("- Price is up but breadth spread goes negative")

st.subheader("Entry Timing + Stop Loss Plan")
entry_col1, entry_col2 = st.columns(2)
with entry_col1:
    st.markdown("**Execution timing**")
    if recommendation == "RSP + URSP":
        st.write("- RSP entry window: 9:45–10:30 ET")
        st.write("- URSP entry window: 10:00–11:00 ET")
        st.write("- Enter RSP first, then add URSP only if confirmation holds through the first 30–60 minutes")
    elif recommendation == "RSP":
        st.write("- Preferred RSP entry window: 9:45–10:30 ET")
        st.write("- Wait for opening noise to settle before entering")
        st.write("- Avoid URSP unless the score improves intraday")
    else:
        st.write("- No long entry window is active right now")
        st.write("- Stay defensive until total score and confirmation improve")
    st.markdown("**Trigger quality**")
    st.write(f"- Current RSP trigger state: {rsp_price_mode}")
    st.write(f"- Current RSP:SPY state: {rsp_spy_mode}")
    st.write(f"- Breadth confirmations: {breadth_confirm_count}/3")
with entry_col2:
    st.markdown("**E*TRADE-style order plan**")
    if recommendation in ["RSP", "RSP + URSP"]:
        st.write("- Enter with a limit order")
        st.write("- After fill, place a protective stop order")
        st.write("- RSP can use a slightly wider stop than URSP")
    else:
        st.write("- No long order plan active")
        st.write("- Keep powder dry until confirmation improves")

rsp_entry_ready = recommendation in ["RSP", "RSP + URSP"] and total_score >= 50 and confirmation_score >= 16
ursp_entry_ready = recommendation == "RSP + URSP" and setup_score >= 22 and confirmation_score >= 22 and regime_score >= 16 and total_score >= 65
rsp_default_stop_pct = 2.0 if recommendation == "RSP" else 1.75
ursp_default_stop_pct = 3.0
stop_col1, stop_col2, stop_col3 = st.columns(3)
with stop_col1:
    st.metric("RSP Entry Active", "Yes" if rsp_entry_ready else "No")
    st.metric("RSP Stop %", f"{rsp_default_stop_pct:.2f}%")
with stop_col2:
    st.metric("URSP Entry Active", "Yes" if ursp_entry_ready else "No")
    st.metric("URSP Stop %", f"{ursp_default_stop_pct:.2f}%")
with stop_col3:
    st.metric("Trade Bias", recommendation)
    st.metric("Execution Posture", "Tradeable" if recommendation in ["RSP", "RSP + URSP"] else "Wait for confirmation")

st.subheader("Input Snapshot")
input_snapshot = {
    "Input_mode": mode,
    "NYMO_today": nymo,
    "NYMO_yesterday": nymo_yesterday,
    "CPCE": cpce,
    "BPSPX_pctB": bpspx_pctb,
    "BPSPX_RSI": bpspx_rsi,
    "BPSPX_ROC": bpspx_roc,
    "BPNYA_pctB": bpnya_pctb,
    "BPNYA_RSI": bpnya_rsi,
    "BPNYA_ROC": bpnya_roc,
    "VXX_mode": vxx_mode,
    "SPXA50R": spxa50r,
    "NYMO_rising_2_days": rising_2_days,
    "NYHL_mode": nyhl_mode,
    "RSP_price_mode": rsp_price_mode,
    "RSP_SPY_mode": rsp_spy_mode,
    "Breadth_confirm_count": breadth_confirm_count,
    "Price_up_breadth_worse": price_up_breadth_worse,
    "Weekly_NYSI": weekly_nysi_mode,
    "Weekly_BP_trend": weekly_bp_trend_mode,
    "Weekly_RSP_structure": weekly_rsp_structure_mode,
    "Weekly_RSP_SPY": weekly_rsp_spy_mode,
    "Risk_appetite": risk_appetite_mode,
    "Setup_score": setup_score,
    "Confirmation_score": confirmation_score,
    "Regime_score": regime_score,
    "Total_score": total_score,
    "Recommendation": recommendation,
}

snapshot_df = pd.DataFrame([input_snapshot])
st.dataframe(snapshot_df, use_container_width=True, hide_index=True)
st.download_button("Download snapshot CSV", data=snapshot_df.to_csv(index=False).encode("utf-8"), file_name="rsp_ursp_breadth_snapshot.csv", mime="text/csv")

st.caption("This validated version can sanity-check URL scraping against the values inside the app. For production use, trust manual or CSV values over scraped public-page numbers when they disagree.")

