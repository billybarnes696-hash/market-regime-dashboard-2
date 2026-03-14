import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="RSP / URSP Chart Scraper Dashboard", layout="wide", page_icon="📈")

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
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_URLS = """https://stockcharts.com/public/3423650/tenpp/1
https://stockcharts.com/public/3423650/tenpp/2
https://stockcharts.com/public/3423650/tenpp/3"""


@dataclass
class ChartItem:
    page_url: str
    page_number: int
    chart_title: str
    chart_link: Optional[str]
    image_url: Optional[str]


@dataclass
class ScoreResult:
    name: str
    score: int
    max_score: int
    description: str


SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://stockcharts.com/",
    }
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@st.cache_data(ttl=900, show_spinner=False)
def fetch_page_html(url: str) -> str:
    response = SESSION.get(url, timeout=20)
    response.raise_for_status()
    return response.text


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def to_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
        return float(value)
    except Exception:
        return default


def parse_stockcharts_page(url: str) -> Tuple[Dict[str, str], List[ChartItem]]:
    html = fetch_page_html(url)
    soup = BeautifulSoup(html, "html.parser")

    page_meta = {
        "page_title": normalize_text(soup.title.get_text(" ", strip=True) if soup.title else ""),
        "updated": "",
    }

    text_blob = soup.get_text(" ", strip=True)
    updated_match = re.search(r"Last Updated:?\s*([A-Za-z]+\s+\d{1,2},\s+\d{4}.*)$", text_blob)
    if updated_match:
        page_meta["updated"] = normalize_text(updated_match.group(1))

    page_match = re.search(r"/tenpp/(\d+)", url)
    page_number = int(page_match.group(1)) if page_match else 0

    items: List[ChartItem] = []
    for link in soup.find_all("a", href=True):
        img = link.find("img")
        if not img:
            continue

        href = link.get("href", "")
        src = img.get("src") or img.get("data-src")
        title = normalize_text(link.get_text(" ", strip=True) or img.get("alt") or "")

        if not src:
            continue
        if not title:
            continue

        looks_like_chart = (
            "/c-sc/" in href
            or "chart" in href.lower()
            or "sharpchart" in href.lower()
            or "stockcharts" in href.lower()
        )
        if not looks_like_chart:
            continue

        items.append(
            ChartItem(
                page_url=url,
                page_number=page_number,
                chart_title=title,
                chart_link=urljoin(url, href),
                image_url=urljoin(url, src),
            )
        )

    deduped: List[ChartItem] = []
    seen = set()
    for item in items:
        key = (item.chart_title, item.chart_link, item.image_url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return page_meta, deduped


@st.cache_data(ttl=900, show_spinner=False)
def scrape_pages(urls: Tuple[str, ...]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    chart_rows: List[Dict[str, object]] = []
    meta_rows: List[Dict[str, object]] = []

    for url in urls:
        meta, charts = parse_stockcharts_page(url)
        meta_rows.append(
            {
                "page_url": url,
                "page_title": meta["page_title"],
                "updated": meta["updated"],
                "chart_count": len(charts),
            }
        )
        for chart in charts:
            chart_rows.append(
                {
                    "page_number": chart.page_number,
                    "page_url": chart.page_url,
                    "chart_title": chart.chart_title,
                    "chart_link": chart.chart_link,
                    "image_url": chart.image_url,
                }
            )

    charts_df = pd.DataFrame(chart_rows)
    meta_df = pd.DataFrame(meta_rows)
    if not charts_df.empty:
        charts_df = charts_df.sort_values(["page_number", "chart_title"]).reset_index(drop=True)
    return charts_df, meta_df


def lookup_chart_url(charts_df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if charts_df.empty:
        return None
    for _, row in charts_df.iterrows():
        title = str(row["chart_title"]).lower()
        for name in names:
            if name.lower() in title or title in name.lower():
                return row["image_url"]
    return None


# -------------------- scoring --------------------
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


def score_vxx(mode: str) -> int:
    return {
        "Complacent / very low": -2,
        "Flat / declining": 0,
        "Elevated but ordinary": 1,
        "Strong spike": 3,
        "Spike above upper Bollinger / panic": 5,
    }[mode]


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


# sidebar
st.sidebar.title("Chart Scraper")
urls_text = st.sidebar.text_area("Public StockCharts URLs (one per line)", value=DEFAULT_URLS, height=120)
run_scrape = st.sidebar.button("Scrape chart pages", type="primary")

charts_df = pd.DataFrame()
meta_df = pd.DataFrame()
urls = tuple([u.strip() for u in urls_text.splitlines() if u.strip()])

if run_scrape and urls:
    try:
        charts_df, meta_df = scrape_pages(urls)
        st.session_state["charts_df"] = charts_df
        st.session_state["meta_df"] = meta_df
    except Exception as exc:
        st.error(f"Scrape failed: {exc}")

if "charts_df" in st.session_state:
    charts_df = st.session_state["charts_df"]
    meta_df = st.session_state.get("meta_df", pd.DataFrame())

st.title("📊 RSP / URSP Chart Scraper Dashboard")
st.caption("Scrape public StockCharts ChartList pages, review chart images, capture the latest readings, and score RSP vs URSP.")

if not meta_df.empty:
    a, b, c = st.columns(3)
    with a:
        st.metric("Pages scraped", len(meta_df))
    with b:
        st.metric("Charts found", len(charts_df))
    with c:
        updated_text = " | ".join([x for x in meta_df["updated"].astype(str).tolist() if x]) or "Not detected"
        st.markdown(f"**Updated:** {updated_text}")

if charts_df.empty:
    st.info("Click **Scrape chart pages** to load the public pages. The scraper pulls chart titles and image links, then you use those chart images to enter the latest values.")
else:
    st.success("Scrape complete. Review the chart inventory and gallery below.")
    st.dataframe(charts_df, use_container_width=True, hide_index=True)

    page_numbers = sorted(charts_df["page_number"].dropna().unique().tolist())
    if page_numbers:
        tabs = st.tabs([f"Page {int(x)}" for x in page_numbers])
        for tab, page_num in zip(tabs, page_numbers):
            with tab:
                subset = charts_df[charts_df["page_number"] == page_num].reset_index(drop=True)
                cols = st.columns(2)
                for idx, row in subset.iterrows():
                    with cols[idx % 2]:
                        st.markdown(f"**{row['chart_title']}**")
                        if pd.notna(row["chart_link"]):
                            st.caption(row["chart_link"])
                        if pd.notna(row["image_url"]):
                            st.image(row["image_url"], use_container_width=True)
                        st.divider()

st.subheader("Key chart references")
ref_cols = st.columns(4)
reference_map = {
    "NYMO": ["NYMO", "$NYMO"],
    "CPCE": ["CPCE", "$CPCE"],
    "BPSPX": ["BPSPX", "$BPSPX"],
    "BPNYA": ["BPNYA", "$BPNYA"],
    "SPXA50R": ["SPXA50R", "$SPXA50R"],
    "RSP": ["RSP"],
    "RSP:SPY": ["RSP:SPY", "RSP/SPY"],
    "NYSI": ["NYSI", "$NYSI"],
    "VXX": ["VXX", "$VXX"],
    "HYG:IEF": ["HYG:IEF", "HYG/IEF"],
    "HYG:TLT": ["HYG:TLT", "HYG/TLT"],
}
for idx, (label, names) in enumerate(reference_map.items()):
    with ref_cols[idx % 4]:
        st.markdown(f"**{label}**")
        image_url = lookup_chart_url(charts_df, names) if not charts_df.empty else None
        if image_url:
            st.image(image_url, use_container_width=True)
        else:
            st.caption("Not detected in scraped pages.")

with st.expander("Enter latest visible chart readings", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        nymo = st.number_input("NYMO", value=-87.72, format="%.2f")
        cpce = st.number_input("CPCE", value=0.58, format="%.2f")
        bpspx_pctb = st.number_input("BPSPX %B", value=0.02, format="%.2f")
        bpspx_rsi = st.number_input("BPSPX RSI", value=24.83, format="%.2f")
        bpspx_roc = st.number_input("BPSPX ROC (%)", value=-30.19, format="%.2f")
    with c2:
        bpnya_pctb = st.number_input("BPNYA %B", value=0.07, format="%.2f")
        bpnya_rsi = st.number_input("BPNYA RSI", value=28.00, format="%.2f")
        bpnya_roc = st.number_input("BPNYA ROC (%)", value=-22.00, format="%.2f")
        spxa50r = st.number_input("SPXA50R", value=31.00, format="%.2f")
        vxx = st.number_input("VXX", value=35.10, format="%.2f")
    with c3:
        nyhl_mode = st.selectbox("NYHL trend", ["Sharp improvement", "Modest improvement", "Flat", "Worsening", "Worsening sharply while price bounces"], index=1)
        nymo_yesterday = st.number_input("NYMO yesterday", value=-92.00, format="%.2f")
        rising_2_days = st.checkbox("NYMO rising 2 straight days from below -60", value=False)
        rsp_open = st.number_input("RSP Open", value=194.10, format="%.2f")
        rsp_close = st.number_input("RSP Close", value=194.95, format="%.2f")
    with c4:
        prior_day_high = st.number_input("Prior day high", value=195.00, format="%.2f")
        prior_day_low = st.number_input("Prior day low", value=193.00, format="%.2f")
        rsp_spy_ratio = st.number_input("RSP:SPY", value=0.2922, format="%.4f")
        rsp_spy_mode = st.selectbox("RSP:SPY mode", ["Up and above 5-day average", "Up on day", "Flat", "Down slightly", "Down hard"], index=1)
        breadth_confirm_count = st.slider("Breadth confirmations", 0, 3, 2)
        price_up_breadth_worse = st.checkbox("Price up but breadth worse underneath", value=False)

    w1, w2, w3 = st.columns(3)
    with w1:
        weekly_nysi_mode = st.selectbox("Weekly NYSI trend", ["Rising and above signal/MA", "Rising but below signal/MA", "Flattening", "Falling", "Falling sharply"], index=2)
        nysi = st.number_input("NYSI", value=120.0, format="%.2f")
    with w2:
        weekly_bp_trend_mode = st.selectbox("Weekly BPSPX/BPNYA trend", ["Both rising", "One rising, one flat", "Both flat", "One falling", "Both falling"], index=1)
        hyg_ief = st.number_input("HYG:IEF", value=0.8450, format="%.4f")
    with w3:
        weekly_rsp_structure_mode = st.selectbox("Weekly RSP structure", ["Above 10W and 40W, both rising", "Above 10W, testing 40W", "Reclaiming 10W after washout", "Below both but stabilizing", "Below both and declining"], index=2)
        weekly_rsp_spy_mode = st.selectbox("Weekly RSP:SPY structure", ["Rising and above key MA", "Bottoming / stabilizing", "Sideways", "Falling", "Breaking support"], index=1)
        hyg_tlt = st.number_input("HYG:TLT", value=0.6350, format="%.4f")
        risk_appetite_mode = st.selectbox("Risk appetite", ["Both HYG ratios rising", "One rising, one flat", "Mixed", "Both weak", "Both falling sharply"], index=2)

if rsp_close > prior_day_high:
    rsp_price_mode = "Close above prior day high"
elif rsp_close > rsp_open and rsp_close < prior_day_high:
    rsp_price_mode = "Close green above open but below prior high"
elif rsp_close < prior_day_low:
    rsp_price_mode = "Close below prior day low"
elif rsp_close < rsp_open:
    rsp_price_mode = "Close red"
else:
    rsp_price_mode = "Doji / no confirmation"

if vxx >= 35:
    vxx_mode = "Spike above upper Bollinger / panic"
elif vxx >= 28:
    vxx_mode = "Strong spike"
elif vxx >= 22:
    vxx_mode = "Elevated but ordinary"
elif vxx >= 17:
    vxx_mode = "Flat / declining"
else:
    vxx_mode = "Complacent / very low"

setup_components = [
    ScoreResult("NYMO setup", score_nymo_setup(nymo), 10, "Daily washout / stretched condition."),
    ScoreResult("CPCE", score_cpce(cpce), 7, "Fear tailwind, not a trigger."),
    ScoreResult("BPSPX core", score_bpspx_pctb_rsi_roc(bpspx_pctb, bpspx_rsi, bpspx_roc), 8, "Bullish percent washout using %B + RSI + ROC."),
    ScoreResult("BPNYA core", score_bpnya_pctb_rsi_roc(bpnya_pctb, bpnya_rsi, bpnya_roc), 5, "NYSE breadth washout."),
    ScoreResult("VXX stress", score_vxx(vxx_mode), 5, "Volatility stress backdrop."),
]
setup_raw = sum(x.score for x in setup_components) + score_participation_damage(spxa50r)
setup_score = int(clamp(setup_raw, 0, 35))

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

st.subheader("Breadth score output")
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(f"<div class='big-card'><div>Setup</div><div style='font-size:2rem;font-weight:800'>{setup_score}/35</div></div>", unsafe_allow_html=True)
with s2:
    st.markdown(f"<div class='big-card'><div>Confirmation</div><div style='font-size:2rem;font-weight:800'>{confirmation_score}/35</div></div>", unsafe_allow_html=True)
with s3:
    st.markdown(f"<div class='big-card'><div>Regime</div><div style='font-size:2rem;font-weight:800'>{regime_score}/30</div></div>", unsafe_allow_html=True)
with s4:
    st.markdown(f"<div class='big-card'><div>Total</div><div style='font-size:2rem;font-weight:800'>{total_score}/100</div><div>{classification}</div></div>", unsafe_allow_html=True)

st.markdown(f"<div class='{card_class}'><div>Recommendation</div><div style='font-size:2rem;font-weight:800'>{recommendation}</div><div>{recommendation_note}</div></div>", unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)
with p1:
    st.metric("Suggested RSP size", sizes["RSP"])
with p2:
    st.metric("Suggested URSP size", sizes["URSP"])
with p3:
    st.metric("RSP price state", rsp_price_mode)

left, right = st.columns(2)
with left:
    comp_df = pd.DataFrame(
        {
            "Bucket": ["Setup"] * len(setup_components) + ["Confirmation"] * len(confirmation_components) + ["Regime"] * len(regime_components),
            "Component": [x.name for x in setup_components + confirmation_components + regime_components],
            "Score": [x.score for x in setup_components + confirmation_components + regime_components],
            "Max": [x.max_score for x in setup_components + confirmation_components + regime_components],
            "Why it matters": [x.description for x in setup_components + confirmation_components + regime_components],
        }
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
with right:
    st.subheader("Notes")
    for note in notes:
        st.write(f"- {note}")
    st.caption(f"Inferred VXX regime: {vxx_mode}")
    st.caption(f"Inferred RSP price trigger: {rsp_price_mode}")

extraction_df = pd.DataFrame(
    [
        {"Field": "NYMO", "Value": nymo},
        {"Field": "CPCE", "Value": cpce},
        {"Field": "BPSPX %B", "Value": bpspx_pctb},
        {"Field": "BPSPX RSI", "Value": bpspx_rsi},
        {"Field": "BPSPX ROC (%)", "Value": bpspx_roc},
        {"Field": "BPNYA %B", "Value": bpnya_pctb},
        {"Field": "BPNYA RSI", "Value": bpnya_rsi},
        {"Field": "BPNYA ROC (%)", "Value": bpnya_roc},
        {"Field": "SPXA50R", "Value": spxa50r},
        {"Field": "VXX", "Value": vxx},
        {"Field": "RSP Open", "Value": rsp_open},
        {"Field": "RSP Close", "Value": rsp_close},
        {"Field": "RSP:SPY", "Value": rsp_spy_ratio},
        {"Field": "NYSI", "Value": nysi},
        {"Field": "HYG:IEF", "Value": hyg_ief},
        {"Field": "HYG:TLT", "Value": hyg_tlt},
        {"Field": "Setup score", "Value": setup_score},
        {"Field": "Confirmation score", "Value": confirmation_score},
        {"Field": "Regime score", "Value": regime_score},
        {"Field": "Total score", "Value": total_score},
        {"Field": "Recommendation", "Value": recommendation},
    ]
)

st.subheader("Extraction sheet")
st.dataframe(extraction_df, use_container_width=True, hide_index=True)

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.download_button(
        "Download extraction sheet CSV",
        data=extraction_df.to_csv(index=False).encode("utf-8"),
        file_name="breadth_extraction_sheet.csv",
        mime="text/csv",
    )
with col_d2:
    if not charts_df.empty:
        st.download_button(
            "Download scraped chart inventory CSV",
            data=charts_df.to_csv(index=False).encode("utf-8"),
            file_name="scraped_stockcharts_inventory.csv",
            mime="text/csv",
        )

st.caption(
    "This app scrapes chart titles and image links from public StockCharts pages, then uses those chart images as a manual extraction surface for your RSP/URSP breadth model. Public pages are image-based, so numeric values still need human entry for reliability."
)

