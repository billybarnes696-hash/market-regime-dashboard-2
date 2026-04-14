import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# =========================
# Config
# =========================
TARGET = "SOXL"
CANARIES = [
    "NVDA", "AMD", "AVGO", "SMH", "SOXX", "TSM",
    "MU", "AMAT", "LRCX", "KLAC", "ASML", "INTC"
]

# Good starting points for your workflow:
# interval="60m", period="730d"  -> strong for hourly style
# interval="15m", period="60d"   -> stronger for very short-term tracking
INTERVAL = "60m"
PERIOD = "730d"

# Test whether canary at t-k predicts SOXL at t+1
MAX_LAG_BARS = 8
TEST_SIZE = 0.30

# Extreme move event study threshold
Z_THRESHOLD = 1.5


@dataclass
class CanaryResult:
    ticker: str
    best_corr_lag: int
    best_corr: float
    best_r2_lag: int
    test_r2: float
    hit_rate: float
    avg_fwd_return_after_strong_up: float
    avg_fwd_return_after_strong_down: float
    sample_size_up: int
    sample_size_down: int


def download_data(tickers: List[str], period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if df.empty:
        raise ValueError("No data downloaded. Check tickers, period, or interval.")

    closes = {}
    for t in tickers:
        if len(tickers) == 1:
            # yfinance single-ticker shape
            closes[t] = df["Close"]
        else:
            if t in df.columns.get_level_values(0):
                closes[t] = df[t]["Close"]

    out = pd.DataFrame(closes).dropna(how="all")
    out = out.ffill().dropna()
    return out


def log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_df / price_df.shift(1)).dropna()


def calc_best_lagged_corr(
    target_ret: pd.Series,
    canary_ret: pd.Series,
    max_lag: int
) -> Tuple[int, float]:
    best_lag = 0
    best_corr = -np.inf

    for lag in range(1, max_lag + 1):
        aligned = pd.concat(
            [
                canary_ret.shift(lag).rename("x"),
                target_ret.rename("y")
            ],
            axis=1
        ).dropna()

        if len(aligned) < 30:
            continue

        corr = aligned["x"].corr(aligned["y"])
        if pd.notna(corr) and abs(corr) > abs(best_corr) if best_corr != -np.inf else True:
            best_corr = corr
            best_lag = lag

    if best_corr == -np.inf:
        return np.nan, np.nan

    return best_lag, best_corr


def out_of_sample_predictive_test(
    target_ret: pd.Series,
    canary_ret: pd.Series,
    max_lag: int,
    test_size: float
) -> Tuple[int, float, float]:
    best_lag = np.nan
    best_r2 = -np.inf
    best_hit = np.nan

    # Predict SOXL next bar using canary lagged return
    y_full = target_ret.shift(-1)

    for lag in range(1, max_lag + 1):
        Xy = pd.concat(
            [
                canary_ret.shift(lag).rename("x"),
                y_full.rename("y")
            ],
            axis=1
        ).dropna()

        if len(Xy) < 100:
            continue

        split_idx = int(len(Xy) * (1 - test_size))
        train = Xy.iloc[:split_idx]
        test = Xy.iloc[split_idx:]

        if len(train) < 30 or len(test) < 20:
            continue

        X_train = train[["x"]].values
        y_train = train["y"].values
        X_test = test[["x"]].values
        y_test = test["y"].values

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)

        # Directional accuracy
        pred_sign = np.sign(pred)
        true_sign = np.sign(y_test)
        hit = (pred_sign == true_sign).mean()

        if r2 > best_r2:
            best_r2 = r2
            best_lag = lag
            best_hit = hit

    if best_r2 == -np.inf:
        return np.nan, np.nan, np.nan

    return best_lag, best_r2, best_hit


def event_study(
    target_ret: pd.Series,
    canary_ret: pd.Series,
    z_threshold: float = 1.5
) -> Tuple[float, float, int, int]:
    # z-score the canary return using rolling stats
    roll_mean = canary_ret.rolling(40).mean()
    roll_std = canary_ret.rolling(40).std()

    z = (canary_ret - roll_mean) / roll_std

    # Forward SOXL next-bar return
    fwd = target_ret.shift(-1)

    strong_up = fwd[z > z_threshold].dropna()
    strong_down = fwd[z < -z_threshold].dropna()

    avg_up = strong_up.mean() if len(strong_up) else np.nan
    avg_down = strong_down.mean() if len(strong_down) else np.nan

    return avg_up, avg_down, len(strong_up), len(strong_down)


def run_backtest(
    target: str,
    canaries: List[str],
    period: str,
    interval: str,
    max_lag: int,
    test_size: float,
    z_threshold: float,
) -> pd.DataFrame:
    tickers = [target] + canaries
    prices = download_data(tickers, period=period, interval=interval)
    rets = log_returns(prices)

    if target not in rets.columns:
        raise ValueError(f"{target} missing from returns data.")

    target_ret = rets[target]
    results: List[CanaryResult] = []

    for ticker in canaries:
        if ticker not in rets.columns:
            continue

        canary_ret = rets[ticker]

        best_corr_lag, best_corr = calc_best_lagged_corr(
            target_ret=target_ret,
            canary_ret=canary_ret,
            max_lag=max_lag
        )

        best_r2_lag, test_r2, hit_rate = out_of_sample_predictive_test(
            target_ret=target_ret,
            canary_ret=canary_ret,
            max_lag=max_lag,
            test_size=test_size
        )

        avg_up, avg_down, n_up, n_down = event_study(
            target_ret=target_ret,
            canary_ret=canary_ret,
            z_threshold=z_threshold
        )

        results.append(
            CanaryResult(
                ticker=ticker,
                best_corr_lag=best_corr_lag,
                best_corr=best_corr,
                best_r2_lag=best_r2_lag,
                test_r2=test_r2,
                hit_rate=hit_rate,
                avg_fwd_return_after_strong_up=avg_up,
                avg_fwd_return_after_strong_down=avg_down,
                sample_size_up=n_up,
                sample_size_down=n_down,
            )
        )

    out = pd.DataFrame([r.__dict__ for r in results])

    # Composite score: emphasizes predictive power + direction + correlation
    out["score"] = (
        out["test_r2"].fillna(0) * 100
        + out["hit_rate"].fillna(0) * 10
        + out["best_corr"].abs().fillna(0) * 5
    )

    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


if __name__ == "__main__":
    results = run_backtest(
        target=TARGET,
        canaries=CANARIES,
        period=PERIOD,
        interval=INTERVAL,
        max_lag=MAX_LAG_BARS,
        test_size=TEST_SIZE,
        z_threshold=Z_THRESHOLD,
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\nTop canaries for SOXL:\n")
    print(results.to_string(index=False))

    print("\nBest practical leader:")
    if not results.empty:
        best = results.iloc[0]
        print(
            f"{best['ticker']} | "
            f"best_corr_lag={best['best_corr_lag']} | "
            f"best_corr={best['best_corr']:.4f} | "
            f"best_r2_lag={best['best_r2_lag']} | "
            f"test_r2={best['test_r2']:.6f} | "
            f"hit_rate={best['hit_rate']:.2%}"
        )
