"""
risk.py  —  bundled with the risk_score skill.

Fetches ~1 year of daily prices for a ticker from Yahoo Finance and computes a
composite 0-100 risk score from annualized volatility, max drawdown, and beta vs a
benchmark index.

The fetch and the maths are kept in separate functions so the maths can be tested
offline without hitting Yahoo Finance.
"""
import argparse
import json

TRADING_DAYS = 252


# ---- data (network) ------------------------------------------------------------
def fetch_close(ticker: str, period: str = "1y"):
    """Daily closing prices from Yahoo Finance. Imported lazily so this file loads
    (and its maths can be tested) on a machine without network access."""
    import yfinance as yf
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    close = df["Close"]
    if hasattr(close, "columns"):          # multi-column frame -> take first column
        close = close.iloc[:, 0]
    return close.dropna()


# ---- maths (pure, testable) ----------------------------------------------------
def daily_returns(close):
    return close.pct_change().dropna()


def annualized_vol(returns) -> float:
    return float(returns.std() * (TRADING_DAYS ** 0.5))


def max_drawdown(close) -> float:
    roll_max = close.cummax()
    return float((close / roll_max - 1.0).min())      # most negative peak-to-trough drop


def beta(stock_returns, bench_returns) -> float:
    import pandas as pd
    import numpy as np
    joined = pd.concat([stock_returns, bench_returns], axis=1).dropna()
    if len(joined) < 2:
        return float("nan")
    s, b = joined.iloc[:, 0], joined.iloc[:, 1]
    var_b = float(np.var(b))
    return float(np.cov(s, b)[0, 1] / var_b) if var_b else float("nan")


def risk_score(vol: float, drawdown: float, bta: float) -> tuple[float, str]:
    """Blend the three metrics into a 0-100 score. Caps and weights are illustrative."""
    vol_s = min(100.0, vol / 0.60 * 100)              # 60% annualized vol -> 100
    dd_s = min(100.0, abs(drawdown) / 0.60 * 100)     # 60% drawdown -> 100
    b = 1.0 if bta != bta else bta                    # NaN-safe (nan != nan)
    beta_s = min(100.0, max(0.0, b) / 2.0 * 100)      # beta 2.0 -> 100
    score = 0.40 * vol_s + 0.35 * dd_s + 0.25 * beta_s
    band = "Low" if score < 33 else "Moderate" if score < 66 else "High"
    return round(score, 1), band


def analyse(ticker: str, benchmark: str = "^NSEI") -> dict:
    close = fetch_close(ticker)
    if close is None or len(close) < 30:
        return {"error": f"not enough price data for {ticker}"}
    rets = daily_returns(close)
    bench_rets = daily_returns(fetch_close(benchmark))
    vol = annualized_vol(rets)
    dd = max_drawdown(close)
    bta = beta(rets, bench_rets)
    score, band = risk_score(vol, dd, bta)
    return {
        "ticker": ticker,
        "annualized_vol": round(vol, 4),
        "max_drawdown": round(dd, 4),
        "beta": round(bta, 3) if bta == bta else None,
        "risk_score": score,
        "band": band,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--benchmark", default="^NSEI")
    a = p.parse_args()
    try:
        print(json.dumps(analyse(a.ticker, a.benchmark)))
    except Exception as e:
        print(json.dumps({"error": str(e)[:300]}))