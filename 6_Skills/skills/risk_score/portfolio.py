"""
portfolio.py — bundled with the portfolio_risk skill.

Computes portfolio volatility from the covariance of daily returns, and the
diversification benefit versus a naive weighted average of individual volatilities.
Fetch and maths are separated so the maths is testable offline.
"""
import argparse
import json

TRADING_DAYS = 252


def fetch_returns(tickers, period="1y"):
    """Daily returns matrix (one column per ticker) from Yahoo Finance."""
    import yfinance as yf
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)["Close"]
    return data.dropna().pct_change().dropna()


def compute(returns, weights) -> dict:
    """Pure maths on a returns DataFrame + weights. No network."""
    import numpy as np
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    cov = returns.cov().values * TRADING_DAYS
    port_vol = float(np.sqrt(w @ cov @ w))
    indiv_vol = (returns.std().values * (TRADING_DAYS ** 0.5))
    weighted_avg_vol = float(w @ indiv_vol)
    benefit = round((weighted_avg_vol - port_vol) / weighted_avg_vol * 100, 1) if weighted_avg_vol else 0.0
    return {
        "portfolio_vol": round(port_vol, 4),
        "weighted_avg_vol": round(weighted_avg_vol, 4),
        "diversification_benefit_pct": benefit,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", required=True, help="comma-separated")
    p.add_argument("--weights", default="", help="comma-separated; equal if omitted")
    a = p.parse_args()
    tickers = [t.strip() for t in a.tickers.split(",")]
    weights = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(tickers)
    try:
        print(json.dumps(compute(fetch_returns(tickers), weights)))
    except Exception as e:
        print(json.dumps({"error": str(e)[:300]}))