"""quote.py — bundled with the market_snapshot skill. Fundamentals from Yahoo Finance."""
import argparse
import json


def snapshot(ticker: str) -> dict:
    import yfinance as yf
    info = yf.Ticker(ticker).info
    return {
        "ticker": ticker,
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "sector": info.get("sector"),
        "market_cap": info.get("marketCap"),
        "pe": info.get("trailingPE"),
        "week52_high": info.get("fiftyTwoWeekHigh"),
        "week52_low": info.get("fiftyTwoWeekLow"),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    a = p.parse_args()
    try:
        print(json.dumps(snapshot(a.ticker)))
    except Exception as e:
        print(json.dumps({"error": str(e)[:300]}))