
---
name: risk_score
description: Compute a 0-100 risk score for a stock from live Yahoo Finance data (volatility, drawdown, beta).
---
# Risk Score

Use this skill when the user asks how risky a single stock is, or asks for a risk
score / risk rating for a ticker.

## How to use

1. Read the ticker symbol from the question. Use the Yahoo Finance format — e.g.
   `RELIANCE.NS`, `INFY.NS`, `TCS.NS` for Indian stocks, or `AAPL`, `MSFT` for US.
2. Run the bundled script (it fetches ~1 year of daily prices from Yahoo Finance and
   computes the metrics — do not estimate by hand):
   `python risk.py --ticker <TICKER>`
   Optionally pass `--benchmark <INDEX>` (default `^NSEI`, the Nifty 50).
   It prints JSON:
   `{"ticker":..., "annualized_vol":..., "max_drawdown":..., "beta":..., "risk_score":..., "band":"Low|Moderate|High"}`
3. Explain the risk score and band in plain language, and say briefly what each metric
   means: volatility = how much the price swings, max drawdown = worst peak-to-trough
   fall, beta = how much it moves relative to the market.

## Notes

- The risk score weights volatility 40%, drawdown 35%, beta 25%. Thresholds are
  illustrative, not financial advice.
- If the script returns an error (e.g. unknown ticker), tell the user the ticker
  could not be found and ask them to check the symbol.
