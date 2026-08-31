
---
name: market_snapshot
description: Fetch a quick fundamentals snapshot for a ticker from Yahoo Finance (price, sector, P/E, 52-week range).
---
# Market Snapshot

Use this skill when the user wants basic context on a stock — its current price,
sector, size, valuation, or where it sits in its 52-week range. Good to pair with a
risk score to explain *why* a stock is risky.

## How to use

1. Read the ticker (Yahoo Finance format, e.g. `INFY.NS`, `AAPL`).
2. Run the bundled script:
   `python quote.py --ticker <TICKER>`
   It prints JSON with: `price`, `sector`, `market_cap`, `pe`, `week52_high`, `week52_low`.
3. Summarise the snapshot in one or two sentences. If a field is missing, just skip it.
