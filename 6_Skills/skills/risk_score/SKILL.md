
---
name: portfolio_risk
description: Compute portfolio volatility and the diversification benefit for several tickers and weights, using Yahoo Finance.
---
# Portfolio Risk

Use this skill when the user gives more than one stock (a portfolio) and asks about
the combined risk, or how much diversification helps.

## How to use

1. Read the tickers and their weights from the question. If weights are not given,
   assume equal weights.
2. Run the bundled script:
   `python portfolio.py --tickers TCS.NS,INFY.NS,RELIANCE.NS --weights 0.4,0.3,0.3`
   It prints JSON with: `portfolio_vol`, `weighted_avg_vol`, `diversification_benefit_pct`.
3. Explain the result: portfolio volatility is the combined risk; the diversification
   benefit is how much lower that combined risk is than simply averaging the individual
   stocks' risk (because the stocks don't all move together).
