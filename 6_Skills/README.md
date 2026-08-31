# Project: Agentic Skills — Live Market-Risk Analyzer

*A richer take on the Skills pattern. Still one agent (Skills has no graph of
agents), but the skills are real `SKILL.md` files on disk with bundled scripts that
pull **live Yahoo Finance data** and compute a **risk score** — and the agent
(GPT-4o) decides, step by step, which skill to load and run.*

---

## 1. What it does

Ask it a market-risk question and it answers using real data:

- *"How risky is INFY.NS?"* → loads `risk_score`, runs `risk.py`, which fetches ~1
  year of prices from Yahoo Finance and returns volatility, max drawdown, beta, and a
  0-100 risk score with a Low/Moderate/High band. The agent explains it in plain words.
- *"Give me a snapshot of TCS.NS"* → loads `market_snapshot`.
- *"How risky is a 40/30/30 portfolio of TCS.NS, INFY.NS, RELIANCE.NS?"* → loads
  `portfolio_risk`, which computes combined volatility and the diversification benefit.

## 2. Why this is the Skills pattern (and not the others)

One agent. No supervisor, no sub-agents, no graph — that is what makes it Skills. What
the agent gains is not helpers but **capabilities loaded on demand**. The interesting
part is that a skill here is not a paragraph of text; it is a folder with instructions
*and a runnable script*, exactly like real skill systems.

## 3. Three levels of progressive disclosure

This is the mechanism, and the code makes each level explicit (`skill_loader.py`):

1. **Level 1 — discover.** At startup the agent reads only the *frontmatter*
   (`name` + `description`) of each `SKILL.md`. Cheap; this is the table of contents.
2. **Level 2 — load.** When a question matches a skill, the agent loads that one
   skill's *full instructions*. The other skills' bodies never enter context.
3. **Level 3 — run.** The instructions tell the agent to run the bundled script; the
   agent executes it (fetching live data) and reads the result.

The offline tests confirm the isolation: a skill's `risk.py` instructions are not in
the agent's context until it explicitly loads that skill.

## 4. The agent loop (`skills_agent.py`)

Each step the agent sees the table of contents, any loaded instructions, and any
script results, then emits ONE action as JSON:

```
{"action":"load","skill":"risk_score"}
{"action":"run","skill":"risk_score","args":{"ticker":"INFY.NS"}}
{"action":"answer","text":"..."}
```

A typical run is `load → run → answer`. For a comparison it may run a skill more than
once before answering.

## 5. The skills (`skills/`)

| Skill | Script | What it computes (from Yahoo Finance) |
|---|---|---|
| `risk_score` | `risk.py` | annualized volatility, max drawdown, beta vs an index → 0-100 risk score |
| `market_snapshot` | `quote.py` | price, sector, market cap, P/E, 52-week range |
| `portfolio_risk` | `portfolio.py` | portfolio volatility and the diversification benefit |

The risk score weights volatility 40%, drawdown 35%, beta 25%. **Thresholds are
illustrative for teaching, not financial advice.**

## 6. Files

```
skills_agentic/
  skills/
    risk_score/       SKILL.md + risk.py
    market_snapshot/  SKILL.md + quote.py
    portfolio_risk/   SKILL.md + portfolio.py
  skill_loader.py     discovery + load-on-demand + run-script (the 3 levels)
  skills_agent.py     the GPT-4o agent loop
  README.md
```

## 7. How to run

```bash
pip install langchain-openai yfinance pandas numpy
export OPENAI_API_KEY=sk-...
python skills_agent.py "How risky is INFY.NS?"
```

Yahoo Finance access needs a live internet connection. Tickers use Yahoo format —
`.NS` suffix for NSE stocks (e.g. `RELIANCE.NS`), bare symbols for US (e.g. `AAPL`).

## 8. Where to take it next

- Add a skill that references a *third-level resource file* (e.g. a sector-risk
  reference table loaded only for certain questions) to show the deepest disclosure level.
- Cache Yahoo Finance responses so repeated questions don't re-fetch.
- Combine with sub-agents (a supervisor whose workers each carry skills) — the
  "deep agents" idea — if you want skills *inside* a true multi-agent system.
