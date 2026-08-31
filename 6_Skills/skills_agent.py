"""
skills_agent.py
===============

A SKILLS workflow over on-disk SKILL.md files, with live Yahoo Finance data.
One agent (GPT-4o), no graph of agents -- this is the Skills pattern.

DESIGN: plan -> execute -> interpret (deterministic, cannot loop)
----------------------------------------------------------------
The model makes exactly TWO decisions; the loading and running in between is plain
Python, so there is no agentic loop to get stuck in:

  1. PLAN     (LLM call 1): given the table of contents (skill names + descriptions),
              decide which skill(s) answer the question and extract each one's args.
  2. EXECUTE  (code): for each chosen skill -- load its full SKILL.md (Level 2) and
              run its bundled script (Level 3, live Yahoo Finance data).
  3. INTERPRET(LLM call 2): given the loaded instructions + the script results, write
              the plain-language answer.

Progressive disclosure still holds: only the chosen skills' bodies are ever loaded.

RUN
---
    pip install langchain-openai yfinance pandas numpy
    export OPENAI_API_KEY=sk-...
    python skills_agent.py "How risky is INFY.NS?"
    python skills_agent.py "Compare the risk of INFY.NS and TCS.NS"
"""

from __future__ import annotations

import json
import os
from dotenv import load_dotenv
load_dotenv(override=True)  # load .env if present, but don't override existing
import sys

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import skill_loader as sl


def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


class SkillsAgent:
    def __init__(self, llm=None, skills_dir=sl.SKILLS_DIR, runner=None):
        self.llm = llm or get_llm()
        self.skills = sl.discover_skills(skills_dir)     # Level 1: names + descriptions
        self.loaded: dict[str, str] = {}                 # bodies loaded on demand (Level 2)
        self.runner = runner or sl.run_script

    def _toc(self) -> str:
        return "\n".join(f"- {n}: {s['description']}" for n, s in self.skills.items())

    # ---- 1. PLAN -------------------------------------------------------------
    def _plan(self, query: str) -> list[dict]:
        system = SystemMessage(content=(
            "You are a market-risk assistant. Available skills:\n"
            f"{self._toc()}\n\n"
            "Decide which skill(s) answer the user's question and extract each one's "
            "arguments. Reply with ONE JSON object and nothing else:\n"
            '{"calls":[{"skill":"<name>","args":{...}}]}\n\n'
            "Argument shapes:\n"
            '  risk_score:      {"ticker":"INFY.NS"}\n'
            '  market_snapshot: {"ticker":"INFY.NS"}\n'
            '  portfolio_risk:  {"tickers":"TCS.NS,INFY.NS","weights":"0.5,0.5"}\n\n'
            "To compare several stocks, include one risk_score call per stock. "
            "Use the exact skill names from the list."
        ))
        plan = _extract_json(self.llm.invoke([system, HumanMessage(content=query)]).content)
        return plan.get("calls", []) if isinstance(plan, dict) else []

    # ---- 2. EXECUTE (plain code -- no model, no loop) ------------------------
    def _execute(self, calls: list[dict]) -> list[dict]:
        executed = []
        for c in calls:
            name, args = c.get("skill"), c.get("args", {})
            if name not in self.skills:
                executed.append({"skill": name, "error": "unknown skill"})
                continue
            if name not in self.loaded:
                self.loaded[name] = sl.load_skill_body(self.skills[name])    # Level 2
            executed.append({"skill": name, "args": args,
                             "result": self.runner(self.skills[name], args)})  # Level 3
        return executed

    # ---- 3. INTERPRET --------------------------------------------------------
    def _interpret(self, query: str, executed: list[dict]) -> str:
        bodies = "\n\n".join(f"[{n} instructions]\n{b}" for n, b in self.loaded.items())
        results = "\n".join(json.dumps(e) for e in executed) or "(no results)"
        system = SystemMessage(content=(
            "You are a market-risk assistant. Using the skill instructions and script "
            "results below, answer the user's question in clear, plain language. Briefly "
            "explain the key numbers. This is analysis, not financial advice.\n\n"
            f"{bodies}\n\nScript results:\n{results}"
        ))
        return self.llm.invoke([system, HumanMessage(content=query)]).content.strip()

    def run(self, query: str) -> dict:
        calls = self._plan(query)
        if not calls:
            return {"answer": "I couldn't map that to any of my skills "
                              f"({list(self.skills)}). Try naming a ticker, e.g. INFY.NS.",
                    "loaded": [], "executed": []}
        executed = self._execute(calls)
        answer = self._interpret(query, executed)
        return {"answer": answer, "loaded": list(self.loaded), "executed": executed}


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running.")

    query = " ".join(sys.argv[1:]) or "How risky is INFY.NS right now?"
    result = SkillsAgent().run(query)

    print("\n--- skills loaded ---", result["loaded"])
    print("\n--- what ran ---")
    for e in result["executed"]:
        print(" ", json.dumps(e))
    print("\n--- answer ---\n", result["answer"])
