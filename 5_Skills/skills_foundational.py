"""
skills_foundational.py
======================

A FOUNDATIONAL, runnable example of the SKILLS pattern, built with OpenAI GPT-4o.
Companion to sub_agents_foundational.py, router_foundational.py, and
handoffs_foundational.py -- and deliberately the odd one out.

--------------------------------------------------------------------------
WHY THIS ONE LOOKS DIFFERENT
--------------------------------------------------------------------------
The other three patterns are multi-agent: they wire several agents together in a
LangGraph graph. Skills is NOT. It is ONE agent, and there is no graph of agents to
build -- which is exactly why this file is plain Python, not LangGraph. A "skill" is
not an agent; it is a bundle of instructions the single agent LOADS when it needs it.

--------------------------------------------------------------------------
THE MECHANISM: PROGRESSIVE DISCLOSURE
--------------------------------------------------------------------------
1. At the start, the agent knows only the skill NAMES and one-line DESCRIPTIONS -- a
   table of contents. That is small and cheap.
2. A question arrives. Using only that table of contents, the agent picks the ONE
   skill it needs.
3. It LOADS that skill's FULL instructions into context, and answers using them. The
   other skills' full instructions are never loaded.

The trade-off (watch the ~context tokens grow): once a skill is loaded it STAYS in
context for the rest of the session. Ask about two skills and both sets of full
instructions now sit in context -- that is the "token bloat" cost of the pattern.

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
    pip install langchain-openai
    export OPENAI_API_KEY=sk-...
    python skills_foundational.py
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ===========================================================================
# THE SKILLS
# ---------------------------------------------------------------------------
# Each skill is a bundle: a name, a one-line description, and the full step-by-step
# instructions. A skill is NOT an agent -- it does nothing on its own. It is material
# the single agent reads when a question calls for it.
# ===========================================================================
SKILLS = {
    "sip-calculation": {
        "description": "Compute the future value and returns of a monthly SIP.",
        "instructions": (
            "To compute SIP future value:\n"
            "1. Monthly rate i = annual_return_percent / 12 / 100.\n"
            "2. n = number of monthly instalments (years * 12).\n"
            "3. FV = P * (((1+i)**n - 1) / i) * (1+i), where P is the monthly amount.\n"
            "4. Total invested = P * n.  Gain = FV - total invested.\n"
            "Report FV, total invested, and gain, rounded to the nearest rupee."
        ),
    },
    "tax-rules": {
        "description": "Explain tax on equity mutual-fund gains and redemptions.",
        "instructions": (
            "Equity mutual-fund taxation (India):\n"
            "- Long-term (held > 1 year): LTCG at 12.5% on gains above Rs 1.25 lakh per year.\n"
            "- Short-term (held <= 1 year): STCG at 20%.\n"
            "Decide which applies from the holding period, then estimate the tax on the stated gain."
        ),
    },
    "redemption-rules": {
        "description": "Explain exit load and redemption/settlement timelines.",
        "instructions": (
            "Redemption rules for equity funds:\n"
            "- Exit load: typically 1% if redeemed within 365 days of investment, otherwise nil.\n"
            "- Settlement: proceeds are credited around T+3 working days.\n"
            "Apply the exit load to the redemption amount if it falls within the load period."
        ),
    },
}

BASE = (
    "You are a mutual-fund assistant. Answer the user using the loaded skill "
    "instructions. If a needed skill is not loaded, say so plainly."
)


def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)


# ===========================================================================
# THE SINGLE AGENT
# ---------------------------------------------------------------------------
# `self.loaded` is the progressive-disclosure state: it starts empty and grows as
# skills are loaded. That growth is the whole story of the pattern -- the benefit
# (only load what you need) and the cost (loaded skills accumulate).
# ===========================================================================
class SkillsAgent:
    def __init__(self, llm, skills=SKILLS):
        self.llm = llm
        self.skills = skills
        self.loaded: dict[str, str] = {}     # name -> full instructions (grows over the session)

    def _toc(self) -> str:
        """The table of contents: names + one-line descriptions ONLY. This is all the
        agent knows before any skill is loaded."""
        return "\n".join(f"- {name}: {s['description']}" for name, s in self.skills.items())

    def _select(self, query: str) -> str | None:
        """STEP 1 (cheap): using ONLY the table of contents, pick the skill needed.
        Note this prompt contains descriptions, never full instructions."""
        system = SystemMessage(content=(
            "You have these skills (names and one-line descriptions only):\n"
            f"{self._toc()}\n\n"
            "Which ONE skill is needed to answer the user's question? "
            "Reply with exactly the skill name, or NONE."
        ))
        raw = self.llm.invoke([system, HumanMessage(content=query)]).content.strip().lower()
        return next((name for name in self.skills if name in raw), None)

    def _answer_system(self) -> str:
        """The system prompt used to ANSWER: base + table of contents + the FULL text of
        every skill loaded so far. This string is what grows as more skills load."""
        loaded_text = "\n\n".join(f"[{n}]\n{instr}" for n, instr in self.loaded.items()) \
            or "(none loaded yet)"
        return (f"{BASE}\n\nAvailable skills:\n{self._toc()}\n\n"
                f"Loaded skill instructions:\n{loaded_text}")

    def ask(self, query: str) -> dict:
        chosen = self._select(query)

        # STEP 2: load the chosen skill's FULL instructions on demand (once).
        if chosen and chosen not in self.loaded:
            self.loaded[chosen] = self.skills[chosen]["instructions"]

        # STEP 3: answer using base + table of contents + whatever is loaded.
        system = self._answer_system()
        answer = self.llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=query)]
        ).content.strip()

        return {
            "query": query,
            "chosen": chosen,
            "loaded": list(self.loaded.keys()),
            "context_tokens": len(system) // 4,      # rough proxy so growth is visible
            "answer": answer,
        }


# ===========================================================================
# RUN IT
# ===========================================================================
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running.")

    agent = SkillsAgent(get_llm())

    turns = [
        "If I invest 10000 rupees monthly for 5 years at 12% annual return, what will it grow to?",
        "And what tax will I pay on the gains if I redeem after those 5 years?",
    ]

    for q in turns:
        r = agent.ask(q)
        print(f"\n=== USER === {q}")
        print(f"skill picked : {r['chosen']}")
        print(f"loaded now   : {r['loaded']}")
        print(f"~context tok : {r['context_tokens']}")
        print(f"ANSWER       : {r['answer']}")

    print("\nAfter turn 2, BOTH skills' full instructions sit in context -- watch the "
          "~context tokens rise. That accumulation is the token-bloat trade-off.")
