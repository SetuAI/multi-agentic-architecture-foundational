"""
router_foundational.py
======================

A FOUNDATIONAL, runnable example of the ROUTER pattern, built with LangGraph and
OpenAI GPT-4o. This is the companion to sub_agents_foundational.py -- read the two
side by side, because the whole lesson is in the difference between them.

--------------------------------------------------------------------------
THE ONE DIFFERENCE FROM THE SUB-AGENT FILE
--------------------------------------------------------------------------
Sub-agent (supervisor): consults specialists ONE AT A TIME in a loop. Each result
comes BACK to the supervisor, which reads it and decides again -- it can re-delegate.

Router: a classifier decides UP FRONT which specialists are relevant, runs them ALL
AT ONCE (in parallel), and merges their outputs ONCE. There is NO loop and NO
supervisor sitting between the specialists. One pass, then done.

    classify  ->  fan out to the chosen specialists (in parallel)  ->  merge  ->  end

Domain: a mutual-fund assistant with three verticals -- performance, holdings, tax.
A question may touch one, two, or all three, and the router runs ONLY the relevant
ones. That selectivity is the router's job.

--------------------------------------------------------------------------
WHAT IS LEFT OUT (on purpose)
--------------------------------------------------------------------------
- No retrieval / RAG. The specialists return small canned strings, so the only thing
  on screen is the control flow -- the parallel fan-out and the single merge.
- The classifier and the merge are real GPT-4o calls (the two real decision points);
  the specialists are stubs.

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
    pip install langgraph langchain-openai
    export OPENAI_API_KEY=sk-...
    python router_foundational.py
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv(override=True)  # load .env if present, but don't override existing env vars

import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# The three verticals this assistant can route to. Each name is also its graph node.
VERTICALS = ["performance", "holdings", "tax"]


# ===========================================================================
# STATE
# ---------------------------------------------------------------------------
# `findings` is special: several specialists write to it AT THE SAME TIME (they run
# in parallel). LangGraph needs to know how to combine those concurrent writes, so we
# annotate it with a reducer -- operator.add -- which concatenates the lists. Without
# the reducer, parallel writes to the same key would raise an error.
# ===========================================================================
class State(TypedDict):
    request: str                              # the user's question
    selected: list                            # verticals the classifier chose
    findings: Annotated[list, operator.add]   # (name, text) pairs; merged across branches
    final_answer: str                         # the merged answer


# ===========================================================================
# CLASSIFIER  (runs ONCE, up front -- this is the router's whole job)
# ===========================================================================
def classify(state: State, llm) -> dict:
    """Decide which verticals the question touches, and return that subset. This is
    the routing decision. It happens once; it does not loop."""
    system = SystemMessage(content=(
        "You route a mutual-fund question to the relevant analysts. "
        f"The analysts are: {VERTICALS}. "
        "List ONLY the analyst names needed to answer, comma-separated. "
        "If unsure, list all of them."
    ))
    human = HumanMessage(content=state["request"])
    raw = llm.invoke([system, human]).content.lower()

    selected = [v for v in VERTICALS if v in raw]
    if not selected:                 # never dispatch nothing -- fall back to all
        selected = list(VERTICALS)
    return {"selected": selected}


# ===========================================================================
# SPECIALISTS  (stubs -- canned strings so the parallelism stays the star)
# ---------------------------------------------------------------------------
# Each returns a ONE-ITEM list. The operator.add reducer on `findings` concatenates
# these as the parallel branches finish.
# ===========================================================================
CANNED = {
    "performance": "Performance: 1Y 22.4%, 3Y 16.1% CAGR; beats the Nifty 100 benchmark over 3Y and 5Y.",
    "holdings":    "Holdings: top positions HDFC Bank, ICICI Bank, Reliance; Financials 34%, IT 15%; top-10 concentration 48%.",
    "tax":         "Tax: equity LTCG 12.5% above Rs 1.25L after 1 year; STCG 20%; exit load 1% if redeemed within 365 days.",
}


def performance_agent(state: State) -> dict:
    return {"findings": [("performance", CANNED["performance"])]}


def holdings_agent(state: State) -> dict:
    return {"findings": [("holdings", CANNED["holdings"])]}


def tax_agent(state: State) -> dict:
    return {"findings": [("tax", CANNED["tax"])]}


# ===========================================================================
# MERGE  (runs ONCE, after all parallel branches finish -- the fan-in)
# ===========================================================================
def merge(state: State, llm) -> dict:
    """Combine the findings from whichever specialists ran into a single answer."""
    system = SystemMessage(content=(
        "Combine the analyst notes into one short, direct answer to the user's "
        "question. Use only what the notes provide. 3-5 sentences."
    ))
    human = HumanMessage(content=(
        f"Question: {state['request']}\n"
        "Notes:\n" + "\n".join(f"- {name}: {text}" for name, text in state["findings"])
    ))
    return {"final_answer": llm.invoke([system, human]).content.strip()}


# ===========================================================================
# ROUTING + GRAPH WIRING
# ===========================================================================
def dispatch(state: State) -> list:
    """Return the LIST of specialist nodes to run. Returning a LIST (not one name) is
    exactly what makes LangGraph run those branches IN PARALLEL."""
    return state["selected"]


def build_graph(llm=None):
    llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)

    g = StateGraph(State)
    g.add_node("classify", lambda s: classify(s, llm))
    g.add_node("performance", performance_agent)
    g.add_node("holdings", holdings_agent)
    g.add_node("tax", tax_agent)
    g.add_node("merge", lambda s: merge(s, llm))

    g.add_edge(START, "classify")

    # Fan-out: the classifier's chosen verticals all run in one parallel step.
    g.add_conditional_edges("classify", dispatch, {v: v for v in VERTICALS})

    # Fan-in: every specialist flows into the single merge. merge waits for all the
    # branches that actually ran, then executes ONCE. There is no edge back to a
    # supervisor -- that missing back-edge is the router vs sub-agent difference.
    for v in VERTICALS:
        g.add_edge(v, "merge")

    g.add_edge("merge", END)
    return g.compile()


# ===========================================================================
# RUN IT
# ===========================================================================
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running.")

    app = build_graph()

    # This question touches performance and tax, but NOT holdings -- so the router
    # should dispatch only two of the three specialists.
    initial: State = {
        "request": "How has the fund performed lately, and what tax will I pay if I redeem now?",
        "selected": [], "findings": [], "final_answer": "",
    }

    for event in app.stream(initial):
        for node, update in event.items():
            print(f"\n=== {node} ===")
            if node == "classify":
                print("router selected ->", update["selected"])
            elif node == "merge":
                print(update["final_answer"])
            else:
                print(update["findings"][0][1])   # the canned note this specialist added
