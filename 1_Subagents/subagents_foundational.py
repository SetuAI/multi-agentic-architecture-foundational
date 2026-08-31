"""
sub_agents_foundational.py
==========================

A FOUNDATIONAL, runnable example of the SUB-AGENT pattern
(also called the supervisor / orchestrator-worker pattern),
built with LangGraph and OpenAI GPT-4o.

--------------------------------------------------------------------------
WHAT THIS TEACHES
--------------------------------------------------------------------------
One SUPERVISOR coordinates several specialised SUB-AGENTS. On each turn the
supervisor:

    1. reads the user's request,
    2. decides which ONE specialist to consult next,
    3. hands that specialist a FOCUSED task,
    4. reads the finding that comes BACK,
    5. decides again -- consult another specialist, or stop and synthesise.

Steps 4 -> 5 are the heart of the pattern. Results return to the supervisor,
and the supervisor RE-DELEGATES based on what came back. That "come back and
decide" loop is exactly what separates a supervisor from a one-pass router
(a router fans out once, merges, and stops -- it never loops).

Domain: mutual-fund analysis. A user asks for a rounded view of a fund, and
the supervisor pulls together four specialists -- performance, holdings, risk,
and fees -- then composes a balanced summary.

--------------------------------------------------------------------------
WHAT IS DELIBERATELY LEFT OUT (and why)
--------------------------------------------------------------------------
- No retrieval / RAG / vector store. The fund data is a small hard-coded dict
  below, so the ONLY thing on screen is the control flow. The next project
  swaps this synthetic data for real retrieval -> a "sub-agentic RAG".
- The supervisor consults specialists ONE AT A TIME and re-decides after each,
  so the loop is visible. Production supervisors often fan several out in
  parallel; that is an optimisation on top of the same idea.

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
    pip install langgraph langchain-openai
    export OPENAI_API_KEY=sk-...
    python sub_agents_foundational.py
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv(override=True)  # read .env if present
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ===========================================================================
# 0. THE MODEL
# ---------------------------------------------------------------------------
# One shared GPT-4o client. Every agent below -- supervisor and specialists --
# is just this same model driven by a DIFFERENT system prompt. In this pattern,
# "specialisation" means a different prompt (and a different slice of data),
# not a different model.
# ===========================================================================
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ===========================================================================
# 1. SYNTHETIC FUND DATA
# ---------------------------------------------------------------------------
# Stands in for what a real system would pull from a database or documents.
# Kept tiny on purpose -- this file is about the pattern, not the data source.
# Each specialist is allowed to see ONLY its own key (see section 4).
# ===========================================================================
FUND_DATA = {
    "Bluechip Equity Fund": {
        "performance": (
            "1Y: 22.4%. 3Y: 16.1% CAGR. 5Y: 14.8% CAGR. "
            "Benchmark (Nifty 100 TRI) 5Y: 13.2%. Beats benchmark over 3Y and 5Y."
        ),
        "holdings": (
            "Top holdings: HDFC Bank 9.1%, ICICI Bank 7.8%, Reliance 6.4%, "
            "Infosys 5.2%, L&T 4.0%. Sectors: Financials 34%, IT 15%, Energy 11%. "
            "Top-10 concentration: 48%."
        ),
        "risk": (
            "Std deviation 13.2% (category avg 14.1%). Sharpe 1.02. "
            "Max drawdown over 5Y: -18.6%. Beta 0.94. Lower risk than category average."
        ),
        "fees": (
            "Expense ratio 1.12% (direct plan 0.68%). Exit load 1% if redeemed within "
            "365 days. Equity taxation: LTCG 12.5% above Rs 1.25L after 1Y; STCG 20%."
        ),
    }
}

# The roster of specialists the supervisor may consult. Each name is also the
# data key it reads AND the graph node it maps to -- kept identical so routing
# stays trivial (see section 6).
SPECIALISTS = ["performance", "holdings", "risk", "fees"]

# A hard cap so the supervisor loop can never run forever. A guardrail, not the
# normal exit -- the normal exit is "all specialists consulted" (see section 3).
MAX_STEPS = 8


# ===========================================================================
# 2. SHARED STATE
# ---------------------------------------------------------------------------
# The single record that travels through the graph. This state is the
# SUPERVISOR's view of the world -- the request, and the findings gathered so
# far. The specialists are STATELESS: they keep nothing between calls. They
# read a focused task, return one finding, and forget. All the memory lives
# here, with the supervisor.
# ===========================================================================
class State(TypedDict):
    request: str        # what the user asked
    fund_name: str      # the fund under analysis
    findings: dict      # specialist name -> its distilled finding (grows over time)
    next_step: str      # the supervisor's routing decision: a specialist name or "DONE"
    final_answer: str   # the synthesised answer, filled in at the very end
    steps: int          # iteration counter (feeds the MAX_STEPS guardrail)


# ===========================================================================
# 3. THE SUPERVISOR
# ---------------------------------------------------------------------------
# Runs every time control returns to the top of the loop. Its ONLY job is to
# decide the next action from what it knows so far:
#     - consult one more specialist  -> return that specialist's name, or
#     - stop and synthesise          -> return "DONE".
# It does NOT do any analysis itself. It reads results and delegates.
# ===========================================================================
def supervisor(state: State) -> dict:
    consulted = list(state["findings"].keys())               # who we've heard from
    remaining = [s for s in SPECIALISTS if s not in consulted]

    # Guardrail exit: stop if we've hit the step cap or already consulted everyone.
    if state["steps"] >= MAX_STEPS or not remaining:
        return {"next_step": "DONE"}

    # Ask GPT-4o to pick the next specialist, given the request and the findings
    # collected so far. THIS is the supervisor "reading results and deciding".
    system = SystemMessage(content=(
        "You are a supervisor coordinating mutual-fund analysts. Given the user's "
        "request and the findings gathered so far, decide the SINGLE next analyst to "
        "consult, or answer DONE if you already have enough for a rounded view.\n"
        f"Analysts still unconsulted: {remaining}\n"
        "Reply with EXACTLY one word: one analyst name from that list, or DONE."
    ))
    human = HumanMessage(content=(
        f"Request: {state['request']}\n"
        f"Fund: {state['fund_name']}\n"
        f"Findings so far: {state['findings'] or 'none yet'}"
    ))
    decision = llm.invoke([system, human]).content.strip().lower()

    # Defensive parse: accept only a known specialist or DONE. If the model says
    # anything unexpected, fall back to the next unconsulted specialist so the
    # graph always makes forward progress.
    if "done" in decision:
        choice = "DONE"
    else:
        choice = next((s for s in remaining if s in decision), remaining[0])

    return {"next_step": choice, "steps": state["steps"] + 1}


# ===========================================================================
# 4. THE SPECIALISTS (the sub-agents)
# ---------------------------------------------------------------------------
# Each is GPT-4o with a focused system prompt and access to ONLY its slice of
# the fund data. Note what a specialist does NOT receive: the full request
# history, the other specialists' findings, or the supervisor's reasoning. It
# gets a focused task, returns one distilled finding, and keeps no memory.
# That is what "stateless sub-agent" means in practice.
# ===========================================================================
def _consult(state: State, role: str, brief: str, data_key: str) -> dict:
    """Run one specialist.

    role      -- its identity, used in the system prompt
    brief     -- the focused instruction for this specialist
    data_key  -- selects the ONLY slice of data this specialist may see
    """
    # Two lookups: first grab this fund's record, then pull out just one field from
    # it. data_key decides which field (e.g. "risk"), so the specialist receives only
    # its own slice of the data -- not the full fund record.
    data_slice = FUND_DATA[state["fund_name"]][data_key]

    system = SystemMessage(content=(
        f"You are a mutual-fund {role}. {brief} "
        "Base your answer ONLY on the data provided. Be concise: 2-3 sentences."
    ))
    human = HumanMessage(content=f"Fund: {state['fund_name']}\nData:\n{data_slice}")
    finding = llm.invoke([system, human]).content.strip()

    # Merge the finding into shared state. Writing it into `findings` is how the
    # result travels BACK to the supervisor for the next decision.
    return {"findings": {**state["findings"], data_key: finding}}


def performance_agent(state: State) -> dict:
    """Judges returns versus the benchmark and consistency."""
    return _consult(state, "performance analyst",
                    "Assess returns versus the benchmark and consistency.",
                    "performance")


def holdings_agent(state: State) -> dict:
    """Reads portfolio composition and concentration."""
    return _consult(state, "portfolio analyst",
                    "Comment on holdings, sector mix, and concentration risk.",
                    "holdings")


def risk_agent(state: State) -> dict:
    """Interprets volatility, drawdown, and risk-adjusted return."""
    return _consult(state, "risk analyst",
                    "Interpret volatility, drawdown, and risk-adjusted return.",
                    "risk")


def fees_agent(state: State) -> dict:
    """Explains cost and tax impact."""
    return _consult(state, "fees and tax analyst",
                    "Explain expense ratio, exit load, and tax on redemption.",
                    "fees")


# ===========================================================================
# 5. SYNTHESIS
# ---------------------------------------------------------------------------
# Reached once the supervisor says DONE. The supervisor now holds every finding
# and composes the final, balanced answer. This is the ONE place the separate
# findings are combined into a single response.
# ===========================================================================
def synthesise(state: State) -> dict:
    system = SystemMessage(content=(
        "You are the lead advisor. Combine the analysts' findings into a short, "
        "balanced view of the fund for an investor. 4-6 sentences. Do not invent data."
    ))
    human = HumanMessage(content=(
        f"Request: {state['request']}\n"
        f"Fund: {state['fund_name']}\n"
        "Analyst findings:\n"
        + "\n".join(f"- {name}: {text}" for name, text in state["findings"].items())
    ))
    answer = llm.invoke([system, human]).content.strip()
    return {"final_answer": answer}


# ===========================================================================
# 6. ROUTING + GRAPH WIRING
# ---------------------------------------------------------------------------
# `route` reads the supervisor's decision and sends control to a specialist
# node or to synthesis. The wiring below has one crucial feature: every
# specialist edge goes BACK to the supervisor, not onward. That back-edge is
# the loop -- it is the sub-agent pattern drawn as a graph.
# ===========================================================================
def route(state: State) -> str:
    if state["next_step"] == "DONE":
        return "synthesise"
    return state["next_step"]        # a specialist node name


def build_graph():
    g = StateGraph(State)

    # One node per agent, plus a synthesis node.
    g.add_node("supervisor", supervisor)
    g.add_node("performance", performance_agent)
    g.add_node("holdings", holdings_agent)
    g.add_node("risk", risk_agent)
    g.add_node("fees", fees_agent)
    g.add_node("synthesise", synthesise)

    # Entry point.
    g.add_edge(START, "supervisor")

    # From the supervisor, branch to whichever specialist it chose, or to synthesis.
    g.add_conditional_edges(
        "supervisor",
        route,
        {
            "performance": "performance",
            "holdings": "holdings",
            "risk": "risk",
            "fees": "fees",
            "synthesise": "synthesise",
        },
    )

    # KEY: each specialist returns to the supervisor. Results come BACK; the
    # supervisor decides again. This is the loop.
    for specialist in SPECIALISTS:
        g.add_edge(specialist, "supervisor")

    # Synthesis ends the run.
    g.add_edge("synthesise", END)

    return g.compile()


# ===========================================================================
# 7. RUN IT
# ---------------------------------------------------------------------------
# Streaming the run lets students watch the supervisor and specialists fire in
# turn -- you can literally see the delegate -> return -> decide loop happen.
# ===========================================================================
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running.")

    app = build_graph()

    initial: State = {
        "request": "I'm considering this fund. Give me a rounded view before I invest.",
        "fund_name": "Bluechip Equity Fund",
        "findings": {},
        "next_step": "",
        "final_answer": "",
        "steps": 0,
    }

    for event in app.stream(initial):
        for node, update in event.items():
            print(f"\n=== {node} ===")
            if node == "supervisor":
                print("supervisor decided ->", update.get("next_step"))
            elif node == "synthesise":
                print(update.get("final_answer"))
            else:
                # A specialist just returned; show the finding it added.
                newest_key = list(update["findings"].keys())[-1]
                print(update["findings"][newest_key])