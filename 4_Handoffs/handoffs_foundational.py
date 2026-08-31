r"""
handoffs_foundational.py
========================

A FOUNDATIONAL, runnable example of the HANDOFFS pattern, built with LangGraph and
OpenAI GPT-4o. Companion to sub_agents_foundational.py and router_foundational.py.

--------------------------------------------------------------------------
WHAT MAKES HANDOFFS DIFFERENT
--------------------------------------------------------------------------
Sub-agents and Router have a coordinator that stays in charge. Handoffs has no
coordinator: the ACTIVE agent changes as the conversation moves through stages, and
each agent TRANSFERS control to the next when its part is done.

Three properties define the pattern, and this file shows all three:

  1. Control transfer  -- each stage agent "hands off" to the next by setting who is
     active next. (In a real system this is a tool the agent calls; here each agent
     sets a `handoff` field, and routing activates that stage.)
  2. State carried forward -- details collected in one stage are available in later
     stages. A MemorySaver checkpointer persists state across the user's turns.
  3. Sequential stage-gating -- a stage unlocks only after the previous one finishes.
     You cannot reach eligibility before intake is complete, or submission before the
     applicant is found eligible. The graph's structure enforces this.

Domain: a loan-application assistant.  intake -> eligibility -> submit

    START -> intake (collect details) --handoff--> eligibility (check)
              |                                        |
         [user turn]                           eligible? --> submit (confirm) --> END
                                                     \--> rejected --> END

--------------------------------------------------------------------------
HOW THE USER TURNS WORK
--------------------------------------------------------------------------
The graph runs as ONE session that PAUSES (via interrupt) whenever it needs input
from the applicant -- once in intake (to collect details) and once in submit (to
confirm). Between pauses, control is handed from agent to agent automatically.

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
    pip install langgraph langchain-openai
    export OPENAI_API_KEY=sk-...
    python handoffs_foundational.py
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv(override=True)  # load .env if present, but don't override existing env vars
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# The details intake must collect before it may hand off to eligibility.
REQUIRED = ["name", "monthly_income", "loan_amount", "tenure_months"]


# ===========================================================================
# STATE
# ---------------------------------------------------------------------------
# Persisted across the user's turns by the checkpointer. `handoff` is the field an
# agent sets to name who should be active next -- it is the handoff mechanism.
# ===========================================================================
class State(TypedDict):
    applicant: dict            # details collected in intake (carried into later stages)
    eligible: Optional[bool]   # set by eligibility
    reference: str             # set by submit on success
    handoff: str               # the stage the current agent hands control to
    reply: str                 # the active agent's message this turn


def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)


def _say(llm, role: str, instruction: str) -> str:
    """Each stage is its own agent with its own persona. This produces that agent's
    user-facing message. (The handoff decision itself is made in code below, not here.)"""
    system = SystemMessage(content=(
        f"You are the {role} agent in a loan-application assistant. {instruction} "
        "Keep it to one or two short, friendly sentences."
    ))
    return llm.invoke([system, HumanMessage(content="Write your message.")]).content.strip()


# ===========================================================================
# STAGE AGENTS
# ---------------------------------------------------------------------------
# Each agent does its stage's work, then HANDS OFF by setting `handoff`. Note that
# every agent reads from the same `applicant` state -- that is state carried forward.
# ===========================================================================
def intake(state: State, llm) -> dict:
    """Collect the applicant's details, then hand off to eligibility."""
    applicant = dict(state.get("applicant", {}))
    if set(REQUIRED) - applicant.keys():
        msg = _say(llm, "intake",
                   "Ask the applicant for their full name, monthly income in INR, the "
                   "loan amount they want in INR, and the tenure in months.")
        # Pause and wait for the applicant. The caller resumes with a dict of the fields.
        provided = interrupt({"agent": "intake", "message": msg, "need": REQUIRED})
        applicant.update(provided)

    # HANDOFF: intake is done -> eligibility becomes the active agent.
    return {"applicant": applicant, "handoff": "eligibility", "reply": "Intake complete."}


def eligibility(state: State, llm) -> dict:
    """Decide eligibility from the details intake collected, then hand off accordingly.
    This agent needs NO user input -- it runs and hands off in the same pass."""
    a = state["applicant"]
    emi = float(a["loan_amount"]) / float(a["tenure_months"])      # simplified, ignores interest
    eligible = emi <= 0.5 * float(a["monthly_income"])             # rule: EMI within 50% of income

    verdict = "eligible" if eligible else "not eligible"
    msg = _say(llm, "eligibility",
               f"Tell the applicant they are {verdict}. Their estimated EMI is about "
               f"{emi:.0f} against a monthly income of {a['monthly_income']}.")

    # HANDOFF depends on the result -- submit if eligible, otherwise rejected. This is
    # the stage-gate: submit is unreachable unless this agent hands off to it.
    return {"eligible": eligible,
            "handoff": "submit" if eligible else "rejected",
            "reply": msg}


def submit(state: State, llm) -> dict:
    """Confirm with the applicant and submit. Only reachable after an eligible verdict."""
    a = state["applicant"]
    msg = _say(llm, "submission",
               f"Ask {a['name']} to confirm they want to submit the application now. "
               "Tell them to reply yes or no.")
    confirm = interrupt({"agent": "submit", "message": msg})       # second user turn

    if str(confirm).strip().lower().startswith("y"):
        return {"reference": "LN-2026-00042", "handoff": "done",
                "reply": "Application submitted."}
    return {"handoff": "cancelled", "reply": "Submission cancelled."}


def rejected(state: State, llm) -> dict:
    """Terminal-ish stage for an ineligible applicant."""
    msg = _say(llm, "eligibility",
               "Politely explain the application can't proceed and suggest reducing the "
               "loan amount or extending the tenure.")
    return {"handoff": "done", "reply": msg}


# ===========================================================================
# ROUTING (the handoff mechanism) + GRAPH WIRING
# ---------------------------------------------------------------------------
# `route` reads the `handoff` field an agent set and activates that stage. The
# path maps below are also the gate: eligibility can hand off ONLY to submit or
# rejected; submit is reachable from nowhere else. Structure enforces the order.
# ===========================================================================
def route(state: State) -> str:
    return state["handoff"]


def build_graph(llm=None):
    llm = llm or get_llm()

    g = StateGraph(State)
    g.add_node("intake", lambda s: intake(s, llm))
    g.add_node("eligibility", lambda s: eligibility(s, llm))
    g.add_node("submit", lambda s: submit(s, llm))
    g.add_node("rejected", lambda s: rejected(s, llm))

    g.add_edge(START, "intake")                                    # the flow always starts at intake
    g.add_conditional_edges("intake", route, {"eligibility": "eligibility"})
    g.add_conditional_edges("eligibility", route,
                            {"submit": "submit", "rejected": "rejected"})
    g.add_conditional_edges("submit", route, {"done": END, "cancelled": END})
    g.add_edge("rejected", END)

    # A checkpointer is required so state survives the interrupt pauses (the user turns).
    return g.compile(checkpointer=MemorySaver())


# ===========================================================================
# RUN IT
# ===========================================================================
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running.")

    app = build_graph()
    config = {"configurable": {"thread_id": "loan-1"}}

    state: State = {"applicant": {}, "eligible": None, "reference": "",
                    "handoff": "", "reply": ""}

    result = app.invoke(state, config)                             # runs until the first pause (intake)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n[{payload['agent']} agent] {payload['message']}")

        if payload["agent"] == "intake":
            # Collect the required fields from the applicant.
            provided = {}
            for field in payload["need"]:
                provided[field] = input(f"  {field}: ").strip()
            # numbers as numbers
            for f in ("monthly_income", "loan_amount", "tenure_months"):
                provided[f] = float(provided[f])
            resume_value = provided
        else:  # submit confirmation
            resume_value = input("  your answer: ").strip()

        result = app.invoke(Command(resume=resume_value), config)  # hand back, continue to next pause/END

    # Finished.
    print("\n----- OUTCOME -----")
    if result.get("reference"):
        print(f"Eligible and submitted. Reference: {result['reference']}")
    elif result.get("eligible") is False:
        print("Not eligible.")
    else:
        print(result.get("reply", "Session ended."))
