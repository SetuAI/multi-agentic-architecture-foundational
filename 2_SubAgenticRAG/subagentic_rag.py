"""
subagentic_rag.py
=================

A SUB-AGENTIC RAG over four quarterly earnings-call transcripts of a fictional
company, Nimbus Renewables Ltd. Built with LangGraph + a local Ollama model.

This is the applied sequel to sub_agents_foundational.py. The supervisor loop is
the SAME idea; the two things that change are:
  - each specialist now RETRIEVES its own evidence from a vector store (that is the
    "RAG" part) instead of reading a hard-coded data slice, and
  - the run is wrapped in guardrails and a human-in-the-loop review gate.

--------------------------------------------------------------------------
THE FLOW (top to bottom)
--------------------------------------------------------------------------
    user request
        -> guard_input      : injection + advice + topic checks
             if unsafe/off-topic -> blocked_response -> END
        -> supervisor  <-------------------+           (decides next specialist)
             |                             |
             v                             |
        a specialist (financials/outlook/risk): retrieves, reasons, returns finding
             |_____________________________|           (result returns to supervisor)
        -> synthesise       : combine findings into a draft answer
        -> guard_output     : grounding check on the draft
        -> human_review     : a person approves / edits / rejects  (HITL)
        -> END

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
    # 1. install and start Ollama, then pull the models:
    ollama pull qwen3:8b
    ollama pull nomic-embed-text
    # 2. install python deps:
    pip install langgraph langchain-ollama langchain-text-splitters
    # 3. run:
    python subagentic_rag.py
"""

from __future__ import annotations

import os
import re
from typing import TypedDict, Callable

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage

import guardrails
from retrieval import build_vectorstore, make_retriever

COMPANY = "Nimbus Renewables"

# Chat model name; override with OLLAMA_MODEL. qwen3:8b is a strong small tool-use
# model; qwen3:30b-a3b is the sweet spot if the machine has 16-24 GB.
CHAT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")


def get_llm():
    """Real Ollama chat client, imported lazily so tests can inject a fake."""
    from langchain_ollama import ChatOllama
    return ChatOllama(model=CHAT_MODEL, temperature=0)


# ===========================================================================
# STATE
# ---------------------------------------------------------------------------
# One record travelling through the graph. As in the foundational file, this is
# the SUPERVISOR's view; the specialists stay stateless.
# ===========================================================================
class State(TypedDict):
    request: str          # the user's question
    findings: dict        # specialist name -> its distilled finding
    sources: list         # quarters/documents the evidence came from
    next_step: str        # supervisor decision: a specialist name or "DONE"
    final_answer: str     # draft, then approved, answer
    steps: int            # loop counter (guardrail against runaway)
    blocked: bool         # set by the input guardrail
    block_reason: str
    flags: list           # notes for the human reviewer (advice / grounding)
    approved: bool        # set by the human reviewer


# Each specialist: (name, focused brief, retrieval query).
SPECIALISTS: dict[str, tuple[str, str]] = {
    "financials": (
        "Track revenue, EBITDA margin, and profit across the quarters and describe the trend.",
        "revenue EBITDA margin profit growth across quarters",
    ),
    "outlook": (
        "Track how management's guidance and forward outlook changed across the quarters.",
        "guidance outlook forward-looking growth expectations next year",
    ),
    "risk": (
        "Surface the key risks, setbacks, delays, and receivable or regulatory concerns.",
        "risk delay receivable discom regulatory input cost setback",
    ),
}

MAX_STEPS = 6


# ===========================================================================
# INPUT GUARDRAIL
# ===========================================================================
def guard_input(state: State, llm) -> dict:
    """Run the input guardrails before any agent work happens."""
    q = state["request"]
    flags: list[str] = []

    # (a) hard blocks: injection attempts
    injection = guardrails.check_injection(q)
    if injection:
        return {"blocked": True, "block_reason": injection, "next_step": "BLOCK"}

    # (b) hard block: off-topic
    on_topic, reason = guardrails.check_topic(q, llm, COMPANY)
    if not on_topic:
        return {"blocked": True, "block_reason": reason, "next_step": "BLOCK"}

    # (c) soft flag: personalised-advice style question. We do NOT block it — we flag
    #     it so the human reviewer sees it and a disclaimer is added downstream.
    if guardrails.check_investment_advice(q):
        flags.append(
            "Query asks for buy/sell/hold advice. This assistant provides analysis "
            "only, not investment advice. Human review required; attach a disclaimer."
        )

    return {"blocked": False, "flags": flags}


def blocked_response(state: State) -> dict:
    """Produce a polite refusal when the input guardrail blocked the request."""
    return {
        "final_answer": (
            f"I can only help with analysis of {COMPANY}'s quarterly earnings, and I "
            f"can't proceed with this request ({state['block_reason']})."
        ),
        "approved": True,
    }


# ===========================================================================
# SUPERVISOR  (same decide-and-loop logic as the foundational file)
# ===========================================================================
def supervisor(state: State, llm) -> dict:
    consulted = list(state["findings"].keys())
    remaining = [s for s in SPECIALISTS if s not in consulted]

    if state["steps"] >= MAX_STEPS or not remaining:
        return {"next_step": "DONE"}

    system = SystemMessage(content=(
        f"You are a supervisor coordinating {COMPANY} earnings analysts. Given the "
        "request and findings so far, decide the SINGLE next analyst to consult, or "
        "answer DONE when you have enough for a complete answer.\n"
        f"Analysts still unconsulted: {remaining}\n"
        "Reply with EXACTLY one word: one analyst name from that list, or DONE."
    ))
    human = HumanMessage(content=(
        f"Request: {state['request']}\n"
        f"Findings so far: {list(state['findings'].keys()) or 'none yet'}"
    ))
    # Strip any <think> block, then read the decision as whole words. Prefer an
    # explicit specialist name; fall back to DONE, then to the next specialist -- so
    # a small model's messy output never stalls the loop.
    decision = guardrails.strip_reasoning(llm.invoke([system, human]).content).lower()
    tokens = re.findall(r"[a-z]+", decision)
    choice = next((s for s in remaining if s in tokens), None)
    if choice is None:
        choice = "DONE" if "done" in tokens else remaining[0]
    return {"next_step": choice, "steps": state["steps"] + 1}


# ===========================================================================
# SPECIALISTS  (each retrieves its own evidence — the "sub-agentic RAG" part)
# ===========================================================================
def run_specialist(state: State, name: str, llm, retrieve: Callable) -> dict:
    """Retrieve evidence for this specialist's focus, reason over it, return one
    finding. The specialist sees ONLY the chunks its own query pulled back — not the
    whole corpus, not the other specialists' findings."""
    brief, query = SPECIALISTS[name]
    hits = retrieve(query, k=4)                     # [(chunk_text, source), ...]
    context = "\n\n".join(f"[{source}] {text}" for text, source in hits)

    system = SystemMessage(content=(
        f"You are a {COMPANY} {name} analyst. {brief} "
        "Use ONLY the evidence provided. Cite the quarter (e.g. Q3 FY26) for each point. "
        "If the evidence doesn't cover something, say so. Be concise: 3-4 sentences."
    ))
    human = HumanMessage(content=f"Evidence:\n{context}")
    finding = guardrails.strip_reasoning(llm.invoke([system, human]).content)

    new_sources = state["sources"] + [s for _, s in hits]
    return {
        "findings": {**state["findings"], name: finding},
        "sources": sorted(set(new_sources)),
    }


# ===========================================================================
# SYNTHESIS + OUTPUT GUARDRAIL
# ===========================================================================
def synthesise(state: State, llm) -> dict:
    system = SystemMessage(content=(
        "You are the lead analyst. Combine the specialists' findings into a clear, "
        "balanced answer to the user's request. Cite quarters. Do not add figures that "
        "are not in the findings. 5-7 sentences."
    ))
    human = HumanMessage(content=(
        f"Request: {state['request']}\n"
        "Findings:\n"
        + "\n".join(f"- {name}: {text}" for name, text in state["findings"].items())
    ))
    return {"final_answer": guardrails.strip_reasoning(llm.invoke([system, human]).content)}


def guard_output(state: State) -> dict:
    """Grounding check on the draft answer; append any flag for the human reviewer."""
    flags = list(state.get("flags", []))
    reference = " ".join(state["findings"].values())
    ok, reason = guardrails.check_grounding(state["final_answer"], reference)
    if not ok:
        flags.append(f"Grounding check flagged: {reason}")
    return {"flags": flags}


# ===========================================================================
# HUMAN-IN-THE-LOOP
# ---------------------------------------------------------------------------
# The graph PAUSES here. `interrupt(...)` stops execution and surfaces the draft,
# the sources, and any guardrail flags to the caller. Nothing is delivered until a
# human resumes the run with a decision: approve, edit, or reject.
# ===========================================================================
def human_review(state: State) -> dict:
    decision = interrupt({
        "draft_answer": state["final_answer"],
        "sources": state["sources"],
        "flags": state["flags"],
        "instructions": "Reply with {'action': 'approve'|'edit'|'reject', "
                        "'edited_answer': '...'} (edited_answer only for 'edit').",
    })
    action = (decision or {}).get("action", "approve")
    if action == "reject":
        return {"final_answer": "[Draft rejected by reviewer; nothing delivered.]",
                "approved": False}
    if action == "edit":
        return {"final_answer": decision.get("edited_answer", state["final_answer"]),
                "approved": True}
    return {"approved": True}


# ===========================================================================
# ROUTING + GRAPH WIRING
# ===========================================================================
def _route_after_input(state: State) -> str:
    return "blocked_response" if state["blocked"] else "supervisor"


def _route_after_supervisor(state: State) -> str:
    return "synthesise" if state["next_step"] == "DONE" else state["next_step"]


def build_app(llm=None, retrieve: Callable | None = None):
    """Assemble the graph. `llm` and `retrieve` are injected so the same code runs
    with real Ollama in production and fakes in tests."""
    llm = llm or get_llm()
    if retrieve is None:
        retrieve = make_retriever(build_vectorstore())

    g = StateGraph(State)

    # Nodes. Small lambdas bind the injected llm/retrieve into each node function.
    g.add_node("guard_input", lambda s: guard_input(s, llm))
    g.add_node("blocked_response", blocked_response)
    g.add_node("supervisor", lambda s: supervisor(s, llm))
    for name in SPECIALISTS:
        g.add_node(name, (lambda n: lambda s: run_specialist(s, n, llm, retrieve))(name))
    g.add_node("synthesise", lambda s: synthesise(s, llm))
    g.add_node("guard_output", guard_output)
    g.add_node("human_review", human_review)

    # Edges.
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", _route_after_input,
                            {"blocked_response": "blocked_response", "supervisor": "supervisor"})
    g.add_edge("blocked_response", END)

    g.add_conditional_edges("supervisor", _route_after_supervisor,
                            {**{n: n for n in SPECIALISTS}, "synthesise": "synthesise"})
    for name in SPECIALISTS:                      # each specialist returns to the boss
        g.add_edge(name, "supervisor")

    g.add_edge("synthesise", "guard_output")
    g.add_edge("guard_output", "human_review")
    g.add_edge("human_review", END)

    # A checkpointer is required for interrupt()/resume to work.
    return g.compile(checkpointer=MemorySaver())


# ===========================================================================
# RUN IT
# ===========================================================================
if __name__ == "__main__":
    app = build_app()
    config = {"configurable": {"thread_id": "session-1"}}

    initial: State = {
        "request": "How did Nimbus Renewables' margins and outlook change through "
                   "FY26, and what were the main risks?",
        "findings": {}, "sources": [], "next_step": "", "final_answer": "",
        "steps": 0, "blocked": False, "block_reason": "", "flags": [], "approved": False,
    }

    # Run until the graph pauses at human_review (or ends early if blocked).
    result = app.invoke(initial, config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n===== HUMAN REVIEW REQUIRED =====")
        print("\nDraft answer:\n", payload["draft_answer"])
        print("\nSources:", payload["sources"])
        print("Flags:", payload["flags"] or "none")

        choice = input("\nApprove this answer? [y = approve / e = edit / r = reject]: ").strip().lower()
        if choice == "e":
            edited = input("Enter the edited answer:\n")
            decision = {"action": "edit", "edited_answer": edited}
        elif choice == "r":
            decision = {"action": "reject"}
        else:
            decision = {"action": "approve"}

        # Resume the paused graph with the human's decision.
        final = app.invoke(Command(resume=decision), config)
        print("\n===== DELIVERED =====\n", final["final_answer"])
    else:
        # Blocked at the input guardrail.
        print("\n===== RESPONSE =====\n", result["final_answer"])