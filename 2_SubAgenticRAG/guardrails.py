"""
guardrails.py
=============

Four guardrails, kept in one place so they are easy to find and reason about.
Three of them are plain rules (deterministic, no model needed) and one uses the
model. Each returns a simple result the graph can act on.

  1. check_injection(query)          -> str | None   (rule)   input safety
  2. check_investment_advice(query)  -> bool          (rule)   input safety
  3. check_topic(query, llm)         -> (bool, str)   (model)  input scope
  4. check_grounding(answer, ref)    -> (bool, str)   (rule)   output safety

Why each exists:
  - Injection: block obvious "ignore your instructions" style attacks before they
    reach the agents.
  - Investment advice: this assistant ANALYSES filings; it must not give personalised
    buy/sell advice (a real regulatory line for a financial firm). We don't block
    these outright — we flag them so a human reviews the answer and a disclaimer is
    attached.
  - Topic/scope: keep the assistant answering only about this company's earnings, not
    weather, politics, or unrelated firms.
  - Grounding: a cheap check that the final answer's figures actually appear in the
    retrieved evidence, catching invented numbers.
"""

from __future__ import annotations

import re

# --- 1. Prompt-injection (rule) -------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+.*instructions",
    r"you\s+are\s+now\s+",
    r"reveal\s+your\s+(system\s+)?(prompt|instructions)",
    r"system\s+prompt",
]


def check_injection(query: str) -> str | None:
    """Return a reason string if the query looks like a prompt-injection attempt,
    else None."""
    q = query.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, q):
            return "possible prompt-injection attempt"
    return None


# --- 2. Investment-advice detection (rule) --------------------------------------
_ADVICE_PATTERNS = [
    r"\bshould\s+i\s+(buy|sell|invest|hold)\b",
    r"\bis\s+(it|this)\s+a\s+good\s+(buy|investment|stock|bet)\b",
    r"\b(buy|sell)\s+(this|the)\s+(stock|share|fund)\b",
    r"\bhow\s+much\s+should\s+i\s+invest\b",
    r"\bwill\s+the\s+(stock|share|price)\s+(go\s+up|rise|fall|crash)\b",
]


def check_investment_advice(query: str) -> bool:
    """True if the query is asking for personalised buy/sell/hold advice."""
    q = query.lower()
    return any(re.search(pattern, q) for pattern in _ADVICE_PATTERNS)


# --- helper: clean reasoning-model output ---------------------------------------
def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models (e.g. qwen3) emit, so
    we parse and display only the final answer -- not the model's scratch thinking."""
    return re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE).strip()


# --- 3. Topic / scope (model) ---------------------------------------------------
def check_topic(query: str, llm, company: str = "Nimbus Renewables") -> tuple[bool, str]:
    """Ask the model whether the query is an earnings/financial-analysis question.

    We judge INTENT (is this about a company's quarterly performance?) rather than
    requiring the company's name, because this assistant only has one company loaded,
    and a small model shouldn't reject "the company" phrasing. Returns (on_topic,
    reason). The parse is deliberately permissive: it reads the FINAL yes/no in the
    cleaned answer and defaults to on-topic if neither is found, so a rambly small
    model doesn't wrongly block a valid question. (Safety is covered by the injection
    and advice guardrails, not by this scope check.)"""
    from langchain_core.messages import SystemMessage, HumanMessage

    system = SystemMessage(content=(
        "You decide whether a user question is about a company's quarterly financial "
        "performance -- its earnings, revenue, margins, profit, guidance, outlook, or "
        "risks. Answer with exactly YES or NO."
    ))
    human = HumanMessage(content=f"Question: {query}")
    cleaned = strip_reasoning(llm.invoke([system, human]).content).lower()

    verdicts = re.findall(r"\b(yes|no)\b", cleaned)
    verdict = verdicts[-1] if verdicts else "yes"      # permissive default: on-topic
    if verdict == "yes":
        return True, ""
    return False, f"question appears to be outside the scope of {company}'s earnings"


# --- 4. Output grounding (rule) -------------------------------------------------
def check_grounding(answer: str, reference: str) -> tuple[bool, str]:
    """Check that numeric figures in the answer appear in the retrieved evidence.

    This is a deliberately simple heuristic: it extracts numbers from the answer and
    flags any not present in the reference text. It catches invented figures; it is
    not a full fact-checker (production would use a claim-level check)."""
    def numbers(text: str) -> set[str]:
        # digit sequences, keeping decimals, dropping thousands commas for comparison
        return {m.replace(",", "") for m in re.findall(r"\d[\d,]*\.?\d*", text)}

    ref = reference.replace(",", "")
    missing = [n for n in numbers(answer) if n not in ref]
    if missing:
        return False, f"figures not found in retrieved evidence: {sorted(missing)}"
    return True, ""