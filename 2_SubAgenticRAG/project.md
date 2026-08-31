# Project: Sub-Agentic RAG over Quarterly Earnings Calls

*A teaching project for AI engineers. Runs fully local. Builds directly on
`sub_agents_foundational.py` — same supervisor pattern, now doing real work.*

---

## 1. What we are building

An assistant that answers questions about one company's financial year by reading
its four quarterly earnings-call transcripts. A **supervisor** agent coordinates
three specialist sub-agents — one for **financials**, one for **outlook**, one for
**risk**. Each specialist retrieves its own evidence from the transcripts, reasons
over it, and reports back. The supervisor decides who to consult, gathers their
findings, and composes a single balanced answer.

Around that core we add the two things a real deployment needs: **guardrails**
(so it stays safe and on-scope) and a **human-in-the-loop review** (so a person
approves the answer before it goes out).

The company (Nimbus Renewables Ltd) and its four transcripts are **synthetic** —
made up for training. The financial story across the year is deliberate: a strong
start, margin pressure, a Q3 setback (a project delay and a stuck receivable), and
a Q4 recovery. That arc gives each specialist something genuinely different to find.

## 2. Why "sub-agentic RAG" and not plain RAG

Plain RAG retrieves once for the whole question, then answers. Here, **retrieval
happens inside each sub-agent**: the financials agent runs its own query, the risk
agent runs a different one, and the supervisor can decide to consult another
specialist based on what came back. Retrieval is distributed across specialists and
driven by a supervisor that reasons over results — that is what makes it *sub-agentic*.

## 3. The flow, top to bottom

*See `subagentic_rag_graph.png` for the same flow as a labelled diagram.*

```
user request
  → guard_input        checks: injection, off-topic, advice-style
        if blocked → blocked_response → END
  → supervisor  ←───────────────────────────┐   decides the next specialist, or DONE
        │                                    │
        ▼                                    │
   a specialist  (financials / outlook / risk)   retrieves its own evidence,
        │        reasons over it, returns one finding                │
        └────────────────────────────────────┘   (result returns to the supervisor)
  → synthesise         combine findings into a draft answer
  → guard_output       grounding check on the draft
  → human_review       a person approves / edits / rejects   ← HUMAN IN THE LOOP
  → END
```

The specialist-back-to-supervisor edge is the loop from the foundational file. The
only additions are the guardrail nodes at the entrance and exit, and the human gate
at the end.

## 4. The sub-agents

Each is the local model driven by a focused prompt, and each sees only the evidence
its own retrieval query pulled back — not the whole corpus, not the other agents'
findings. They keep no memory between calls (stateless). They are told to cite the
quarter for every point, so findings are traceable.

- **Financials** — revenue, EBITDA margin, and profit trend across the quarters.
- **Outlook** — how management's guidance changed quarter to quarter.
- **Risk** — delays, the overdue receivable, input-cost and regulatory concerns.

## 5. Guardrails (in `guardrails.py`)

Four checks, three of them plain rules (no model call, so they are fast and
predictable) and one that uses the model.

- **Prompt-injection (rule, input).** Blocks obvious "ignore your instructions"
  style attacks before they reach the agents.
- **Investment-advice (rule, input).** Detects "should I buy/sell/hold" style
  questions. It does **not** block them — a financial assistant can discuss a
  company, but it must not give personalised investment advice. So it **flags** the
  request, which forces human review and attaches a disclaimer. This is the
  compliance-sensitive guardrail for a financial firm.
- **Topic / scope (model, input).** A single YES/NO check that the question is about
  this company's earnings. Off-topic questions (weather, other firms) are blocked.
- **Grounding (rule, output).** Checks that numeric figures in the draft answer
  actually appear in the retrieved evidence, catching invented numbers. It is a
  simple heuristic, not a full fact-checker — enough to flag a hallucinated figure
  for the reviewer.

## 6. Human-in-the-loop

Before any answer is delivered, the graph **pauses** at `human_review`. The reviewer
sees the draft answer, the source quarters it used, and any guardrail flags, then
chooses **approve**, **edit** (supply a corrected answer), or **reject**. Nothing
reaches the user until the reviewer acts. In LangGraph this uses `interrupt()` with
a checkpointer, so the run genuinely stops and resumes on the human's decision.

## 7. Data

Four synthetic transcripts in `data/` — `nimbus_q1_fy26.md` … `nimbus_q4_fy26.md`.
Each has prepared remarks (CEO + CFO) and an analyst Q&A, with consistent numbers
across the year so the specialists' findings line up into a coherent story.

## 8. Model choice (local, via Ollama)

Local keeps the data on the machine (a real plus for a financial firm), costs
nothing per query, and runs offline.

- **Agents + supervisor:** `qwen3:8b` — a strong small tool-use model (~5 GB, fits
  8 GB RAM). If the machine has 16–24 GB, `qwen3:30b-a3b` makes the supervisor's
  routing more reliable.
- **Embeddings:** `nomic-embed-text`.

Small models are weaker at the supervisor's decide-and-route job, so the supervisor
is written defensively: it can only pick from the unconsulted specialists, it parses
the model's answer loosely, and it is guaranteed to terminate (once all specialists
are consulted, it stops). Change models with the `OLLAMA_MODEL` and
`OLLAMA_EMBED_MODEL` environment variables — no code change.

## 9. How to run

```bash
# 1. install and start Ollama, then pull the models
ollama pull qwen3:8b
ollama pull nomic-embed-text

# 2. python dependencies
pip install langgraph langchain-ollama langchain-text-splitters

# 3. run
python subagentic_rag.py
```

The script asks a sample question, runs the agents, pauses for your review, and
prints the delivered answer once you approve.

## 10. Files

- `data/` — the four synthetic quarterly transcripts.
- `retrieval.py` — reads the transcripts, chunks and embeds them, exposes `retrieve`.
- `guardrails.py` — the four guardrail checks.
- `subagentic_rag.py` — the LangGraph orchestration (guardrails, supervisor,
  specialists, grounding, human review).
- `subagentic_rag_graph.png` — the graph as a labelled diagram for students.
- `make_graph_image.py` — regenerates that diagram if the graph changes.

## 11. Where to take it next

- Swap the in-memory vector store for a persistent one (Chroma) — one line in
  `retrieval.py`.
- Let specialists run in parallel instead of one at a time (a speed optimisation on
  the same pattern).
- Add a fourth specialist (e.g. valuation) to show how the roster extends.
- Replace synthetic transcripts with real filings once the pattern is understood.
