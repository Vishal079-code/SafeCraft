# Privacy-Preserving Agentic Code Generation — 20-Day Execution Plan (Final)

**Rule:** Come back and say "day N" — we build that day's piece together, with real code and debugging. Don't rush a day; continue it next session if needed.

---

## Phase 1: Parsing & Codebase Understanding (Days 1–4)

**Day 1 — Workspace Setup & Core Architecture**
- Set up repository, Python virtual environment, `.env` management, and project directory structure (`indexer/`, `retriever/`, `privacy/`, `router/`, `agent/`, `planner/`, `verifier/`)
- Pick a target sample repository (10–30 Python files) for evaluation

**Day 2 — AST Extraction (Single File)**
- Write the single-file AST extractor using Python's built-in `ast` module
- Extract function signatures, arguments, class definitions, imports, and docstrings into a structured dictionary

**Day 3 — Whole-Repo Mapping & Summarization**
- Walk the target codebase directory recursively and run the AST extractor on all `.py` files
- Construct a unified `repo_map.json` mapping file paths to their internal code structural entities

**Day 4 — Call Graph Construction**
- Extract inter-file dependencies and function call relationships
- Build a directed call graph using `networkx` to map file dependencies

---

## Phase 2: Retrieval & Embeddings (Days 5–8)

**Day 5 — Vector DB Setup & Embeddings**
- Set up `chromadb` (or `faiss-cpu`) using a local embedding model (`sentence-transformers/all-MiniLM-L6-v2`)
- Embed function signatures, docstrings, and summaries into the vector store

**Day 6 — Semantic Search Engine**
- Implement query embedding and k-NN retrieval against indexed function summaries
- Verify candidate file retrieval against sample natural language coding prompts

**Day 7 — Graph-Expanded Context Retrieval**
- Combine vector retrieval with the `networkx` call graph
- Traverse direct 1-hop dependencies of retrieved files to ensure structural context is preserved

**Day 8 — Retrieval Benchmark Evaluation**
- Create a dataset of 10–15 test prompts with ground-truth file mappings
- Calculate Precision and Recall metrics for the hybrid retrieval engine

---

## Phase 3: Privacy & Anonymization Layer (Days 9–10)

**Day 9 — Scope-Aware Identifier Anonymization**
- Write an AST `NodeTransformer` to rename functions, variables, and classes to deterministic placeholders (e.g., `func_a1`, `var_b2`)
- Preserve Python built-ins, standard library calls, and keyword parameters
- Generate and store an in-memory `IdentifierMap` (original ↔ placeholder)

**Day 10 — De-anonymization & Round-Trip Validation**
- Build the inverse mapping engine to restore placeholders back to original domain identifiers
- Test round-trip integrity: Raw Code → Anonymize → De-anonymize → AST Equality Check

---

## Phase 4: Agent Loop, Routing & Deduplication (Days 11–13)

**Day 11 — Single-Turn LLM Integration**
- Connect anonymized payload construction to a single LLM API call
- Intercept model response, run de-anonymization, and inspect generated code diffs

**Day 12 — Agent Loop & Context Deduplication**
- Build tool-calling execution loop (`read_file`, `search_codebase`)
- **Context Deduplication:** intercept `read_file` requests to check if a file was already loaded in earlier turns; return a reference token instead of re-sending full source code

**Day 13 — Multi-Model Router & Rate-Limit Fallback**
- Implement `ModelRouter` to assign fast models (e.g., Gemini Flash) for utility tasks and high-reasoning models (e.g., Claude Sonnet / Qwen Coder) for code edits
- Add try/except HTTP 429 rate-limit catches with automatic secondary model failover

---

## Phase 5: Planning, Verification & Cost (Days 14–16)

**Day 14 — Cost Tracking & Token Budgeting**
- Log input/output token usage per subtask and calculate real-time estimated inference costs
- Implement pre-flight budget checks to abort or re-route subtasks exceeding token thresholds

**Day 15 — Sequential Subtask Planner**
- Build a planner component that takes a high-level request (e.g., "Add authentication endpoint and link to user model") and outputs an ordered sequence of subtasks

**Day 16 — Integrated Agentic Pipeline Run**
- Wire Planner → Router → Anonymizer → Agent Loop → De-anonymizer into a complete end-to-end flow
- Execute a multi-file task across the target codebase

---

## Phase 6: Automated Verification & Optimization (Days 17–18)

**Day 17 — Deterministic Verification Engine**
- Implement `ast.parse` syntax checks on de-anonymized output code
- Add automated import resolution and run unit tests (`pytest`) before committing code to disk

**Day 18 — System Evaluation & Benchmarking**
- Run comparative benchmarks:
  - Naive context dump vs. Graph-RAG token savings
  - Code correctness on Anonymized vs. Raw code inputs
  - Failed subtask recovery rate via automated verification

---

## Phase 7: Polish & Documentation (Days 19–20)

**Day 19 — CLI Interface & Error Handling**
- Build a clean terminal CLI (`main.py`) with configurable repository paths and task flags
- Handle edge cases (empty search results, model timeouts, unparseable code)

**Day 20 — Repository Finalization & Architecture Writeup**
- Finalize the updated `README.md`, record a demo walkthrough
- Document future design considerations (DAG parallel execution, atomic locking)

---

*Note: multi-language (JS/React/SQL) support via `tree-sitter` is a post-MVP extension — see README for scoping notes. This plan targets Python-only for the 20-day build.*