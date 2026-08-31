# SafeCraft — Privacy-Preserving Agentic Code Generation

An AI coding agent that edits a local codebase using a remote LLM (Gemini),
without ever sending the LLM more of your code than it needs, and without
sending real identifier names or sensitive literals you've marked as
protected. Code is anonymized locally via AST transformation, only the
target file plus its real dependencies (structural + semantic) are sent,
the LLM returns a SEARCH/REPLACE diff, the diff is applied and
de-anonymized locally, verified, and only written to disk after you
explicitly accept it.

---

## 1. What this actually does, in one paragraph

You point the tool at a local Python repo. It indexes the repo once
(AST-parses every file, builds a dependency graph, embeds function/class
summaries into a vector store, builds a keyword index). From then on, you
describe a task in plain English. The tool figures out which file(s) are
relevant (via import analysis + semantic search), sends **only those
files** to Gemini — with any identifier names or literal values you've
marked as sensitive replaced by placeholders before anything leaves your
machine — gets back a proposed code change, verifies it's syntactically
valid, shows you the diff, and only writes it to disk if you say yes.

---

## 2. Architecture

```
   User task (plain English)
          |
          v
  +-------------------+       +----------------------+
  |  ContextBundler    |       |   HybridRetriever     |
  |  (import graph,    |       |   (vector + BM25       |
  |   deterministic,   |  ---> |    semantic search,    |
  |   always runs)     |  UNION|    optional)           |
  +-------------------+       +----------------------+
          |
          v
  Anonymize (only the bundled/retrieved subset - never the whole repo)
          |
          v
  Send to Gemini (target file full-source, dependencies skeletonized)
          |
          v
  Parse SEARCH/REPLACE diff  --(retry up to N times on mismatch)-->
          |
          v
  Apply patch against the ANONYMIZED representation (not the original -
  see Section 5, bug #5, for why this matters)
          |
          v
  De-anonymize the result (placeholders -> real names, once, on the
  whole staged result)
          |
          v
  Verify (ast.parse - syntax must still be valid)
          |
          v
  Show diff to user -> ACCEPT (write to disk) / REJECT (discard)
          |
          v
  Re-index the changed file (incremental - only this file re-embedded)
```

Two independent things decide which files get sent for a given task, and
they are **unioned**, not either/or:

- **`ContextBundler`** — deterministic. Walks the target file's real
  Python `import` statements, breadth-first, up to `max_dependency_depth`
  hops. Catches "this file literally cannot be edited correctly without
  also seeing that file" (a base class, a shared constant, a function
  being called).
- **`HybridRetriever`** — semantic (vector embeddings + BM25 keyword
  search, fused via Reciprocal Rank Fusion). Optional — only used if a
  retriever is configured. Catches files that are *conceptually*
  relevant to the task but have no import relationship at all (e.g. "a
  similar validation pattern exists in an unrelated file").

Only the **target file** is ever sent full-source. Every dependency or
retrieval-hit file is **skeletonized** first (signatures + docstrings
only, function bodies replaced with `...`) — this reduces both token
cost and privacy exposure, since the LLM only needs to know a
dependency's *shape*, not its full implementation, to use it correctly.

**Important scope note on anonymization:** only names/values you
explicitly list in `protected_identifiers` / `protected_literals` are
hidden. Ordinary code structure, arithmetic, control flow, and any name
not on your list are sent as plain, readable text — the LLM has to
understand the logic to edit it correctly, so the actual computation
being performed is never hidden, only specific names and constants you
flag. This tool protects proprietary *naming* and specific *sensitive
values*, not proprietary *algorithms* — if the logic itself (not just its
name) is the trade secret, this tool's anonymization does not protect it;
that would require a local model instead of a remote API call.

---

## 3. Module reference

### `privacy/` — anonymization layer

- **`anonymizer.py`** — `CodeAnonymizer(ast.NodeTransformer)` renames
  protected identifiers (function names, parameters, attributes, keyword
  arguments) to placeholders (`__ANON_FUNC_N__`, `__ANON_VAR_N__`,
  `__ANON_ATTR_N__`, `__ANON_CONST_N__`). Literals are protected **per
  occurrence**, not per value — two identical numbers at different
  source locations get different placeholders. `anonymize_workspace(...)`
  anonymizes multiple files using **one shared `CodeAnonymizer`
  instance**, so the same real name always maps to the same placeholder
  across file boundaries, and internal counters never collide between
  files.

- **`deanonymizer.py`** — `deanonymize_code(text, privacy_maps)` reverses
  placeholders via `re.sub`, using a **callable** replacement (not a
  string), because a string replacement in `re.sub` interprets
  backslashes as regex backreferences, which would corrupt any restored
  value containing a backslash (Windows paths, regex patterns). After
  substitution, scans for any leftover `__ANON_*__`-shaped token that
  didn't get resolved (a hallucinated or mangled placeholder the LLM
  emitted) and raises `DeanonymizationError` rather than silently
  returning a placeholder-contaminated result.

### `agent/` — pipeline

- **`orchestrator.py`** — `AgentOrchestrator.run_cycle(...)` is the
  central pipeline function. Takes an optional `retriever` — if `None`
  (the default), behavior is `ContextBundler`-only, identical to before
  retrieval existed. If a retriever is passed, its hits are queried and
  unioned into the bundle. Retrieval failures (e.g. an unindexed repo)
  are caught and logged, never crash the run — retrieval is an
  enhancement, never a hard requirement. Handles the full retry loop,
  patch staging, de-anonymization, and optional disk commit.

- **`context_bundler.py`** — `ContextBundler(workspace_files, max_depth)`.
  `build_bundled_context(target_filepath)` does a real breadth-first
  walk: resolves the target's imports, then those files' imports, up to
  `max_depth` hops, stopping early if a hop finds nothing new. Always
  resolves against each file's full original source (not an
  already-skeletonized version), since only function *bodies* are
  stripped by skeletonization — import statements stay intact regardless.

- **`skeletonizer.py`** — `skeletonize_code(code_str)` replaces every
  function/method body with its docstring (if present) plus a single
  `...`, keeping signatures, decorators, and class structure intact.

- **`diff_parser.py`** — extracts `File: ... <<<<<<< SEARCH ... =======
  ... >>>>>>> REPLACE` blocks via regex. Tolerates blank lines after
  `File:`, backtick/quote-wrapped filenames, and carries the last-seen
  filepath forward across blocks that omit `File:`.

- **`patch_applier.py`** — `apply_patch_to_content`: exact match first,
  then a normalized match (quote style unified, trailing whitespace
  stripped — for *comparison* only; the real file's original text is
  what actually gets used in the replace). Falls back to a `difflib`
  diagnostic (best-matching region + similarity score) if neither
  matches. Rejects ambiguous (multiple-match) or syntax-breaking patches.

- **`disk_committer.py`** — `commit_staged_workspace_to_disk`: resolves
  every path against the repo root and checks it via
  `full_path.relative_to(resolved_root)` — if that raises, the path
  escaped the root (absolute-path override or `../` traversal), and the
  **entire commit is aborted before any file is written.**

- **`llm_client.py`** — `build_llm_caller(provider, api_key,
  mock_fallback)`. Live Gemini (`gemini-3.6-flash`) and Anthropic
  providers, both with `temperature=0.0`. Falls back to a provided mock
  only if no API key is set.

- **`tool_loop.py`** — `AgentToolLoop`: gives the model `read_file` and
  `search_codebase` tools via function-calling, for cases where the
  model needs to explore beyond what was pre-bundled. Deduplicates
  repeated reads of the same file within one conversation.

### `planner/` — multi-file task breakdown

- **`subtask_planner.py`** — `generate_plan(user_request,
  available_files, llm_caller_fn)` asks the LLM for a JSON array of
  `{"description", "target_filepath"}` subtasks. Validates every
  `target_filepath` actually exists, raising on a hallucinated path or
  malformed JSON.

- **`multi_step_executor.py`** — `execute_plan(...)` runs each subtask
  through `orchestrator.run_cycle(...)` in order, threading the evolving
  workspace from one subtask into the next. **Stops immediately on the
  first subtask failure** — later files are never touched.

### `verifier/`

- **`ast_verifier.py`** — `verify_code_syntax` wraps `ast.parse`.
- **`test_runner.py`** — `run_pytest_suite` runs the project's own test
  suite via subprocess. Guards against infinite recursion (if the
  default `test_dir="tests/"` includes the very test calling this
  function) via an env-var check that breaks the chain at depth 1.
- **`self_corrector.py`** — `validate_staged_workspace`: checks every
  modified file is actually present in the staged result (previously
  this silently passed on a missing file, since `ast.parse("")` is
  valid Python), runs syntax verification, and optionally the full test
  suite.

### `indexer/` — repo parsing and change tracking

- **`ast_extractor.py`** — `extract_file(filepath)` parses one file into
  structured metadata: functions (with full argument info — positional,
  vararg, keyword-only, kwarg, defaults, annotations), classes (with
  base classes, methods), imports, docstrings.
- **`repo_indexer.py`** — `RepoIndexer` / `build_repo_map(repo_path,
  output_path)` walks a directory (ignoring `.git`, `venv`,
  `__pycache__`, etc.), extracts every `.py` file, produces
  `repo_map.json`.
- **`repo_onboarder.py`** — `onboard_repo(repo_path)` — see Section 4.
  The real "set this repo up for use" entry point.
- **`graph_builder.py`** — a second, `networkx`-based import-dependency
  graph, built from `repo_map.json`. **Not currently wired into the live
  pipeline** — `ContextBundler` does the equivalent job directly from
  source. Kept for reference / possible future use; see Section 6.

### `retriever/` — semantic + keyword search

- **`summary_builder.py`** — `build_summary_units(repo_map)` turns every
  function/class/method into one embeddable text unit (docstring if
  present, else a signature-based description).
- **`vector_store.py`** — `VectorStore`: embeds units with
  `sentence-transformers` (`all-MiniLM-L6-v2`), stores/queries them in a
  persistent local ChromaDB collection (cosine similarity).
- **`bm25_store.py`** — `BM25Index`: sparse keyword search with a
  code-aware tokenizer (splits `camelCase`/`snake_case`, preserves full
  identifier names too).
- **`hybrid_retriever.py`** — `HybridRetriever`: fuses vector + BM25
  results via Reciprocal Rank Fusion (`1/(k_rrf + rank)` per source,
  summed). `.retrieve(query, top_k)` is the interface `AgentOrchestrator`
  expects.
- **`context_assembler.py`** — `ContextAssembler`: groups retrieved
  units by file, formats into markdown blocks, enforces a token budget
  (truncates if needed).

### `eval/`

- **`inspect_pipeline.py`** — `load_repo_as_workspace` +
  `inspect_run(...)`: runs a full plan through the pipeline, printing
  every intermediate step (plan, prompt sent, patches, final files) —
  the primary manual-verification tool.
- **`benchmark.py`** — `BenchmarkRunner`: real, live-API side-by-side
  comparison of RAW (whole-workspace, unscoped, unanonymized) vs.
  SAFECRAFT (dependency + retrieval scoped, anonymized) — reports token
  counts, pass/fail, and which files were actually bundled. Saves a
  **timestamped** JSON report per run (see Section 5, bug list below,
  for why timestamping matters).

### Top level

- **`main.py`** — original single-file, single-task CLI. Reads a target
  file, prompts for a task + protected identifiers/literals, shows the
  sanitized prompt and staged result, writes to disk only on explicit
  `y` confirmation. Good for a one-off edit; no session memory, no
  retrieval, no multi-file support.
- **`session.py`** — the real interactive tool. See Section 4.

---

## 4. Setup and usage

### One-time environment setup

```powershell
pip install -r requirements.txt   # or install individually: google-genai,
                                   # sentence-transformers, chromadb,
                                   # rank_bm25, networkx, tiktoken, python-dotenv
```

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```
Get a free key at https://aistudio.google.com/. **Free tier is 20
requests/day per model** (as of this writing, for `gemini-3.6-flash`) —
resets at midnight Pacific Time. Each `session.py` task costs 1 request
per file touched (plus 1 extra for planning, on multi-file tasks).

### Running a session

```powershell
python session.py
```

Flow:
1. **Repo path** — first run, type the path. Later runs, it remembers
   the last repo used and offers to continue with it.
2. **Onboarding** — automatic. First time on a repo, does a full index
   (AST parse → embed → BM25). Later runs, only re-indexes files that
   actually changed (hash-compared) — unchanged files aren't
   re-embedded, keeping repeat runs fast and free.
3. **Protected identifiers/literals** — asked once per repo, then
   remembered (saved alongside that repo's index) and offered again
   next time, so you don't retype them every session.
4. **Task** — type anything, in your own words.
5. **File selection** — if you don't specify one, the tool suggests
   candidates (via semantic retrieval) as a numbered list; pick one,
   several (comma-separated), `all`, or `0` to type a path manually.
6. **Per file, in order**: the change is generated, verified
   (`ast.parse` must succeed) *before* you ever see it, then the diff is
   shown. **Accept** writes to disk immediately and triggers an
   incremental re-index of that file. **Reject** discards it and stops
   the rest of that task (later files in a multi-file plan are not
   attempted after a rejection).
7. Back to step 4 for the next task, same session, same loaded repo.

Type `exit` to end the session.

### Onboarding a repo manually (without a full session)

```python
from indexer.repo_onboarder import onboard_repo
result = onboard_repo("/path/to/any/local/python/repo")
print(result["files_total"], result["files_added"], result["files_changed"])
retriever = result["retriever"]  # ready to use immediately
```

Indexed data for each repo lives under
`.safecraft_data/repos/<folder_name>_<8-char hash of the resolved
path>/` — the hash suffix prevents collisions between two different
repos that happen to share a folder name.

---

## 5. Real bugs found and fixed (development history)

Kept because these are substantive, specific, and demonstrate the actual
engineering work — better material for a writeup than generic
"things I'd improve" statements.

1. **`deanonymize_code` + backslashes.** A string replacement argument to
   `re.sub` interprets backslashes as regex backreferences — would
   corrupt any restored string containing one (Windows paths, regex
   patterns). Fixed with a callable replacement instead.

2. **Diff parser blank-line tolerance.** LLMs commonly add a blank line
   after `File:` before the `SEARCH` marker; regex updated to tolerate it.

3. **Import resolution solved twice.** Relative-import and
   `from package import submodule` resolution was solved once in
   `indexer/graph_builder.py`, then independently re-discovered and
   re-solved later in `agent/context_bundler.py`. Lesson: check a
   codebase's own prior solutions before re-deriving a fix.

4. **LLM non-determinism causing patch mismatches.** An identical prompt
   against identical anonymized code produced a matching SEARCH block on
   one live call and a non-matching one on the next. Motivated the
   retry loop (`max_attempts`).

5. **The most significant bug: anonymization restructures code, not just
   renames it.** `ast.unparse()` collapses multi-line literals onto one
   line and normalizes ALL string quote style throughout the whole file
   — including parts anonymization never touched. The original
   architecture de-anonymized each patch fragment and matched it against
   the *original, differently-formatted* file — a structural mismatch,
   not fixable by whitespace/quote normalization alone. **Fix:** patches
   are applied against the *anonymized* representation (the same text
   the model actually saw); de-anonymization happens exactly once, on
   the whole final staged result.

6. **Multi-file placeholder collision.** A fresh `CodeAnonymizer` per
   file (counters restarting at 0 each time) could let two unrelated
   names in two different files both become `__ANON_FUNC_1__`, silently
   corrupting de-anonymization when maps were merged. Fixed by sharing
   one `CodeAnonymizer` instance's state across every file in a
   workspace.

7. **Model deprecation, twice.** `claude-3-5-sonnet-20241022` was
   retired; `gemini-2.5-flash` was later found blocked for new users via
   a live 404, confirmed by direct check rather than assumption, updated
   to `gemini-3.6-flash`.

8. **`run_pytest_suite` recursion trap.** If the default `test_dir=
   "tests/"` is used and the *calling* test itself lives inside
   `tests/`, a naive subprocess spawn would re-collect and re-run that
   same test, which would call `run_pytest_suite` again — infinite
   recursion. Fixed via an env-var check that breaks the chain at depth 1.

9. **Orchestrator originally anonymized and sent the ENTIRE workspace on
   every call**, regardless of size — confirmed directly by running
   `eval/inspect_pipeline.py` against a 14-file test repo and seeing
   every file in the prompt. Meant the context-bundling components built
   earlier were never actually wired into the live edit path. **Fix:**
   `ContextBundler` scopes to target + real dependencies before
   anonymizing/sending anything.

10. **`ContextBundler` only resolved one hop**, not truly recursive,
    despite being described as recursive — caught by direct code
    inspection, not by trusting the description. Fixed to real
    breadth-first multi-hop resolution up to `max_depth`.

11. **No check for leftover, unresolved placeholders after
    de-anonymization.** A hallucinated or mangled `__ANON_*__` token from
    the LLM would previously pass through silently into the final code.
    Fixed: `deanonymize_code` now scans for any leftover placeholder
    pattern and raises `DeanonymizationError` if one survives.

12. **Benchmark report file silently overwritten every run** — lost the
    actual error detail from a failed run three times in a row before
    being caught. Fixed with a timestamped filename per run.

13. **Benchmark's own `_failed_result` dropped `bundled_files` and
    `deanon_success`** on any failure, making failed runs impossible to
    debug (`Bundled: None` even though the orchestrator had actually
    populated the real value). Fixed by threading those fields through
    on failure too.

14. **A hardcoded test mock used the wrong anonymized placeholder name**
    (`__ANON_ATTR_1__` instead of the correct `__ANON_FUNC_1__` for a
    function name) — initially looked like live-model non-determinism or
    a pipeline bug; turned out to be a stale, hand-typed mock. Lesson:
    always check whether a "flaky" result is coming from a real API call
    or a hardcoded string before investigating the pipeline itself.

15. **Live Gemini free-tier quota exhaustion (429) initially
    misdiagnosed as model non-determinism**, because early error-message
    handling swallowed the real cause. Once real error text was
    surfaced (see bug #12/#13 fixes), the actual cause — 20
    requests/day on the free tier — became immediately obvious. Several
    earlier "flaky" results in this project's history may retroactively
    be quota-related rather than genuine model variance; this was never
    confirmed either way because the information wasn't visible at the
    time.

16. **`session.py` initially missing `load_dotenv()`**, unlike `main.py`
    and `benchmark.py` — caused a `GEMINI_API_KEY not set` error even
    with a correctly configured `.env` file, since the key was simply
    never loaded into the environment. One-line fix.

17. **Session's semantic index going silently stale mid-session.** A
    file edited and accepted within a session wasn't automatically
    re-indexed, so a subsequent task in the *same* session could fail to
    find content that had just been added, with no warning that the
    index was out of date. Fixed: an incremental re-index now runs
    automatically after every accepted write.

---

## 6. Known limitations / explicitly deferred scope

**Not a bug, a scope boundary (see Section 2):** anonymization hides only
names/literals you explicitly list. The logic/structure of code is
always visible to the LLM — this tool cannot hide an algorithm that is
itself the sensitive asset, only its naming and specific flagged values.

**Deliberately not built, considered out of MVP scope:**
- A larger, formal benchmark suite (20–50 tasks; current suite is 3
  tasks against `sample_repo`, run manually).
- Rollback / checkpoint system for accepted changes.
- Automatic sensitive-data detection (currently, you manually specify
  `protected_identifiers`/`protected_literals` — nothing infers what
  *should* be protected).
- Cost/latency dashboards.
- Exponential backoff on API rate limits (currently: fixed 2-attempt
  retry, no backoff — directly relevant given the 429/quota issue found
  in bug #15 above).
- A UI beyond the terminal session (`session.py` is a text/numbered-list
  interface, not a GUI or web app).
- Deployment / hosting (this is a local, single-user tool as built —
  no auth, no multi-tenancy, no remote access).
- `graph_builder.py`'s `networkx`-based dependency graph is not wired
  into the live pipeline — `ContextBundler` performs the equivalent
  resolution directly. Kept in the codebase but currently redundant;
  a future cleanup could either retire one or find a reason to use both.
- `retrieval_top_k` (currently 5 in `session.py`, 3 as the
  `AgentOrchestrator` default) is an untuned guess, not validated
  against measured retrieval quality.

---

## 7. Testing

```powershell
python -m pytest tests/ -v
```

Test files, one per module area: anonymizer, de-anonymizer, diff parser,
patch applier, disk committer, LLM client, agent tool loop, orchestrator
(+ retry behavior + retrieval integration), multi-step executor,
multifile orchestrator, subtask planner, AST verifier, verifier,
context bundler, CLI.

`tests/test_orchestrator_retrieval.py` specifically covers the
retrieval-integration logic with a fake retriever (no live API, no real
embeddings needed) — the union/dedup behavior, stale-index handling, and
confirmation that omitting a retriever entirely reproduces pre-retrieval
behavior exactly.

Live-API confirmation (outside the automated suite, since it costs real
quota) has been run manually via `eval/inspect_pipeline.py`,
`eval/benchmark.py`, and ad hoc scripts confirming: dependency bundling
scopes correctly on a 13-file test repo, semantic retrieval surfaces
genuinely import-unconnected relevant files, the full pipeline succeeds
end-to-end on a live Gemini call with retrieval-augmented context, and
`onboard_repo`'s incremental re-indexing correctly skips unchanged files
and re-embeds only what changed.