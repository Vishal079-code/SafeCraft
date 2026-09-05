"""
Sequential Subtask Planner: takes a high-level user request and breaks it
into an ORDERED list of subtasks, each targeting a specific file and,
where determinable, a specific function/method within it. No
parallelism/DAG scheduling here (documented post-MVP scope) - just
correct ordering, so dependent edits (e.g. "add a route" before "link a
button to that route") happen in the right sequence.

V2 change: generate_plan can optionally take retrieval_candidates - a
list of METADATA-ONLY dicts from HybridRetriever.retrieve_metadata_only()
(never raw retrieve(), which includes source text). When provided, the
planner prompt includes these candidates (filepath, symbol name, score)
so the model can name a target_symbol directly instead of guessing blind
from filenames alone. retrieval_candidates=None (the default) reproduces
the exact prior prompt/behavior - fully backward compatible.
"""
import json
import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Subtask:
    order: int
    description: str
    target_filepath: str
    target_symbol: Optional[str] = None


def build_planner_prompt(
    user_request: str,
    available_files: List[str],
    retrieval_candidates: Optional[List[Dict[str, Any]]] = None,
) -> str:
    files_list = "\n".join(f"- {f}" for f in available_files)

    candidates_block = ""
    if retrieval_candidates:
        lines = []
        for c in retrieval_candidates:
            meta = c.get("metadata", {})
            symbol_id = c.get("id", "")
            filepath = meta.get("filepath", "")
            name = meta.get("name", "")
            score = c.get("score", 0.0)
            lines.append(f"- {symbol_id} (file: {filepath}, symbol: {name}, relevance: {score:.3f})")
        candidates_block = (
            "\n\nRelevant symbols found by search (name/location only - NOT their "
            "implementation), most relevant first:\n" + "\n".join(lines) +
            "\n\nWhen a subtask clearly corresponds to one of these symbols, set "
            "\"target_symbol\" to its bare name (or \"ClassName.method_name\" for a "
            "method) EXACTLY as shown above. If no candidate clearly matches, or the "
            "subtask concerns the file as a whole, omit target_symbol or set it to null."
        )

    return (
        "You are a software planning assistant. Break the following user "
        "request into an ORDERED list of small, atomic subtasks. Each "
        "subtask must target exactly ONE file from the available files "
        "list below. Order matters: a subtask that depends on another "
        "(e.g. linking to a route) must come AFTER the subtask that "
        "creates the thing it depends on."
        f"{candidates_block}\n\n"
        f"Available files:\n{files_list}\n\n"
        f"User request: {user_request}\n\n"
        "Respond with ONLY a JSON array, no other text, in this exact shape:\n"
        '[{"description": "...", "target_filepath": "...", "target_symbol": "..." or null}, ...]'
    )


def _extract_json_array(text: str) -> str:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in planner response.")
    return match.group(0)


def parse_subtask_response(llm_response: str, available_files: List[str]) -> List[Subtask]:
    """
    Parses and validates the planner LLM's response. "target_symbol" is
    OPTIONAL in the response - a missing key, explicit null, or empty
    string are all treated as None (no symbol determined), preserving
    full backward compatibility with a planner response that predates
    this field entirely.
    """
    json_str = _extract_json_array(llm_response)
    try:
        raw_items = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Planner response was not valid JSON: {e}")

    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Planner response must be a non-empty JSON array.")

    subtasks = []
    for i, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"Subtask {i} is not a JSON object: {item!r}")
        description = item.get("description")
        target_filepath = item.get("target_filepath")
        if not description or not target_filepath:
            raise ValueError(f"Subtask {i} missing 'description' or 'target_filepath': {item!r}")
        if target_filepath not in available_files:
            raise ValueError(
                f"Subtask {i} targets '{target_filepath}', which is not in the workspace. "
                f"Available files: {available_files}"
            )
        target_symbol = item.get("target_symbol") or None  # covers missing key, null, and ""
        subtasks.append(Subtask(
            order=i + 1, description=description,
            target_filepath=target_filepath, target_symbol=target_symbol,
        ))

    return subtasks


def generate_plan(
    user_request: str,
    available_files: List[str],
    llm_caller_fn,
    retrieval_candidates: Optional[List[Dict[str, Any]]] = None,
) -> List[Subtask]:
    """
    llm_caller_fn: Callable[[str], str] - same simple string-in/string-out
    shape as AgentOrchestrator's llm_caller_fn.

    retrieval_candidates: output of HybridRetriever.retrieve_metadata_only()
    - metadata-only by construction, so passing it here never risks
    leaking source code into the planner prompt. Pass None (default) for
    the exact prior behavior.
    """
    prompt = build_planner_prompt(user_request, available_files, retrieval_candidates=retrieval_candidates)
    llm_response = llm_caller_fn(prompt)
    return parse_subtask_response(llm_response, available_files)