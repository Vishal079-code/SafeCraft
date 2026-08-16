"""
Sequential Subtask Planner: takes a high-level user request and breaks it
into an ORDERED list of subtasks, each targeting a specific file. No
parallelism/DAG scheduling here (that's documented post-MVP scope) - just
correct ordering, so dependent edits (e.g. "add a route" before "link a
button to that route") happen in the right sequence.
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


def build_planner_prompt(user_request: str, available_files: List[str]) -> str:
    files_list = "\n".join(f"- {f}" for f in available_files)
    return (
        "You are a software planning assistant. Break the following user "
        "request into an ORDERED list of small, atomic subtasks. Each "
        "subtask must target exactly ONE file from the available files "
        "list below. Order matters: a subtask that depends on another "
        "(e.g. linking to a route) must come AFTER the subtask that "
        "creates the thing it depends on.\n\n"
        f"Available files:\n{files_list}\n\n"
        f"User request: {user_request}\n\n"
        "Respond with ONLY a JSON array, no other text, in this exact shape:\n"
        '[{"description": "...", "target_filepath": "..."}, ...]'
    )


def _extract_json_array(text: str) -> str:
    """
    LLMs frequently wrap JSON in markdown code fences or add a sentence
    before/after it even when explicitly told not to. Pull out just the
    JSON array substring rather than assuming the whole response is clean.
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in planner response.")
    return match.group(0)


def parse_subtask_response(llm_response: str, available_files: List[str]) -> List[Subtask]:
    """
    Parses and validates the planner LLM's response. Raises ValueError on
    malformed JSON, missing required fields, or a target_filepath that
    isn't actually in the workspace - fail loudly here rather than let a
    bad plan silently reach the execution loop later.
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
        subtasks.append(Subtask(order=i + 1, description=description, target_filepath=target_filepath))

    return subtasks


def generate_plan(user_request: str, available_files: List[str], llm_caller_fn) -> List[Subtask]:
    """
    llm_caller_fn: Callable[[str], str] - same simple string-in/string-out
    shape as AgentOrchestrator's llm_caller_fn, so this can reuse the exact
    same LLM client wiring (get_gemini_client / mock) already built.
    """
    prompt = build_planner_prompt(user_request, available_files)
    llm_response = llm_caller_fn(prompt)
    return parse_subtask_response(llm_response, available_files)