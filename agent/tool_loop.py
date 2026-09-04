import json
import logging
from typing import Dict, Any, List, Optional, Callable

from privacy.anonymizer import CodeAnonymizer
from agent.diff_parser import process_llm_diff_response, CodePatch

logger = logging.getLogger(__name__)

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the full (anonymized) contents of a file in the workspace. Use this when you need to see a dependency's implementation, not just its signature.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {"filepath": {"type": "string", "description": "Workspace-relative file path, e.g. 'auth/login.py'"}},
            "required": ["filepath"],
        },
    },
    {
        "name": "search_codebase",
        "description": "Semantic search over the codebase for relevant functions/files given a natural language query.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}},
            "required": ["query"],
        },
    },
]


class AgentToolLoop:
    def __init__(
        self,
        workspace_files: Dict[str, str],
        protected_identifiers: Optional[List[str]] = None,
        protected_literals: Optional[List[Any]] = None,
        search_fn: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
        max_rounds: int = 8,
    ):
        self.workspace_files = workspace_files
        self.protected_identifiers = protected_identifiers or []
        self.protected_literals = protected_literals or []
        self.search_fn = search_fn
        self.max_rounds = max_rounds

        self._files_already_sent: set = set()
        # ONE CodeAnonymizer instance, shared across every file read_file
        # touches during a run() call - same reasoning as
        # anonymize_workspace()'s shared transformer (README bug #6): a
        # fresh anonymizer per file lets the same placeholder string get
        # assigned to two DIFFERENT real names in two different files,
        # and merging the resulting maps afterward silently corrupts
        # whichever file's mapping gets overwritten. Recreated fresh at
        # the start of each run() call, not __init__, so consecutive
        # unrelated tasks on the same AgentToolLoop instance don't leak
        # placeholder state between them.
        self._shared_transformer: Optional[CodeAnonymizer] = None

    def _anonymize_and_track(self, filepath: str) -> str:
        content = self.workspace_files.get(filepath)
        if content is None:
            return f"[ERROR] File '{filepath}' does not exist in the workspace."

        import ast
        try:
            parsed = ast.parse(content)
        except SyntaxError as e:
            return f"[ERROR] Syntax error in '{filepath}': {e}"

        self._shared_transformer.filepath = filepath
        transformed = self._shared_transformer.visit(parsed)
        ast.fix_missing_locations(transformed)
        anonymized_code = ast.unparse(transformed)

        self._files_already_sent.add(filepath)
        return anonymized_code

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "read_file":
            filepath = tool_args.get("filepath", "")
            if filepath in self._files_already_sent:
                return {"result": f"[Already provided earlier in this conversation - see above for '{filepath}']"}
            anonymized = self._anonymize_and_track(filepath)
            return {"result": anonymized}

        elif tool_name == "search_codebase":
            if not self.search_fn:
                return {"result": "[search_codebase is not available - no search function was configured]"}
            query = tool_args.get("query", "")
            top_k = tool_args.get("top_k", 5)
            results = self.search_fn(query, top_k)
            summary = [{"id": r.get("id"), "text": r.get("text")} for r in results]
            return {"result": json.dumps(summary)}

        return {"result": f"[Unknown tool: {tool_name}]"}

    def _merge_all_privacy_maps(self) -> Dict[str, Any]:
        # No longer a per-file merge of independent maps - there's ONE
        # shared transformer now, so this just reads its single map
        # directly. Kept as a method for interface stability.
        if self._shared_transformer is None:
            return {"reverse_identifier_map": {}, "reverse_literal_map": {}}
        return {
            "reverse_identifier_map": self._shared_transformer.reverse_identifier_map,
            "reverse_literal_map": self._shared_transformer.reverse_literal_map,
        }

    def run(self, user_request: str, target_filepath: str, llm_tool_caller_fn) -> Dict[str, Any]:
        self._files_already_sent = set()
        self._shared_transformer = CodeAnonymizer(
            filepath=target_filepath,
            protected_identifiers=self.protected_identifiers,
            protected_literals=self.protected_literals,
        )

        initial_context = self._anonymize_and_track(target_filepath)
        if initial_context.startswith("[ERROR]"):
            return {"success": False, "patches": [], "rounds_used": 0, "error_message": initial_context}

        system_prompt = (
            "You are an expert AI code assistant with tool access. Use read_file "
            "to see any dependency's implementation before editing, and "
            "search_codebase to find relevant code by description. When ready, "
            "respond with your final edit using SEARCH/REPLACE block format:\n"
            "File: <filepath>\n<<<<<<< SEARCH\n<exact code>\n=======\n<new code>\n>>>>>>> REPLACE"
        )

        contents = [
            {"role": "user", "parts": [{"text": (
                f"{system_prompt}\n\nUser Task: {user_request}\n\n"
                f"File: {target_filepath}\n```python\n{initial_context}\n```"
            )}]}
        ]

        for round_num in range(1, self.max_rounds + 1):
            response = llm_tool_caller_fn(contents, TOOL_DEFINITIONS)
            function_calls = getattr(response, "function_calls", None) or []

            if function_calls:
                model_parts = [{"function_call": {"name": fc.name, "args": fc.args}} for fc in function_calls]
                contents.append({"role": "model", "parts": model_parts})

                response_parts = []
                for fc in function_calls:
                    tool_result = self._execute_tool(fc.name, dict(fc.args))
                    response_parts.append({"function_response": {"name": fc.name, "response": tool_result}})
                contents.append({"role": "user", "parts": response_parts})
                continue

            final_text = getattr(response, "text", "") or ""
            merged_privacy_maps = self._merge_all_privacy_maps()
            try:
                patches = process_llm_diff_response(
                    llm_response=final_text, privacy_maps=merged_privacy_maps, default_filepath=target_filepath,
                )
            except Exception as e:
                return {"success": False, "patches": [], "rounds_used": round_num, "error_message": f"Diff parsing failed: {e}"}

            return {"success": True, "patches": patches, "rounds_used": round_num, "error_message": None}

        return {"success": False, "patches": [], "rounds_used": self.max_rounds, "error_message": f"Exceeded max_rounds ({self.max_rounds}) without a final answer."}


def get_gemini_tool_caller(api_key: Optional[str] = None, model: str = "gemini-3.6-flash"):
    import os
    from google import genai
    from google.genai import types

    key = api_key or os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=key)

    function_declarations = [types.FunctionDeclaration(**t) for t in TOOL_DEFINITIONS]
    tool = types.Tool(function_declarations=function_declarations)

    def caller(contents, tool_definitions):
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[tool],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )

    return caller
def run_and_commit(
    loop: "AgentToolLoop",
    user_request: str,
    target_filepath: str,
    llm_tool_caller_fn,
    commit_to_disk: bool = False,
    disk_root: Optional[str] = None,
):
    """
    Runs the full tool-calling loop, then feeds its resulting patches
    through the SAME staging/verification/disk-commit pipeline used by
    AgentOrchestrator - so the agentic (multi-turn, tool-using) path and
    the simple (single-turn) path both end up going through identical
    safety checks (exact-match validation, AST syntax check, path
    containment) before anything touches disk.

    Returns a dict shaped consistently with AgentOrchestrator's AgentResult
    fields, so callers don't need to branch on which path was used.
    """
    from agent.patch_applier import apply_patches_staged, PatchApplyError
    from agent.disk_committer import commit_staged_workspace_to_disk, DiskCommitError

    loop_result = loop.run(user_request, target_filepath, llm_tool_caller_fn)

    if not loop_result["success"]:
        return {
            "success": False,
            "staged_workspace": loop.workspace_files,
            "patches": loop_result["patches"],
            "rounds_used": loop_result["rounds_used"],
            "error_message": loop_result["error_message"],
            "committed_to_disk": False,
            "files_written": 0,
        }

    try:
        staged_workspace = apply_patches_staged(loop.workspace_files, loop_result["patches"])
    except PatchApplyError as e:
        return {
            "success": False,
            "staged_workspace": loop.workspace_files,
            "patches": loop_result["patches"],
            "rounds_used": loop_result["rounds_used"],
            "error_message": f"Patch Staging Failed: {e}",
            "committed_to_disk": False,
            "files_written": 0,
        }

    if not commit_to_disk:
        return {
            "success": True,
            "staged_workspace": staged_workspace,
            "patches": loop_result["patches"],
            "rounds_used": loop_result["rounds_used"],
            "error_message": None,
            "committed_to_disk": False,
            "files_written": 0,
        }

    if not disk_root:
        return {
            "success": False,
            "staged_workspace": staged_workspace,
            "patches": loop_result["patches"],
            "rounds_used": loop_result["rounds_used"],
            "error_message": "commit_to_disk=True but no disk_root was provided.",
            "committed_to_disk": False,
            "files_written": 0,
        }

    try:
        commit_success, files_written = commit_staged_workspace_to_disk(staged_workspace, disk_root)
    except DiskCommitError as e:
        return {
            "success": False,
            "staged_workspace": staged_workspace,
            "patches": loop_result["patches"],
            "rounds_used": loop_result["rounds_used"],
            "error_message": f"Disk Commit Blocked: {e}",
            "committed_to_disk": False,
            "files_written": 0,
        }

    return {
        "success": True,
        "staged_workspace": staged_workspace,
        "patches": loop_result["patches"],
        "rounds_used": loop_result["rounds_used"],
        "error_message": None,
        "committed_to_disk": commit_success,
        "files_written": files_written,
    }

