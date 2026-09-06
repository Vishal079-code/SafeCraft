"""
Agent Tool-Calling Loop (Gemini): gives the LLM read_file / search_codebase
as real tools via the google-genai SDK's function-calling interface, so it
can request more context mid-task instead of needing every relevant file
guessed correctly upfront.

Privacy/context-minimization (V2): read_file is SYMBOL-AWARE. Calling
read_file(filepath, symbol="function_name" or "Class.method") returns the
file with ONLY that symbol's real body intact - every other function in
it is skeletonized (signature + docstring only), via the same
skeletonize_except() used by ContextBundler's target-scope extraction
(Step 1). A symbol-less read_file(filepath) call returns the WHOLE file
and is a deliberate, explicit fallback - the tool description below
steers the model toward symbol-scoped reads first.

Loop pattern:
    Gemini -> (function_calls present? -> execute each -> feed results back
               as function_response parts) -> repeat
           -> (response.text present, no function_calls -> parse as
               SEARCH/REPLACE diff) -> done

Includes context deduplication at two levels: a (file, symbol) pair
already sent isn't re-sent, and once a file has been sent IN FULL, any
later read of it (symbol-scoped or not) is deduped too, since nothing
new would be revealed.
"""
import json
import logging
from typing import Dict, Any, List, Optional, Callable

from privacy.anonymizer import CodeAnonymizer
from agent.skeletonizer import skeletonize_except
from agent.diff_parser import process_llm_diff_response, CodePatch

logger = logging.getLogger(__name__)

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": (
            "Read a function or method's implementation. PREFER passing "
            "'symbol' (e.g. 'apply_discount' or 'Product.apply_discount' - "
            "use the part of a search_codebase result's id AFTER '::') to "
            "get just that function's real body, with the rest of the file "
            "shown only as signatures/docstrings. Omit 'symbol' only as a "
            "fallback when you genuinely need the whole file's content."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Workspace-relative file path, e.g. 'auth/login.py'"},
                "symbol": {"type": "string", "description": "Optional. A function name or 'ClassName.method_name' to read just that symbol's real implementation."},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "search_codebase",
        "description": "Semantic search over the codebase for relevant functions/files given a natural language query. Each result's id is 'filepath::symbol_name' - pass symbol_name to read_file's 'symbol' argument to fetch just that implementation.",
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

        self._files_fully_sent: set = set()          # filepaths sent WITHOUT a symbol filter
        self._symbols_sent: set = set()               # (filepath, symbol) pairs sent
        # ONE CodeAnonymizer instance, shared across every read during a
        # run() call - prevents the same placeholder being assigned to
        # two different real names in two different files/reads (see
        # README bug #6; this loop had the same latent bug until fixed).
        self._shared_transformer: Optional[CodeAnonymizer] = None

    def _anonymize_and_track(self, filepath: str, symbol: Optional[str] = None) -> str:
        raw_content = self.workspace_files.get(filepath)
        if raw_content is None:
            return f"[ERROR] File '{filepath}' does not exist in the workspace."

        import ast
        content_to_send = raw_content if symbol is None else skeletonize_except(raw_content, symbol)

        try:
            parsed = ast.parse(content_to_send)
        except SyntaxError as e:
            return f"[ERROR] Could not process '{filepath}': {e}"

        self._shared_transformer.filepath = filepath
        transformed = self._shared_transformer.visit(parsed)
        ast.fix_missing_locations(transformed)
        anonymized_code = ast.unparse(transformed)

        if symbol is None:
            self._files_fully_sent.add(filepath)
        else:
            self._symbols_sent.add((filepath, symbol))

        return anonymized_code

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "read_file":
            filepath = tool_args.get("filepath", "")
            symbol = tool_args.get("symbol")  # may be None

            if filepath in self._files_fully_sent:
                return {"result": f"[Already provided earlier in this conversation - see above for '{filepath}']"}
            if symbol is not None and (filepath, symbol) in self._symbols_sent:
                return {"result": f"[Already provided earlier in this conversation - see above for '{filepath}::{symbol}']"}

            anonymized = self._anonymize_and_track(filepath, symbol=symbol)
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
        if self._shared_transformer is None:
            return {"reverse_identifier_map": {}, "reverse_literal_map": {}}
        return {
            "reverse_identifier_map": self._shared_transformer.reverse_identifier_map,
            "reverse_literal_map": self._shared_transformer.reverse_literal_map,
        }

    def run(
            self,
            user_request: str,
            target_filepath: str,
            llm_tool_caller_fn,
            target_symbol: Optional[str] = None,
            context_bundler=None,
            max_dependency_depth: int = 2,
        ) -> Dict[str, Any]:
            """
            context_bundler (V2): an optional ContextBundler instance (or
            None). When provided, the INITIAL context sent to the model is
            SEEDED from context_bundler.build_bundled_context(target_filepath,
            target_symbol) - the target file (symbol-scoped if target_symbol
            is given) PLUS its real import dependencies (skeletonized) - not
            just the target file alone. Progressive tools (read_file,
            search_codebase) remain available on top of this seed for
            anything the bundle doesn't cover. context_bundler=None (default)
            preserves prior behavior exactly: only the target file is sent
            initially, and the agent must ask for everything else via tools.

            Every bundled file (including dependencies) is anonymized through
            the SAME shared CodeAnonymizer instance used for everything else
            in this run - a protected name appearing in a dependency gets the
            identical placeholder it would get anywhere else, and dependency
            files are marked as "already sent" so a later read_file call for
            one of them is correctly deduped rather than re-fetched.
            """
            self._files_fully_sent = set()
            self._symbols_sent = set()
            self._shared_transformer = CodeAnonymizer(
                filepath=target_filepath,
                protected_identifiers=self.protected_identifiers,
                protected_literals=self.protected_literals,
            )

            initial_context = self._anonymize_and_track(target_filepath, symbol=target_symbol)
            if initial_context.startswith("[ERROR]"):
                return {"success": False, "patches": [], "rounds_used": 0, "error_message": initial_context}

            seeded_dependency_blocks = []
            if context_bundler is not None:
                bundled = context_bundler.build_bundled_context(
                    target_filepath, target_symbol=target_symbol, skeletonize_dependencies=True
                )
                for filepath, content in bundled.items():
                    if filepath == target_filepath:
                        continue  # already handled above, don't re-anonymize/re-add
                    import ast
                    try:
                        self._shared_transformer.filepath = filepath
                        parsed = ast.parse(content)
                        transformed = self._shared_transformer.visit(parsed)
                        ast.fix_missing_locations(transformed)
                        anonymized_dep = ast.unparse(transformed)
                    except SyntaxError:
                        continue  # skip a dependency that fails to parse rather than aborting the whole seed
                    self._files_fully_sent.add(filepath)  # already skeletonized/sent - dedupe future read_file calls
                    seeded_dependency_blocks.append(f"File: {filepath} (dependency, structural only)\n```python\n{anonymized_dep}\n```")

            system_prompt = (
                "You are an expert AI code assistant with tool access. Use read_file "
                "(preferably WITH a symbol argument, to see just one function's real "
                "implementation) to inspect any dependency before editing, and "
                "search_codebase to find relevant code by description. When ready, "
                "respond with your final edit using SEARCH/REPLACE block format:\n"
                "File: <filepath>\n<<<<<<< SEARCH\n<exact code>\n=======\n<new code>\n>>>>>>> REPLACE"
            )

            dependency_context = ("\n\n" + "\n\n".join(seeded_dependency_blocks)) if seeded_dependency_blocks else ""
            initial_prompt_text = (
                f"{system_prompt}\n\nUser Task: {user_request}\n\n"
                f"File: {target_filepath}\n```python\n{initial_context}\n```"
                f"{dependency_context}"
            )

            contents = [{"role": "user", "parts": [{"text": initial_prompt_text}]}]

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
    target_symbol: Optional[str] = None,
):
    from agent.patch_applier import apply_patches_staged, PatchApplyError
    from agent.disk_committer import commit_staged_workspace_to_disk, DiskCommitError

    loop_result = loop.run(user_request, target_filepath, llm_tool_caller_fn, target_symbol=target_symbol)

    if not loop_result["success"]:
        return {
            "success": False, "staged_workspace": loop.workspace_files, "patches": loop_result["patches"],
            "rounds_used": loop_result["rounds_used"], "error_message": loop_result["error_message"],
            "committed_to_disk": False, "files_written": 0,
        }

    try:
        # Applied directly against the REAL original files - never a
        # separately-staged skeletonized copy - so unlike run_cycle()'s
        # pre-bundled path, no splice-back step is needed here: patches
        # only succeed if their (de-anonymized) SEARCH text matches real,
        # untouched original content.
        staged_workspace = apply_patches_staged(loop.workspace_files, loop_result["patches"])
    except PatchApplyError as e:
        return {
            "success": False, "staged_workspace": loop.workspace_files, "patches": loop_result["patches"],
            "rounds_used": loop_result["rounds_used"], "error_message": f"Patch Staging Failed: {e}",
            "committed_to_disk": False, "files_written": 0,
        }

    if not commit_to_disk:
        return {
            "success": True, "staged_workspace": staged_workspace, "patches": loop_result["patches"],
            "rounds_used": loop_result["rounds_used"], "error_message": None,
            "committed_to_disk": False, "files_written": 0,
        }

    if not disk_root:
        return {
            "success": False, "staged_workspace": staged_workspace, "patches": loop_result["patches"],
            "rounds_used": loop_result["rounds_used"], "error_message": "commit_to_disk=True but no disk_root was provided.",
            "committed_to_disk": False, "files_written": 0,
        }

    try:
        commit_success, files_written = commit_staged_workspace_to_disk(staged_workspace, disk_root)
    except DiskCommitError as e:
        return {
            "success": False, "staged_workspace": staged_workspace, "patches": loop_result["patches"],
            "rounds_used": loop_result["rounds_used"], "error_message": f"Disk Commit Blocked: {e}",
            "committed_to_disk": False, "files_written": 0,
        }

    return {
        "success": True, "staged_workspace": staged_workspace, "patches": loop_result["patches"],
        "rounds_used": loop_result["rounds_used"], "error_message": None,
        "committed_to_disk": commit_success, "files_written": files_written,
    }