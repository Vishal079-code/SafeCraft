"""
Agent Diff Parser: Extracts, validates, and de-anonymizes SEARCH/REPLACE blocks
from LLM edit responses.
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from privacy.deanonymizer import deanonymize_code


@dataclass
class CodePatch:
    filepath: str
    search_text: str
    replace_text: str


def parse_search_replace_blocks(llm_response: str, default_filepath: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """
    Extracts raw (filepath, search_text, replace_text) blocks from an LLM response.

    Supports:
    - File: path/to/file.py  (with optional backtick/quote wrapping: `path`, 'path', "path")
    - Blank lines between the File: header and the SEARCH marker
    - Multiple SEARCH/REPLACE blocks for the SAME file without repeating
      the File: header each time - the filepath carries forward from the
      most recent block that specified one, seeded by default_filepath.
    """
    pattern = re.compile(
        r"(?:File:\s*[`'\"]?([^\n`'\"]+)[`'\"]?\s*\n)?"
        r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        re.DOTALL,
    )

    blocks = []
    current_filepath = default_filepath or ""

    for match in pattern.finditer(llm_response):
        matched_file, search_block, replace_block = match.groups()
        if matched_file:
            current_filepath = matched_file.strip()
        blocks.append((current_filepath, search_block, replace_block))

    return blocks


def process_llm_diff_response(llm_response: str, privacy_maps: Dict[str, Any], default_filepath: Optional[str] = None) -> List[CodePatch]:
    raw_blocks = parse_search_replace_blocks(llm_response, default_filepath=default_filepath)
    if not raw_blocks:
        raise ValueError("Malformed patch: No valid SEARCH/REPLACE blocks found in LLM response.")
    patches = []
    for filepath, search_anon, replace_anon in raw_blocks:
        target_path = filepath or default_filepath
        if not target_path:
            raise ValueError("Ambiguous patch: Target filepath was not specified in the diff block or context.")
        if not search_anon.strip():
            raise ValueError(f"Invalid patch for '{target_path}': SEARCH block cannot be empty.")
        search_restored = deanonymize_code(search_anon, privacy_maps)
        replace_restored = deanonymize_code(replace_anon, privacy_maps)
        patches.append(CodePatch(filepath=target_path, search_text=search_restored, replace_text=replace_restored))
    return patches