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


def parse_search_replace_blocks(llm_response: str) -> List[Tuple[str, str, str]]:
    """
    Extracts raw (filepath, search_text, replace_text) blocks from an LLM response.
    
    Supports formats:
    File: path/to/file.py
    <<<<<<< SEARCH
    old code
    =======
    new code
    >>>>>>> REPLACE
    """
    # \s* after the File: line's \n tolerates blank lines between the
    # File: header and the SEARCH marker - a common LLM formatting habit.
    pattern = r"(?:File:\s*([^\n]+)\n\s*)?<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
    matches = re.findall(pattern, llm_response, re.DOTALL)
    
    blocks = []
    for filepath, search_block, replace_block in matches:
        blocks.append((filepath.strip() if filepath else "", search_block, replace_block))
    
    return blocks


def process_llm_diff_response(
    llm_response: str,
    privacy_maps: Dict[str, Any],
    default_filepath: Optional[str] = None,
) -> List[CodePatch]:
    """
    Parses LLM output for SEARCH/REPLACE blocks, validates patch structures,
    de-anonymizes the contents of each block, and returns structured CodePatch objects.
    """
    raw_blocks = parse_search_replace_blocks(llm_response)
    
    if not raw_blocks:
        raise ValueError("Malformed patch: No valid SEARCH/REPLACE blocks found in LLM response.")

    patches = []
    for filepath, search_anon, replace_anon in raw_blocks:
        target_path = filepath or default_filepath
        
        if not target_path:
            raise ValueError("Ambiguous patch: Target filepath was not specified in the diff block or context.")

        if not search_anon.strip():
            raise ValueError(f"Invalid patch for '{target_path}': SEARCH block cannot be empty.")

        # De-anonymize the anonymized placeholders inside both SEARCH and REPLACE blocks
        search_restored = deanonymize_code(search_anon, privacy_maps)
        replace_restored = deanonymize_code(replace_anon, privacy_maps)

        patches.append(
            CodePatch(
                filepath=target_path,
                search_text=search_restored,
                replace_text=replace_restored,
            )
        )

    return patches