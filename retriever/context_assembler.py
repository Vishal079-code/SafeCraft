"""
Context Assembler: Groups, deduplicates, and formats retrieved code units
into a token-budgeted prompt context block for LLM generation.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ContextAssembler:
    def __init__(self, max_tokens: int = 3000, approx_chars_per_token: float = 4.0):
        self.max_tokens = max(1, max_tokens)
        self.max_chars = int(self.max_tokens * approx_chars_per_token)

    def assemble_context(self, retrieved_units: List[Dict[str, Any]], query: Optional[str] = None) -> str:
        """
        Deduplicates units by file path, formats snippets into structured markdown blocks,
        and enforces maximum token character limits.
        """
        if not retrieved_units:
            return "No relevant context found."

        # Filter out invalid or None entries
        valid_units = [u for u in retrieved_units if isinstance(u, dict)]
        if not valid_units:
            return "No valid context units found."

        # Group units by file path
        file_grouped: Dict[str, List[Dict[str, Any]]] = {}
        for unit in valid_units:
            meta = unit.get("metadata") or {}
            filepath = meta.get("filepath") or meta.get("file_path")
            
            if not filepath:
                unit_id = str(unit.get("id", ""))
                filepath = unit_id.split("::")[0] if "::" in unit_id else "unknown_file"

            if filepath not in file_grouped:
                file_grouped[filepath] = []
            file_grouped[filepath].append(unit)

        assembled_blocks = []
        current_length = 0

        if query and query.strip():
            header = f"### User Query\n{query.strip()}\n\n### Relevant Code Context\n"
            assembled_blocks.append(header)
            current_length += len(header)

        for filepath, units in file_grouped.items():
            if current_length >= self.max_chars:
                logger.warning("Token budget exhausted. Stopping context assembly.")
                break

            file_header = f"File: `{filepath}`\n```python\n"
            file_body_parts = []

            for u in units:
                snippet = str(u.get("text", "")).strip()
                if not snippet:
                    continue

                meta = u.get("metadata") or {}
                lineno = meta.get("lineno")
                line_info = f"# Line {lineno}\n" if lineno is not None else ""
                file_body_parts.append(f"{line_info}{snippet}")

            if not file_body_parts:
                continue

            file_content = "\n\n".join(file_body_parts) + "\n```\n"
            full_file_block = file_header + file_content

            # Truncate content if it exceeds remaining character budget
            remaining_budget = self.max_chars - current_length
            if len(full_file_block) > remaining_budget:
                if remaining_budget > len(file_header) + 20:
                    truncated_content = full_file_block[:remaining_budget - 15] + "\n... [Truncated]\n```\n"
                    assembled_blocks.append(truncated_content)
                break
            else:
                assembled_blocks.append(full_file_block)
                current_length += len(full_file_block)

        return "\n".join(assembled_blocks) if assembled_blocks else "No relevant context assembled."