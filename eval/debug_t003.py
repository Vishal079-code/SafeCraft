"""
One-off diagnostic: reruns ONLY T003 through the real Gemini pipeline and
prints the anonymized prompt sent + the raw LLM response received, so we
can see EXACTLY what the SEARCH block looked like vs the actual
anonymized file content - rather than guessing why the match failed.
"""
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from privacy.anonymizer import anonymize_code
from agent.llm_client import build_llm_caller
from agent.orchestrator import build_system_prompt

TARGET_FILE = "models/user.py"
PROTECTED_IDS = ["to_dict", "password_hash"]
USER_REQUEST = "Add a docstring to the User.to_dict method explaining it excludes password_hash"

original_code = Path(f"sample_repo/{TARGET_FILE}").read_text(encoding="utf-8")
anonymized_code, privacy_maps = anonymize_code(
    code_snippet=original_code, filepath=TARGET_FILE,
    protected_identifiers=PROTECTED_IDS, protected_literals=[],
)

print("=" * 60)
print("  ANONYMIZED CODE ACTUALLY IN THE FILE (what SEARCH must match)")
print("=" * 60)
print(repr(anonymized_code))
print()
print("--- Human readable ---")
print(anonymized_code)

prompt = f"{build_system_prompt()}\n\nUser Task: {USER_REQUEST}\n\nFile: {TARGET_FILE}\n```python\n{anonymized_code}\n```"

caller = build_llm_caller(provider="gemini")
response = caller(prompt)

print("\n" + "=" * 60)
print("  RAW LLM RESPONSE")
print("=" * 60)
print(response)

# Extract just the SEARCH block for direct comparison
import re
match = re.search(r"<<<<<<< SEARCH\n(.*?)\n=======", response, re.DOTALL)
if match:
    search_block = match.group(1)
    print("\n" + "=" * 60)
    print("  EXTRACTED SEARCH BLOCK (repr, to see exact whitespace)")
    print("=" * 60)
    print(repr(search_block))
    print()
    print("Is this SEARCH block an exact substring of the anonymized code?", search_block in anonymized_code)
else:
    print("\nNo SEARCH block found in response at all.")