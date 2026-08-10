"""
Central config. Every module imports settings from here instead of
calling os.getenv() in scattered places — one source of truth,
easy to change later.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TARGET_REPO_PATH = Path(os.getenv("TARGET_REPO_PATH", "./sample_repo"))
MAX_TOOL_CALL_ROUNDS = int(os.getenv("MAX_TOOL_CALL_ROUNDS", "8"))
TOKEN_BUDGET_PER_TASK = int(os.getenv("TOKEN_BUDGET_PER_TASK", "50000"))

PROJECT_ROOT = Path(__file__).parent
