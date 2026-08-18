"""
Test Runner: executes pytest via subprocess, with a recursion guard that
generalizes to ANY test file (not just one hardcoded name).

Fix: set an environment variable in the SUBPROCESS before it starts. If a
test running INSIDE that subprocess calls run_pytest_suite() again (with
the default test_dir), it inherits the env var, sees it's already inside
a spawned run, and short-circuits immediately instead of spawning yet
another subprocess - breaking any recursion chain at depth 1, regardless
of which file triggered it.
"""
import os
import subprocess
import sys
from typing import Dict, Any

RECURSION_GUARD_ENV_VAR = "AGENT_VERIFIER_IN_SUBPROCESS"


def run_pytest_suite(test_dir: str = "tests/", timeout_sec: int = 30) -> Dict[str, Any]:
    """
    Runs pytest via subprocess and captures stdout, stderr, and return code.
    """
    if os.environ.get(RECURSION_GUARD_ENV_VAR) == "1":
        return {
            "success": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "traceback": "",
            "skipped_recursion_guard": True,
        }

    # DEFENSIVE FIX: on some Windows setups, sys.executable can resolve to
    # None or an empty string (seen in practice on this project - PATH/App
    # Execution Alias quirks were an issue as far back as initial setup).
    # subprocess.run's Windows-specific list2cmdline() requires every
    # argument to be a str/bytes/PathLike, and silently fails with a
    # confusing TypeError deep in the stdlib if any element is None -
    # falling back to the plain string "python" (PATH-resolved) avoids
    # that crash outright, and the explicit validation below turns any
    # OTHER unexpected None into a clear, actionable error instead of an
    # opaque stdlib traceback.
    python_exe = sys.executable or "python"

    try:
        cmd = [python_exe, "-m", "pytest", test_dir, "-v", "--tb=short"]

        missing = [i for i, arg in enumerate(cmd) if arg is None]
        if missing:
            raise ValueError(
                f"run_pytest_suite: cmd list contains None at position(s) {missing}: {cmd}. "
                f"sys.executable={sys.executable!r}, test_dir={test_dir!r}"
            )

        env = os.environ.copy()
        env[RECURSION_GUARD_ENV_VAR] = "1"
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
        passed = (completed.returncode == 0)
        return {
            "success": passed,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "traceback": completed.stdout if not passed else ""
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout_sec} seconds.",
            "traceback": f"Test runner timed out after {timeout_sec} seconds."
        }
def run_pytest_suite(test_dir: str = "tests/", timeout_sec: int = 30) -> Dict[str, Any]:
    if test_dir is None:
        test_dir = "tests/"
    if os.environ.get(RECURSION_GUARD_ENV_VAR) == "1":
        return {
            "success": True, "returncode": 0, "stdout": "", "stderr": "",
            "traceback": "", "skipped_recursion_guard": True,
        }
    try:
        cmd = [sys.executable, "-m", "pytest", test_dir, "-v", "--tb=short"]
        env = os.environ.copy()
        env[RECURSION_GUARD_ENV_VAR] = "1"
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, env=env)
        passed = (completed.returncode == 0)
        return {
            "success": passed, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
            "traceback": completed.stdout if not passed else ""
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False, "returncode": -1, "stdout": "",
            "stderr": f"Execution timed out after {timeout_sec} seconds.",
            "traceback": f"Test runner timed out after {timeout_sec} seconds."
        }