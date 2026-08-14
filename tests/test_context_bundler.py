"""
Test Day 16: Multi-File Context Bundling & Dependency Tracking
"""

from agent.context_bundler import (
    ContextBundler,
    extract_local_imports,
    resolve_import_to_filepath,
)


def test_import_extraction_and_bundling():
    workspace = {
        "main.py": "import auth.models\nfrom utils.helpers import calculate_tax\nfrom .config import settings",
        "config.py": "settings = {}",
        "auth/models.py": "class User: pass",
        "auth/service.py": "from .models import User",
        "utils/helpers.py": "def calculate_tax(): pass",
        "unrelated.py": "x = 100",
    }

    print("=" * 60)
    print("  TEST 1: Extract Local & Relative Imports")
    print("=" * 60)
    imports_main = extract_local_imports(workspace["main.py"], current_filepath="main.py")
    print(f"Main Discovered Imports: {imports_main}")
    assert "auth.models" in imports_main
    assert "utils.helpers" in imports_main

    imports_service = extract_local_imports(
        workspace["auth/service.py"], current_filepath="auth/service.py"
    )
    print(f"Service Discovered Imports: {imports_service}")
    assert "auth.models" in imports_service
    print("[PASS] Absolute and relative imports successfully extracted!")

    print("\n" + "=" * 60)
    print("  TEST 2: Context Bundling Assembly")
    print("=" * 60)
    bundler = ContextBundler(workspace_files=workspace)
    bundled = bundler.build_bundled_context("main.py")

    print(f"Bundled Files for main.py: {list(bundled.keys())}")
    assert "main.py" in bundled
    assert "auth/models.py" in bundled
    assert "utils/helpers.py" in bundled
    assert "config.py" in bundled
    assert "unrelated.py" not in bundled
    print("[PASS] Multi-file dependency context successfully assembled!")


if __name__ == "__main__":
    test_import_extraction_and_bundling()
    print("\n" + "=" * 60)
    print("  ALL DAY 16 TESTS PASSED")
    print("=" * 60)