import ast
from agent.skeletonizer import skeletonize_code
from agent.context_bundler import ContextBundler

def test_skeletonizer():
    raw_code = '''
class AuthVault:
    """Vault class docstring."""
    def verify_token(self, token: str) -> bool:
        """Check secret token validity."""
        secret_pass = "DATABASE_SECRET_KEY_12345"
        return token == secret_pass
'''
    skeleton = skeletonize_code(raw_code)
    
    # Assert secret implementation is stripped
    assert "DATABASE_SECRET_KEY_12345" not in skeleton, "Failed: Implementation body was not stripped!"
    # Assert signature and docstrings are preserved
    assert "def verify_token(self, token: str) -> bool:" in skeleton
    assert "Check secret token validity." in skeleton
    assert "..." in skeleton
    # Assert valid Python AST syntax
    ast.parse(skeleton)
    print("✅ Test 1 Passed: Skeletonizer correctly strips bodies and preserves signatures/docstrings.")

def test_context_bundler():
    workspace = {
        "main.py": "from auth import AuthVault\n\ndef run():\n    vault = AuthVault()\n    return vault.verify_token('test')",
        "auth.py": '''
class AuthVault:
    def verify_token(self, token: str) -> bool:
        db_key = "INTERNAL_SECRET"
        return token == db_key
'''
    }
    
    bundler = ContextBundler(workspace_files=workspace, token_budget=5000)
    
    # 1. Test build_bundled_context
    raw_bundle = bundler.build_bundled_context("main.py")
    assert "def run():" in raw_bundle["main.py"], "Target file full source missing!"
    assert "INTERNAL_SECRET" not in raw_bundle["auth.py"], "Dependency file was not skeletonized!"
    assert "..." in raw_bundle["auth.py"]
    print("✅ Test 2 Passed: Primary target kept full; dependency skeletonized.")

    # 2. Test build_bundled_prompt with budget
    prompt = bundler.build_bundled_prompt("main.py", user_request="Refactor login flow")
    assert "File: `main.py`" in prompt
    assert "File: `auth.py`" in prompt
    print("✅ Test 3 Passed: Budget-enforced prompt generated successfully.")

if __name__ == "__main__":
    test_skeletonizer()
    test_context_bundler()
    print("\n🎉 All verification checks passed with zero regressions!")