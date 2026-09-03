import ast
from agent.skeletonizer import skeletonize_code
from agent.context_bundler import ContextBundler
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

    # 1. Target file full source, dependency skeletonized (unchanged
    # behavior when target_symbol is not given)
    raw_bundle = bundler.build_bundled_context("main.py")
    assert "def run():" in raw_bundle["main.py"], "Target file full source missing!"
    assert "INTERNAL_SECRET" not in raw_bundle["auth.py"], "Dependency file was not skeletonized!"
    assert "..." in raw_bundle["auth.py"]
    print("✅ Test 1 Passed: Primary target kept full; dependency skeletonized.")

    # 2. Token budget enforcement: an unreasonably small budget should
    # drop the dependency file but NEVER the target file itself.
    tiny_budget_bundler = ContextBundler(workspace_files=workspace, token_budget=1)
    budget_bundle = tiny_budget_bundler.build_bundled_context("main.py")
    assert "main.py" in budget_bundle, "Target file must never be dropped for budget."
    assert "auth.py" not in budget_bundle, "Dependency should be dropped under a tiny budget."
    print("✅ Test 2 Passed: Token budget drops dependencies, never the target file.")

    # 3. Target-scope extraction (V2): with target_symbol set, only that
    # function's real body should be present in the target file itself -
    # everything else in it is skeletonized too.
    multi_func_workspace = {
        "product.py": (
            "class Product:\n"
            "    def apply_discount(self, percent):\n"
            "        self.price = round(self.price * (1 - percent / 100), 2)\n"
            "        return self.price\n\n"
            "    def restock(self, qty):\n"
            "        self.stock += qty\n"
            "        return self.stock\n"
        )
    }
    scoped_bundler = ContextBundler(workspace_files=multi_func_workspace, token_budget=5000)
    scoped_bundle = scoped_bundler.build_bundled_context("product.py", target_symbol="apply_discount")
    assert "self.price = round" in scoped_bundle["product.py"], "Target function body missing!"
    assert "self.stock += qty" not in scoped_bundle["product.py"], "Non-target function should be skeletonized!"
    assert "..." in scoped_bundle["product.py"]
    print("✅ Test 3 Passed: Target-scope extraction keeps only the target function's real body.")

    # 4. Splice-back: editing the scoped view and splicing into the
    # ORIGINAL must preserve every untouched function's real body -
    # this is the safety mechanism that makes target_symbol safe to use
    # at all (see project README, bug write-up on ast.unparse reformatting).
    from agent.context_bundler import splice_function_into_original

    patched_scoped_view = scoped_bundle["product.py"].replace(
        "self.price = round(self.price * (1 - percent / 100), 2)",
        "if percent > 90:\n            raise ValueError('too much')\n        self.price = round(self.price * (1 - percent / 100), 2)",
    )
    spliced = splice_function_into_original(
        original_content=multi_func_workspace["product.py"],
        target_symbol="apply_discount",
        patched_content_with_target=patched_scoped_view,
    )
    assert "raise ValueError('too much')" in spliced, "Splice did not apply the target function's edit!"
    assert "self.stock += qty" in spliced, "Splice must preserve untouched functions' real bodies!"
    assert "return self.stock" in spliced
    print("✅ Test 4 Passed: Splice-back applies the edit while preserving untouched functions.")