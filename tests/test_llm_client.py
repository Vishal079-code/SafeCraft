"""
Test Day 15: LLM Client Wrapper & Fallback Architecture (Gemini + Anthropic)
"""

import os
from unittest.mock import patch
from agent.llm_client import build_llm_caller, LLMClientError


def mock_provider(prompt: str) -> str:
    return "MOCK_RESPONSE"


def test_gemini_missing_api_key_fallback():
    # Force clean environment to ensure fallback is triggered
    with patch.dict(os.environ, {}, clear=True):
        caller = build_llm_caller(
            provider="gemini",
            api_key=None,
            mock_fallback=mock_provider,
        )

        print("=" * 60)
        print("  TEST 1: Fallback to Mock Provider when Gemini API Key Missing")
        print("=" * 60)

        response = caller("Test Prompt")
        assert response == "MOCK_RESPONSE"
        print("\n[PASS] Correctly fell back to mock provider when Gemini key was absent!")


def test_gemini_missing_api_key_error():
    # Force clean environment with no key and no fallback
    with patch.dict(os.environ, {}, clear=True):
        caller = build_llm_caller(
            provider="gemini",
            api_key=None,
            mock_fallback=None,
        )

        print("\n" + "=" * 60)
        print("  TEST 2: Raise LLMClientError when No Gemini Key & No Fallback")
        print("=" * 60)

        try:
            caller("Test Prompt")
            assert False, "Should have raised LLMClientError!"
        except LLMClientError as e:
            print(f"[PASS] Correctly raised LLMClientError: {e}")


def test_anthropic_missing_api_key_fallback():
    # Force clean environment to ensure fallback is triggered
    with patch.dict(os.environ, {}, clear=True):
        caller = build_llm_caller(
            provider="anthropic",
            api_key=None,
            mock_fallback=mock_provider,
        )

        print("\n" + "=" * 60)
        print("  TEST 3: Fallback to Mock Provider when Anthropic API Key Missing")
        print("=" * 60)

        response = caller("Test Prompt")
        assert response == "MOCK_RESPONSE"
        print("\n[PASS] Correctly fell back to mock provider when Anthropic key was absent!")


def test_unsupported_provider():
    print("\n" + "=" * 60)
    print("  TEST 4: Reject Invalid Provider")
    print("=" * 60)

    try:
        caller = build_llm_caller(provider="unsupported_llm")
        caller("Test Prompt")
        assert False, "Should have raised ValueError!"
    except ValueError as e:
        print(f"[PASS] Correctly rejected unsupported provider: {e}")


if __name__ == "__main__":
    test_gemini_missing_api_key_fallback()
    test_gemini_missing_api_key_error()
    test_anthropic_missing_api_key_fallback()
    test_unsupported_provider()
    print("\n" + "=" * 60)
    print("  ALL DAY 15 TESTS PASSED")
    print("=" * 60)