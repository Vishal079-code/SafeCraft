"""
Agent LLM Client: Unified wrapper for live LLM API invocations (Gemini & Anthropic)
with fallback and error resilience.
"""

import os
from typing import Optional, Callable

from dotenv import load_dotenv

load_dotenv()


class LLMClientError(Exception):
    """Custom exception for LLM invocation failures."""
    pass


def get_gemini_client(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gemini-3.6-flash",
    temperature: float = 0.0,
) -> str:
    """
    Calls the Google Gemini API using the official google-genai SDK.
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise LLMClientError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key at https://aistudio.google.com/ and set GEMINI_API_KEY."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise LLMClientError(
            "The 'google-genai' package is not installed. Install it via 'pip install google-genai'."
        )

    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
            ),
        )
        return response.text or ""
    except Exception as e:
        raise LLMClientError(f"Gemini API request failed: {str(e)}")


def get_anthropic_client(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-5",
    temperature: float = 0.0,
) -> str:
    """
    Calls the Anthropic Claude API using the official `anthropic` SDK.
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise LLMClientError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set the environment variable or pass a valid key."
        )

    try:
        import anthropic
    except ImportError:
        raise LLMClientError(
            "The 'anthropic' package is not installed. Install it via 'pip install anthropic'."
        )

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        
        # Concatenate text blocks from output
        output_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return output_text
    except Exception as e:
        raise LLMClientError(f"Anthropic API request failed: {str(e)}")


def build_llm_caller(
    provider: str = "gemini",
    api_key: Optional[str] = None,
    mock_fallback: Optional[Callable[[str], str]] = None,
) -> Callable[[str], str]:
    """
    Factory function returning a uniform llm_caller_fn usable by AgentOrchestrator.
    Defaults to 'gemini'. If no API key is found and mock_fallback is provided,
    it gracefully falls back to the mock handler.
    """
    def caller(prompt: str) -> str:
        if provider == "gemini":
            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key and mock_fallback:
                print("⚠️ No GEMINI_API_KEY found in environment. Falling back to provided mock handler.")
                return mock_fallback(prompt)
            return get_gemini_client(prompt, api_key=key)

        if provider == "anthropic":
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key and mock_fallback:
                print("⚠️ No ANTHROPIC_API_KEY found in environment. Falling back to provided mock handler.")
                return mock_fallback(prompt)
            return get_anthropic_client(prompt, api_key=key)

        raise ValueError(f"Unsupported provider: '{provider}'")

    return caller