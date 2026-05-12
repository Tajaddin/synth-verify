"""LLM client protocol with three implementations: Mock, Groq, Anthropic."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable


@dataclass
class CompletionResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 800) -> CompletionResult: ...


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Deterministic LLM for tests + offline demos."""

    def __init__(self, model: str = "mock", handler=None, canned: dict[str, str] | None = None) -> None:
        self.model = model
        self.handler = handler
        self.canned = canned or {}
        self.calls: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 800) -> CompletionResult:
        self.calls.append((prompt, system))
        if self.handler is not None:
            text = self.handler(prompt, system)
        else:
            text = ""
            for needle, ans in self.canned.items():
                if needle in prompt:
                    text = ans
                    break
        return CompletionResult(
            text=text,
            input_tokens=max(1, (len(prompt) + len(system or "")) // 4),
            output_tokens=max(1, len(text) // 4),
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------


class GroqClient:
    """Rate-limited Groq client. Defaults to llama-3.1-8b-instant.

    Honors a minimum interval between calls so the 30 RPM free-tier ceiling
    isn't hit. Tune via ``min_interval_secs`` or the
    ``SYNTH_VERIFY_GROQ_MIN_INTERVAL`` env var.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        min_interval_secs: float | None = None,
    ) -> None:
        from groq import Groq  # lazy

        self.model = model or os.environ.get("SYNTH_VERIFY_GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        env = os.environ.get("SYNTH_VERIFY_GROQ_MIN_INTERVAL")
        self.min_interval_secs = (
            min_interval_secs if min_interval_secs is not None else (float(env) if env else 2.1)
        )
        self._last_t = 0.0

    def _wait(self) -> None:
        now = time.monotonic()
        delta = now - self._last_t
        if delta < self.min_interval_secs:
            time.sleep(self.min_interval_secs - delta)
        self._last_t = time.monotonic()

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 800) -> CompletionResult:
        from groq import RateLimitError

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        backoff = 4.0
        for attempt in range(5):
            self._wait()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                text = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                return CompletionResult(
                    text=text,
                    input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                    model=self.model,
                )
            except RateLimitError:
                if attempt == 4:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicClient:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from anthropic import Anthropic  # lazy

        self.model = model or os.environ.get("SYNTH_VERIFY_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 800) -> CompletionResult:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return CompletionResult(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
        )
