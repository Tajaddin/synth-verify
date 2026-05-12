"""Verifier: a *separate* LLM scores each example against the TaskSpec.

The verifier prompt asks for a one-line ``VERIFIED: YES|NO`` + ``REASON``.
A NO with a parseable reason is more useful than a NO with no explanation,
so when the parser can't find a reason it records the raw model output
so failures can be triaged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from synth_verify.llm import LLMClient
from synth_verify.spec import TaskSpec

_VERIFIED_RE = re.compile(r"VERIFIED\s*:\s*(YES|NO)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


@dataclass
class VerifierVerdict:
    verified: bool
    reason: str
    raw_text: str
    parse_failed: bool = False


VERIFIER_SYSTEM = (
    "You are a strict data-quality reviewer for synthetic training examples. "
    "Reject anything ambiguous, label-incorrect, malformed, or unfaithful to the task description. "
    "Do not write more than the two-line VERIFIED / REASON response."
)


class Verifier:
    """Wrap an LLM into a single :meth:`verify` per example."""

    def __init__(self, llm: LLMClient, spec: TaskSpec) -> None:
        if llm is None:
            raise ValueError("Verifier requires an LLM (should be a *different* model from the generator).")
        self.llm = llm
        self.spec = spec

    def verify(self, example: dict) -> VerifierVerdict:
        prompt = self.spec.render_for_verifier(example)
        completion = self.llm.complete(prompt, system=VERIFIER_SYSTEM, max_tokens=120)
        text = completion.text or ""
        m = _VERIFIED_RE.search(text)
        if m is None:
            return VerifierVerdict(verified=False, reason="(unparseable)", raw_text=text, parse_failed=True)
        verified = m.group(1).upper() == "YES"
        r = _REASON_RE.search(text)
        reason = r.group(1).strip() if r else ""
        return VerifierVerdict(verified=verified, reason=reason, raw_text=text)
