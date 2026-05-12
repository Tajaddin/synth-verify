"""Generator: prompt an LLM to produce a batch of examples for a TaskSpec.

The generator rotates through ``TaskSpec.diversity_seeds`` per batch so the
LLM doesn't keep producing examples about the same subject. The output is
parsed as a JSON array; any element that fails ``json.loads`` is recorded
but never reaches the rest of the pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterator

from synth_verify.llm import LLMClient
from synth_verify.spec import TaskSpec

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _parse_batch(text: str) -> list[dict]:
    text = _strip_fences(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


GEN_SYSTEM = (
    "You generate synthetic training data. Strictly follow the format and "
    "schema described. Return JSON only — no commentary, no markdown fences."
)


@dataclass
class GenerationStats:
    requested: int = 0
    received: int = 0
    parse_failures: int = 0


class Generator:
    """Drive an LLM to produce ``n_examples`` for a :class:`TaskSpec`."""

    def __init__(self, llm: LLMClient, spec: TaskSpec, *, batch_size: int = 5) -> None:
        self.llm = llm
        self.spec = spec
        self.batch_size = batch_size
        self.stats = GenerationStats()

    def generate(self, n_examples: int) -> list[dict]:
        return list(self.iter_generate(n_examples))

    def iter_generate(self, n_examples: int) -> Iterator[dict]:
        """Yield examples one at a time. Calls the LLM in batches of ``batch_size``."""
        self.stats = GenerationStats(requested=n_examples)
        n_batches = (n_examples + self.batch_size - 1) // self.batch_size
        seeds = self.spec.diversity_seeds or [None]  # type: ignore[list-item]
        yielded = 0
        for batch_idx in range(n_batches):
            if yielded >= n_examples:
                break
            seed = seeds[batch_idx % len(seeds)]
            system = GEN_SYSTEM
            user_prompt = self.spec.render_for_prompt(
                batch_size=min(self.batch_size, n_examples - yielded), seed=seed
            )
            completion = self.llm.complete(user_prompt, system=system, max_tokens=1500)
            items = _parse_batch(completion.text)
            if not items:
                self.stats.parse_failures += 1
                continue
            for it in items:
                if yielded >= n_examples:
                    break
                self.stats.received += 1
                yielded += 1
                yield it
