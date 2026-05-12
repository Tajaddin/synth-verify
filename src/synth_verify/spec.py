"""TaskSpec: the contract that defines what synthesis + verification do.

A spec carries:

* a human-readable description (shown to both generator and verifier prompts)
* a Pydantic schema each generated example must conform to
* a list of valid label values, when the task has categorical labels
* an optional list of "diversity seeds" — short topical hints that the
  generator must rotate through so it doesn't keep producing examples about
  the same subject
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskSpec:
    """Description of the data-generation task.

    Example::

        TaskSpec(
            name="nli",
            description=(
                "Generate Natural Language Inference triples. Each triple has "
                "a premise, a hypothesis, and a label in {entailment, "
                "contradiction, neutral}. The label must accurately describe "
                "whether the hypothesis follows from the premise."
            ),
            label_field="label",
            valid_labels=["entailment", "contradiction", "neutral"],
            text_fields=["premise", "hypothesis"],
            diversity_seeds=["cooking", "sports", "weather", "history", ...],
        )
    """

    name: str
    description: str
    label_field: str | None = None
    valid_labels: list[str] | None = None
    text_fields: list[str] = field(default_factory=list)
    diversity_seeds: list[str] = field(default_factory=list)
    # The dict-shape each generated example should have. Keys are field names;
    # values are short type descriptions ("string", "string in {...}", etc.).
    schema_hint: dict[str, str] = field(default_factory=dict)

    def render_for_prompt(self, *, batch_size: int = 5, seed: str | None = None) -> str:
        """Format the spec into a system prompt fragment for the generator."""
        seed_clause = f"\nThis batch should focus on: {seed}." if seed else ""
        fields = "\n".join(
            f"  - {k}: {v}" for k, v in self.schema_hint.items()
        ) or "  (no field hints)"
        labels = (
            f"\nThe `{self.label_field}` field MUST be one of: {self.valid_labels}."
            if self.label_field and self.valid_labels
            else ""
        )
        return (
            f"Task: {self.description}{seed_clause}\n\n"
            f"Each example is a JSON object with these fields:\n{fields}{labels}\n\n"
            f"Return a JSON array of exactly {batch_size} distinct examples. "
            "No prose, no markdown fences."
        )

    def render_for_verifier(self, example: dict[str, Any]) -> str:
        """Format a verifier prompt asking a different LLM to score one example."""
        import json as _json

        body = _json.dumps(example, ensure_ascii=False)
        labels = (
            f"The valid labels are: {self.valid_labels}.\n"
            if self.label_field and self.valid_labels
            else ""
        )
        return (
            f"You verify a single synthetic training example for the task:\n\n"
            f"{self.description}\n\n"
            f"{labels}"
            f"Example to verify:\n{body}\n\n"
            "Reply with EXACTLY one line in the format:\n"
            'VERIFIED: <YES or NO>\n'
            'REASON: <one short sentence>\n'
            "Output nothing else."
        )
