"""HeuristicChecks: cheap structural checks the verifier might miss.

Each check is stateless and returns a :class:`CheckResult`. The pipeline
runs every enabled check and aggregates the results into the per-example
:class:`QualityScore`.

Built-in checks:

* ``format_valid``     — required fields present, value types match
* ``label_valid``      — label is in TaskSpec.valid_labels
* ``no_leakage``       — premise/hypothesis (and any two text fields)
  don't share a verbatim substring above the threshold ratio
* ``length_ok``        — text fields meet a min character length
* ``no_near_duplicate``— example isn't a near-duplicate of one already seen
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Iterable

from synth_verify.spec import TaskSpec


@dataclass
class CheckResult:
    check: str
    passed: bool
    reason: str = ""


@dataclass
class HeuristicChecks:
    """Collection of structural quality checks.

    Pass ``enabled=False`` to disable any individual check at runtime.
    """

    spec: TaskSpec
    leakage_ratio: float = 0.7
    min_text_len: int = 5
    near_duplicate_ratio: float = 0.92
    enabled: dict[str, bool] = field(default_factory=dict)
    _seen_signatures: list[str] = field(default_factory=list, init=False, repr=False)

    def _is_on(self, name: str) -> bool:
        return self.enabled.get(name, True)

    def run(self, example: dict) -> list[CheckResult]:
        results: list[CheckResult] = []
        if self._is_on("format_valid"):
            results.append(self._format_valid(example))
        if self._is_on("label_valid"):
            results.append(self._label_valid(example))
        if self._is_on("no_leakage"):
            results.append(self._no_leakage(example))
        if self._is_on("length_ok"):
            results.append(self._length_ok(example))
        if self._is_on("no_near_duplicate"):
            results.append(self._no_near_duplicate(example))
        return results

    def reset(self) -> None:
        self._seen_signatures = []

    # ------------------------------------------------------------------
    # individual checks
    # ------------------------------------------------------------------
    def _format_valid(self, ex: dict) -> CheckResult:
        missing = [k for k in self.spec.schema_hint.keys() if k not in ex]
        if missing:
            return CheckResult("format_valid", False, f"missing fields: {missing}")
        for k in self.spec.text_fields:
            if k in ex and not isinstance(ex[k], str):
                return CheckResult("format_valid", False, f"field {k!r} not a string")
        return CheckResult("format_valid", True)

    def _label_valid(self, ex: dict) -> CheckResult:
        if self.spec.label_field is None or self.spec.valid_labels is None:
            return CheckResult("label_valid", True, "no label_field configured")
        v = ex.get(self.spec.label_field)
        if v in self.spec.valid_labels:
            return CheckResult("label_valid", True)
        return CheckResult("label_valid", False, f"label {v!r} not in {self.spec.valid_labels}")

    def _no_leakage(self, ex: dict) -> CheckResult:
        """When two text fields share a verbatim substring above the ratio threshold, flag."""
        if len(self.spec.text_fields) < 2:
            return CheckResult("no_leakage", True, "only one text field")
        a_key, b_key = self.spec.text_fields[:2]
        a, b = str(ex.get(a_key, "")).lower(), str(ex.get(b_key, "")).lower()
        if not a or not b:
            return CheckResult("no_leakage", True, "empty field")
        # Longest common substring ratio.
        matcher = difflib.SequenceMatcher(None, a, b)
        match = matcher.find_longest_match(0, len(a), 0, len(b))
        ratio = match.size / max(min(len(a), len(b)), 1)
        if ratio >= self.leakage_ratio:
            return CheckResult(
                "no_leakage", False,
                f"{a_key!r} and {b_key!r} overlap by {ratio:.2f} of shorter field",
            )
        return CheckResult("no_leakage", True)

    def _length_ok(self, ex: dict) -> CheckResult:
        for k in self.spec.text_fields:
            v = ex.get(k)
            if not isinstance(v, str) or len(v.strip()) < self.min_text_len:
                return CheckResult("length_ok", False, f"{k!r} too short")
        return CheckResult("length_ok", True)

    def _no_near_duplicate(self, ex: dict) -> CheckResult:
        sig = self._signature(ex)
        for prev in self._seen_signatures:
            if difflib.SequenceMatcher(None, sig, prev).ratio() >= self.near_duplicate_ratio:
                return CheckResult(
                    "no_near_duplicate", False, "matches a previous example",
                )
        self._seen_signatures.append(sig)
        return CheckResult("no_near_duplicate", True)

    def _signature(self, ex: dict) -> str:
        parts = [str(ex.get(k, "")).lower().strip() for k in self.spec.text_fields]
        return " | ".join(parts)


def aggregate_pass(checks: Iterable[CheckResult]) -> bool:
    """All checks must pass for an example to clear the heuristic gate."""
    return all(c.passed for c in checks)
