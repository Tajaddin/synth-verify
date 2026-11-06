"""End-to-end pipeline: generator → heuristic gate → verifier → quality score.

Each example gets a :class:`QualityScore` that records:

* ``heuristic_passed`` — every enabled heuristic check passed
* ``verifier_verified`` — the verifier LLM said YES
* ``passed`` — both of the above

Order matters: heuristics run first so we don't spend verifier-LLM tokens on
malformed or duplicate examples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterator

from synth_verify.checks import CheckResult, HeuristicChecks, aggregate_pass
from synth_verify.generator import Generator
from synth_verify.spec import TaskSpec
from synth_verify.verifier import Verifier, VerifierVerdict


@dataclass
class QualityScore:
    """Per-example quality signals."""

    heuristic_passed: bool
    verifier_verified: bool
    passed: bool
    failed_checks: list[str] = field(default_factory=list)
    verifier_reason: str = ""


@dataclass
class Example:
    """One synthesized + scored example."""

    data: dict
    score: QualityScore
    checks: list[CheckResult] = field(default_factory=list)
    verifier: VerifierVerdict | None = None

    def to_dict(self) -> dict:
        return {
            "data": self.data,
            "score": asdict(self.score),
            "checks": [{"check": c.check, "passed": c.passed, "reason": c.reason} for c in self.checks],
            "verifier": (
                {
                    "verified": self.verifier.verified,
                    "reason": self.verifier.reason,
                    "raw_text": self.verifier.raw_text,
                    "parse_failed": self.verifier.parse_failed,
                }
                if self.verifier is not None
                else None
            ),
        }


@dataclass
class PipelineResult:
    examples: list[Example]

    @property
    def pass_rate(self) -> float:
        if not self.examples:
            return 0.0
        n_pass = sum(1 for e in self.examples if e.score.passed)
        return n_pass / len(self.examples)

    @property
    def heuristic_pass_rate(self) -> float:
        if not self.examples:
            return 0.0
        return sum(1 for e in self.examples if e.score.heuristic_passed) / len(self.examples)

    @property
    def verifier_pass_rate(self) -> float:
        if not self.examples:
            return 0.0
        # Only examples that reached the verifier — i.e. cleared the heuristic gate.
        reached = [e for e in self.examples if e.verifier is not None]
        if not reached:
            return 0.0
        return sum(1 for e in reached if e.score.verifier_verified) / len(reached)

    def verifier_rejections_by_label(self, label_field: str) -> dict[str, dict[str, float]]:
        """Verifier outcome per generated label, over the examples that reached the verifier.

        When rejection rates differ by label, the verified set has a different
        label mix from what was generated, which a single pass rate hides.
        """
        by_label: dict[str, dict[str, float]] = {}
        for ex in self.examples:
            if ex.verifier is None:
                continue
            row = by_label.setdefault(str(ex.data.get(label_field, "?")), {"reached": 0, "rejected": 0})
            row["reached"] += 1
            if not ex.score.verifier_verified:
                row["rejected"] += 1
        for row in by_label.values():
            row["rejection_rate"] = row["rejected"] / row["reached"]
        return by_label

    def failure_breakdown(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ex in self.examples:
            if ex.score.passed:
                continue
            for failed in ex.score.failed_checks:
                counts[failed] = counts.get(failed, 0) + 1
            if not ex.score.verifier_verified and ex.verifier is not None:
                counts["verifier_rejected"] = counts.get("verifier_rejected", 0) + 1
        return counts


class Pipeline:
    """Synthesize ``n_examples`` for ``spec``, gate with heuristics, verify with a separate LLM."""

    def __init__(
        self,
        spec: TaskSpec,
        generator: Generator,
        verifier: Verifier,
        checks: HeuristicChecks | None = None,
    ) -> None:
        if generator.llm is verifier.llm:
            raise ValueError(
                "Generator and verifier must use *different* LLMClient instances. "
                "Pass two different clients (or two different models)."
            )
        self.spec = spec
        self.generator = generator
        self.verifier = verifier
        self.checks = checks or HeuristicChecks(spec=spec)

    def run(self, n_examples: int) -> PipelineResult:
        examples: list[Example] = []
        for raw in self.generator.iter_generate(n_examples):
            check_results = self.checks.run(raw)
            heuristic_ok = aggregate_pass(check_results)
            failed = [c.check for c in check_results if not c.passed]
            verdict: VerifierVerdict | None = None
            verifier_ok = False
            if heuristic_ok:
                verdict = self.verifier.verify(raw)
                verifier_ok = verdict.verified
            score = QualityScore(
                heuristic_passed=heuristic_ok,
                verifier_verified=verifier_ok,
                passed=heuristic_ok and verifier_ok,
                failed_checks=failed,
                verifier_reason=verdict.reason if verdict else "",
            )
            examples.append(Example(data=raw, score=score, checks=check_results, verifier=verdict))
        return PipelineResult(examples=examples)
