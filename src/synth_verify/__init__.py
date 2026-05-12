"""Synthetic training-data pipeline with separate-model verification.

Three building blocks:

* :class:`Generator` — prompts an LLM to produce examples for a task spec
* :class:`Verifier`  — sends each example to a *different* LLM that scores
  whether the example matches the spec
* :class:`HeuristicChecks` — cheap structural checks (leakage, near-duplicate,
  label balance, format) that catch failures the verifier might miss

Wire them together with :class:`Pipeline`. Outputs a JSONL of every example
with its full set of quality signals so a reviewer can audit + filter.
"""

from synth_verify.checks import CheckResult, HeuristicChecks
from synth_verify.generator import Generator
from synth_verify.pipeline import Example, Pipeline, PipelineResult, QualityScore
from synth_verify.spec import TaskSpec
from synth_verify.verifier import VerifierVerdict, Verifier
from synth_verify.llm import LLMClient, MockLLMClient, GroqClient, AnthropicClient

__version__ = "0.1.0"

__all__ = [
    "Generator",
    "Verifier",
    "VerifierVerdict",
    "HeuristicChecks",
    "CheckResult",
    "Pipeline",
    "PipelineResult",
    "Example",
    "QualityScore",
    "TaskSpec",
    "LLMClient",
    "MockLLMClient",
    "GroqClient",
    "AnthropicClient",
]
