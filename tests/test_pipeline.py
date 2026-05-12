"""Pipeline end-to-end tests with mock LLMs."""

from __future__ import annotations

import json

import pytest

from synth_verify import (
    Generator,
    HeuristicChecks,
    MockLLMClient,
    Pipeline,
    TaskSpec,
    Verifier,
)


def _spec() -> TaskSpec:
    return TaskSpec(
        name="nli",
        description="NLI triples.",
        label_field="label",
        valid_labels=["entailment", "contradiction", "neutral"],
        text_fields=["premise", "hypothesis"],
        schema_hint={"premise": "str", "hypothesis": "str", "label": "str"},
    )


def test_pipeline_requires_distinct_llm_instances() -> None:
    spec = _spec()
    shared = MockLLMClient(handler=lambda p, s: "[]")
    gen = Generator(shared, spec)
    ver = Verifier(shared, spec)
    with pytest.raises(ValueError):
        Pipeline(spec, gen, ver)


def test_pipeline_pass_rate_with_clean_inputs() -> None:
    spec = _spec()
    examples = [
        {"premise": "Dogs bark loudly.", "hypothesis": "Canines make noise.", "label": "entailment"},
        {"premise": "The sky is blue.", "hypothesis": "The sky is green.", "label": "contradiction"},
    ]
    gen = Generator(MockLLMClient(handler=lambda p, s: json.dumps(examples)), spec, batch_size=2)
    ver = Verifier(
        MockLLMClient(model="v", handler=lambda p, s: "VERIFIED: YES\nREASON: ok"),
        spec,
    )
    result = Pipeline(spec, gen, ver).run(2)
    assert result.pass_rate == 1.0
    assert result.verifier_pass_rate == 1.0


def test_pipeline_catches_leakage_before_verifier() -> None:
    spec = _spec()
    bad = [
        {"premise": "same text here", "hypothesis": "same text here", "label": "entailment"},
    ]
    ver_calls = []

    def ver_handler(prompt, system):
        ver_calls.append(prompt)
        return "VERIFIED: YES\nREASON: x"

    gen = Generator(MockLLMClient(handler=lambda p, s: json.dumps(bad)), spec, batch_size=1)
    ver = Verifier(MockLLMClient(model="v", handler=ver_handler), spec)
    result = Pipeline(spec, gen, ver, checks=HeuristicChecks(spec=spec)).run(1)
    # Heuristic caught the leakage; verifier was never called.
    assert result.pass_rate == 0.0
    assert "no_leakage" in result.failure_breakdown()
    assert len(ver_calls) == 0


def test_pipeline_failure_breakdown_counts_failures() -> None:
    spec = _spec()
    examples = [
        {"premise": "Dogs bark.", "hypothesis": "Canines vocalize.", "label": "entailment"},
        {"premise": "x", "hypothesis": "y", "label": "ufo-test"},  # bad label + short
        {"premise": "same", "hypothesis": "same", "label": "entailment"},  # leakage
    ]
    gen = Generator(MockLLMClient(handler=lambda p, s: json.dumps(examples)), spec, batch_size=3)
    ver = Verifier(MockLLMClient(model="v", handler=lambda p, s: "VERIFIED: YES\nREASON: ok"), spec)
    result = Pipeline(spec, gen, ver).run(3)
    breakdown = result.failure_breakdown()
    assert breakdown.get("label_valid", 0) >= 1
    assert breakdown.get("no_leakage", 0) >= 1


def test_pipeline_records_verifier_reason_on_rejection() -> None:
    spec = _spec()
    examples = [
        {"premise": "Dogs bark.", "hypothesis": "Cats meow.", "label": "entailment"},
    ]
    gen = Generator(MockLLMClient(handler=lambda p, s: json.dumps(examples)), spec, batch_size=1)
    ver = Verifier(
        MockLLMClient(model="v", handler=lambda p, s: "VERIFIED: NO\nREASON: hypothesis is unrelated"),
        spec,
    )
    result = Pipeline(spec, gen, ver).run(1)
    ex = result.examples[0]
    assert ex.score.verifier_verified is False
    assert "unrelated" in ex.score.verifier_reason


def test_pipeline_to_dict_round_trip() -> None:
    spec = _spec()
    examples = [
        {"premise": "Dogs bark.", "hypothesis": "Canines vocalize.", "label": "entailment"},
    ]
    gen = Generator(MockLLMClient(handler=lambda p, s: json.dumps(examples)), spec, batch_size=1)
    ver = Verifier(MockLLMClient(model="v", handler=lambda p, s: "VERIFIED: YES\nREASON: ok"), spec)
    result = Pipeline(spec, gen, ver).run(1)
    d = result.examples[0].to_dict()
    assert d["data"] == examples[0]
    assert d["score"]["passed"] is True
    json.dumps(d)  # round-trip serializable


def test_generator_parses_fenced_output() -> None:
    spec = _spec()
    examples = [{"premise": "p", "hypothesis": "h", "label": "neutral"}]
    fenced = f"```json\n{json.dumps(examples)}\n```"
    gen = Generator(MockLLMClient(handler=lambda p, s: fenced), spec, batch_size=1)
    out = list(gen.iter_generate(1))
    assert len(out) == 1
    assert out[0]["premise"] == "p"


def test_generator_handles_unparseable_output() -> None:
    spec = _spec()
    gen = Generator(MockLLMClient(handler=lambda p, s: "this is not JSON"), spec, batch_size=2)
    out = list(gen.iter_generate(2))
    assert out == []
    assert gen.stats.parse_failures >= 1


def test_verifier_handles_unparseable_response() -> None:
    spec = _spec()
    ver = Verifier(MockLLMClient(model="v", handler=lambda p, s: "what?"), spec)
    verdict = ver.verify({"premise": "p", "hypothesis": "h", "label": "neutral"})
    assert verdict.verified is False
    assert verdict.parse_failed is True
