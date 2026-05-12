"""Heuristic check tests + property tests via hypothesis."""

from __future__ import annotations

from hypothesis import given, strategies as st

from synth_verify import HeuristicChecks, TaskSpec


def _nli_spec() -> TaskSpec:
    return TaskSpec(
        name="nli",
        description="NLI triples.",
        label_field="label",
        valid_labels=["entailment", "contradiction", "neutral"],
        text_fields=["premise", "hypothesis"],
        schema_hint={"premise": "str", "hypothesis": "str", "label": "str"},
    )


def test_format_valid_missing_field() -> None:
    spec = _nli_spec()
    res = HeuristicChecks(spec=spec).run({"premise": "p"})
    failed = [r for r in res if r.check == "format_valid"][0]
    assert failed.passed is False
    assert "hypothesis" in failed.reason or "label" in failed.reason


def test_label_valid_rejects_unknown() -> None:
    spec = _nli_spec()
    res = HeuristicChecks(spec=spec).run({"premise": "a long premise here", "hypothesis": "a long hyp here", "label": "ufo-test"})
    label = [r for r in res if r.check == "label_valid"][0]
    assert label.passed is False


def test_no_leakage_catches_identical_fields() -> None:
    spec = _nli_spec()
    same = "a perfectly identical sentence used twice"
    res = HeuristicChecks(spec=spec).run({"premise": same, "hypothesis": same, "label": "entailment"})
    leak = [r for r in res if r.check == "no_leakage"][0]
    assert leak.passed is False


def test_no_leakage_allows_different_fields() -> None:
    spec = _nli_spec()
    res = HeuristicChecks(spec=spec).run(
        {"premise": "The cat sat on the mat.", "hypothesis": "An animal is resting.", "label": "entailment"}
    )
    leak = [r for r in res if r.check == "no_leakage"][0]
    assert leak.passed is True


def test_length_ok_rejects_short_fields() -> None:
    spec = _nli_spec()
    res = HeuristicChecks(spec=spec, min_text_len=10).run(
        {"premise": "hi", "hypothesis": "longer hypothesis here", "label": "entailment"}
    )
    length = [r for r in res if r.check == "length_ok"][0]
    assert length.passed is False


def test_no_near_duplicate_within_one_session() -> None:
    spec = _nli_spec()
    checks = HeuristicChecks(spec=spec)
    ex = {"premise": "The boy ran.", "hypothesis": "A child moved fast.", "label": "entailment"}
    r1 = checks.run(ex)
    dup_check_1 = [r for r in r1 if r.check == "no_near_duplicate"][0]
    assert dup_check_1.passed is True

    r2 = checks.run(ex)  # exact same example
    dup_check_2 = [r for r in r2 if r.check == "no_near_duplicate"][0]
    assert dup_check_2.passed is False


@given(
    premise=st.text(min_size=10, max_size=200),
    hypothesis=st.text(min_size=10, max_size=200),
)
def test_format_valid_passes_when_all_fields_present(premise: str, hypothesis: str) -> None:
    spec = _nli_spec()
    res = HeuristicChecks(spec=spec).run(
        {"premise": premise, "hypothesis": hypothesis, "label": "entailment"}
    )
    fmt = [r for r in res if r.check == "format_valid"][0]
    assert fmt.passed is True


def test_disabled_check_is_skipped() -> None:
    spec = _nli_spec()
    checks = HeuristicChecks(spec=spec, enabled={"no_leakage": False})
    same = "a perfectly identical sentence"
    res = checks.run({"premise": same, "hypothesis": same, "label": "entailment"})
    # no_leakage should not appear
    assert not any(r.check == "no_leakage" for r in res)
