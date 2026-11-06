"""Generate 200 NLI training examples, verify each, report pass rate + failure modes.

Generator:  ``llama-3.1-8b-instant``      via Groq (fast, free)
Verifier:   ``llama-3.3-70b-versatile``   via Groq (stronger, *different* model)

Two Groq clients are constructed separately to satisfy the "different LLM"
guard in :class:`Pipeline`. Both are rate-limited so the run stays under
Groq's 30 RPM free-tier ceiling.

Outputs:
* ``bench/nli_examples.jsonl`` — every example with full scoring metadata
* ``bench/nli_summary.json``   — pass rate + per-failure-mode counts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synth_verify import (
    Generator,
    GroqClient,
    HeuristicChecks,
    Pipeline,
    TaskSpec,
    Verifier,
)


BENCH = Path(__file__).resolve().parent
EXAMPLES_OUT = BENCH / "nli_examples.jsonl"
SUMMARY_OUT = BENCH / "nli_summary.json"


NLI_SEEDS = [
    "cooking and food",
    "sports and athletics",
    "weather and seasons",
    "history and historical events",
    "music and musicians",
    "animals and wildlife",
    "science experiments",
    "technology and computers",
    "travel and geography",
    "literature and books",
    "movies and television",
    "transportation and vehicles",
]


def build_spec() -> TaskSpec:
    return TaskSpec(
        name="nli",
        description=(
            "Generate Natural Language Inference triples. Each triple has a premise "
            "sentence, a hypothesis sentence, and a label in {entailment, contradiction, "
            "neutral}. The label must accurately describe whether the hypothesis follows "
            "from the premise. Premises should be 5-25 word factual statements. "
            "Hypotheses should be 5-15 word claims that are entailed by, contradict, "
            "or are neutral with respect to the premise. Premise and hypothesis must "
            "not share more than half of their characters verbatim."
        ),
        label_field="label",
        valid_labels=["entailment", "contradiction", "neutral"],
        text_fields=["premise", "hypothesis"],
        diversity_seeds=NLI_SEEDS,
        schema_hint={
            "premise": "a 5-25 word factual sentence",
            "hypothesis": "a 5-15 word claim about the premise",
            "label": "string in {entailment, contradiction, neutral}",
        },
    )


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--n-examples", type=int, default=200)
    p.add_argument("--gen-model", type=str, default="llama-3.1-8b-instant")
    p.add_argument("--ver-model", type=str, default="llama-3.3-70b-versatile")
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--min-interval", type=float, default=2.1)
    p.add_argument("--out-examples", type=str, default=str(EXAMPLES_OUT))
    p.add_argument("--out-summary", type=str, default=str(SUMMARY_OUT))
    args = p.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set", file=sys.stderr)
        return 2

    spec = build_spec()
    gen_client = GroqClient(model=args.gen_model, min_interval_secs=args.min_interval)
    ver_client = GroqClient(model=args.ver_model, min_interval_secs=args.min_interval)
    generator = Generator(gen_client, spec, batch_size=args.batch_size)
    verifier = Verifier(ver_client, spec)
    pipeline = Pipeline(spec, generator, verifier)

    print(
        f"Generator: {gen_client.model}\nVerifier:  {ver_client.model}\n"
        f"Generating {args.n_examples} examples in batches of {args.batch_size}...\n"
    )
    result = pipeline.run(args.n_examples)
    print(f"\nGenerated: {len(result.examples)} (gen stats: {generator.stats})")
    print(f"Heuristic pass rate: {result.heuristic_pass_rate:.3f}")
    print(f"Verifier pass rate (of those that reached verifier): {result.verifier_pass_rate:.3f}")
    print(f"Overall pass rate: {result.pass_rate:.3f}")
    print(f"Failure breakdown: {result.failure_breakdown()}")

    Path(args.out_examples).write_text(
        "\n".join(json.dumps(ex.to_dict(), ensure_ascii=False) for ex in result.examples),
        encoding="utf-8",
    )

    label_counts: dict[str, int] = {}
    for ex in result.examples:
        if ex.score.passed:
            lab = ex.data.get(spec.label_field, "?")
            label_counts[lab] = label_counts.get(lab, 0) + 1
    summary = {
        "n_generated": len(result.examples),
        "heuristic_pass_rate": round(result.heuristic_pass_rate, 4),
        "verifier_pass_rate_of_those_reached": round(result.verifier_pass_rate, 4),
        "overall_pass_rate": round(result.pass_rate, 4),
        "failure_breakdown": result.failure_breakdown(),
        "verified_label_counts": label_counts,
        "verifier_rejections_by_label": {
            label: {**row, "rejection_rate": round(row["rejection_rate"], 4)}
            for label, row in result.verifier_rejections_by_label(spec.label_field).items()
        },
        "gen_model": gen_client.model,
        "ver_model": ver_client.model,
        "n_target": args.n_examples,
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out_examples} and {args.out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
