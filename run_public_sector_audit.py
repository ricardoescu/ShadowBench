import json
from pathlib import Path

from audits.services.attacks import (
    FRUSTRATED_TONE,
    NEUTRAL_CONTEXT,
    NOISY_WRITING,
)
from audits.services.inference_cache import (
    cached_model_run,
)
from audits.services.public_sector_evaluator import (
    evaluate_public_sector_variants,
)
from audits.services.prompts import (
    CONSTITUENT_TRIAGE_PROMPT,
)
from audits.services.target_model import (
    OllamaTargetModel,
)


BASE_DIR = Path(__file__).resolve().parent

CASES_FILE = (
    BASE_DIR
    / "data"
    / "public_sector_cases.json"
)

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "results"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "qwen_public_sector_audit.json"
)


ATTACKS = [
    FRUSTRATED_TONE,
    NEUTRAL_CONTEXT,
    NOISY_WRITING,
]


def load_cases():
    with open(
        CASES_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_results(results):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "model": "qwen3:1.7b",
                "benchmark":
                    "constituent_treatment_consistency",
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def summarize(results):
    total = len(results)

    if total == 0:
        return {}

    affected_cases = sum(
        result["any_change"]
        for result in results
    )

    variant_changes = {}

    for attack in ATTACKS:
        variant_changes[attack.name] = sum(
            result["variants"]
            [attack.name]
            ["changed"]
            for result in results
        )

    total_variant_tests = (
        total * len(ATTACKS)
    )

    total_variant_changes = sum(
        variant_changes.values()
    )

    mean_max_priority_shift = (
        sum(
            result["max_priority_swing"]
            for result in results
        )
        / total
    )

    return {
        "cases_tested": total,

        "affected_cases":
            affected_cases,

        "case_instability_rate":
            affected_cases / total,

        "variant_changes":
            variant_changes,

        "variant_tests":
            total_variant_tests,

        "total_variant_changes":
            total_variant_changes,

        "variant_change_rate":
            (
                total_variant_changes
                / total_variant_tests
            ),

        "mean_max_priority_shift":
            mean_max_priority_shift,
    }


def main():
    cases = load_cases()

    model = OllamaTargetModel(
        model_name="qwen3:1.7b",
        system_prompt=
            CONSTITUENT_TRIAGE_PROMPT,
    )

    print(
        f"Loaded {len(cases)} "
        "constituent cases."
    )

    print(
        f"Target model: "
        f"{model.model_name}"
    )

    print()

    results = []

    # Resume support
    if RESULTS_FILE.exists():
        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(f)

        results = existing.get(
            "results",
            [],
        )

        print(
            f"Found {len(results)} "
            "existing results."
        )

        print()

    completed_ids = {
        result["case_id"]
        for result in results
    }

    for index, case in enumerate(
        cases,
        start=1,
    ):
        if case["id"] in completed_ids:
            print(
                f"[{index}/{len(cases)}] "
                f"{case['id']} "
                "— already complete"
            )

            continue

        print(
            f"[{index}/{len(cases)}] "
            f"Testing {case['id']}..."
        )

        original_text = case["text"]

        original, original_cached = (
            cached_model_run(
                model,
                original_text,
            )
        )

        variant_outputs = {}
        variant_metadata = {}

        for attack in ATTACKS:
            variant_text = attack.apply(
                original_text
            )

            response, cached = (
                cached_model_run(
                    model,
                    variant_text,
                )
            )

            variant_outputs[
                attack.name
            ] = response

            variant_metadata[
                attack.name
            ] = {
                "text": variant_text,
                "action": response["action"],
                "reason": response["reason"],
                "cached": cached,
            }

        evaluation = (
            evaluate_public_sector_variants(
                original=original,
                variants=variant_outputs,
            )
        )

        result = {
            "case_id": case["id"],
            "domain": case["domain"],

            "original_text":
                original_text,

            "original_reason":
                original["reason"],

            "original_cached":
                original_cached,

            "variant_metadata":
                variant_metadata,

            **evaluation,
        }

        results.append(result)

        save_results(results)

        print(
            f"    Original:         "
            f"{result['original_action']}"
        )

        for attack in ATTACKS:
            name = attack.name

            action = (
                result["variants"]
                [name]
                ["action"]
            )

            print(
                f"    {name:18} "
                f"{action}"
            )

        if result["any_change"]:
            print(
                "    >>> DECISION INSTABILITY"
            )

            print(
                "    Changed variants: "
                + ", ".join(
                    result[
                        "changed_variants"
                    ]
                )
            )

        else:
            print(
                "    Stable across variants"
            )

        print()

    summary = summarize(results)

    print("=" * 55)

    print(
        "SHADOWBENCH — "
        "CONSTITUENT TREATMENT CONSISTENCY"
    )

    print("=" * 55)

    print(
        f"Cases tested: "
        f"{summary['cases_tested']}"
    )

    print(
        f"Cases with any decision change: "
        f"{summary['affected_cases']}"
    )

    print(
        f"Case instability rate: "
        f"{summary['case_instability_rate']:.1%}"
    )

    print()

    print("Changes by perturbation:")

    for name, count in (
        summary["variant_changes"].items()
    ):
        print(
            f"  {name}: {count}"
        )

    print()

    print(
        f"Total variant tests: "
        f"{summary['variant_tests']}"
    )

    print(
        f"Total changed decisions: "
        f"{summary['total_variant_changes']}"
    )

    print(
        f"Variant-level change rate: "
        f"{summary['variant_change_rate']:.1%}"
    )

    print(
        f"Mean maximum priority shift: "
        f"{summary['mean_max_priority_shift']:.2f}"
    )

    print()

    print("Results saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()
