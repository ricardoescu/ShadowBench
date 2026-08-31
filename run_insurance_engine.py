import json
from pathlib import Path

from audits.services.audit_engine import (
    ShadowBenchAudit,
)
from audits.services.attacks import (
    FRUSTRATED_TONE,
)
from audits.services.counterfactual_evaluator import (
    evaluate_standard_counterfactual,
)
from audits.services.prompts import (
    INSURANCE_CLAIMS_PROMPT,
)
from audits.services.target_model import (
    InsureLLMTargetModel,
)


BASE_DIR = Path(__file__).resolve().parent

CASES_FILE = (
    BASE_DIR
    / "data"
    / "insurance_cases.json"
)

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "results"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "insurellm_insurance_engine_run_v1.json"
)


ATTACKS = [
    FRUSTRATED_TONE,
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
                "model":
                    "piyushptiwari/InsureLLM-4B",
                "benchmark":
                    "insurance_claims_consistency",
                "engine":
                    "ShadowBenchAudit",
                "attacks": [
                    attack.name
                    for attack in ATTACKS
                ],
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
        result["evaluation"]["any_change"]
        for result in results
    )

    changes_by_attack = {}

    for attack in ATTACKS:
        changes_by_attack[
            attack.name
        ] = sum(
            result["evaluation"]["variants"][
                attack.name
            ]["changed"]
            for result in results
        )

    total_variant_tests = (
        total * len(ATTACKS)
    )

    total_changed_decisions = sum(
        changes_by_attack.values()
    )

    return {
        "cases_tested": total,
        "affected_cases": affected_cases,
        "case_instability_rate":
            affected_cases / total,
        "changes_by_attack":
            changes_by_attack,
        "total_variant_tests":
            total_variant_tests,
        "total_changed_decisions":
            total_changed_decisions,
        "variant_change_rate":
            total_changed_decisions
            / total_variant_tests,
    }


def main():
    cases = load_cases()

    # Loads the insurance-specialised model once.
    model = InsureLLMTargetModel(
        system_prompt=INSURANCE_CLAIMS_PROMPT,
    )

    # Generic ShadowBench orchestration layer.
    audit = ShadowBenchAudit(
        model=model,
        attacks=ATTACKS,
        evaluator=evaluate_standard_counterfactual,
    )

    print(
        f"Loaded {len(cases)} insurance claims."
    )
    print(
        f"Target model: {model.model_name}"
    )
    print(
        "Engine: ShadowBenchAudit"
    )
    print()

    results = []

    # -------------------------
    # Resume support
    # -------------------------

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
            "existing engine results."
        )
        print()

    completed_ids = {
        result["case_id"]
        for result in results
    }

    # -------------------------
    # Benchmark
    # -------------------------

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

        result = audit.run_case(case)

        results.append(result)

        # Save after every case.
        save_results(results)

        original = (
            result["original"]["response"]
        )

        print(
            f"    Original: "
            f"{original['decision']} "
            f"({original['action']})"
        )

        for attack in ATTACKS:
            variant = (
                result["variants"]
                [attack.name]
            )

            response = variant["response"]

            cache_marker = (
                " [cached]"
                if variant["cached"]
                else ""
            )

            print(
                f"    {attack.name:20} "
                f"{response['decision']} "
                f"({response['action']})"
                f"{cache_marker}"
            )

        evaluation = result["evaluation"]

        if evaluation["any_change"]:
            print(
                "    >>> DECISION INSTABILITY"
            )

            print(
                "    Changed variants: "
                + ", ".join(
                    evaluation[
                        "changed_variants"
                    ]
                )
            )
        else:
            print(
                "    Stable across variants"
            )

        print()

    # -------------------------
    # Summary
    # -------------------------

    summary = summarize(results)

    print("=" * 60)
    print(
        "SHADOWBENCH — "
        "INSURANCE CLAIMS CONSISTENCY"
    )
    print("=" * 60)

    print(
        f"Cases tested: "
        f"{summary['cases_tested']}"
    )

    print(
        "Cases with any decision change: "
        f"{summary['affected_cases']}"
    )

    print(
        "Case instability rate: "
        f"{summary['case_instability_rate']:.1%}"
    )

    print()

    print("Changes by perturbation:")

    for name, count in (
        summary[
            "changes_by_attack"
        ].items()
    ):
        print(
            f"  {name}: {count}"
        )

    print()

    print(
        "Total variant tests: "
        f"{summary['total_variant_tests']}"
    )

    print(
        "Total changed decisions: "
        f"{summary['total_changed_decisions']}"
    )

    print(
        "Variant-level change rate: "
        f"{summary['variant_change_rate']:.1%}"
    )

    print()

    print("Results saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()
