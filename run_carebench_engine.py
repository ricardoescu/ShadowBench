import json
from pathlib import Path

from audits.services.audit_engine import (
    ShadowBenchAudit,
)
from audits.services.attacks import (
    SCHIZOPHRENIA_HISTORY,
)
from audits.services.carebench import (
    load_carebench_cases,
)
from audits.services.clinical_evaluator import (
    evaluate_clinical_counterfactual,
)
from audits.services.target_model import (
    OllamaClinicalModel,
)


BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "results"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "medgemma_carebench_engine_v1.json"
)

ATTACKS = [
    SCHIZOPHRENIA_HISTORY,
]


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
                "model": "medgemma",
                "benchmark": "CARE-Bench",
                "engine": "ShadowBenchAudit",
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

    baseline_correct = sum(
        result["evaluation"].get(
            "baseline_correct",
            False,
        )
        for result in results
    )

    outputs_changed = sum(
        result["evaluation"]["any_change"]
        for result in results
    )

    under_escalations = sum(
        variant[
            "under_escalated_vs_original"
        ]
        for result in results
        for variant in (
            result["evaluation"]
            ["variants"]
            .values()
        )
    )

    confirmed_degradations = sum(
        variant.get(
            "benchmark_confirmed_degradation",
            False,
        )
        for result in results
        for variant in (
            result["evaluation"]
            ["variants"]
            .values()
        )
    )

    counterfactual_correct = sum(
        variant.get(
            "counterfactual_correct",
            False,
        )
        for result in results
        for variant in (
            result["evaluation"]
            ["variants"]
            .values()
        )
    )

    return {
        "cases_tested": total,
        "baseline_correct":
            baseline_correct,
        "counterfactual_correct":
            counterfactual_correct,
        "outputs_changed":
            outputs_changed,
        "under_escalations":
            under_escalations,
        "benchmark_confirmed_degradations":
            confirmed_degradations,
    }


def main():

    cases = load_carebench_cases()

    model = OllamaClinicalModel()

    audit = ShadowBenchAudit(
        model=model,
        attacks=ATTACKS,
        evaluator=
            evaluate_clinical_counterfactual,
    )

    print(
        f"Loaded {len(cases)} "
        "frozen CARE-Bench cases."
    )

    print(
        f"Target model: {model.model_name}"
    )

    print(
        "Engine: ShadowBenchAudit"
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
            "existing engine results."
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

        result = audit.run_case(case)

        results.append(result)

        save_results(results)

        evaluation = result["evaluation"]

        original_action = (
            result["original"]
            ["response"]
            ["action"]
        )

        gold_action = (
            evaluation.get("gold_action")
        )

        variant = (
            result["variants"]
            [SCHIZOPHRENIA_HISTORY.name]
        )

        counterfactual_action = (
            variant["response"]["action"]
        )

        cached_marker = (
            " [cached]"
            if variant["cached"]
            else ""
        )

        print(
            f"    Gold: {gold_action} | "
            f"Original: {original_action} | "
            f"Shadow: {counterfactual_action}"
            f"{cached_marker}"
        )

        clinical_result = (
            evaluation["variants"]
            [SCHIZOPHRENIA_HISTORY.name]
        )

        if clinical_result.get(
            "benchmark_confirmed_degradation"
        ):
            print(
                "    >>> BENCHMARK-CONFIRMED "
                "DEGRADATION"
            )

        elif clinical_result[
            "under_escalated_vs_original"
        ]:
            print(
                "    >>> Recommendation became "
                "less urgent"
            )

        elif clinical_result["changed"]:
            print(
                "    >>> DECISION INSTABILITY"
            )

        else:
            print(
                "    Stable across shadow case"
            )

        print()

    summary = summarize(results)

    print("=" * 60)

    print(
        "SHADOWBENCH — "
        "CLINICAL CARE-BENCH"
    )

    print("=" * 60)

    print(
        f"Cases tested: "
        f"{summary['cases_tested']}"
    )

    print(
        f"Baseline correct: "
        f"{summary['baseline_correct']}"
        f"/{summary['cases_tested']}"
    )

    print(
        f"Counterfactual correct: "
        f"{summary['counterfactual_correct']}"
        f"/{summary['cases_tested']}"
    )

    print(
        f"Outputs changed: "
        f"{summary['outputs_changed']}"
    )

    print(
        f"Under-escalations: "
        f"{summary['under_escalations']}"
    )

    print(
        "Benchmark-confirmed degradations: "
        f"{summary['benchmark_confirmed_degradations']}"
    )

    print()

    print("Results saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()
