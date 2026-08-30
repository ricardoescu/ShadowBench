import json
from pathlib import Path

from audits.services.carebench import load_carebench_cases
from audits.services.runner import (
    run_case_audit,
    summarize_audits,
)
from audits.services.target_model import (
    OllamaClinicalModel,
)


BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "data" / "results"
#RESULTS_FILE = RESULTS_DIR / "gpt_oss_carebench.json"
RESULTS_FILE = RESULTS_DIR / "medgemma_carebench.json"


def save_results(model_name, results):

    summary = summarize_audits(results)

    payload = {
        "model": model_name,
        "summary": summary,
        "results": results,
    }

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
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )


def main():

    cases = load_carebench_cases()

    #model = HuggingFaceClinicalModel()
    model = OllamaClinicalModel()

    print(
        f"Loaded {len(cases)} frozen CARE-Bench cases."
    )

    print(
        f"Target model: {model.model_name}"
    )

    print()

    # --------------------------------------------------
    # Resume support
    # --------------------------------------------------

    results = []

    if RESULTS_FILE.exists():

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            existing = json.load(f)

        results = existing.get(
            "results",
            []
        )

        print(
            f"Found {len(results)} existing results."
        )
        print(
            "Already completed cases will be skipped."
        )
        print()

    completed_ids = {
        result["case_id"]
        for result in results
    }

    # --------------------------------------------------
    # Run cases sequentially
    # --------------------------------------------------

    for index, case in enumerate(
        cases,
        start=1,
    ):

        if case["id"] in completed_ids:

            print(
                f"[{index}/{len(cases)}] "
                f"{case['id']} — already complete"
            )

            continue

        print(
            f"[{index}/{len(cases)}] "
            f"Testing {case['id']}..."
        )

        try:

            result = run_case_audit(
                case=case,
                model=model,
            )

        except Exception as exc:

            print()
            print("ERROR while testing:")
            print(case["id"])
            print()
            print(exc)
            print()
            print(
                "Previous successful results "
                "have been preserved."
            )

            raise

        results.append(result)

        # IMPORTANT:
        # checkpoint after every individual case
        save_results(
            model.model_name,
            results,
        )

        gold = result["gold_action"]
        original = result["original_action"]
        counterfactual = (
            result["counterfactual_action"]
        )

        print(
            f"    Gold: {gold} | "
            f"Original: {original} | "
            f"Counterfactual: {counterfactual}"
        )

        if result[
            "benchmark_confirmed_degradation"
        ]:
            print(
                "    >>> BENCHMARK-CONFIRMED "
                "DEGRADATION"
            )

        elif result[
            "under_escalated_vs_original"
        ]:
            print(
                "    >>> Recommendation became "
                "less urgent"
            )

        else:
            print(
                "    No under-escalation"
            )

        print()

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    summary = summarize_audits(results)

    print("=" * 50)
    print("SHADOWBENCH RESULTS")
    print("=" * 50)

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
    print("Full results saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()
