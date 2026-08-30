from .counterfactuals import add_schizophrenia_history
from .evaluator import compare_outputs


def run_case_audit(case: dict, model) -> dict:

    original_text = case["text"]

    counterfactual_text = add_schizophrenia_history(
        original_text
    )

    original_result = model.run(
        original_text
    )

    counterfactual_result = model.run(
        counterfactual_text
    )

    comparison = compare_outputs(
        original=original_result,
        counterfactual=counterfactual_result,
        gold_action=case.get("gold_action"),
    )

    return {
        "case_id": case["id"],
        "prefix_id": case.get("prefix_id"),
        "source_dataset": case.get("source_dataset"),
        "source_case_id": case.get("source_case_id"),
        "specialty": case.get("specialty"),
        "age_group": case.get("age_group"),

        "original_case": original_text,
        "counterfactual_case": counterfactual_text,

        "original_response": original_result["reason"],
        "counterfactual_response":
            counterfactual_result["reason"],

        **comparison,
    }


def summarize_audits(results: list[dict]) -> dict:

    total = len(results)

    if total == 0:
        return {
            "cases_tested": 0,
        }

    baseline_correct = sum(
        result.get("baseline_correct", False)
        for result in results
    )

    counterfactual_correct = sum(
        result.get("counterfactual_correct", False)
        for result in results
    )

    outputs_changed = sum(
        result["output_changed"]
        for result in results
    )

    under_escalations = sum(
        result["under_escalated_vs_original"]
        for result in results
    )

    confirmed_degradations = sum(
        result.get(
            "benchmark_confirmed_degradation",
            False,
        )
        for result in results
    )

    return {
        "cases_tested": total,

        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / total,

        "counterfactual_correct": counterfactual_correct,
        "counterfactual_accuracy":
            counterfactual_correct / total,

        "outputs_changed": outputs_changed,
        "under_escalations": under_escalations,

        "benchmark_confirmed_degradations":
            confirmed_degradations,
    }
