from .counterfactuals import add_schizophrenia_history
from .evaluator import compare_outputs
from .inference_cache import cached_model_run

def run_case_audit(
    case: dict,
    model,
    attack,
) -> dict:

    original_text = case["text"]

    counterfactual_text = attack.apply(
        original_text
    )

    original_result, original_cached = (
        cached_model_run(
            model,
            original_text,
        )
    )

    counterfactual_result, counterfactual_cached = (
        cached_model_run(
            model,
            counterfactual_text,
        )
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

        "attack_name": attack.name,
        "attack_family": attack.family,

        "original_case": original_text,
        "counterfactual_case": counterfactual_text,

        "original_response": original_result["reason"],
        "counterfactual_response":
            counterfactual_result["reason"],

        "original_cached": original_cached,
        "counterfactual_cached": counterfactual_cached,

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

    over_escalations = sum(
        result["counterfactual_action"] >
        result["original_action"]
        for result in results
    )

    under_escalations = sum(
        result["counterfactual_action"] <
        result["original_action"]
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
        "over_escalations": over_escalations,
        "under_escalations": under_escalations,
        "counterfactual_changes": outputs_changed,
        "counterfactual_change_rate": outputs_changed / total,

        "benchmark_confirmed_degradations":
            confirmed_degradations,
    }
