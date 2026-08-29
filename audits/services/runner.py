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
        original_result,
        counterfactual_result,
    )

    return {
        "case_id": case["id"],

        "original_case": original_text,
        "counterfactual_case": counterfactual_text,

        "original_response": original_result["reason"],
        "counterfactual_response": counterfactual_result["reason"],

        **comparison,
    }
