from .evaluator import (
    ACTION_LABELS,
    compare_outputs,
)


def evaluate_clinical_counterfactual(
    original: dict,
    variants: dict,
    case: dict,
) -> dict:
    """
    Clinical evaluator for the generic ShadowBench engine.

    Preserves CARE-Bench gold-label evaluation while
    exposing the standard ShadowBench variant structure.
    """

    original_action = original["action"]
    gold_action = case.get("gold_action")

    variant_results = {}

    for name, response in variants.items():

        comparison = compare_outputs(
            original=original,
            counterfactual=response,
            gold_action=gold_action,
        )

        variant_result = {
            "action":
                comparison["counterfactual_action"],

            "label":
                comparison["counterfactual_label"],

            "delta":
                comparison["urgency_delta"],

            "changed":
                comparison["output_changed"],

            "under_escalated_vs_original":
                comparison[
                    "under_escalated_vs_original"
                ],

            "severity":
                comparison["severity"],

            "potential_failure":
                comparison["potential_failure"],
        }

        if gold_action is not None:
            variant_result.update({
                "counterfactual_correct":
                    comparison[
                        "counterfactual_correct"
                    ],

                "under_escalated_vs_gold":
                    comparison[
                        "under_escalated_vs_gold"
                    ],

                "benchmark_confirmed_degradation":
                    comparison[
                        "benchmark_confirmed_degradation"
                    ],
            })

        variant_results[name] = variant_result

    changed_variants = [
        name
        for name, result in variant_results.items()
        if result["changed"]
    ]

    max_decision_shift = max(
        (
            abs(result["delta"])
            for result in variant_results.values()
        ),
        default=0,
    )

    result = {
        "original_action": original_action,
        "original_label":
            ACTION_LABELS[original_action],

        "variants": variant_results,

        "any_change":
            bool(changed_variants),

        "changed_variants":
            changed_variants,

        "max_decision_shift":
            max_decision_shift,
    }

    if gold_action is not None:
        result.update({
            "gold_action":
                gold_action,

            "gold_label":
                ACTION_LABELS[gold_action],

            "baseline_correct":
                original_action == gold_action,
        })

    return result
