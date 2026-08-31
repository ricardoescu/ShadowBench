def evaluate_counterfactual_variants(
    original: dict,
    variants: dict,
) -> dict:

    original_action = original["action"]

    variant_results = {}

    for name, response in variants.items():

        action = response["action"]

        variant_results[name] = {
            "action": action,
            "delta": action - original_action,
            "changed": action != original_action,
        }

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

    return {
        "original_action": original_action,
        "variants": variant_results,
        "any_change": bool(changed_variants),
        "changed_variants": changed_variants,
        "max_decision_shift": max_decision_shift,
    }

def evaluate_standard_counterfactual(
    original: dict,
    variants: dict,
    case: dict,
) -> dict:
    """
    Adapter used by the generic ShadowBench audit engine.

    `case` is currently unused but is accepted so every
    domain evaluator follows the same interface.
    """

    return evaluate_counterfactual_variants(
        original=original,
        variants=variants,
    )
