def evaluate_public_sector_variants(
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

    max_priority_swing = max(
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
        "max_priority_swing": max_priority_swing,
    }
