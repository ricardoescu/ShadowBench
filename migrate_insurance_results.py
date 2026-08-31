import json
from pathlib import Path

from audits.services.attacks import (
    FRUSTRATED_TONE,
)


BASE_DIR = Path(__file__).resolve().parent

OLD_FILE = (
    BASE_DIR
    / "data"
    / "results"
    / "insurellm_insurance_audit_v1.json"
)

NEW_FILE = (
    BASE_DIR
    / "data"
    / "results"
    / "insurellm_insurance_engine_v1.json"
)


ATTACK_MAP = {
    FRUSTRATED_TONE.name: FRUSTRATED_TONE,
}


with open(
    OLD_FILE,
    "r",
    encoding="utf-8",
) as f:
    old_data = json.load(f)


new_results = []


for old in old_data["results"]:

    variants = {}

    for name, metadata in (
        old["variant_metadata"].items()
    ):

        attack = ATTACK_MAP[name]

        variants[name] = {
            "family": attack.family,
            "description": attack.description,
            "text": metadata["text"],

            "response": {
                "action": metadata["action"],
                "decision": metadata["decision"],
                "raw_response":
                    metadata["raw_response"],
            },

            "cached": metadata["cached"],
        }

    result = {
        "case_id": old["case_id"],

        "metadata": {
            "domain": old["domain"],
        },

        "original": {
            "text": old["original_text"],

            "response": {
                "action":
                    old["original_action"],

                "decision":
                    old["original_decision"],

                "raw_response":
                    old["original_raw_response"],
            },

            "cached":
                old["original_cached"],
        },

        "variants": variants,

        "evaluation": {
            "original_action":
                old["original_action"],

            "variants":
                old["variants"],

            "any_change":
                old["any_change"],

            "changed_variants":
                old["changed_variants"],

            "max_decision_shift":
                old["max_decision_shift"],
        },
    }

    new_results.append(result)


new_data = {
    "model":
        "piyushptiwari/InsureLLM-4B",

    "benchmark":
        "insurance_claims_consistency",

    "engine":
        "ShadowBenchAudit",

    "results":
        new_results,
}


with open(
    NEW_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        new_data,
        f,
        indent=2,
        ensure_ascii=False,
    )


print(
    f"Migrated {len(new_results)} results."
)

print(f"Saved to: {NEW_FILE}")
