import json
import random
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATASET_NAME = "ningkko/CARE-Bench"

OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "carebench_selected.json"

RANDOM_SEED = 42


LABEL_TO_ACTION = {
    "0A_NO_ESCALATION_INFO_NEEDED": 0,
    "0B_NO_ESCALATION_SELF_CARE_MONITOR": 1,
    "1A_ESCALATION_NONURGENT_CARE": 2,
    "1B_ESCALATION_URGENT_CARE": 3,
}


print("Loading CARE-Bench...")

evaluation = load_dataset(
    DATASET_NAME,
    "evaluation_prefixes",
    split="public",
)

# --------------------------------------------------
# Filter to clean first-turn adult physical cases
# --------------------------------------------------

EXCLUDED_TERMS = [
    "panic attack",
    "anxiety disorder",
    "schizophrenia",
    "psychosis",
    "hallucination",
    "hallucinations",
]

EXCLUDED_PREFIX_IDS = {
    "meb_v1_case_0266_r01",
}


eligible = []

for row in evaluation:

    if row["round_id"] != 1:
        continue

    if row["case_age_group"] not in {"adult", "older_adult"}:
        continue

    if row["specialty_or_domain"] == "mental_health_behavior":
        continue

    if row["gold_label"] not in {
        "1A_ESCALATION_NONURGENT_CARE",
        "1B_ESCALATION_URGENT_CARE",
    }:
        continue

    if row["prefix_id"] in EXCLUDED_PREFIX_IDS:
        continue

    patient_turns = row["input_patient_turns"]

    if not patient_turns:
        continue

    text = "\n".join(patient_turns)
    lower_text = text.lower()

    if any(term in lower_text for term in EXCLUDED_TERMS):
        continue

    eligible.append({
        "prefix_id": row["prefix_id"],
        "case_id": row["case_id"],
        "source_dataset": row["source_dataset"],
        "source_case_id": row["source_case_id"],
        "specialty": row["specialty_or_domain"],
        "age_group": row["case_age_group"],
        "gold_label": row["gold_label"],
        "gold_action": LABEL_TO_ACTION[row["gold_label"]],
        "text": text,
    })

print(f"Eligible cases: {len(eligible)}")


# --------------------------------------------------
# Split by benchmark label
# --------------------------------------------------

nonurgent = [
    case for case in eligible
    if case["gold_action"] == 2
]

urgent = [
    case for case in eligible
    if case["gold_action"] == 3
]

print(f"Eligible non-urgent: {len(nonurgent)}")
print(f"Eligible urgent: {len(urgent)}")


# --------------------------------------------------
# Reproducible sample
# --------------------------------------------------

rng = random.Random(RANDOM_SEED)

selected_nonurgent = rng.sample(
    nonurgent,
    min(10, len(nonurgent)),
)

selected_urgent = rng.sample(
    urgent,
    min(10, len(urgent)),
)

selected = selected_nonurgent + selected_urgent

if len(selected) < 20:
    print(
        f"WARNING: Only selected {len(selected)} cases "
        f"(nonurgent={len(selected_nonurgent)}, urgent={len(selected_urgent)})"
    )

# Stable ordering for readability
selected.sort(
    key=lambda case: (
        case["gold_action"],
        case["prefix_id"],
    )
)


# --------------------------------------------------
# Save locally
# --------------------------------------------------

OUTPUT_DIR.mkdir(exist_ok=True)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        selected,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print(f"Saved {len(selected)} cases to:")
print(OUTPUT_FILE)


# --------------------------------------------------
# Show exactly what we selected
# --------------------------------------------------

print("\n=== SELECTED CASES ===")

for i, case in enumerate(selected, start=1):

    print()
    print(
        f"{i:02d}. "
        f"{case['prefix_id']} | "
        f"{case['specialty']} | "
        f"gold={case['gold_action']}"
    )

    print(case["text"])
