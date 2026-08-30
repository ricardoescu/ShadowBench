from collections import Counter
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATASET_NAME = "ningkko/CARE-Bench"


# --------------------------------------------------
# Evaluation labels
# --------------------------------------------------

evaluation = load_dataset(
    DATASET_NAME,
    "evaluation_prefixes",
    split="public",
)

print("=== EVALUATION DATA ===")
print("Rows:", len(evaluation))
print("Columns:", evaluation.column_names)


print("\n=== GOLD LABEL COUNTS ===")

for label, count in Counter(evaluation["gold_label"]).items():
    print(label, ":", count)


print("\n=== AGE GROUPS ===")

for age_group, count in Counter(evaluation["case_age_group"]).items():
    print(age_group, ":", count)


print("\n=== SPECIALTIES / DOMAINS ===")

for specialty, count in Counter(
    evaluation["specialty_or_domain"]
).most_common():
    print(specialty, ":", count)


# --------------------------------------------------
# Actual model inputs
# --------------------------------------------------

model_inputs = load_dataset(
    DATASET_NAME,
    "model_inputs",
)

print("\n=== MODEL INPUT SPLITS ===")
print(model_inputs)


split_name = list(model_inputs.keys())[0]
model_split = model_inputs[split_name]

print("\nUsing split:", split_name)
print("Rows:", len(model_split))
print("Columns:", model_split.column_names)


print("\n=== FIRST MODEL INPUT ===")

print(model_split[0])
