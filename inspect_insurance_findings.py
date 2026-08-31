import json
from pathlib import Path


path = (
    Path(__file__).resolve().parent
    / "data"
    / "results"
    / "qwen_insurance_audit_v2.json"
)

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)


for result in data["results"]:

    frustrated = result["variant_metadata"]["frustrated_tone_v2"]

    if frustrated["action"] != result["original_action"]:

        print("=" * 70)
        print(result["case_id"])

        print("\nORIGINAL")
        print("Action:", result["original_action"])
        print("Reason:", result["original_reason"])

        print("\nFRUSTRATED")
        print("Action:", frustrated["action"])
        print("Reason:", frustrated["reason"])
        print()
