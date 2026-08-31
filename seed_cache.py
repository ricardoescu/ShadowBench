import json
from pathlib import Path

from audits.services.inference_cache import (
    store_cached_result,
)


BASE_DIR = Path(__file__).resolve().parent

RESULTS_FILE = (
    BASE_DIR
    / "data"
    / "results"
    / "medgemma_carebench.json"
)


with open(
    RESULTS_FILE,
    "r",
    encoding="utf-8",
) as f:
    payload = json.load(f)


model_name = payload["model"]
results = payload["results"]

stored = 0


for result in results:

    store_cached_result(
        model_name=model_name,
        prompt=result["original_case"],
        result={
            "action": result["original_action"],
            "reason": result["original_response"],
        },
    )

    stored += 1

    store_cached_result(
        model_name=model_name,
        prompt=result["counterfactual_case"],
        result={
            "action": result["counterfactual_action"],
            "reason": result["counterfactual_response"],
        },
    )

    stored += 1


print(
    f"Imported {stored} MedGemma responses into cache."
)
