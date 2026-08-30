import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / "data" / "carebench_selected.json"


def load_carebench_cases() -> list[dict]:

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Give each case the generic ID expected by ShadowBench.
    for case in cases:
        case["id"] = case["prefix_id"]

    return cases
