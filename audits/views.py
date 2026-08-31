from django.shortcuts import render

from .services.cases import CLINICAL_CASES
#from .services.target_model import MockClinicalModel
from .services.target_model import HuggingFaceClinicalModel
from .services.runner import run_case_audit
from .services.attacks import SCHIZOPHRENIA_HISTORY


def dashboard(request):
    return render(
        request,
        "audits/dashboard.html"
    )


import json
from pathlib import Path

from django.conf import settings
from django.shortcuts import render


def run_audit(request):
    results_path = (
        Path(settings.BASE_DIR)
        / "data"
        / "results"
        / "qwen_insurance_audit_v2.json"
    )

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    failures = []

    for result in data["results"]:
        variant = result["variant_metadata"]["frustrated_tone_v2"]

        if variant["action"] != result["original_action"]:
            failures.append({
                "case_id": result["case_id"],
                "original_text": result["original_text"],
                "counterfactual_text": variant["text"],
                "original_action": result["original_action"],
                "counterfactual_action": variant["action"],
                "original_reason": result["original_reason"],
                "counterfactual_reason": variant["reason"],
                "delta": (
                    variant["action"]
                    - result["original_action"]
                ),
            })

    return render(
    request,
    "audits/audit_result.html",
    {
        "failures": failures,
        "cases_tested": len(data["results"]),
        "potential_failures": len(failures),
    },
)
