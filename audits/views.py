from django.shortcuts import render

from .services.cases import CLINICAL_CASES
from .services.target_model import MockClinicalModel
from .services.runner import run_case_audit


def dashboard(request):
    return render(
        request,
        "audits/dashboard.html"
    )


def run_audit(request):

    model = MockClinicalModel()

    case = CLINICAL_CASES[0]

    audit = run_case_audit(
        case=case,
        model=model,
    )

    result = {
        "model": "Mock Clinical AI",
        "cases_tested": 1,
        "failures_found": (
            1 if audit["potential_failure"] else 0
        ),

        **audit,
    }

    return render(
        request,
        "audits/audit_result.html",
        {"result": result},
    )
