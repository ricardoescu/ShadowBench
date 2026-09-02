from django.shortcuts import render

from .services.cases import CLINICAL_CASES
#from .services.target_model import MockClinicalModel
from .services.target_model import HuggingFaceClinicalModel
from .services.runner import run_case_audit
from .services.attacks import SCHIZOPHRENIA_HISTORY


def dashboard(request):
    tree_path = (
        Path(settings.BASE_DIR)
        / "data"
        / "results"
        / "qwen_public_sector_adversarial_tree_v1.json"
    )

    with open(
        tree_path,
        "r",
        encoding="utf-8",
    ) as f:
        tree = json.load(f)

    nodes = [
        node
        for node in tree["nodes"]
        if node["id"] != "root"
    ]

    communication_nodes = [
        node
        for node in nodes
        if node.get("family") == "communication_style"
    ]

    context_nodes = [
        node
        for node in nodes
        if node.get("family") == "context_control"
    ]

    writing_nodes = [
        node
        for node in nodes
        if node.get("family") == "communication_quality"
    ]

    deployment_path = (
        Path(settings.BASE_DIR)
        / "deployments"
        / "sepolia.json"
    )

    with open(
        deployment_path,
        "r",
        encoding="utf-8",
    ) as f:
        deployment = json.load(f)

    deployment["audit_url"] = (
        "https://sepolia.etherscan.io/tx/"
        + deployment["audit_transaction"]
    )

    return render(
        request,
        "audits/dashboard.html",
        {
            "search": tree,
            "communication_nodes": communication_nodes,
            "context_nodes": context_nodes,
            "writing_nodes": writing_nodes,
            "deployment": deployment,
        },
    )
def insurance_audit(request):

    results_path = (
        Path(settings.BASE_DIR)
        / "data"
        / "results"
        / "insurellm_insurance_engine_run_v1.json"
    )

    with open(
        results_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    findings = []

    for result in data["results"]:

        variant = result["variants"][
            "frustrated_tone_v2"
        ]

        if result["evaluation"]["any_change"]:

            findings.append({
                "case_id":
                    result["case_id"],

                "domain":
                    result["metadata"]["domain"],

                "original_text":
                    result["original"]["text"],

                "shadow_text":
                    variant["text"],

                "original_decision":
                    result["original"]
                    ["response"]["decision"],

                "shadow_decision":
                    variant["response"]["decision"],
            })

    return render(
        request,
        "audits/insurance_audit.html",
        {
            "model":
                "InsureLLM-4B",

            "cases_tested":
                len(data["results"]),

            "findings":
                findings,

            "finding_count":
                len(findings),
        },
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

    deployment_path = (
        Path(settings.BASE_DIR)
        / "deployments"
        / "sepolia.json"
    )

    with open(
        deployment_path,
        "r",
        encoding="utf-8",
    ) as f:
        deployment = json.load(f)

    deployment["audit_url"] = (
        "https://sepolia.etherscan.io/tx/"
        + deployment["audit_transaction"]
    )

    return render(
        request,
        "audits/audit_result.html",
        {
            "failures": failures,
            "cases_tested": len(data["results"]),
            "potential_failures": len(failures),
            "deployment": deployment,
        },
    )

def public_sector_audit(request):

    results_path = (
        Path(settings.BASE_DIR)
        / "data"
        / "results"
        / "qwen_public_sector_engine_v1.json"
    )

    with open(
        results_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    action_labels = {
        0: "Routine information / signposting",
        1: "Standard casework",
        2: "Priority casework",
        3: "Urgent / severe time-critical harm",
    }

    findings = []
    affected_case_ids = set()

    for result in data["results"]:

        original_action = (
            result["original"]["response"]["action"]
        )

        for variant_name, variant in (
            result["variants"].items()
        ):

            variant_action = (
                variant["response"]["action"]
            )

            if variant_action == original_action:
                continue

            affected_case_ids.add(
                result["case_id"]
            )

            findings.append({
                "case_id":
                    result["case_id"],

                "change_name":
                    variant_name
                    .replace("_", " ")
                    .title(),

                "change_description":
                    variant["description"],

                "original_text":
                    result["original"]["text"],

                "shadow_text":
                    variant["text"],

                "original_decision":
                    action_labels[
                        original_action
                    ],

                "shadow_decision":
                    action_labels[
                        variant_action
                    ],

                "original_action":
                    original_action,

                "shadow_action":
                    variant_action,

                "original_reason":
                    result["original"]
                    ["response"].get(
                        "reason",
                        "",
                    ),

                "shadow_reason":
                    variant["response"].get(
                        "reason",
                        "",
                    ),
            })

    return render(
        request,
        "audits/domain_audit.html",
        {
            "eyebrow":
                "Public-sector audit",

            "title":
                "Can irrelevant context alter casework triage?",

            "lead":
                (
                    "ShadowBench tested Qwen3 on synthetic "
                    "constituency casework while changing only "
                    "controlled presentation or contextual details."
                ),

            "model":
                "Qwen3 1.7B",

            "cases_tested":
                len(data["results"]),

            "affected_cases":
                len(affected_case_ids),

            "decision_changes":
                len(findings),

            "headline":
                (
                    f"{len(affected_case_ids)} of "
                    f"{len(data['results'])} constituent cases "
                    "showed at least one decision change under "
                    "controlled counterfactual perturbations."
                ),

            "findings":
                findings,

            "method_note":
                (
                    "These are synthetic constituency cases. "
                    "The audit demonstrates decision instability "
                    "under controlled perturbations; it is not an "
                    "estimate of real-world failure prevalence."
                ),
        },
    )


def medical_audit(request):

    results_path = (
        Path(settings.BASE_DIR)
        / "data"
        / "results"
        / "medgemma_carebench_engine_v1.json"
    )

    with open(
        results_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    action_labels = {
        0: "More information required",
        1: "Self-care / monitor",
        2: "Non-urgent medical care",
        3: "Urgent / emergency care",
    }

    findings = []
    benchmark_degradations = 0

    for result in data["results"]:

        variant = result["variants"][
            "schizophrenia_history"
        ]

        evaluation = result[
            "evaluation"
        ]["variants"][
            "schizophrenia_history"
        ]

        if evaluation.get(
            "benchmark_confirmed_degradation"
        ):
            benchmark_degradations += 1

        if not result["evaluation"]["any_change"]:
            continue

        original_action = (
            result["original"]["response"]["action"]
        )

        variant_action = (
            variant["response"]["action"]
        )

        findings.append({
            "case_id":
                result["case_id"],

            "change_name":
                "Schizophrenia history",

            "change_description":
                variant["description"],

            "original_text":
                result["original"]["text"],

            "shadow_text":
                variant["text"],

            "original_decision":
                action_labels[
                    original_action
                ],

            "shadow_decision":
                action_labels[
                    variant_action
                ],

            "original_action":
                original_action,

            "shadow_action":
                variant_action,

            "original_reason":
                result["original"]
                ["response"].get(
                    "reason",
                    "",
                ),

            "shadow_reason":
                variant["response"].get(
                    "reason",
                    "",
                ),
        })

    return render(
        request,
        "audits/domain_audit.html",
        {
            "eyebrow":
                "Medical audit",

            "title":
                "Can irrelevant patient history alter triage?",

            "lead":
                (
                    "ShadowBench tested MedGemma on a frozen "
                    "20-case CARE-Bench subset. A diagnosed "
                    "history of schizophrenia was added without "
                    "changing the presenting complaint."
                ),

            "model":
                "MedGemma",

            "cases_tested":
                len(data["results"]),

            "affected_cases":
                len(findings),

            "decision_changes":
                len(findings),

            "headline":
                (
                    f"{len(findings)} of "
                    f"{len(data['results'])} triage outputs "
                    "changed after schizophrenia history was "
                    "added. No benchmark-confirmed degradation "
                    "was observed."
                ),

            "findings":
                findings,

            "method_note":
                (
                    "A changed output is not automatically a "
                    "clinical failure. This experiment found "
                    f"{benchmark_degradations} benchmark-confirmed "
                    "degradations, so the result should be "
                    "interpreted as sensitivity to added history "
                    "rather than evidence of diagnostic bias."
                ),
        },
    )
