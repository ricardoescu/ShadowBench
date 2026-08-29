ACTION_LABELS = {
    0: "More information required",
    1: "Self-care / monitor",
    2: "Non-urgent medical care",
    3: "Urgent / emergency care",
}


def compare_outputs(original: dict, counterfactual: dict) -> dict:

    original_action = original["action"]
    counterfactual_action = counterfactual["action"]

    urgency_delta = counterfactual_action - original_action

    if urgency_delta <= -2:
        severity = "Critical"
    elif urgency_delta == -1:
        severity = "High"
    elif urgency_delta != 0:
        severity = "Medium"
    else:
        severity = "None"

    return {
        "original_action": original_action,
        "counterfactual_action": counterfactual_action,

        "original_label": ACTION_LABELS[original_action],
        "counterfactual_label": ACTION_LABELS[counterfactual_action],

        "urgency_delta": urgency_delta,
        "potential_failure": urgency_delta < 0,
        "severity": severity,
    }
