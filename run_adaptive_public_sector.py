import json
from pathlib import Path

from audits.services.adaptive_controller import (
    ThompsonAttackController,
)
from audits.services.audit_engine import (
    ShadowBenchAudit,
)
from audits.services.attacks import (
    FRUSTRATED_TONE,
    NEUTRAL_CONTEXT,
    NOISY_WRITING,
)
from audits.services.counterfactual_evaluator import (
    evaluate_standard_counterfactual,
)
from audits.services.prompts import (
    CONSTITUENT_TRIAGE_PROMPT,
)
from audits.services.target_model import (
    OllamaTargetModel,
)


BASE_DIR = Path(__file__).resolve().parent

CASES_FILE = (
    BASE_DIR
    / "data"
    / "public_sector_cases.json"
)

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "results"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "qwen_public_sector_adaptive_v1.json"
)


ATTACKS = [
    FRUSTRATED_TONE,
    NEUTRAL_CONTEXT,
    NOISY_WRITING,
]


BUDGET = 10


def load_cases():

    with open(
        CASES_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def main():

    cases = load_cases()

    model = OllamaTargetModel(
        model_name="qwen3:1.7b",
        system_prompt=
            CONSTITUENT_TRIAGE_PROMPT,
    )

    audit = ShadowBenchAudit(
        model=model,
        attacks=ATTACKS,
        evaluator=
            evaluate_standard_counterfactual,
    )

    controller = (
        ThompsonAttackController(
            attacks=ATTACKS,
            seed=42,
        )
    )

    print("=" * 60)
    print(
        "SHADOWBENCH — "
        "ADAPTIVE ADVERSARIAL AUDIT"
    )
    print("=" * 60)

    print(
        f"Cases available: {len(cases)}"
    )

    print(
        f"Inference budget: {BUDGET}"
    )

    print()

    results = []

    # Keep track of which attacks have already
    # been used against each case.
    tested = {
        case["id"]: set()
        for case in cases
    }

    step = 0

    while step < BUDGET:

        # Round-robin through cases.
        case = cases[
            step % len(cases)
        ]

        case_id = case["id"]

        allowed_attacks = {
            attack.name
            for attack in ATTACKS
            if attack.name
            not in tested[case_id]
        }

        # If this case has exhausted all attacks,
        # move to another available case.
        if not allowed_attacks:

            available_cases = [
                candidate
                for candidate in cases
                if len(
                    tested[candidate["id"]]
                ) < len(ATTACKS)
            ]

            if not available_cases:
                break

            case = available_cases[0]
            case_id = case["id"]

            allowed_attacks = {
                attack.name
                for attack in ATTACKS
                if attack.name
                not in tested[case_id]
            }

        attack = (
            controller.select_attack(
                allowed_names=
                    allowed_attacks
            )
        )

        print(
            f"[{step + 1}/{BUDGET}] "
            f"{case_id}"
        )

        print(
            f"    Selected attack: "
            f"{attack.name}"
        )

        result = (
            audit.run_single_attack(
                case=case,
                attack=attack,
            )
        )

        finding = (
            result["evaluation"]
            ["any_change"]
        )

        controller.update(
            attack_name=attack.name,
            finding=finding,
        )

        tested[
            case_id
        ].add(
            attack.name
        )

        result["adaptive"].update({
            "budget_step":
                step + 1,
            "finding":
                finding,
        })

        results.append(result)

        variant = (
            result["variants"]
            [attack.name]
        )

        original_action = (
            result["original"]
            ["response"]
            ["action"]
        )

        variant_action = (
            variant["response"]
            ["action"]
        )

        cache_marker = (
            " [cached]"
            if variant["cached"]
            else ""
        )

        print(
            f"    Decision: "
            f"{original_action}"
            f" -> "
            f"{variant_action}"
            f"{cache_marker}"
        )

        print(
            "    Result: "
            + (
                "FINDING"
                if finding
                else "stable"
            )
        )

        print()

        step += 1

    summary = controller.summary()

    payload = {
        "model": "qwen3:1.7b",
        "benchmark":
            "public_sector_consistency",
        "mode":
            "adaptive_thompson_sampling",
        "budget":
            BUDGET,
        "controller":
            "Beta-Bernoulli Thompson sampling",
        "controller_summary":
            summary,
        "results":
            results,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 60)
    print("ADAPTIVE ALLOCATION")
    print("=" * 60)
    print()

    total_findings = 0

    for name, stats in summary.items():

        print(name)

        print(
            f"  Tests: "
            f"{stats['tests']}"
        )

        print(
            f"  Findings: "
            f"{stats['findings']}"
        )

        print(
            f"  Posterior mean: "
            f"{stats['posterior_mean']:.3f}"
        )

        print()

        total_findings += (
            stats["findings"]
        )

    print(
        f"Failures discovered: "
        f"{total_findings}"
        f"/{len(results)} tests"
    )

    print()

    print("Results saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()
