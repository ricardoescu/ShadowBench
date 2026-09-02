import random

from audits.services.adaptive_controller import (
    ThompsonAttackController,
)

from audits.services.attacks import (
    FRUSTRATED_TONE,
    NEUTRAL_CONTEXT,
    NOISY_WRITING,
)


attacks = [
    FRUSTRATED_TONE,
    NEUTRAL_CONTEXT,
    NOISY_WRITING,
]


controller = ThompsonAttackController(
    attacks=attacks,
    seed=42,
)


# Fake underlying failure rates.
# We KNOW frustration is the most productive arm.
TRUE_FAILURE_RATES = {
    FRUSTRATED_TONE.name: 0.70,
    NEUTRAL_CONTEXT.name: 0.10,
    NOISY_WRITING.name: 0.20,
}


environment = random.Random(123)


for step in range(100):

    attack = controller.select_attack()

    finding = (
        environment.random()
        <
        TRUE_FAILURE_RATES[
            attack.name
        ]
    )

    controller.update(
        attack_name=attack.name,
        finding=finding,
    )


print()
print("=" * 60)
print("SHADOWBENCH ADAPTIVE TESTING")
print("=" * 60)
print()

for name, stats in (
    controller.summary().items()
):

    print(name)

    print(
        f"  Tests allocated: "
        f"{stats['tests']}"
    )

    print(
        f"  Failures found: "
        f"{stats['findings']}"
    )

    print(
        f"  Observed yield: "
        f"{stats['finding_rate']:.1%}"
    )

    print(
        f"  Posterior mean: "
        f"{stats['posterior_mean']:.3f}"
    )

    print()
