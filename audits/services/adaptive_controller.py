from dataclasses import dataclass
import random

from .attacks import CounterfactualAttack


@dataclass
class AttackState:
    attack: CounterfactualAttack
    alpha: float = 1.0
    beta: float = 1.0
    tests: int = 0
    findings: int = 0

    @property
    def empirical_rate(self) -> float:
        if self.tests == 0:
            return 0.0

        return self.findings / self.tests

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (
            self.alpha + self.beta
        )


class ThompsonAttackController:
    """
    Beta-Bernoulli Thompson sampling over
    adversarial attack families.

    Reward = 1 when an attack changes the
    target model's decision.
    """

    def __init__(
        self,
        attacks: list[CounterfactualAttack],
        seed: int = 42,
    ):
        self.random = random.Random(seed)

        self.states = {
            attack.name: AttackState(
                attack=attack
            )
            for attack in attacks
        }

    def select_attack(
        self,
        allowed_names: set[str] | None = None,
    ) -> CounterfactualAttack:

        candidates = [
            state
            for name, state in self.states.items()
            if (
                allowed_names is None
                or name in allowed_names
            )
        ]

        if not candidates:
            raise ValueError(
                "No attacks available."
            )

        samples = {
            state.attack.name:
                self.random.betavariate(
                    state.alpha,
                    state.beta,
                )
            for state in candidates
        }

        selected_name = max(
            samples,
            key=samples.get,
        )

        return self.states[
            selected_name
        ].attack

    def update(
        self,
        attack_name: str,
        finding: bool,
    ) -> None:

        state = self.states[
            attack_name
        ]

        state.tests += 1

        if finding:
            state.findings += 1
            state.alpha += 1
        else:
            state.beta += 1

    def summary(self) -> dict:

        return {
            name: {
                "family":
                    state.attack.family,
                "tests":
                    state.tests,
                "findings":
                    state.findings,
                "finding_rate":
                    state.empirical_rate,
                "posterior_mean":
                    state.posterior_mean,
                "alpha":
                    state.alpha,
                "beta":
                    state.beta,
            }
            for name, state
            in self.states.items()
        }
