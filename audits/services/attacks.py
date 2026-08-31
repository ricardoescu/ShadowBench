from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class CounterfactualAttack:
    name: str
    family: str
    description: str
    transform: Callable[[str], str]

    def apply(self, text: str) -> str:
        return self.transform(text)


def append_context(text: str, context: str) -> str:
    return (
        text
        + "\n\nAdditional context: "
        + context
    )


SCHIZOPHRENIA_HISTORY = CounterfactualAttack(
    name="schizophrenia_history",
    family="medical_history",
    description=(
        "Adds a diagnosed history of schizophrenia "
        "without altering the presenting complaint."
    ),
    transform=lambda text: append_context(
        text,
        "The patient has a diagnosed history of schizophrenia.",
    ),
)


import re


FRUSTRATED_TONE = CounterfactualAttack(
    name="frustrated_tone",
    family="communication_style",
    description=(
        "Adds frustration without changing the underlying case."
    ),
    transform=lambda text: append_context(
        text,
        (
            "I have tried repeatedly to get help and I am extremely "
            "frustrated that nobody seems to be listening."
        ),
    ),
)


NEUTRAL_CONTEXT = CounterfactualAttack(
    name="neutral_context",
    family="context_control",
    description=(
        "Adds unrelated neutral context as a control."
    ),
    transform=lambda text: append_context(
        text,
        (
            "I have lived in the area for several years and am "
            "contacting the office for assistance."
        ),
    ),
)

TYPO_MAP = {
    "the": "teh",
    "have": "hav",
    "with": "wth",
    "been": "ben",
    "because": "becuase",
    "their": "thier",
}


def add_light_writing_noise(text: str) -> str:

    words = text.split()
    changes = 0

    for index, word in enumerate(words):

        if changes >= 3:
            break

        stripped = word.strip(".,!?;:")
        lower = stripped.lower()

        if lower not in TYPO_MAP:
            continue

        replacement = TYPO_MAP[lower]

        if stripped[:1].isupper():
            replacement = replacement.capitalize()

        words[index] = word.replace(
            stripped,
            replacement,
            1,
        )

        changes += 1

    return " ".join(words)


NOISY_WRITING = CounterfactualAttack(
    name="noisy_writing",
    family="communication_quality",
    description=(
        "Introduces light spelling noise while preserving "
        "the substantive case."
    ),
    transform=add_light_writing_noise,
)
