from typing import Callable

from .attacks import CounterfactualAttack
from .inference_cache import cached_model_run


class ShadowBenchAudit:
    """
    Generic ShadowBench audit engine.

    A domain supplies:
    - cases
    - a target model
    - counterfactual attacks
    - an evaluator

    The engine handles:
    - original inference
    - counterfactual generation
    - cached inference
    - evaluation
    - structured audit results
    """

    def __init__(
        self,
        model,
        attacks: list[CounterfactualAttack],
        evaluator: Callable,
    ):
        self.model = model
        self.attacks = attacks
        self.evaluator = evaluator

    def run_case(self, case: dict) -> dict:
        """
        Run one original case and all configured
        counterfactual variants.
        """

        case_id = case["id"]
        original_text = case["text"]

        # -------------------------
        # Original inference
        # -------------------------

        original_response, original_cached = cached_model_run(
            self.model,
            original_text,
        )

        # -------------------------
        # Counterfactual inference
        # -------------------------

        variant_responses = {}
        variants = {}

        for attack in self.attacks:

            variant_text = attack.apply(
                original_text
            )

            response, cached = cached_model_run(
                self.model,
                variant_text,
            )

            variant_responses[
                attack.name
            ] = response

            variants[
                attack.name
            ] = {
                "family": attack.family,
                "description": attack.description,
                "text": variant_text,
                "response": response,
                "cached": cached,
            }

        # -------------------------
        # Domain evaluation
        # -------------------------

        evaluation = self.evaluator(
            original_response,
            variant_responses,
            case,
        )

        # Preserve arbitrary domain metadata.
        metadata = {
            key: value
            for key, value in case.items()
            if key not in {"id", "text"}
        }

        return {
            "case_id": case_id,
            "metadata": metadata,

            "original": {
                "text": original_text,
                "response": original_response,
                "cached": original_cached,
            },

            "variants": variants,

            "evaluation": evaluation,
        }
