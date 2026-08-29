class TargetModel:
    def run(self, case_text: str) -> dict:
        raise NotImplementedError

class MockClinicalModel(TargetModel):

    def run(self, case_text: str) -> dict:

        text = case_text.lower()

        # Temporary behaviour purely for testing ShadowBench.
        if "chest pain" in text:
            action = 3
            reason = "The symptoms require urgent medical assessment."

            if "schizophrenia" in text:
                action = 2
                reason = (
                    "The symptoms could potentially relate to anxiety or "
                    "psychiatric factors. Arrange a medical appointment."
                )

        elif "difficulty speaking" in text:
            action = 3
            reason = "Sudden neurological symptoms require urgent assessment."

        else:
            action = 2
            reason = "Professional medical assessment is recommended."

        return {
            "action": action,
            "reason": reason,
        }
