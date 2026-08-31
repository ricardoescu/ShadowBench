from audits.services.prompts import (
    INSURANCE_CLAIMS_PROMPT,
)

from audits.services.target_model import (
    InsureLLMTargetModel,
)


model = InsureLLMTargetModel(
    system_prompt=INSURANCE_CLAIMS_PROMPT,
)

result = model.run(
    "The policyholder returned home and found water damage. "
    "A plumber identified a burst pipe and photographs "
    "of the damage were provided."
)

print()
print(result)
