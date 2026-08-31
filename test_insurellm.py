import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_ID = "piyushptiwari/InsureLLM-4B"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

print("Loading InsureLLM-4B...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print("Model loaded.")

messages = [
    {
        "role": "user",
        "content": """
        Assess this insurance claim for investigation requirements.
        
        Claim:
        The policyholder returned home and found water damage.
        A plumber identified a burst pipe and photos were provided.
        
        Choose exactly ONE label:
        
        NONE = no additional investigation needed
        ROUTINE = normal claims-processing checks only
        ENHANCED = additional evidence or enhanced review required
        FRAUD = specialist fraud investigation required
        
        Return only the label. No explanation.
        """,
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Required by the InsureLLM model card:
# prevent it spending the output budget in thinking mode
text += "<think>\n</think>\n"

inputs = tokenizer(
    text,
    return_tensors="pt",
)

print("Generating...")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=False,
    )

generated = output[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated,
    skip_special_tokens=True,
)

print()
print("=== INSURELLM RESPONSE ===")
print(response)
