# predict.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# This is your Hugging Face model repo name, e.g. "username/merged_llm"
HF_MODEL_ID = "alexxi19/ft-v1-nemo-base-merge-v1"

# Load tokenizer and model at global scope so it’s loaded only once
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # automatically place on GPU if available
)

def predict(prompt: str = "Hello, world!", max_new_tokens: int = 50) -> str:
    """
    Called by Replicate for each inference. 
    :param prompt: The input text prompt
    :param max_new_tokens: How many tokens to generate
    :return: The generated text
    """
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
    # Decode the outputs
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
