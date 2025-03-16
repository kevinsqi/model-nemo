# predict.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def setup(self):
        """
        This runs once when the container starts.
        Load your Hugging Face model & tokenizer here.
        """
        self.model_id = "alexxi19/ft-v1-nemo-base-merge-v1"
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,  # or whatever precision you want
            device_map="auto"            # place on GPU if available
        )
        
        # If the model is private on HF, handle auth tokens or configuration.
        # e.g. from huggingface_hub import login
        # login(os.environ["HUGGINGFACE_HUB_TOKEN"])

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="Hello, world!"),
        max_new_tokens: int = Input(description="Max tokens to generate", default=50),
        temperature: float = Input(description="Temperature for sampling", default=0.7),
    ) -> str:
        """
        Run inference on a single input. Returns the generated text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature
            )

        # Decode and return the text
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)