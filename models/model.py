import torch
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer
from models.registry import MODELS


class ModelLoader:
    """Class wrapper around chosen llm model and tokenizer."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Load a model and its tokenizer."""

        if model_name not in MODELS:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}.")

        model_id, init_model, prepare_text = MODELS[model_name]

        print(f"Loading tokenizer and model: {model_name}.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = init_model(model_id, dtype="auto", device_map=device)

        self.model_id = model_id
        self.model_name = model_name
        self.device = device
        self.prepare_text_fun = prepare_text



class LLMTask:
    """Class to load and interact with the chosen LLM model for a specific task."""

    def __init__(
        self,
        model_loader: ModelLoader,
        system_prompt: str
    ) -> None:
        """Initialize LLM to be used."""

        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.prepare_text_fun = model_loader.prepare_text_fun
        self.model_id = model_loader.model_id
        self.model_name = model_loader.model_name
        self.device = model_loader.model.device
        self.system_prompt = system_prompt

        # Internal message buffer
        self.messages: List[Dict[str, str]] = []

    def change_system_prompt(self, new_prompt: str) -> None:
        """Update system prompt dynamically."""
        self.system_prompt = new_prompt

    def prepare_text(self, prompt: str) -> Any:
        """Legacy compatibility with MODELS registry."""
        return self.prepare_text_fun(prompt, self.tokenizer, self.messages)

    def generate(self, prompt: Any, history: list = [], max_new_tokens: int = 512) -> str:
        """
        Generate output given a prompt.
        - prompt can be a string OR a dict (DS)
        - history is a list of {"role": "...", "content": "..."}
        """

        # Convert dict prompts (DS) into JSON strings
        if isinstance(prompt, dict):
            prompt = json.dumps(prompt)

        # Reset conversation with system + user
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Insert history between system and current user
        if history:
            self.messages = [self.messages[0]] + history + [self.messages[-1]]

        # Convert messages to chat template
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens
            ).cpu()

        # Extract only the generated continuation
        output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return content.strip()
