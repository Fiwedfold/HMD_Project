import torch
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer
from models.registry import MODELS


class ModelLoader:
    """Loads a Qwen3 model and tokenizer."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        if model_name not in MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}."
            )

        model_id, init_model, prepare_text = MODELS[model_name]

        print(f"Loading tokenizer and model: {model_name}.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model
        self.model = init_model(model_id, dtype="auto", device_map=device)

        self.model_id = model_id
        self.model_name = model_name
        self.device = device
        self.prepare_text_fun = prepare_text


class LLMTask:
    """Wraps a loaded LLM and tokenizer for a specific task with a system prompt."""

    def __init__(self, model_loader: ModelLoader, system_prompt: str) -> None:
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.prepare_text_fun = model_loader.prepare_text_fun
        self.model_id = model_loader.model_id
        self.model_name = model_loader.model_name
        self.device = model_loader.device
        self.system_prompt = system_prompt

    def change_system_prompt(self, new_prompt: str) -> None:
        self.system_prompt = new_prompt

    def generate(
        self,
        prompt: Any,
        history: List[Dict[str, str]] = [],
        max_new_tokens: int = 512,
    ) -> str:

        # Convert dict prompts (DS) into JSON strings
        if isinstance(prompt, dict):
            prompt = json.dumps(prompt, indent=2)

        # -----------------------------------------
        # BUILD QWEN3 CHAT MESSAGES
        # -----------------------------------------
        messages = []

        # SYSTEM
        messages.append({"role": "system", "content": self.system_prompt})

        # HISTORY
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # USER
        messages.append({"role": "user", "content": prompt})

        # -----------------------------------------
        # APPLY QWEN3 CHAT TEMPLATE
        # -----------------------------------------
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        model_inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
            ).cpu()

        # Extract only the assistant continuation
        output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return text.strip()
