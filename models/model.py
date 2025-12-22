import torch
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer
from models.registry import MODELS


class ModelLoader:
    """Loads a model and tokenizer, and registers Qwen special tokens."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        if model_name not in MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}."
            )

        model_id, init_model, prepare_text = MODELS[model_name]

        print(f"Loading tokenizer and model: {model_name}.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # -----------------------------------------
        # ADD QWEN SPECIAL TOKENS (CRITICAL)
        # -----------------------------------------
        special_tokens = {
            "additional_special_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                "<|assistant|>",
                "<|user|>",
                "<|system|>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        # Load model
        self.model = init_model(model_id, dtype="auto", device_map=device)

        # Resize embeddings to include new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

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
        # BUILD QWEN CHAT TEMPLATE MANUALLY
        # -----------------------------------------
        chat = ""

        # SYSTEM
        chat += "<|im_start|>system\n"
        chat += self.system_prompt.strip() + "\n"
        chat += "<|im_end|>\n"

        # HISTORY
        for msg in history:
            role = msg["role"]
            content = msg["content"]

            chat += f"<|im_start|>{role}\n"
            chat += content.strip() + "\n"
            chat += "<|im_end|>\n"

        # USER
        chat += "<|im_start|>user\n"
        chat += prompt.strip() + "\n"
        chat += "<|im_end|>\n"

        # ASSISTANT (generation starts here)
        chat += "<|im_start|>assistant\n"

        # Tokenize
        model_inputs = self.tokenizer(chat, return_tensors="pt").to(self.device)

        # Generate with <|im_end|> as EOS
        eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id
            ).cpu()

        # Extract only the assistant continuation
        output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=False)

        # Stop at <|im_end|>
        if "<|im_end|>" in text:
            text = text.split("<|im_end|>")[0]

        # Remove any leftover special tokens
        text = text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

        return text
