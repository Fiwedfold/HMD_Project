import torch
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer
from models.registry import MODELS


class ModelLoader:
    """Class wrapper around chosen LLM model and tokenizer."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """
        Load a model and its tokenizer.

        Args:
            model_name (str): name of the model to load, must exist in MODELS.
            device (str): device where to load the model ("cpu", "cuda", or mapping).
        """
        if model_name not in MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}."
            )

        model_id, init_model, prepare_text = MODELS[model_name]

        print(f"Loading tokenizer and model: {model_name}.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # init_model is expected to handle dtype/device_map internally
        self.model = init_model(model_id, dtype="auto", device_map=device)

        self.model_id = model_id
        self.model_name = model_name
        self.device = device
        self.prepare_text_fun = prepare_text


class LLMTask:
    """Wraps a loaded LLM and tokenizer for a specific task with a system prompt."""

    def __init__(
        self,
        model_loader: ModelLoader,
        system_prompt: str,
    ) -> None:
        """
        Initialize LLM to be used for a specific task.

        Args:
            model_loader (ModelLoader): loader that handles model and tokenizer.
            system_prompt (str): detailed description of the task for the LLM.
        """
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.prepare_text_fun = model_loader.prepare_text_fun
        self.model_id = model_loader.model_id
        self.model_name = model_loader.model_name
        # device is stored as string in ModelLoader
        self.device = model_loader.device
        self.system_prompt = system_prompt

        # Internal message buffer (for potential multi-turn usage)
        self.messages: List[Dict[str, str]] = []

    def change_system_prompt(self, new_prompt: str) -> None:
        """Change the system prompt to dynamically adjust the task."""
        self.system_prompt = new_prompt

    def prepare_text(self, prompt: str) -> Any:
        """
        Legacy compatibility wrapper for MODELS.prepare_text_fun.

        Not used in the chat-style generate() for Qwen, but kept so that
        other components relying on this interface do not break.
        """
        return self.prepare_text_fun(prompt, self.tokenizer, self.messages)

    def generate(
        self,
        prompt: Any,
        history: List[Dict[str, str]] = [],
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate output given a prompt.

        Args:
            prompt (Any): user prompt. Can be a string or a dict (e.g. DS).
                          If dict, it will be converted to JSON string.
            history (list): optional list of previous messages, each:
                            {"role": "user" | "assistant", "content": str}
            max_new_tokens (int): maximum number of tokens to generate.

        Returns:
            str: generated response text.
        """

        # If prompt is a dict (e.g. DS), convert to JSON so model sees structured input
        if isinstance(prompt, dict):
            prompt = json.dumps(prompt, indent=2)

        # Build messages: system + history + current user
        # We always reset messages per call to keep behaviour explicit.
        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Insert history if provided
        if history:
            # history is expected as a list of {"role": "...", "content": "..."}
            self.messages.extend(history)

        # Append the current user turn
        self.messages.append({"role": "user", "content": prompt})

        # Turn messages into a single chat-formatted string using the tokenizer's template
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize the chat template
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
            ).cpu()

        # Remove the prompt tokens to keep only the continuation
        prompt_length = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][prompt_length:]
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return content.strip()
