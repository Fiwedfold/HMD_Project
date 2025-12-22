import torch
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer
from models.registry import MODELS


class ModelLoader:
    """Charge un modèle et son tokenizer pour un nom donné dans MODELS."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        if model_name not in MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}."
            )

        model_id, init_model, prepare_text = MODELS[model_name]

        print(f"Loading tokenizer and model: {model_name}.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Le modèle Qwen3 gère lui-même le mode instruct/chat via apply_chat_template
        self.model = init_model(model_id, dtype="auto", device_map=device)

        self.model_id = model_id
        self.model_name = model_name
        self.device = device


class LLMTask:
    """
    Wrapper pour un LLM + tokenizer, avec un system prompt spécifique à la tâche.
    Utilisé pour PREPROC, NLU, DM, NLG, SA, etc.
    """

    def __init__(self, model_loader: ModelLoader, system_prompt: str) -> None:
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.model_id = model_loader.model_id
        self.model_name = model_loader.model_name
        self.device = model_loader.device
        self.system_prompt = system_prompt

    def change_system_prompt(self, new_prompt: str) -> None:
        """Permet de changer dynamiquement le system prompt (ex: selon l'intent DM/NLG)."""
        self.system_prompt = new_prompt

    def generate(
        self,
        prompt: Any,
        history: List[Dict[str, str]] = [],
        max_new_tokens: int = 512,
    ) -> str:
        """
        Génère une réponse pour un prompt donné.

        Args:
            prompt: soit une string, soit un dict (ex: DS).
                    Si dict, il est converti en JSON pretty.
            history: liste optionnelle de messages précédents au format
                     [{"role": "user"|"assistant", "content": str}, ...]
            max_new_tokens: nb max de tokens générés.

        Returns:
            str: texte généré (contenu assistant uniquement).
        """

        # Si le prompt est un DS ou autre dict, on le sérialise proprement
        if isinstance(prompt, dict):
            prompt = json.dumps(prompt, indent=2)

        # Construction de la liste de messages pour Qwen3
        messages: List[Dict[str, str]] = []

        # SYSTEM
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # HISTORY (optionnelle, mais tu peux l'utiliser si tu veux un vrai multi-turn)
        for msg in history:
            # On suppose history déjà au bon format
            messages.append(
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }
            )

        # USER
        messages.append({"role": "user", "content": prompt})

        # Utilisation du chat template officiel Qwen3
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # ajoute le tour assistant à compléter
        )

        # Tokenisation
        model_inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.device)

        # Génération
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
            ).cpu()

        # On supprime les tokens du prompt pour ne garder que la continuation assistant
        prompt_len = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][prompt_len:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return text.strip()
