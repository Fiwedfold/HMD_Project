from agent.dst import DST
from models.model import ModelLoader, LLMTask
from data.kb import KnowledgeBase
from collections import deque
from typing import Deque, Dict
import yaml
import os
import json
import re


class DialogueAgent:
    def __init__(self, model: Dict[str, str], device: str = "cuda", n_exchanges: int = 3) -> None:
        """Initialize dialogue agent."""
        self.model_name = model
        self.device = device
        self.n_exchanges = n_exchanges
        
        # Load knowledge base
        self.kb = KnowledgeBase()

        # Model loader cache
        self.loaders = {}

        # Load prompts
        self.system_prompt = self._load_prompt()

        # Instantiate components
        self.preproc = LLMTask(self._get_loader("preproc"), self.system_prompt["preproc"]["prompt"])
        self.nlu = LLMTask(self._get_loader("nlu"), self.system_prompt["nlu"]["prompt"])
        self.dm = LLMTask(self._get_loader("dm"), self.system_prompt["dm"]["prompt"]["main"])
        self.nlg = LLMTask(self._get_loader("nlg"), self.system_prompt["nlg"]["prompt"]["main"])
        self.sa = LLMTask(self._get_loader("sa"), self.system_prompt["sa"]["prompt"])

        # Dialogue State Tracker
        self.dst = DST()

        # Conversation history
        max_len = self.n_exchanges * 2
        self.history: Deque[Dict[str, str]] = deque(maxlen=max_len)

    # ---------------------------------------------------------
    # Model loader
    # ---------------------------------------------------------
    def _get_loader(self, component: str) -> ModelLoader:
        default_model = self.model_name.get("default")
        model_name = self.model_name.get(component, default_model)
        if not model_name:
            raise ValueError(f"model_name dict must contain a 'default' key or a key for '{component}'")

        if model_name not in self.loaders:
            self.loaders[model_name] = ModelLoader(model_name, self.device)
        return self.loaders[model_name]

    # ---------------------------------------------------------
    # Load prompts
    # ---------------------------------------------------------
    def _load_prompt(self) -> dict:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_dir = os.path.join(base_dir, "prompt")

        prompts = {}
        for filename in os.listdir(prompt_dir):
            if filename.endswith(".yaml"):
                file_path = os.path.join(prompt_dir, filename)
                key_name = os.path.splitext(filename)[0]
                with open(file_path, "r", encoding="utf-8") as file:
                    prompts[key_name] = yaml.safe_load(file)
        return prompts

    # ---------------------------------------------------------
    # Sentiment Analysis
    # ---------------------------------------------------------
    def get_review_sa(self, reviews: dict) -> dict:
        report = {"positive": 0, "negative": 0, "neutral": 0}

        if not reviews or "sample_reviews" not in reviews:
            return report

        for review in reviews["sample_reviews"]:
            if not isinstance(review, str):
                continue
            if len(review.strip()) < 10:
                continue

            label = self.sa.generate(review)
            label = label.strip().lower()
            label = label.replace("[", "").replace("]", "")
            label = label.replace('"', "").replace("'", "")
            label = label.strip()

            if label not in report:
                label = "neutral"

            report[label] += 1

        return report

    # ---------------------------------------------------------
    # External Knowledge Retrieval
    # ---------------------------------------------------------
    def get_knowledge(self, nba: str, ds: dict) -> dict:
        pattern = r'^([a-zA-Z_]\w*)\s*\('
        match = re.match(pattern, nba)
        if not match:
            return {}

        action_name = match.group(1)
        if action_name in ["get_info", "fallback"]:
            return {}

        intent = ds.get("intent")
        slots = ds.get("slots", {})

        match intent:
            case "get_game_info":
                title = slots.get("title")
                info = slots.get("info", "summary")
                data = self.kb.get_game_info(title, info)
                if "review" in data:
                    sa = self.get_review_sa(data["review"])
                    data["review"] = sa

            case "discover_game":
                allowed = [
                    "genre", "price", "release_year", "platform", "mode",
                    "required_age", "publisher", "developer", "similar_title"
                ]
                slots_for_kb = {k: v for k, v in slots.items() if k in allowed}

                # Souls-like normalization at the agent level
                sim = slots_for_kb.get("similar_title")
                if sim:
                    sim_str = str(sim).lower()
                    if "soul" in sim_str:  # matches "souls", "soulslike", etc.
                        slots_for_kb["similar_title"] = "dark souls iii"
                        # If NLU biased genre to "indie" for Souls-like, correct it to "action"
                        if slots_for_kb.get("genre") == "indie":
                            slots_for_kb["genre"] = "action"

                data = self.kb.discover_game(**slots_for_kb)

            case "compare_games":
                slots_for_kb = {k: v for k, v in slots.items() if k != "comparison_mode"}
                data = self.kb.compare_games(**slots_for_kb)

                if "review" in data:
                    enriched = {}
                    for title, reviews in data["review"].items():
                        enriched[title] = {
                            "summary": reviews.get("summary"),
                            "sentiment": self.get_review_sa(reviews)
                        }
                    data["review"] = enriched

            case "get_term_explained":
                data = self.kb.get_term_explained(**slots)

            case "get_friend_games":
                data = self.kb.get_friend_games(**slots)

            case "add_to_wishlist":
                data = self.kb.add_wishlist(**slots)

            case "remove_from_wishlist":
                data = self.kb.remove_wishlist(**slots)

            case "get_wishlist":
                data = self.kb.get_wishlist()

            case _:
                data = {"error": "Invalid intent."}

        return data

    # ---------------------------------------------------------
    # Chat Pipeline
    # ---------------------------------------------------------
    def chat(self, user_input: str) -> str:
        print(user_input, list(self.history))

        multiple_intents = False

        # PREPROC
        preproc_out = self.preproc.generate(user_input)
        split_input = json.loads(preproc_out)
        print("PREPROC->", split_input)

        if len(split_input) > 1:
            multiple_intents = True
            for input in split_input[:-1]:
                self.history.append({"role": "user", "content": input})

        nlu_input = user_input if len(split_input) == 0 else split_input[-1]

        # NLU
        nlu_out = self.nlu.generate(nlu_input)
        print(f"NLU OUT->{nlu_out}")

        self.dst.update_ds(nlu_out)
        ds = self.dst.get_ds()
        print(f"DST OUT->{ds}")

        # DM
        intent_name = self.dst.ds["intent"]
        self.dm.change_system_prompt(
            self.system_prompt["dm"]["prompt"]["main"]
            + self.system_prompt["dm"]["prompt"][intent_name]
        )
        nba = self.dm.generate(ds)
        print(f"DM OUT->{nba}")

        # KB
        ek = self.get_knowledge(nba, self.dst.ds)
        if "error" in ek:
            nba = "fallback()"
            ek = None

        # NLG
        self.nlg.change_system_prompt(
            self.system_prompt["nlg"]["prompt"]["main"]
            + self.system_prompt["nlg"]["prompt"][intent_name]
        )

        nlg_input = (
            f"NBA: {nba}\n"
            f"DS: {self.dst.get_ds()}\n"
            f"EK: {ek}\n"
            f"MI: {multiple_intents}"
        )
        print("NLG Input -> ", nlg_input)

        response = self.nlg.generate(nlg_input)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response
