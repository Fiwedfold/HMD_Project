from copy import deepcopy
import json
import re
import unicodedata
from typing import Any, Dict, Optional

# Valid values for enum slots
VALID_GENRES = [
    "action", "adventure", "casual", "early access", "education",
    "free to play", "game development", "gore", "indie",
    "massively multiplayer", "movie", "rpg", "racing",
    "simulation", "sports", "strategy", "violent"
]
VALID_PLATFORMS = ["windows", "mac", "linux"]
VALID_INFO_TYPES = [
    "summary", "genre", "mode", "platform", "required_age", "publisher", "developer",
    "review", "price"
]
VALID_CRITERIA = ["price", "review", "genre"]
VALID_MODES = ["singleplayer", "multiplayer"]


class DST:
    """Dialogue State Tracker to track current user requests."""

    def __init__(self):
        """Initialize Dialogue State Tracker module."""
        # Empty dialogue state at the start
        self.ds: Dict[str, Any] = {
            "intent": None,
            "slots": {}
        }

    def get_ds(self) -> Dict[str, Any]:
        """
        Return the dialogue state as a Python dict.

        IMPORTANT: Previously this returned a JSON string, which caused
        type errors downstream. Now it always returns a dict so that
        agent.py, DM, NLG, etc. can safely access ["slots"].
        """
        return deepcopy(self.ds)

    @staticmethod
    def _normalize_names(name: str) -> str:
        """
        Game titles, publisher and developer names must be normalized.
        Args:
          name (str): name to normalize.
        Returns:
          str: normalized name.
        """
        name = ''.join(c for c in name if unicodedata.category(c) != 'So')
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
        name = name.lower()
        name = re.sub(r'[^a-z0-9\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def _clean_slot_values(self, slot_name: str, val: Any) -> Optional[Any]:
        """Clean specific slots according to some rules."""
        if val is None:
            return None

        # Normalize names (titles, people, similar titles)
        if slot_name in ["title", "title1", "title2", "similar_title", "publisher", "developer", "name"]:
            return self._normalize_names(str(val))

        # Value is a numeral
        if slot_name == "price":
            try:
                cleaned_price = re.sub(r'[^\d.]', '', str(val))
                return float(cleaned_price)
            except ValueError:
                return None  # Drop invalid price

        if slot_name in ["release_date", "required_age"]:
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        # Value is a str and not a name
        val_str = str(val).lower().strip()

        if slot_name == "info":
            return val_str if val_str in VALID_INFO_TYPES else None

        if slot_name == "genre":
            return val_str if val_str in VALID_GENRES else None

        if slot_name == "platform":
            if "win" in val_str:
                return "windows"
            if "osx" in val_str or "mac" in val_str:
                return "mac"
            return val_str if val_str in VALID_PLATFORMS else None

        if slot_name == "criteria":
            return val_str if val_str in VALID_CRITERIA else None

        if slot_name == "mode":
            return val_str if val_str in VALID_MODES else None

        if isinstance(val, str):
            return val.strip()
        return val

    def _clean_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean the NLU output by validating the output.
        Args:
          response (Dict[str, Any]): dict nlu response.
        Returns:
          Dict[str, Any]: cleaned response.
        """

        def remove_nulls(obj: Any) -> Any:
            """
            Normalize null strings but keep slot keys in dialogue state.
            Args:
              obj (Any): input object.
            Returns:
              Any: normalized object.
            """
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    # If it's a string "null", convert to None
                    if isinstance(v, str) and v.lower().strip() == "null":
                        v = None
                    # Recursively clean nested dicts
                    if isinstance(v, dict):
                        nested = remove_nulls(v)
                        cleaned[k] = nested
                    else:
                        cleaned[k] = v
                return cleaned
            return obj

        # Create a clean copy of response
        clean_data = remove_nulls(deepcopy(response))

        # Clean slot values
        if "slots" in clean_data and isinstance(clean_data["slots"], dict):
            cleaned_slots = {}
            for slot_name, slot_val in clean_data["slots"].items():
                cleaned_val = self._clean_slot_values(slot_name, slot_val)
                # We keep the slot key, even if value is None (explicitly unknown)
                cleaned_slots[slot_name] = cleaned_val
            clean_data["slots"] = cleaned_slots

        return clean_data

    def update_ds(self, nlu_response: Any) -> None:
        """
        Update dialogue state by validating the NLU response.
        Args:
          nlu_response: NLU response. May be a JSON string or a dict.
        """
        # Accept both dict and JSON string
        if isinstance(nlu_response, str):
            try:
                parsed = json.loads(nlu_response)
            except Exception:
                # If NLU output is malformed, do not crash; keep DS unchanged
                return
        elif isinstance(nlu_response, dict):
            parsed = deepcopy(nlu_response)
        else:
            # Unsupported type, ignore
            return

        cleaned = self._clean_response(parsed)

        # Get intents
        new_intent = cleaned.get("intent")
        current_intent = self.ds["intent"]

        # If intent changes, discard previous slots (context switch)
        if new_intent is not None and new_intent != current_intent:
            self.ds["intent"] = new_intent
            self.ds["slots"] = {}

        # Merge cleaned slots into the current slots
        if "slots" in cleaned and isinstance(cleaned["slots"], dict):
            for k, v in cleaned["slots"].items():
                self.ds["slots"][k] = v
