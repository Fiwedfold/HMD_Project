import pandas as pd
import json
import os
from typing import Any, Optional, List
import numpy as np
from difflib import get_close_matches
from data.review_api import get_reviews as fetch_steam_reviews


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_PATH = os.path.join(DATA_DIR, "steam_dataset.feather")
USER_PROFILE_PATH = os.path.join(DATA_DIR, "mock_user.json")
GLOSSARY_PATH = os.path.join(DATA_DIR, "video_game_glossary.json")


class KnowledgeBase:
    """KnowledgeBase class used to get data to return to the user."""

    def __init__(self):
        """Initialize external knowledge module."""
        # Load games dataset
        self.game_database = pd.read_feather(GAMES_PATH)
        # Load user profile
        self.user_profile = self._load_json(USER_PROFILE_PATH)
        # Load glossary
        self.glossary = self._load_json(GLOSSARY_PATH)

    # ---------------------------------------------------------
    # JSON utilities
    # ---------------------------------------------------------
    def _load_json(self, path: str) -> Any:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _save_json(self, data: dict, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    # ---------------------------------------------------------
    # Title normalization + fuzzy matching
    # ---------------------------------------------------------
    def _normalize_title(self, title: str) -> str:
        """Normalize user-provided title to match dataset."""
        if not title:
            return ""

        t = title.lower().strip()
        t = t.replace("™", "").replace("®", "")
        t = t.replace(":", "").replace("-", " ")
        t = " ".join(t.split())  # collapse multiple spaces
        return t

    def game_by_title(self, title: str) -> Optional[dict]:
        """Get a game given the title (robust matching)."""
        title_norm = self._normalize_title(title)

        # 1) exact match on name_normalized
        exact = self.game_database[self.game_database["name_normalized"] == title_norm]
        if not exact.empty:
            return exact.iloc[0].to_dict()

        # 2) substring match
        substring = self.game_database[
            self.game_database["name_normalized"].str.contains(title_norm, case=False, na=False)
        ]
        if not substring.empty:
            return substring.iloc[0].to_dict()

        # 3) fuzzy match
        candidates = self.game_database["name_normalized"].tolist()
        match = get_close_matches(title_norm, candidates, n=1, cutoff=0.6)
        if match:
            row = self.game_database[self.game_database["name_normalized"] == match[0]]
            return row.iloc[0].to_dict()

        return None

    # ---------------------------------------------------------
    # get_game_info
    # ---------------------------------------------------------
    def get_game_info(self, title: str, info: str) -> dict:
        """Extract information for get_game_info intent."""
        game = self.game_by_title(title)
        if not game:
            return {"error": f"No game found with title '{title}'"}

        match info:
            case "summary":
                data = game.get("about_the_game")
            case "genre":
                data = game.get("genres", np.ndarray(1))
                data = data.tolist()
            case "mode":
                cats = game.get("categories", [])
                data = {
                    "singleplayer": any("single" in c for c in cats),
                    "multiplayer": any("multi" in c for c in cats)
                }
            case "required_age":
                data = game.get("required_age")
            case "platform":
                data = [p for p in ["windows", "mac", "linux"] if game.get(p) is True]
            case "price":
                data = game.get("price")
            case "review":
                appid = game.get("appid", 0)
                data = self.get_reviews(appid)
            case _:
                data = {"error": "Invalid info"}

        return {info: data}

    # ---------------------------------------------------------
    # discover_game
    # ---------------------------------------------------------
    def discover_game(
        self,
        genre: Optional[str],
        price: Optional[float],
        release_year: Optional[int],
        platform: Optional[str],
        mode: Optional[str],
        required_age: Optional[int],
        publisher: Optional[str],
        developer: Optional[str]
    ) -> dict:
        """Get games that satisfy a set of characteristics."""
        filtered_games = self.game_database.copy()

        if genre:
            filtered_games = filtered_games[
                filtered_games["genres"].astype(str).str.contains(genre, case=False, na=False)
            ]
        if price:
            filtered_games = filtered_games[filtered_games["price"] <= price]
        if release_year:
            filtered_games = filtered_games[
                filtered_games["release_date"].apply(lambda x: x.year) == release_year
            ]
        if platform:
            filtered_games = filtered_games[filtered_games[platform] is True]
        if mode:
            filtered_games = filtered_games[
                filtered_games["categories"].astype(str).str.contains(mode, case=False, na=False)
            ]
        if required_age:
            filtered_games = filtered_games[filtered_games["required_age"] == required_age]
        if publisher:
            filtered_games = filtered_games[
                filtered_games["publishers_normalized"].str.contains(publisher, case=False, na=False)
            ]
        if developer:
            filtered_games = filtered_games[
                filtered_games["developers_normalized"].str.contains(developer, case=False, na=False)
            ]

        results = filtered_games.head(5)
        if results.empty:
            return {"error": "No matches found with characteristics."}

        return {"games": results["name"].tolist()}

    # ---------------------------------------------------------
    # compare_games
    # ---------------------------------------------------------
    def compare_games(self, title1: str, title2: str, criteria: str) -> dict:
        """Get data to compare two games."""
        game1 = self.game_by_title(title1)
        game2 = self.game_by_title(title2)

        if not game1 or not game2:
            return {"error": "Could not retrieve data on one of the two titles"}

        match criteria:
            case "genre":
                data = {
                    title1: game1.get("genres", np.ndarray(1)).tolist(),
                    title2: game2.get("genres", np.ndarray(1)).tolist()
                }
            case "price":
                data = {
                    title1: game1.get("price"),
                    title2: game2.get("price")
                }
            case "review":
                id1 = game1.get("appid", 0)
                id2 = game2.get("appid", 0)
                reviews1 = self.get_reviews(id1)
                reviews2 = self.get_reviews(id2)
                data = {title1: reviews1, title2: reviews2}
            case _:
                return {"error": "Invalid criteria"}

        return {criteria: data}

    # ---------------------------------------------------------
    # friend games
    # ---------------------------------------------------------
    def get_friend_games(self, name: str) -> dict:
        """Get the games of a friend."""
        friends = self.user_profile.get("friends", [])
        for friend in friends:
            if friend["username"].lower() == name.lower():
                return {"friend_games": friend.get("owned", [])}
        return {"error": f"Friend '{name}' not found in friends list"}

    # ---------------------------------------------------------
    # glossary
    # ---------------------------------------------------------
    def get_term_explained(self, term: str) -> dict:
        """Get explanation of a term from the glossary."""
        definition = self.glossary.get(term)
        if not definition:
            return {"error": f"Invalid term '{term}'"}
        return {"definition": definition}

    # ---------------------------------------------------------
    # wishlist
    # ---------------------------------------------------------
    def add_wishlist(self, title: str) -> dict:
        """Add a game to wishlist verifying it also exists."""
        game = self.game_by_title(title)
        if not game:
            return {"error": f"No game found with title '{title}'"}

        current_wishlist = self.user_profile.get("wishlist", [])
        if game["name_normalized"] in current_wishlist:
            return {"error": f"'{game['name']}' is already in wishlist"}

        current_wishlist.append(game["name_normalized"])
        self.user_profile["wishlist"] = current_wishlist
        self._save_json(self.user_profile, USER_PROFILE_PATH)
        return {"confirmation": f"Added '{game['name']}' to your wishlist"}

    def remove_wishlist(self, title: str) -> dict:
        """Remove a game from wishlist."""
        current_wishlist = self.user_profile.get("wishlist", [])
        match = next((g for g in current_wishlist if g == title), None)

        if not match:
            return {"error": f"'{title}' was not found in your wishlist"}

        current_wishlist.remove(match)
        self.user_profile["wishlist"] = current_wishlist
        self._save_json(self.user_profile, USER_PROFILE_PATH)
        return {"confirmation": f"Removed '{match}' from your wishlist"}

    def get_wishlist(self) -> dict:
        """Get user wishlist."""
        wl = self.user_profile.get("wishlist", [])
        return {"wishlist": wl}

    # ---------------------------------------------------------
    # reviews (correctly attached to the class)
    # ---------------------------------------------------------
    def get_reviews(self, id: int, num_reviews: int = 50) -> List[str]:
        """
        Using API get up to date reviews on a game.
        """
        if id is None or id == 0:
            return []
        return fetch_steam_reviews(id, num_reviews=num_reviews)


if __name__ == "__main__":
    kb = KnowledgeBase()

    # Basic manual tests if you want to run kb.py directly

    # print(kb.game_by_title("elden ring"))
    # slots = {"title": "postal 2", "info": "price"}
    # print(kb.get_game_info(**slots))

    # slots = {
    #     "genre": None,
    #     "price": None,
    #     "release_year": 2021,
    #     "platform": "linux",
    #     "mode": "multiplayer",
    #     "required_age": None,
    #     "publisher": None,
    #     "developer": None
    # }
    # print(kb.discover_game(**slots))

    # slots = {"title1": "elden ring", "title2": "dark souls remastered", "criteria": "review"}
    # print(kb.compare_games(**slots))

    # slots = {"name": "Alex"}
    # print(kb.get_friend_games(**slots))

    # slots = {"term": "adventure game"}
    # print(kb.get_term_explained(**slots))

    # print(kb.get_wishlist())
