import pandas as pd
import json
import os
from typing import Any, Optional, List, Dict
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
        self.game_database = pd.read_feather(GAMES_PATH)
        self.user_profile = self._load_json(USER_PROFILE_PATH)
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
        if not title:
            return ""
        t = title.lower().strip()
        t = t.replace("™", "").replace("®", "")
        t = t.replace(":", "").replace("-", " ")
        t = " ".join(t.split())
        return t

    def game_by_title(self, title: str) -> Optional[dict]:
        title_norm = self._normalize_title(title)

        exact = self.game_database[self.game_database["name_normalized"] == title_norm]
        if not exact.empty:
            return exact.iloc[0].to_dict()

        substring = self.game_database[
            self.game_database["name_normalized"].str.contains(title_norm, case=False, na=False)
        ]
        if not substring.empty:
            return substring.iloc[0].to_dict()

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
        game = self.game_by_title(title)
        if not game:
            return {"error": f"No game found with title '{title}'"}

        match info:
            case "summary":
                data = game.get("about_the_game")
            case "genre":
                data = game.get("genres", np.ndarray(1)).tolist()
            case "mode":
                cats = game.get("categories", [])
                data = {
                    "singleplayer": any("single" in c for c in cats),
                    "multiplayer": any("multi" in c for c in cats),
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
    # discover_game (FULLY FIXED VERSION)
    # ---------------------------------------------------------
    def discover_game(
    self,
    genre=None,
    price=None,
    release_year=None,
    platform=None,
    mode=None,
    required_age=None,
    publisher=None,
    developer=None,
    similar_title=None
):
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

    # ✅ NEW : similar_title support
    if similar_title:
        ref = self.game_by_title(similar_title)
        if ref:
            ref_genres = set(ref.get("genres", []))
            filtered_games["similarity"] = filtered_games["genres"].apply(
                lambda g: len(ref_genres.intersection(set(g))) if isinstance(g, list) else 0
            )
            filtered_games = filtered_games.sort_values("similarity", ascending=False)

    # ✅ NEW : fallback si aucun résultat
    if filtered_games.empty and similar_title:
        ref = self.game_by_title(similar_title)
        if ref:
            ref_genres = set(ref.get("genres", []))
            filtered_games = self.game_database.copy()
            filtered_games["similarity"] = filtered_games["genres"].apply(
                lambda g: len(ref_genres.intersection(set(g))) if isinstance(g, list) else 0
            )
            filtered_games = filtered_games.sort_values("similarity", ascending=False)

    results = filtered_games.head(5)
    if results.empty:
        return {"error": "No matches found with characteristics."}

    return {"games": results["name"].tolist()}

    # ---------------------------------------------------------
    # genre similarity helper
    # ---------------------------------------------------------
    def compute_genre_similarity(self, game1: dict, game2: dict) -> Dict[str, Any]:
        g1 = game1.get("genres", np.ndarray(0))
        g2 = game2.get("genres", np.ndarray(0))
        g1_set = set(g1.tolist() if isinstance(g1, np.ndarray) else g1)
        g2_set = set(g2.tolist() if isinstance(g2, np.ndarray) else g2)

        overlap = list(g1_set & g2_set)
        only1 = list(g1_set - g2_set)
        only2 = list(g2_set - g1_set)

        union = g1_set | g2_set
        jaccard = float(len(overlap) / len(union)) if union else 0.0

        return {
            "overlap": overlap,
            "only_game1": only1,
            "only_game2": only2,
            "jaccard": jaccard,
        }

    # ---------------------------------------------------------
    # metadata comparison helper
    # ---------------------------------------------------------
    def get_comparison_metadata(self, game1: dict, game2: dict) -> Dict[str, Any]:
        def platforms(g: dict) -> List[str]:
            return [p for p in ["windows", "mac", "linux"] if g.get(p) is True]

        def modes(g: dict) -> Dict[str, bool]:
            cats = g.get("categories", [])
            return {
                "singleplayer": any("single" in c for c in cats),
                "multiplayer": any("multi" in c for c in cats),
            }

        meta = {
            "price": {
                game1["name_normalized"]: game1.get("price"),
                game2["name_normalized"]: game2.get("price"),
            },
            "platforms": {
                game1["name_normalized"]: platforms(game1),
                game2["name_normalized"]: platforms(game2),
            },
            "modes": {
                game1["name_normalized"]: modes(game1),
                game2["name_normalized"]: modes(game2),
            },
            "release_year": {
                game1["name_normalized"]: game1.get("release_date").year
                if game1.get("release_date") is not None
                else None,
                game2["name_normalized"]: game2.get("release_date").year
                if game2.get("release_date") is not None
                else None,
            },
            "publisher": {
                game1["name_normalized"]: game1.get("publishers_normalized"),
                game2["name_normalized"]: game2.get("publishers_normalized"),
            },
            "developer": {
                game1["name_normalized"]: game1.get("developers_normalized"),
                game2["name_normalized"]: game2.get("developers_normalized"),
            },
        }
        return meta

    # ---------------------------------------------------------
    # compare_games
    # ---------------------------------------------------------
    def compare_games(self, title1: str, title2: str, criteria: str) -> dict:
        game1 = self.game_by_title(title1)
        game2 = self.game_by_title(title2)

        if not game1 or not game2:
            return {"error": "Could not retrieve data on one of the two titles"}

        title1_norm = game1["name_normalized"]
        title2_norm = game2["name_normalized"]

        match criteria:
            case "genre":
                data = {"genre": self.compute_genre_similarity(game1, game2)}

            case "price":
                data = {
                    "price": {
                        title1_norm: game1.get("price"),
                        title2_norm: game2.get("price"),
                    }
                }

            case "metadata":
                data = {"metadata": self.get_comparison_metadata(game1, game2)}

            case "review":
                id1 = game1.get("appid", 0)
                id2 = game2.get("appid", 0)
                reviews1 = self.get_reviews(id1)
                reviews2 = self.get_reviews(id2)
                data = {
                    "review": {
                        title1_norm: reviews1,
                        title2_norm: reviews2,
                    }
                }

            case "all":
                id1 = game1.get("appid", 0)
                id2 = game2.get("appid", 0)
                reviews1 = self.get_reviews(id1)
                reviews2 = self.get_reviews(id2)
                data = {
                    "review": {
                        title1_norm: reviews1,
                        title2_norm: reviews2,
                    },
                    "genre": self.compute_genre_similarity(game1, game2),
                    "metadata": self.get_comparison_metadata(game1, game2),
                }

            case _:
                return {"error": "Invalid criteria"}

        return data

    # ---------------------------------------------------------
    # friend games
    # ---------------------------------------------------------
    def get_friend_games(self, name: str) -> dict:
        friends = self.user_profile.get("friends", [])
        for friend in friends:
            if friend["username"].lower() == name.lower():
                return {"friend_games": friend.get("owned", [])}
        return {"error": f"Friend '{name}' not found in friends list"}

    # ---------------------------------------------------------
    # glossary
    # ---------------------------------------------------------
    def get_term_explained(self, term: str) -> dict:
        definition = self.glossary.get(term)
        if not definition:
            return {"error": f"Invalid term '{term}'"}
        return {"definition": definition}

    # ---------------------------------------------------------
    # wishlist
    # ---------------------------------------------------------
    def add_wishlist(self, title: str) -> dict:
        game = self.game_by_title(title)
        if not game:
            return {"error": f"No game found with title '{title}'"}

        wl = self.user_profile.get("wishlist", [])
        if game["name_normalized"] in wl:
            return {"error": f"'{game['name']}' is already in wishlist"}

        wl.append(game["name_normalized"])
        self.user_profile["wishlist"] = wl
        self._save_json(self.user_profile, USER_PROFILE_PATH)
        return {"confirmation": f"Added '{game['name']}' to your wishlist"}

    def remove_wishlist(self, title: str) -> dict:
        wl = self.user_profile.get("wishlist", [])
        match = next((g for g in wl if g == title), None)

        if not match:
            return {"error": f"'{title}' was not found in your wishlist"}

        wl.remove(match)
        self.user_profile["wishlist"] = wl
        self._save_json(self.user_profile, USER_PROFILE_PATH)
        return {"confirmation": f"Removed '{match}' from your wishlist"}

    def get_wishlist(self) -> dict:
        wl = self.user_profile.get("wishlist", [])
        return {"wishlist": wl}

    # ---------------------------------------------------------
    # reviews
    # ---------------------------------------------------------
    def get_reviews(self, id: int, num_reviews: int = 50) -> dict:
        if id is None or id == 0:
            return {"summary": None, "sample_reviews": []}
        return fetch_steam_reviews(id, num_reviews=num_reviews)


if __name__ == "__main__":
    kb = KnowledgeBase()
