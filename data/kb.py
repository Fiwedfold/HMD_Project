import pandas as pd
import json
import os
from typing import Any, Optional
import numpy as np
from typing import Any, Optional, List
from data.review_api import get_reviews as fetch_steam_reviews


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_PATH = os.path.join(DATA_DIR,"steam_dataset.feather")
USER_PROFILE_PATH = os.path.join(DATA_DIR,"mock_user.json")
GLOSSARY_PATH = os.path.join(DATA_DIR,"video_game_glossary.json")


class KnowledgeBase:
  """KnowledgeBase class used to get data to return to the user."""

  def __init__(self):
    """Initialize external knowledge module."""
    # Load games dataset
    self.game_database = pd.read_feather(GAMES_PATH)
    # Load user profile
    self.user_profile = self._load_json(USER_PROFILE_PATH)
    self.glossary = self._load_json(GLOSSARY_PATH)
  
  def _load_json(self, path: str) -> Any:
    """Utility to load json files.
    Args:
      path (str): path to load.
    Returns:
      Any: loaded data.
    """
    with open(path, "r", encoding="utf-8") as file:
      return json.load(file)

  def _save_json(self, data:dict, path: str) -> None:
    """Save json file.
    Args:
      data (dict): data to save.
      path (str): path where to save it.
    """
    with open(path, "w", encoding="utf-8") as file:
      json.dump(data, file, indent=2)

  def game_by_title(self, title: str) -> Optional[dict]:
    """Get a game given the title.
    Args:
      title (str): title of the game already normalized.
    Returns:
      Optional[dict]: returns the data of that game.
    """
    match = self.game_database[self.game_database['name_normalized'] == title]
      
    if match.empty:
      return None
    
    result = match.iloc[0]
    return result.to_dict()
  

  def get_game_info(self, title: str, info: str) -> dict:
    """Extract information for get_game_info intent.
    Args:
      slots (dict): slots of the get_game_info intent.
    Returns:
      dict: requested information or errror message.
    """
    # First get game in the database 
    game = self.game_by_title(title)
    if not game:
      # If there is no match, error message
      return {"error": f"No game found with title '{title}'"}

    # Based on info, different information is returned
    match info:
      case "summary":
        data = game.get("about_the_game")
      case "genre":
        data = game.get("genres", np.ndarray(1))
        data = data.tolist()
      case "mode":
        # Extract a dict telling which modes are available
        data = game.get("categories", [])
        data = {
          "singleplayer": any("single" in c for c in data),
          "multiplayer": any("multi" in c for c in data)
        }
      case "required_age":
        data = game.get("required_age")
      case "platform":
        data = [p for p in ["windows", "mac", "linux"] if game.get(p) is True]
      case "price":
        data = game.get("price")
      case "review":
        id = game.get("appid", 0)
        data = self.get_reviews(id)
      case _:
        data = {"error": "Invalid info"}

    return {info: data}
        
  
  def discover_game(self, genre: Optional[str], price: Optional[float], release_year: Optional[int], platform: Optional[str], mode: Optional[str], required_age: Optional[int], publisher: Optional[str], developer: Optional[str]) -> dict:
    """Get games that satisfy a set of characteristics.
    Args:
      genre (Optional[str]): genre of the games to discover.
      price (Optional[float]): higher bound of the games to discover.
      release_year (Optional[int]): year of release of the games to discover.
      platform (Optional[str]): platform of the games to discover.
      mode (Optional[str]): if the games must have either singleplayer or multiplayer.
      required_age (Optional[int]): required age to play the games to discover.
      publisher (Optional[str]): name of the publisher of the games to discover.
      developer (Optional[str]): name of the developer of the games to discover.
    Returns:
      dict: result containing matches.
    """
    # Copy the game database
    filtered_games = self.game_database.copy()

    # Filtering genre
    if genre:
      filtered_games = filtered_games[filtered_games['genres'].astype(str).str.contains(genre, case=False, na=False)]
    # Filtering the price as an upper bound
    if price:
      filtered_games = filtered_games[filtered_games['price'] <= price]
    # Filter release year
    if release_year:
      # From date take the year
      filtered_games = filtered_games[filtered_games['release_date'].apply(lambda x: x.year) == release_year]
    # Filter platform
    if platform:
      filtered_games = filtered_games[filtered_games[platform] == True]
    # Filter game mode
    if mode:
      filtered_games = filtered_games[filtered_games['categories'].astype(str).str.contains(mode, case=False, na=False)]
    # Filter required age
    if required_age:
      filtered_games = filtered_games[filtered_games['required_age'] == required_age]
    # Filter publisher and developer using the normalized fields
    if publisher:
      filtered_games = filtered_games[filtered_games['publishers_normalized'].str.contains(publisher, case=False, na=False)]
    if developer:
      filtered_games = filtered_games[filtered_games['developers_normalized'].str.contains(developer, case=False, na=False)]

    # Take first 5 matches
    results = filtered_games.head(5)

    if results.empty:
      return {"error": "No matches found with characteristics."}

    return {"games": results['name'].tolist()}


  def compare_games(self, title1: str, title2: str, criteria: str) -> dict:
    """Get data to compare two games.
    Args:
      title1 (str): title of first game to compare.
      title2 (str): title of the second game to compare.
      criteria (str): criteria to use for comparison, either 'price', 'genre' or 'review'.
    Returns:
      dict: data to use for comparison.
    """
    # Get the two games data
    game1 = self.game_by_title(title1)
    game2 = self.game_by_title(title2)
    # If one game was not found, return an error
    if not game1 or not game2:
      return {"error": "Could not retrieve data on one of the two titles"}

    # Based on criteria, different info is returned
    match criteria:
      case "genre":
        data = {title1: game1.get("genres", np.ndarray(1)).tolist(), title2: game2.get("genres", np.ndarray(1)).tolist()}
      case "price":
        data = {title1: game1.get('price'), title2: game2.get('price')}
      case "review":
        id1 = game1.get("appid", 0)
        id2 = game2.get("appid", 0)
        reviews1 = self.get_reviews(id1)
        reviews2 = self.get_reviews(id2)
        data = {title1: reviews1, title2: reviews2}
      case _:
        return {"error": "Invalid criteria"}

    return {criteria: data}


  def get_friend_games(self, name: str) -> dict:
    """Get the games of a friend.
    Args:
      name (str): friend username.
    Returns:
      dict: friend games or error message.
    """        
    # Search in loaded user profile mock
    friends = self.user_profile.get('friends', [])
        
    # Search for friend
    for friend in friends:
      if friend['username'].lower() == name.lower():
        return {"friend_games": friend.get('owned', [])}
        
    return {"error": f"Friend '{name}' not found in friends list"}


  def get_term_explained(self, term: str) -> dict:
    """Get explaination of a term from the glossary.
    Args:
      term (str): term to explain.
    Returns:
      dict: explaination of a term or error message.
    """        
    # Search for direct match
    definition = self.glossary.get(term)
        
    if not definition:
      return {"error": f"Invalid term '{term}'"}
        
    return {"definition": definition}
  

  def add_wishlist(self, title: str) -> dict:
    """Add a game to wishlist verifying it also exists.
    Args:
      title (str): title of game to add.
    Returns:
      dict: confirm or error message
    """
    # Search for game in database
    game = self.game_by_title(title)
        
    if not game:
      return {"error": f"No game found with title '{title}'"}
        
    # Check if already in wishlist
    current_wishlist = self.user_profile.get('wishlist', [])
    if game['name_normalized'] in current_wishlist:
      return {"error": f"'{game['name']}' is already in wishlist"}

    self.user_profile['wishlist'].append(game['name_normalized'])
    self._save_json(self.user_profile, USER_PROFILE_PATH)
    return {"confirmation": f"Added '{game['name']}' to your wishlist"}


  def remove_wishlist(self, title: str) -> dict:
    """Remove a game from wishlist.
    Args:
      title (str): title of game to remove.
    Returns:
      dict: confirm or error message
    """

    current_wishlist = self.user_profile.get('wishlist', [])
    # Find a match in current wishlist of the title
    match = next((g for g in current_wishlist if g == title), None)
        
    if not match:
      return {"error": f"'{title}' was not found in your wishlist"}
    
    self.user_profile['wishlist'].remove(match)
    self._save_json(self.user_profile, USER_PROFILE_PATH)
    return {"confirmation": f"Removed '{match}' from your wishlist"}
    
  

  def get_wishlist(self) -> dict:
    """Get user wishlist.
    Returns:
      dict: user wishlist.
    """
    wl = self.user_profile.get('wishlist', [])
    return {"wishlist": wl}


    def get_reviews(self, id: int, num_reviews: int = 50) -> List[str]:
        """
        Using API get up to date reviews on a game.

        Args:
            id (int): app id of game.
            num_reviews (int): maximum number of reviews to retrieve.

        Returns:
            List[str]: list of recent reviews for that given game (can be empty).
        """
        if id is None or id == 0:
            return []

        return fetch_steam_reviews(id, num_reviews=num_reviews)


if __name__ == "__main__":
  kb = KnowledgeBase()

  #print(kb.game_by_title("darkwood"))

  #slots = {"title": "postal 2", "info": "release_date"}
  #print(kb.get_game_info(**slots))

  #slots = {"genre": None, "price": None, "release_year": 2021, "platform": "linux", 
  #         "mode": "multiplayer", "required_age": None, "publisher": None, "developer": None}
  #print(kb.discover_game(**slots))

  #slots = {"title1": "terraria", "title2": "rust", "criteria": "price"}
  #print(kb.compare_games(**slots))

  #slots = {"name": "Alex"}
  #print(kb.get_friend_games(**slots))

  #slots = {"term": "adventure game"}
  #print(kb.get_term_explained(**slots))

  #slots = {"title": "stardew valley"}
  #print(kb.get_wishlist())


