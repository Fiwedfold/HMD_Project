from agent.dst import DST
from models.model import ModelLoader, LLMTask
from data.kb import KnowledgeBase
from collections import deque
from typing import Deque, Dict, Optional, Union
import yaml
import os
import json
import re

class DialogueAgent:
  def __init__(self, model: Dict[str, str], device: str = "cuda", n_exchanges: int = 3) -> None:
    """Initialize dialogue agent.
    Args:
      model (Dict[str, str]): model names to load for each component.
      device (str): device where to run the model on.
      n_exchanges (int): number of exchanges to keep in conversation history.
    """
    self.model_name = model
    self.device = device
    self.n_exchanges = n_exchanges
    
    # Load knowledge base
    self.kb = KnowledgeBase()

    # Load model
    # Model loader cache
    self.loaders = {}
    #model_loader = ModelLoader(self.model_name, self.device)

    # Load prompts
    self.system_prompt = self._load_prompt()

    # Instantiate components
    self.preproc = LLMTask(self._get_loader("preproc"), self.system_prompt["preproc"]["prompt"])
    self.nlu = LLMTask(self._get_loader("nlu"), self.system_prompt["nlu"]["prompt"])
    self.dm = LLMTask(self._get_loader("dm"), self.system_prompt["dm"]["prompt"]["main"])
    self.nlg = LLMTask(self._get_loader("nlg"), self.system_prompt["nlg"]["prompt"]["main"])
    self.sa = LLMTask(self._get_loader("sa"), self.system_prompt["sa"]["prompt"])

    # Create Dialogue State Tracker
    self.dst = DST()

    # Conversation history
    max_len = self.n_exchanges * 2
    self.history: Deque[Dict[str, str]] = deque(maxlen=max_len)

  def _get_loader(self, component: str) -> ModelLoader:
    """Given model choices from different components, give the right model for the right component.
    Args:

    """
    default_model = self.model_name.get("default")
    model_name = self.model_name.get(component, default_model)
    if not model_name:
      raise ValueError(f"model_name dict must contain a 'default' key or a key for '{component}'")

    # Load model only it is not already loaded
    if model_name not in self.loaders:
      self.loaders[model_name] = ModelLoader(model_name, self.device)
    return self.loaders[model_name]

  def _load_prompt(self) -> dict:
    """Load a yaml files and get system prompts.
    Returns:
      dict: dict containing the prompts.
    """
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
  

  def get_review_sa(self, reviews: dict) -> dict:
    """
    Compute sentiment analysis on a batch of reviews.
    reviews is expected to be a dict:
    {
        "summary": {...},
        "sample_reviews": [str, ...]
    }
    """
    # Initialize counters
    report = {"positive": 0, "negative": 0, "neutral": 0}

    if not reviews or "sample_reviews" not in reviews:
        return report

    for review in reviews["sample_reviews"]:
        if not isinstance(review, str):
            continue

        label = self.sa.generate(review)

        # Normalize label (remove brackets, quotes, spaces)
        label = label.strip().lower()
        label = label.replace("[", "").replace("]", "")
        label = label.replace('"', "").replace("'", "")
        label = label.strip()

        # Map unexpected outputs to neutral
        if label not in report:
            label = "neutral"

        report[label] += 1

    return report



  def get_knowledge(self, nba: str, ds: dict) -> dict:
    """Get external knowledge given nba and ds.
    Args:
      nba (str): next best action.
      ds (dict): dialogue state,
    Returns:
      dict: external knowledge inside a json object.
    """
    pattern = r'^([a-zA-Z_]\w*)\s*\('
    match = re.match(pattern, nba)
    # Error if dm gave an incorrect output
    if not match:
      return {}
    action_name = match.group(1)
    # If action does not require knowledge, agent does not request it.
    if action_name in ["get_info", "fallback"]:
      return {}
    
    # Handle different intents
    intent = ds.get("intent")
    slots = ds.get("slots", {})

    # Based on different intents different data is requested
    match intent:
      case "get_game_info":
        # Default info is summary if not specified
        title = slots.get("title")
        info = slots.get("info", "summary")
        data = self.kb.get_game_info(title, info)
        # If data has reviews, the sa component will be called
        if "review" in data:
          sa = self.get_review_sa(data["review"])
          data["review"] = sa
      case "discover_game":
        data = self.kb.discover_game(**slots)
      case "compare_games":
        data = self.kb.compare_games(**slots)
        if "review" in data:
          # Execute sa if reviews are requested
          for title, review_list in data["review"].items():
            sa = self.get_review_sa(review_list)
            data["review"][title] = sa
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


  def chat(self, user_input: str) -> str:
    """Chat with the model giving a user input.
    Args:
      user_input (str): user input.
    Returns:
      str: assistant response.
    """
    print(user_input, list(self.history))

    # Keep track if there are multiple intents
    multiple_intents = False

    # Go through the preprocess to split multiple intents
    preproc_out = self.preproc.generate(user_input)
    
    # Convert preproc_out from string to list
    split_input = json.loads(preproc_out)
    print("PREPROC->", split_input)

    # If there are multiple intents the agent will only attend to the last one
    # For better user experience the agent will know when there are multiple intents with a 
    # flag and also add to history other inputs for context
    if len(split_input) > 1:
      multiple_intents = True
      for input in split_input[:-1]:
        self.history.append({"role": "user", "content": input})
    
    # Handle the case where preproc discards entire input
    if len(split_input) == 0:
      nlu_input = user_input
    else:
      # Agent will always attend to last user intent
      nlu_input = split_input[-1]
    
    # Pass through nlu to get intents
    nlu_out = self.nlu.generate(nlu_input, list(self.history))

    print(f"NLU OUT->{nlu_out}")
    # Merge dialogue state and get new state
    self.dst.update_ds(nlu_out)
    ds = self.dst.get_ds()
    print(f"DST OUT->{ds}")


    # Get intent based prompt for dm
    intent_name = self.dst.ds["intent"]
    self.dm.change_system_prompt(self.system_prompt["dm"]["prompt"]["main"] + self.system_prompt["dm"]["prompt"][intent_name])

    # Pass through dm to get next best action
    nba = self.dm.generate(ds)
    print(f"DM OUT->{nba}")

    # Get external knowledge if needed
    ek = self.get_knowledge(nba, self.dst.ds)
    # If error in request, action becomes fallback()
    if "error" in ek:
      nba = "fallback()"
      # Error is not seen by the nlg
      ek = None

    # Get intent based prompt for dm
    self.nlg.change_system_prompt(self.system_prompt["nlg"]["prompt"]["main"] + self.system_prompt["nlg"]["prompt"][intent_name])

    # Prepare the nlg input and get natural language output
    nlg_input = f"NBA: {nba}\nDS: {self.dst.get_ds()}\nEK: {ek}\n MI: {multiple_intents}"
    print("NLG Input -> ", nlg_input)
    response = self.nlg.generate(nlg_input)

    # Update history
    self.history.append({"role": "user", "content": user_input})
    self.history.append({"role": "assistant", "content": response})

    return response
  



