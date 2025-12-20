import requests
from typing import List, Dict, Any, Optional


STEAM_APPREVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"


def _fetch_appreviews(
    appid: int,
    num_reviews: int = 50,
    language: str = "english",
) -> Optional[Dict[str, Any]]:
    """
    Low-level helper to call Steam appreviews endpoint once.
    """
    if appid is None or appid == 0:
        return None

    params = {
        "json": 1,
        "language": language,
        "filter": "recent",
        "num_per_page": max(0, min(num_reviews, 100)),  # Steam cap
        "purchase_type": "all",
    }

    try:
        resp = requests.get(STEAM_APPREVIEWS_URL.format(appid=appid), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception:
        # In production tu loggerais l’erreur, ici on renvoie None
        return None


def get_reviews(appid: int, num_reviews: int = 50) -> Dict[str, Any]:
    """
    Public API used by KnowledgeBase.
    Returns a lightweight summary + a small sample of review texts.

    Args:
        appid (int): Steam app id.
        num_reviews (int): maximum number of reviews to retrieve (for sampling).

    Returns:
        Dict[str, Any]: {
            "summary": {
                "total_positive": int,
                "total_negative": int,
                "total_reviews": int,
                "score_desc": str
            },
            "sample_reviews": [str, ...]
        }
        or empty structure if nothing available.
    """
    data = _fetch_appreviews(appid, num_reviews=num_reviews)

    if not data:
        return {
            "summary": None,
            "sample_reviews": [],
        }

    qs = data.get("query_summary", {}) or {}

    summary = {
        "total_positive": qs.get("total_positive", 0),
        "total_negative": qs.get("total_negative", 0),
        "total_reviews": qs.get("total_reviews", 0),
        "score_desc": qs.get("review_score_desc", None),
    }

    reviews_raw = data.get("reviews", []) or []
    sample_texts: List[str] = []
    for r in reviews_raw:
        txt = r.get("review")
        if isinstance(txt, str):
            sample_texts.append(txt.strip())
        if len(sample_texts) >= num_reviews:
            break

    return {
        "summary": summary,
        "sample_reviews": sample_texts,
    }
