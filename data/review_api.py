import steamreviews
from typing import List


def get_reviews(appid: int, num_reviews: int = 50) -> List[str]:
    """
    Given an appid downloads most recent reviews (just text) and returns
    a list of them (last `num_reviews`).

    Args:
        appid (int): Steam app id of the game.
        num_reviews (int): Maximum number of reviews to return.

    Returns:
        List[str]: List of review texts (can be empty if nothing is available or on error).
    """
    if appid is None or appid == 0:
        return []

    # Basic request parameters: most recent, English reviews
    request_params = {
        "language": "english",
        "review_type": "all",
        "purchase_type": "all",
        "num_per_page": 100,  # steamreviews uses pagination
    }

    try:
        # Download review data for this app id
        review_dict, query_count = steamreviews.download_reviews_for_app_id(
            appid, chosen_request_params=request_params
        )

        if "reviews" not in review_dict or not isinstance(review_dict["reviews"], list):
            return []

        texts: List[str] = []
        for entry in review_dict["reviews"]:
            text = entry.get("review", "")
            if text:
                texts.append(text)
            if len(texts) >= num_reviews:
                break

        return texts

    except Exception as e:
        # In production you might want to log this instead of printing
        print(f"[review_api.get_reviews] Error while fetching reviews for appid={appid}: {e}")
        return []
