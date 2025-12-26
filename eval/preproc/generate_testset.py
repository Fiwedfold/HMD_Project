import json
from typing import List, Dict


def build_preproc_testset() -> List[Dict]:
    """
    Returns a list of test examples for PREPROC.
    Each example has:
      - input: raw user utterance
      - expected: list of normalized sub-utterances (what PREPROC should output)
    """

    examples: List[Dict] = []

    # --- Comparaison de jeux (reviews, prix, etc.) ---
    examples.append({
        "input": "Between Elden Ring and Dark Souls II which one is the best?",
        "expected": ["Compare the reviews of Elden Ring and Dark Souls II"]
    })

    examples.append({
        "input": "Which one is cheaper between Elden Ring and Dark Souls Remastered?",
        "expected": ["Compare the price of Elden Ring and Dark Souls Remastered"]
    })

    examples.append({
        "input": "Compare the reviews and the price of Hades and Dead Cells.",
        "expected": [
            "Compare the reviews of Hades and Dead Cells",
            "Compare the price of Hades and Dead Cells"
        ]
    })

    # --- Infos sur un seul jeu ---
    examples.append({
        "input": "Is Elden Ring multiplayer?",
        "expected": ["Get the mode of Elden Ring"]
    })

    examples.append({
        "input": "What is the price of Hollow Knight on PC?",
        "expected": ["Get the price of Hollow Knight on PC"]
    })

    examples.append({
        "input": "Tell me about the genre and publisher of Celeste.",
        "expected": [
            "Get the genre of Celeste",
            "Get the publisher of Celeste"
        ]
    })

    # --- Découverte de jeux ---
    examples.append({
        "input": "Show me some cheap indie roguelike games on PC.",
        "expected": [
            "Discover cheap indie roguelike games on PC"
        ]
    })

    examples.append({
        "input": "Find games like Dark Souls but less punishing.",
        "expected": [
            "Discover games similar to Dark Souls but less punishing"
        ]
    })

    # --- Multiples intentions dans une seule phrase ---
    examples.append({
        "input": "First, tell me the price of Elden Ring, then compare it with Dark Souls III.",
        "expected": [
            "Get the price of Elden Ring",
            "Compare the price of Elden Ring and Dark Souls III"
        ]
    })

    examples.append({
        "input": "What is the genre of Hades and is it multiplayer?",
        "expected": [
            "Get the genre of Hades",
            "Get the mode of Hades"
        ]
    })

    # --- Cas plus bruités / vagues ---
    examples.append({
        "input": "Any good soulslike cheaper than Elden Ring?",
        "expected": [
            "Discover soulslike games cheaper than Elden Ring"
        ]
    })

    examples.append({
        "input": "Is Dark Souls Remastered worth it compared to Elden Ring?",
        "expected": [
            "Compare the reviews of Dark Souls Remastered and Elden Ring"
        ]
    })

    return examples


def main() -> None:
    testset = build_preproc_testset()

    output_path = "eval/preproc/test.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in testset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"PREPROC test set written to {output_path} with {len(testset)} examples.")


if __name__ == "__main__":
    main()
