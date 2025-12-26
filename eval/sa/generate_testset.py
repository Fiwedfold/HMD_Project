import json
import random
import os

OUTPUT_PATH = "eval/sa/test.jsonl"
N = 300

# -------------------------------------------------------------------
# 1. Base templates for each sentiment
# -------------------------------------------------------------------

positive_templates = [
    "I absolutely loved this game, it was fantastic.",
    "One of the best experiences I've had in years.",
    "Amazing gameplay and beautiful graphics.",
    "I enjoyed every minute of it.",
    "This game exceeded all my expectations.",
    "A masterpiece, truly unforgettable.",
    "The combat feels great and the story is wonderful.",
    "I can't stop playing, it's so good.",
    "Brilliant design and super fun to play.",
    "This is easily a 10/10 for me."
]

negative_templates = [
    "This game is terrible and a complete waste of money.",
    "I regret buying this, it's awful.",
    "The gameplay is clunky and frustrating.",
    "I hate how buggy this game is.",
    "The worst experience I've had in a long time.",
    "Nothing works properly, it's a disaster.",
    "The story is boring and the combat is awful.",
    "I uninstalled it after one hour, horrible.",
    "Terrible performance and constant crashes.",
    "This is trash, do not buy it."
]

neutral_templates = [
    "It's okay, nothing special.",
    "Not bad, not great, just average.",
    "Some parts are good, others not so much.",
    "I don't really feel strongly about this game.",
    "It's fine, but I wouldn't recommend it strongly.",
    "Mixed feelings overall.",
    "The game is decent but forgettable.",
    "I guess it's alright.",
    "Hard to say if I like it or not.",
    "It's acceptable, nothing more."
]

# -------------------------------------------------------------------
# 2. Variation helpers
# -------------------------------------------------------------------

def add_variation(sentence):
    """Adds small natural variations to avoid duplicates."""
    variations = [
        "",
        " Honestly.",
        " To be honest.",
        " Really.",
        " Overall.",
        " In my opinion.",
        " Just saying.",
        " For real.",
        " Seriously.",
        " No joke."
    ]
    return sentence + random.choice(variations)


def generate_example(sentiment):
    """Generate one example with the correct expected label."""
    if sentiment == "positive":
        base = random.choice(positive_templates)
    elif sentiment == "negative":
        base = random.choice(negative_templates)
    else:
        base = random.choice(neutral_templates)

    return {
        "input": add_variation(base),
        "expected": [sentiment]
    }


# -------------------------------------------------------------------
# 3. Generate dataset
# -------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    dataset = []

    # Balanced distribution: 100 per class
    sentiments = ["positive"] * (N // 3) + \
                 ["negative"] * (N // 3) + \
                 ["neutral"] * (N - 2 * (N // 3))

    random.shuffle(sentiments)

    for s in sentiments:
        dataset.append(generate_example(s))

    # Write JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(dataset)} SA test examples → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
