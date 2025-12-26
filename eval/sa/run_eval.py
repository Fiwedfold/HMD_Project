import os
import sys
import json

# -------------------------------------------------------------------
# 1. Add project root to PYTHONPATH (safe for Colab & local)
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))  # /eval/sa → /eval → /root

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agent.agent import DialogueAgent


def normalize(x):
    """
    Normalize SA output:
    - lowercase
    - trim
    - ensure list structure
    """
    if isinstance(x, list):
        return [normalize(e) for e in x]
    if isinstance(x, str):
        return x.strip().lower()
    return x


def main():
    # Load agent with SA model
    agent = DialogueAgent(
        model={"default": "qwen3", "sa": "qwen3"},
        device="cuda"
    )

    test_path = os.path.join("eval", "sa", "test.jsonl")
    total = 0
    correct = 0
    errors = []

    print(f"Loading SA test set from {test_path}...")

    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            review = item["input"]
            expected = item["expected"]

            expected_norm = normalize(expected)

            # Run SA
            out_raw = agent.sa.generate(review)

            try:
                out = json.loads(out_raw)
            except Exception:
                out = ["neutral"]  # fallback

            out_norm = normalize(out)

            # Compare
            is_correct = (out_norm == expected_norm)
            if is_correct:
                correct += 1
            else:
                errors.append({
                    "input": review,
                    "expected": expected_norm,
                    "got": out_norm
                })

            total += 1

    # Results
    accuracy = correct / total if total > 0 else 0

    print("\n==================== SA EVALUATION ====================")
    print(f"Total examples : {total}")
    print(f"Correct        : {correct}")
    print(f"Accuracy       : {accuracy:.2%}")
    print("=======================================================\n")

    if errors:
        print("❌ ERRORS:")
        for e in errors[:20]:  # show first 20
            print("\n---")
            print("Input:    ", e["input"])
            print("Expected: ", e["expected"])
            print("Got:      ", e["got"])

        print(f"\n({len(errors)} errors total — showing first 20)")
    else:
        print("🎉 No errors — perfect score!")


if __name__ == "__main__":
    main()
