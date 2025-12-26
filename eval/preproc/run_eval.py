import json
from agent.agent import DialogueAgent


def normalize(x):
    """
    Normalise la sortie PREPROC :
    - trim
    - lower
    - suppression des espaces multiples
    """
    if isinstance(x, list):
        return [normalize(e) for e in x]
    if isinstance(x, str):
        return " ".join(x.strip().split()).lower()
    return x


def main():
    # Charge l’agent uniquement avec PREPROC
    agent = DialogueAgent(
        model={"default": "qwen3", "preproc": "qwen3"},
        device="cuda"
    )

    test_path = "eval/preproc/test.jsonl"
    total = 0
    correct = 0
    errors = []

    print(f"Loading test set from {test_path}...")

    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            user_input = item["input"]
            expected = item["expected"]

            # Normalisation du gold
            expected_norm = normalize(expected)

            # Exécution PREPROC
            out_raw = agent.preproc.generate(user_input)
            try:
                out = json.loads(out_raw)
            except Exception:
                out = []

            out_norm = normalize(out)

            # Comparaison
            is_correct = (out_norm == expected_norm)
            if is_correct:
                correct += 1
            else:
                errors.append({
                    "input": user_input,
                    "expected": expected_norm,
                    "got": out_norm
                })

            total += 1

    # Résultats
    accuracy = correct / total if total > 0 else 0

    print("\n==================== PREPROC EVALUATION ====================")
    print(f"Total examples : {total}")
    print(f"Correct        : {correct}")
    print(f"Accuracy       : {accuracy:.2%}")
    print("============================================================\n")

    if errors:
        print("❌ ERRORS:")
        for e in errors[:20]:  
            print("\n---")
            print("Input:    ", e["input"])
            print("Expected: ", e["expected"])
            print("Got:      ", e["got"])

        print(f"\n({len(errors)} errors total — showing first 20)")
    else:
        print("🎉 No errors — perfect score!")


if __name__ == "__main__":
    main()
