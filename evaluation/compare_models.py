import os
import pandas as pd

models = [
    "rag",
    "llama-3.1-8b-instant",
    "openai_gpt-oss-20b",
    "moonshotai_kimi-k2-instruct-0905",
    "meta-llama_llama-4-maverick-17b-128e-instruct",
    "qwen_qwen3-32b"
]

for model in models:
    file_path = f"evaluation/results/results_{model}_labeled.csv"

    if not os.path.exists(file_path):
        print(f"Missing labeled file for model: {model}")
        continue

    df = pd.read_csv(file_path)

    required_cols = ["correctness", "abstention"]
    if not all(col in df.columns for col in required_cols):
        print(f"Model {model}: required columns missing (correctness, abstention)")
        continue

    total = len(df)

    # # Accuracy = fully correct answers only
    # correct = (df["correctness"] == "correct").sum()
    correct = (df["correctness"] == "correct").sum()
    partial = (df["correctness"] == "partial").sum()

    weighted_accuracy = (correct + 0.5 * partial) / total


    # Hallucination definition:
    # Incorrect answer AND did not abstain
    halluc = (
        (df["correctness"] == "incorrect") &
        (df["abstention"] == "no")
    ).sum()

    # Abstention rate
    abstain = (df["abstention"] == "yes").sum()

    # print(f"\nModel: {model}")
    # print("Total Questions:", total)
    # print("Accuracy:", round(correct / total, 3))
    # print("Hallucination Rate:", round(halluc / total, 3))
    # print("Abstention Rate:", round(abstain / total, 3))
    print(f"\nModel: {model}")
    print("Total Questions:", total)
    print("Weighted Accuracy:", round(weighted_accuracy, 3))
    print("Hallucination Rate:", round(halluc / total, 3))
    print("Abstention Rate:", round(abstain / total, 3))
