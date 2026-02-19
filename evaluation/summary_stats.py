import os
import sys
import pandas as pd


def main(path: str | None = None):
    candidates = [path] if path else []
    candidates += [
        os.path.join("evaluation", "results_rag_labeled.csv"),
        os.path.join("evaluation", "results_rag.csv"),
        "results_rag_labeled.csv",
        "results_rag.csv",
    ]

    input_path = None
    for p in candidates:
        if p and os.path.exists(p):
            input_path = p
            break

    if input_path is None:
        print("No results file found. Expected evaluation/results_rag_labeled.csv or results_rag_labeled.csv")
        sys.exit(1)

    df = pd.read_csv(input_path)

    if "correctness" not in df.columns or "final_trust" not in df.columns:
        print("Required columns 'correctness' and 'final_trust' not found. Run labeling step first.")
        sys.exit(1)

    print("\nCorrectness distribution:")
    print(df["correctness"].value_counts())

    print("\nAverage trust by correctness:")
    print(df.groupby("correctness")["final_trust"].mean())

    print("\nHow many answers have trust >= 0.6?")
    print((df["final_trust"] >= 0.6).sum())


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
