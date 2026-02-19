import os
import sys
import pandas as pd


def find_input_path(provided: str | None) -> str | None:
    candidates = [provided] if provided else []
    candidates += [
        os.path.join("evaluation", "results_rag_labeled.csv"),
        os.path.join("evaluation", "results_rag.csv"),
        "results_rag_labeled.csv",
        "results_rag.csv",
    ]

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def main(input_path: str | None = None):
    path = find_input_path(input_path)
    if path is None:
        print("No results file found. Expected one of: evaluation/results_rag_labeled.csv, evaluation/results_rag.csv, results_rag_labeled.csv, results_rag.csv")
        sys.exit(1)

    df = pd.read_csv(path)

    if "correctness" not in df.columns:
        print("Column 'correctness' not found. Run evaluation/label_results.py first to add labels.")
        sys.exit(1)

    # support multiple possible trust column names
    for candidate in ("trust", "final_trust", "final_trust_score"):
        if candidate in df.columns:
            trust_col = candidate
            break
    else:
        print("No trust column found ('trust' or 'final_trust').")
        sys.exit(1)

    # Compute accuracy using weighted partial credit:
    # (correct + 0.5 * partial) / total
    total = len(df)
    num_correct = int((df["correctness"] == "correct").sum())
    num_partial = int((df["correctness"] == "partial").sum())

    accuracy = ((num_correct + 0.5 * num_partial) / total) if total > 0 else 0.0
    print("True RAG Accuracy:", round(accuracy, 3))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
