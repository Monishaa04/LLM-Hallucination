import os
import pandas as pd


def assign_correctness(row):
    try:
        trust = float(row.get("final_trust", 0))
    except Exception:
        trust = 0.0
    try:
        sim = float(row.get("acc_sim", 0))
    except Exception:
        sim = 0.0

    if trust >= 0.75 and sim >= 0.6:
        return "correct"
    elif trust >= 0.5 or sim >= 0.4:
        return "partial"
    else:
        return "incorrect"


def assign_grounded(row):
    try:
        trust = float(row.get("final_trust", 0))
    except Exception:
        trust = 0.0
    try:
        coverage = float(row.get("sentence_coverage", 0))
    except Exception:
        coverage = 0.0
    try:
        overlap = float(row.get("concept_overlap", 0))
    except Exception:
        overlap = 0.0

    if trust >= 0.6 and coverage >= 0.5:
        return "yes"
    elif overlap >= 0.5 and coverage >= 0.4:
        return "yes"
    else:
        return "no"


def main():
    # prefer evaluation/results_rag.csv if it exists
    candidates = [
        os.path.join("evaluation", "results_rag.csv"),
        "results_rag.csv",
        os.path.join("evaluation", "results_rag.csv"),
    ]

    input_path = None
    for p in candidates:
        if os.path.exists(p):
            input_path = p
            break

    if input_path is None:
        print("No results_rag.csv found. Please run evaluation and place the file at evaluation/results_rag.csv or results_rag.csv")
        return

    df = pd.read_csv(input_path)

    df["correctness"] = df.apply(assign_correctness, axis=1)
    df["grounded"] = df.apply(assign_grounded, axis=1)
    # --- Abstention based on final_trust ---
    def assign_abstention(row):
        try:
            trust = float(row.get("final_trust", row.get("trust", 0)))
        except Exception:
            trust = 0.0
        return "yes" if trust < 0.4 else "no"

    df["abstention"] = df.apply(assign_abstention, axis=1)

    out_dir = os.path.dirname(input_path) or "."
    out_path = os.path.join(out_dir, "results_rag_labeled.csv")
    df.to_csv(out_path, index=False)

    print("Labeled file created:", out_path)


if __name__ == "__main__":
    main()
